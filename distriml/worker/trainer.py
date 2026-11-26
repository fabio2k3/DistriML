"""Ejecución de un attempt: descarga, entrenamiento por batches, checkpoints y cierre.

Herramientas ya hechas en common/ (no reescribir):
    dataset.load_for_training(data, meta, task)          -> (X estandarizada, y)
    batching.batch_indices(n_rows, batch_size, seed, k)  -> filas del batch k (determinista)
    algorithms.init_state / train_batch / evaluate_full / serialize_state / deserialize_state
    ids.sha256_hex

Referencia de comportamiento en el caso sin fallos: tests/helpers/reference_worker.py (~40 líneas,
sin leases ni manejo de errores). Lo que falta aquí: leases, abortos, errores y reintentos de descarga.

Registro: usar un logger `get_logger(f"worker.{worker_id}")` y escribir una línea por evento
(descarga, batch k de N, checkpoint confirmado/rechazado, lease renovado/perdido, abort, error, fin).
"""
from __future__ import annotations

import threading
import numpy as np
from dataclasses import dataclass
import time

from distriml.common import algorithms, dataset, batching, constants as C
from distriml.common.config import Config
from distriml.common.interfaces import Assignment, DBService
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import ContractError, unavailable
from distriml.common.ids import sha256_hex
from distriml.common.log import get_logger

_log = get_logger("worker.trainer")

class AttemptAborted(Exception):
    """El attempt fue revocado (RENEW_REVOKED / CKPT_REJECTED) o el lease local venció:
    se abandona sin reportar error."""

DOWNLOAD_RETRIES = 5        # reintentos de descarga de un artefacto
DOWNLOAD_BASE_S = 0.5
DOWNLOAD_MAX_S = 8.0
MAX_ERROR_LEN = 300         # longitud máxima del mensaje enviado en report_failure

class ArtifactUnavailable(Exception):
    """No se pudo obtener un binario íntegro: `run_attempt` lo convertirá en db.report_unavailable (WK-6)."""

    def __init__(self, sha256: str, detail: str = ""):
        super().__init__(f"artefacto {sha256[:12]}… no disponible: {detail}")
        self.sha256 = sha256

def download_artifact(db: DBService, sha256: str, abort: threading.Event, retries: int | None = None, base_delay: float | None = None, max_delay: float | None = None) -> bytes:
    """Descarga con db.get_artifact y VERIFICA el sha256.

    - 503 de la DB o hash que no coincide: se reintenta con espera creciente (base_delay · 2^i, tope max_delay).
    - Otro error del contrato (400/404): no tiene sentido reintentar -> ArtifactUnavailable.
    - Si `abort` se activa (antes de un intento o durante una espera) -> AttemptAborted.
    - Agotados los `retries` reintentos -> ArtifactUnavailable.
    """

    retries = DOWNLOAD_RETRIES if retries is None else retries
    base_delay = DOWNLOAD_BASE_S if base_delay is None else base_delay
    max_delay = DOWNLOAD_MAX_S if max_delay is None else max_delay
    last = "sin intentos"

    for i in range(retries + 1):
        if abort.is_set():
            raise AttemptAborted("abortado durante la descarga")
        try:
            data = db.get_artifact(sha256)
        except ContractError as exc:
            if exc.status != 503:
                raise ArtifactUnavailable(sha256, f"{exc.status} {exc.reason}") from exc
            last = f"{exc.status} {exc.reason}"
        else:
            if sha256_hex(data) == sha256:
                return data
            last = "el hash del contenido no coincide"
        if i == retries:
            break
        delay = min(base_delay * (2 ** i), max_delay)
        _log.warning("descarga de %s… fallida (%s); reintento %d/%d en %.1fs", sha256[:12], last, i + 1, retries, delay)
        if abort.wait(delay):
            raise AttemptAborted("abortado durante la espera de un reintento de descarga")
    raise ArtifactUnavailable(sha256, last)


@dataclass
class Prepared:
    """Todo lo necesario para empezar a entrenar un attempt."""
    meta: DatasetMeta
    task: str
    X: np.ndarray
    y: np.ndarray
    state: dict


def prepare_attempt(a: Assignment, db: DBService, abort: threading.Event, **download_kwargs) -> Prepared:
    """Descarga el dataset (y el checkpoint si continúa) y deja listo el estado."""
    meta = DatasetMeta.from_dict(a.dataset_meta)
    task = algorithms.task_of(a.algorithm)
    data = download_artifact(db, a.dataset_sha256, abort, **download_kwargs)
    X, y = dataset.load_for_training(data, meta, task)
    if a.next_batch > 1:
        if not a.checkpoint_sha256:
            raise ValueError(f"next_batch={a.next_batch} pero la asignación no trae checkpoint")
        state = algorithms.deserialize_state(download_artifact(db, a.checkpoint_sha256, abort, **download_kwargs))
    else:
        state = algorithms.init_state(a.algorithm, a.hyperparams, meta, dataset.class_names_of(data))
    return Prepared(meta, task, X, y, state)


# Reintentos ante 503 (módulo-nivel para poder acortarlos en los tests)
RETRY_BASE_S = 0.2          # espera inicial; se duplica en cada reintento
RETRY_MAX_S = 4.0
RETRY_COUNT = 8             # reintentos ante 503 de la DB
CKPT_RETRY_COUNT = 10       # veces que se repite un checkpoint si la DB contesta CKPT_RETRY


def _retry_unavailable(fn, abort: threading.Event, what: str):
    """Ejecuta `fn()` reintentando con espera creciente mientras la DB responda 503.

    Se interrumpe con AttemptAborted si `abort` se activa (antes de un intento o durante una espera).
    Otros errores del contrato, o agotar los reintentos, se propagan tal cual.
    """
    for i in range(RETRY_COUNT + 1):
        if abort.is_set():
            raise AttemptAborted(f"abortado antes de {what}")
        try:
            return fn()
        except ContractError as exc:
            if exc.status != 503 or i == RETRY_COUNT:
                raise
            delay = min(RETRY_BASE_S * (2 ** i), RETRY_MAX_S)
            _log.warning("%s: la DB respondió %s %s; reintento %d/%d en %.1fs",
                         what, exc.status, exc.reason, i + 1, RETRY_COUNT, delay)
        if abort.wait(delay):
            raise AttemptAborted(f"abortado durante la espera de {what}")


class LeaseKeeper(threading.Thread):
    """Hilo que renueva el lease del attempt cada cfg.renew_interval_ms, con independencia de lo que
    dure cada batch.

    - RENEW_REVOKED -> `abort.set()` y termina (el attempt ya no es el vigente).
    - 503 / error de red / cualquier excepción -> sigue intentando en el siguiente intervalo.
    - El lease local se mide DESDE EL ENVÍO de la última renovación con éxito (time.monotonic()):
      si pasan lease_duration_ms sin éxito -> `abort.set()`. Así el worker se retira ANTES de que la DB
      lo dé por perdido (lease + epsilon) y no queda un attempt «zombi» escribiendo.
    - Termina cuando `done` se activa (el attempt acabó).
    """

    def __init__(self, db: DBService, worker_id: str, attempt_id: str, cfg: Config, abort: threading.Event, done: threading.Event):
        super().__init__(name=f"lease-{attempt_id[:8]}", daemon=True)
        self.db, self.worker_id, self.attempt_id, self.cfg = db, worker_id, attempt_id, cfg
        self.abort, self.done = abort, done
        self.log = get_logger(f"worker.{worker_id}")


    def run(self) -> None:
        interval = self.cfg.renew_interval_ms / 1000
        lease = self.cfg.lease_duration_ms / 1000
        last_ok = time.monotonic()      # la DB abre el lease al crear el attempt: arranca «ahora»
        
        while not self.done.is_set():
            sent = time.monotonic()
            result = None
            try:
                result = self.db.renew_lease(self.worker_id, self.attempt_id)
            except ContractError as exc:
                self.log.warning("renovación de lease fallida (%s %s)", exc.status, exc.reason)
            except Exception:  # noqa: BLE001  (el hilo no debe morir: un fallo de red es una renovación perdida)
                self.log.exception("error inesperado al renovar el lease")
            
            if result == C.RENEW_REVOKED:
                self.log.warning("attempt %s revocado por la DB: aborto", self.attempt_id[:8])
                self.abort.set()
                return
            
            if result == C.RENEW_CONTINUE:
                last_ok = sent
                self.log.debug("lease renovado (attempt %s)", self.attempt_id[:8])
            
            remaining = last_ok + lease - time.monotonic()
            if remaining <= 0:
                self.log.error("lease local vencido (%.1fs sin renovar): aborto el attempt %s",
                               lease, self.attempt_id[:8])
                self.abort.set()
                return
            
            self.done.wait(min(interval, remaining))      # despierta antes si el lease está a punto de vencer


def _confirm_checkpoint(db: DBService, worker_id: str, attempt_id: str, k: int, state: dict, metrics: dict, abort: threading.Event) -> None:
    """Checkpoint en dos pasos: a) put_artifact del estado; b) upload_checkpoint.

    CKPT_CONFIRMED -> listo | CKPT_RETRY -> repite (a) y (b) con espera | CKPT_REJECTED -> AttemptAborted.
    Si `abort` está activo NO se llama a la DB: un attempt revocado no escribe.
    """
    blob = algorithms.serialize_state(state)
    sha = sha256_hex(blob)

    for i in range(CKPT_RETRY_COUNT + 1):
        _retry_unavailable(lambda: db.put_artifact(sha, blob), abort, f"guardar el checkpoint {k}")
        r = _retry_unavailable(lambda: db.upload_checkpoint(worker_id, attempt_id, k, sha, metrics),
                               abort, f"confirmar el checkpoint {k}")
        if r == C.CKPT_CONFIRMED:
            return
        if r == C.CKPT_REJECTED:
            raise AttemptAborted(f"checkpoint {k} rechazado: el attempt ya no es el vigente")
        _log.warning("checkpoint %d: la DB pide reintentar (aún no ve el binario) %d/%d", k, i + 1, CKPT_RETRY_COUNT)
        if abort.wait(min(RETRY_BASE_S * (2 ** i), RETRY_MAX_S)):
            raise AttemptAborted(f"abortado durante la espera del checkpoint {k}")
        
    raise unavailable("no_confirm", f"la DB no confirmó el checkpoint {k} tras {CKPT_RETRY_COUNT} intentos")



def _report(log, abort: threading.Event, what: str, fn) -> None:
    """Informa a la DB de un fallo del attempt (mejor esfuerzo): reintenta los 503 y, si no lo consigue,
    solo lo registra: el lease acabará venciendo y la DB reasignará el job igualmente.
    """
    if abort.is_set():
        return
    try:
        _retry_unavailable(fn, abort, what)
    except AttemptAborted:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("no se pudo %s (%s); el lease vencerá y el job se reasignará", what, exc)


def _fail(log, abort: threading.Event, db: DBService, worker_id: str, a: Assignment, exc: Exception) -> str:
    """Error del propio entrenamiento: se reporta con report_failure (cuenta para max_attempts)."""
    if abort.is_set():               # ya estaba revocado: el error es una consecuencia, no el motivo
        log.info("attempt %s abortado (con error posterior: %s)", a.attempt_id[:8], exc)
        return "ABORTED"
    
    msg = " ".join(f"{type(exc).__name__}: {exc}".split())[:MAX_ERROR_LEN]     # una línea, acotada
    log.error("attempt %s FALLIDO: %s", a.attempt_id[:8], msg, exc_info=exc)
    _report(log, abort, "reportar el fallo", lambda: db.report_failure(worker_id, a.attempt_id, msg))

    return "FAILED"



def run_attempt(a: Assignment, worker_id: str, db: DBService, cfg: Config, abort: threading.Event | None = None) -> str:
    """Ejecuta un attempt completo. Devuelve "COMPLETED" | "ABORTED" | "FAILED".

    WK-2/WK-3  Lanza el LeaseKeeper ANTES de descargar (la DB cuenta el lease desde que crea el attempt),
               y prepara datos y estado (`prepare_attempt`).
    WK-4       Bucle de batches k = next_batch … total_batches, con checkpoint cada cfg.k_ckpt batches y en el último.
    WK-5       Cierre: evaluate_full, put_artifact del modelo y complete_job.
    WK-6       Errores y abortos. Qué se le dice a la DB según el origen del problema:
                 - AttemptAborted (revocado, cancelado, lease vencido, checkpoint rechazado) -> nada; "ABORTED".
                 - ArtifactUnavailable (no se pudo obtener un binario íntegro) -> report_unavailable; "ABORTED"
                   (no cuenta para max_attempts: no es culpa del entrenamiento).
                 - 503 que se agotó tras los reintentos (la DB no responde) -> nada: la DB no puede oírnos
                   y el lease vencerá (WORKER_LOST); "ABORTED".
                 - Cualquier otra excepción (datos corruptos, error del algoritmo, 400…) -> report_failure; "FAILED".
               Un attempt ya revocado NO reporta nada (la DB lo rechazaría y podría pisar al sucesor).
    SIEMPRE    Se para el LeaseKeeper (done.set()) por cualquier camino.

    `abort` (opcional) es la señal de aborto del attempt: la activan el LeaseKeeper (revocado / lease vencido)
    y, desde fuera, `Worker.stop()` (retirada ordenada, hito 9). Si no se pasa, se crea una propia.
    """
    log = get_logger(f"worker.{worker_id}")
    abort = abort if abort is not None else threading.Event()
    done = threading.Event()
    keeper = LeaseKeeper(db, worker_id, a.attempt_id, cfg, abort, done)
    keeper.start()

    try:
        p = prepare_attempt(a, db, abort)
        log.info("attempt %s: %d filas, batches %d..%d de %d",
                 a.attempt_id[:8], len(p.y), a.next_batch, a.total_batches, a.total_batches)
        hp = a.hyperparams

        for k in range(a.next_batch, a.total_batches + 1):
            if abort.is_set():
                raise AttemptAborted("revocado o lease vencido; se descarta el progreso no confirmado")
            idx = batching.batch_indices(p.meta.n_rows, hp["batch_size"], hp["seed"], k)
            metrics = algorithms.train_batch(p.state, a.algorithm, p.X[idx], p.y[idx])
            if k % cfg.k_ckpt == 0 or k == a.total_batches:
                _confirm_checkpoint(db, worker_id, a.attempt_id, k, p.state, metrics, abort)
                log.debug("batch %d/%d confirmado (score=%.3f)", k, a.total_batches, metrics["score"])

        final = algorithms.evaluate_full(p.state, a.algorithm, p.X, p.y)
        blob = algorithms.serialize_state(p.state)
        sha = sha256_hex(blob)
        _retry_unavailable(lambda: db.put_artifact(sha, blob), abort, "guardar el modelo final")
        res = _retry_unavailable(lambda: db.complete_job(worker_id, a.attempt_id, sha, final), abort, "cerrar el job")

        if res != C.COMPLETE_OK:
            # Si la respuesta de un intento anterior se perdió, la DB puede haber cerrado ya el job: no es un error.
            log.warning("complete_job devolvió %s: el attempt %s ya no es el vigente", res, a.attempt_id[:8])
            return "ABORTED"
        log.info("job %s COMPLETADO (attempt %s, %s)", a.job_id[:8], a.attempt_id[:8], final)
        return "COMPLETED"
    except AttemptAborted as exc:
        log.info("attempt %s abortado: %s", a.attempt_id[:8], exc)
        return "ABORTED"
    except ArtifactUnavailable as exc:
        log.error("attempt %s: %s", a.attempt_id[:8], exc)
        if not abort.is_set():
            _report(log, abort, "avisar de que falta un artefacto",
                    lambda: db.report_unavailable(worker_id, a.attempt_id, exc.sha256))
        return "ABORTED"
    except ContractError as exc:
        if exc.status == 503:        # la DB no respondió tras los reintentos
            log.error("attempt %s: la DB no responde (%s %s); lo abandono y dejo que venza el lease",
                      a.attempt_id[:8], exc.status, exc.reason)
            return "ABORTED"
        return _fail(log, abort, db, worker_id, a, exc)
    except Exception as exc:  # noqa: BLE001
        return _fail(log, abort, db, worker_id, a, exc)
    finally:
        done.set()
        keeper.join(timeout=2.0)