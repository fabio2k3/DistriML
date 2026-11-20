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

from distriml.common.config import Config
from distriml.common.interfaces import Assignment, DBService


class AttemptAborted(Exception):
    """El attempt fue revocado (RENEW_REVOKED / CKPT_REJECTED) o el lease local venció:
    se abandona sin reportar error."""


def download_artifact(db: DBService, sha256: str, abort: threading.Event, retries: int = 5) -> bytes:
    """TODO(WK-2) Descarga con db.get_artifact y VERIFICA el sha256 (ids.sha256_hex).

    Reintentar con espera creciente (la DB puede devolver 503). Si tras los reintentos no hay
    artefacto, o el hash no coincide: levantar una excepción que `run_attempt` convierta en
    db.report_unavailable(...), que revoca el attempt con ARTIFACT_UNAVAILABLE (no cuenta como fallo).
    """
    raise NotImplementedError("TODO(WK-2)")


class LeaseKeeper(threading.Thread):
    """Hilo que renueva el lease del attempt cada cfg.renew_interval_ms, con independencia de lo que
    dure cada batch.

    TODO(WK-3)
      - Cada intervalo: r = db.renew_lease(worker_id, attempt_id)
          RENEW_REVOKED -> `abort.set()` y terminar
          503 / error de red -> seguir intentando; el lease local se mide DESDE EL ENVÍO de la última
          renovación que tuvo éxito (reloj monotónico: time.monotonic()).
      - Si pasa lease_duration_ms sin éxito -> `abort.set()` (la DB ya habrá dado por perdido el attempt).
      - Parar cuando `done` esté activado (el attempt terminó).
    """

    def __init__(self, db: DBService, worker_id: str, attempt_id: str, cfg: Config,
                 abort: threading.Event, done: threading.Event):
        super().__init__(name=f"lease-{attempt_id[:8]}", daemon=True)
        self.db, self.worker_id, self.attempt_id, self.cfg = db, worker_id, attempt_id, cfg
        self.abort, self.done = abort, done

    def run(self) -> None:
        raise NotImplementedError("TODO(WK-3)")


def run_attempt(a: Assignment, worker_id: str, db: DBService, cfg: Config) -> str:
    """Ejecuta un attempt completo. Devuelve "COMPLETED" | "ABORTED" | "FAILED".

    TODO(WK-2) Preparación:
      1. Lanzar LeaseKeeper (hilo) con un `abort = threading.Event()`.
      2. data = download_artifact(dataset); X, y = dataset.load_for_training(data, DatasetMeta.from_dict(a.dataset_meta), task)
      3. Si a.next_batch > 1: state = deserialize_state(download_artifact(a.checkpoint_sha256));
         si no: state = algorithms.init_state(a.algorithm, a.hyperparams, meta).

    TODO(WK-4) Bucle de batches, para k = a.next_batch … a.total_batches:
      - si abort está activo -> AttemptAborted (descartar el progreso no confirmado)
      - idx = batching.batch_indices(n_rows, hp["batch_size"], hp["seed"], k)
      - metrics = algorithms.train_batch(state, a.algorithm, X[idx], y[idx])
      - si k % cfg.k_ckpt == 0 o k == total_batches: subir el checkpoint en dos pasos:
            blob = serialize_state(state); sha = sha256_hex(blob)
            db.put_artifact(sha, blob)                                          (paso a)
            r = db.upload_checkpoint(worker_id, attempt_id, k, sha, metrics)    (paso b)
            CKPT_CONFIRMED -> seguir | CKPT_RETRY -> reintentar con espera | CKPT_REJECTED -> AttemptAborted

    TODO(WK-5) Cierre: final = algorithms.evaluate_full(state, algorithm, X, y);
      subir el modelo final con put_artifact y llamar db.complete_job(worker_id, attempt_id, sha, final).

    TODO(WK-6) Errores:
      - excepción de entrenamiento (datos malformados, etc.) -> db.report_failure(worker_id, attempt_id, str(exc)); "FAILED"
      - artefacto no disponible -> db.report_unavailable(...)
      - SIEMPRE: parar el LeaseKeeper (done.set()) al terminar, por cualquier camino.
    """
    raise NotImplementedError("TODO(WK-2..6)")
