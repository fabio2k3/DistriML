"""Servicio de base de datos: implementa `DBService` (common/interfaces.py).

Corre como un único actor de Ray (db/main.py). Un actor ejecuta una llamada a la vez, así que las
operaciones quedan serializadas y no hacen falta bloqueos entre ellas. Persistencia: tablas CSV
(db/tables.py) y artefactos binarios por hash (common/artifacts.py).

Responsabilidades:
  - ids derivados e idempotencia por (client_id, client_request_id)
  - validaciones del catálogo y del dataset al crear un job
  - attempts con su base, cadena de checkpoints y lease
  - presupuestos de reintento (max_attempts / max_lost) y revocaciones
  - versiones de modelo: version_id = attempt_id, con alias `version = 1` para la canónica
  - aislamiento por client_id: un recurso ajeno responde 404
  - inferencia y contadores de uso

Estado: las tablas se cargan completas en memoria (diccionarios por id, en orden de creación) y cada
cambio se escribe en su CSV antes de responder. Al arrancar (también tras un reinicio del actor) se
recarga todo y se repara cualquier escritura que quedara a medias (`_reconcile`).
Estado volátil (no se guarda): último renew de cada attempt, retardos de reasignación y workers vistos.

Registro: cada llamada pública queda en el log (`distriml.db`) gracias al decorador `@traced`.
Además, cada cambio de estado de un job o attempt se registra con `self.log`.
"""
from __future__ import annotations

import re
import threading
import time

from distriml.common import catalog, constants as C, dataset, ids
from distriml.common.artifacts import ArtifactStore
from distriml.common.config import Config
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import bad_request, conflict, not_found, unavailable
from distriml.common.interfaces import Assignment, WorkerConfig
from distriml.common.log import get_logger, setup_logging, short, traced
from distriml.db import scheduler
from distriml.db.csvtable import dumps
from distriml.db.inference import InferenceEngine
from distriml.db.records import decode, encode
from distriml.db.tables import Tables

_ENTRY = re.compile(r"^db:(\d+)$")


@traced("db")
class Database:
    def __init__(self, cfg: Config, clock=None, wall_clock=None):
        """`clock` (ms, monótono) mide leases y retardos; `wall_clock` (ms desde epoch) sella los registros.
        Ambos se pueden sustituir en las pruebas."""
        setup_logging(cfg.log_level)          # el actor corre en su propio proceso: configura su log
        self.log = get_logger("db")
        self.cfg = cfg
        self._clock = clock or (lambda: int(time.monotonic() * 1000))
        self._wall = wall_clock or ids.now_ms
        self.store = ArtifactStore(cfg.data_dir)      # artefactos en <data_dir>/artifacts/
        self.tables = Tables(cfg.data_dir)            # tablas CSV en <data_dir>/db/
        self._lock = threading.RLock()                # por si se añaden hilos auxiliares
        self._entry_counter = 0
        # Estado persistente, en memoria y en orden de creación.
        self._datasets: dict[str, dict] = {}
        self._jobs: dict[str, dict] = {}
        self._attempts: dict[str, dict] = {}
        self._batches: dict[str, dict[int, dict]] = {}      # job_id -> batch_idx -> fila
        self._models: dict[str, dict] = {}
        self._versions: dict[str, list[dict]] = {}          # model_id -> versiones
        self._requests: dict[str, dict] = {}                # request_key -> respuesta ya dada
        # Estado volátil.
        self._last_renew_ms: dict[str, int] = {}            # SOLO attempts ACTIVOS -> último renew
        self._started: set[str] = set()                     # attempts activos que ya renovaron o confirmaron
        self._not_before_ms: dict[str, int] = {}            # job_id -> no reasignar antes de este instante
        self._workers: dict[str, int] = {}                  # worker_id -> última vez visto
        self._load()
        self.inference = InferenceEngine(self.store, self.tables, self._wall)
        self.log.info("DB iniciada (data_dir=%s): %d datasets, %d jobs, %d attempts, %d modelos",
                      cfg.data_dir, len(self._datasets), len(self._jobs), len(self._attempts), len(self._models))

    # ------------------------------------------------------------------ artefactos y catálogo
    def ping(self) -> dict:
        return {"status": "ok"}

    def list_model_definitions(self) -> list[dict]:
        return catalog.list_definitions()

    def put_artifact(self, sha256: str, data: bytes) -> None:
        self.store.put(sha256, data)

    def get_artifact(self, sha256: str) -> bytes:
        return self.store.get(sha256)

    # ------------------------------------------------------------------ carga y reparación al iniciar
    def _rows(self, name: str) -> list[dict]:
        return [decode(name, r) for r in getattr(self.tables, name).load()]

    def _load(self) -> None:
        for r in self._rows("datasets"):
            self._datasets[r["dataset_id"]] = r
        for r in self._rows("jobs"):
            self._jobs[r["job_id"]] = r
        for r in self._rows("attempts"):
            self._attempts[r["attempt_id"]] = r
        for r in self._rows("job_batches"):
            self._batches.setdefault(r["job_id"], {})[r["batch_idx"]] = r
        for r in self._rows("models"):
            self._models[r["model_id"]] = r
        for r in self._rows("model_versions"):
            self._versions.setdefault(r["model_id"], []).append(r)
        for r in self._rows("requests"):
            self._requests[r["request_key"]] = r
        self._reconcile()
        self._entry_counter = self._recover_entry_counter()

    def _recover_entry_counter(self) -> int:
        """Mayor `entry_id` ya entregado (en respuestas guardadas, batches y versiones): el contador sigue desde ahí."""
        found = [r["response"].get("entry_id") for r in self._requests.values() if isinstance(r["response"], dict)]
        found += [b["entry_id"] for per_job in self._batches.values() for b in per_job.values()]
        found += [v["entry_id"] for vs in self._versions.values() for v in vs]
        numbers = [int(m.group(1)) for e in found if e and (m := _ENTRY.match(e))]
        return max(numbers, default=0)

    def _reconcile(self) -> None:
        """Deja el estado coherente tras un arranque. Las escrituras de una operación se hacen en un orden
        fijo (primero lo que otras tablas referencian), así que un corte a mitad solo deja casos conocidos:

          - batches confirmados que `jobs` no refleja       -> last_confirmed_batch sale de job_batches
          - modelo creado pero job sin cerrar               -> el job se marca COMPLETADO
          - attempt ACTIVO que su job ya no considera       -> se revoca (sin contar en ningún presupuesto)
          - job EN_EJECUCION sin attempt activo             -> vuelve a PENDIENTE
          - attempts activos legítimos                      -> reciben un plazo nuevo completo: sus workers
                                                               pudieron no poder renovar mientras la DB estaba caída
        """
        changed = False
        now = self._clock()
        for job in self._jobs.values():
            done = self._batches.get(job["job_id"], {})
            last = 0
            while done.get(last + 1, {}).get("estado") == C.BATCH_CONFIRMED:
                last += 1
            if job["last_confirmed_batch"] != last:
                job["last_confirmed_batch"] = last
                changed = True
            model_id = ids.derive_model_id(job["job_id"])
            canonical = next((v for v in self._versions.get(model_id, []) if v["canonical"]), None)
            if canonical is not None and job["estado"] in (C.PENDIENTE, C.EN_EJECUCION):
                job.update(estado=C.COMPLETADO, model_id=model_id, canonical_version_id=canonical["version_id"],
                           model_version=canonical["version"], attempt_vigente=None, completed_at=self._wall())
                self.log.warning("recuperado: job %s tenía su modelo pero seguía abierto -> COMPLETADO", short(job["job_id"]))
                changed = True
        for a in self._attempts.values():
            if a["estado"] != C.ATTEMPT_ACTIVO:
                continue
            job = self._jobs.get(a["job_id"])
            if job is None or job["attempt_vigente"] != a["attempt_id"] or job["estado"] != C.EN_EJECUCION:
                a["estado"] = C.ATTEMPT_REVOCADO
                a["revoke_reason"] = self._orphan_reason(a, job)
                self.log.warning("recuperado: attempt %s ya no era el vigente -> REVOCADO (%s)",
                                 short(a["attempt_id"]), a["revoke_reason"] or "completó el job")
                changed = True
            else:
                self._last_renew_ms[a["attempt_id"]] = now
                if any(b["attempt_id"] == a["attempt_id"] for b in self._batches.get(job["job_id"], {}).values()):
                    self._started.add(a["attempt_id"])
        for job in self._jobs.values():
            if job["estado"] == C.EN_EJECUCION and job["attempt_vigente"] not in self._last_renew_ms:
                job.update(estado=C.PENDIENTE, attempt_vigente=None)
                self.log.warning("recuperado: job %s estaba EN_EJECUCION sin attempt activo -> PENDIENTE", short(job["job_id"]))
                changed = True
        if changed:
            self._save_jobs()
            self._save_attempts()

    @staticmethod
    def _orphan_reason(attempt: dict, job: dict | None) -> str | None:
        if job is not None and job["canonical_version_id"] == attempt["attempt_id"]:
            return None                                   # fue el attempt que completó el job
        if job is not None and job["estado"] == C.CANCELADO:
            return C.CANCEL
        return C.NEVER_STARTED

    # ------------------------------------------------------------------ persistencia
    def _save_jobs(self) -> None:
        self.tables.jobs.rewrite(encode("jobs", r) for r in self._jobs.values())

    def _save_attempts(self) -> None:
        self.tables.attempts.rewrite(encode("attempts", r) for r in self._attempts.values())

    def _persist_state(self) -> None:
        """Guarda jobs y attempts (en ese orden: ver `_reconcile`)."""
        self._save_jobs()
        self._save_attempts()

    # ------------------------------------------------------------------ utilidades internas
    def _next_entry_id(self) -> str:
        """Identificador de una mutación ("db:<contador>"). Se devuelve al cliente en las escrituras.
        El contador se recupera al iniciar a partir de los entry_id ya guardados."""
        self._entry_counter += 1
        return f"db:{self._entry_counter}"

    @staticmethod
    def _own(table: dict, client_id: str, key) -> dict:
        row = table.get(key) if isinstance(key, str) else None
        if row is None or row["client_id"] != client_id:
            raise not_found()                  # ajeno e inexistente se responden igual
        return row

    @staticmethod
    def _page(rows: list[dict], key: str, limit: int, cursor: str | None) -> list[dict]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise bad_request("invalid_limit", "limit debe ser un entero >= 1")
        if cursor:
            keys = [r[key] for r in rows]
            rows = rows[keys.index(cursor) + 1:] if cursor in keys else []
        return rows[:limit]

    def _touch_worker(self, worker_id: str) -> None:
        if worker_id not in self._workers:
            self.log.info("worker %s visto por primera vez", worker_id)
        self._workers[worker_id] = self._clock()

    def _response_for(self, key: str, kind: str, payload_hash: str, build) -> dict:
        """Respuesta de una mutación de cliente: la ya guardada si se repite la petición; si no, se construye
        con un `entry_id` nuevo y se guarda (así un reintento recibe exactamente lo mismo)."""
        prior = self._requests.get(key)
        if prior is not None:
            return dict(prior["response"])
        response = build(self._next_entry_id())
        row = {"request_key": key, "kind": kind, "payload_hash": payload_hash, "response": response}
        self.tables.requests.append(encode("requests", row))
        self._requests[key] = row
        return dict(response)

    # ------------------------------------------------------------------ leases y revocaciones
    def _vigente(self, attempt_id: str) -> dict | None:
        """El attempt si está ACTIVO y es el vigente de su job; si no, None."""
        a = self._attempts.get(attempt_id) if isinstance(attempt_id, str) else None
        if a is None or a["estado"] != C.ATTEMPT_ACTIVO:
            return None
        job = self._jobs.get(a["job_id"])
        return a if job is not None and job["attempt_vigente"] == attempt_id else None

    def _revoke(self, attempt: dict, reason: str, lost_at_ms: int | None = None) -> None:
        """Revoca un attempt activo y decide el estado de su job (solo en memoria; quien llama persiste).
        `lost_at_ms` es el instante en que se perdió al worker (el vencimiento de su lease); el retardo de
        reasignación cuenta desde ahí y no desde que la DB lo detecta, porque la revisión es perezosa."""
        if attempt["estado"] != C.ATTEMPT_ACTIVO:
            return
        aid = attempt["attempt_id"]
        attempt["estado"], attempt["revoke_reason"] = C.ATTEMPT_REVOCADO, reason
        self._last_renew_ms.pop(aid, None)
        self._started.discard(aid)
        job = self._jobs[attempt["job_id"]]
        if job["attempt_vigente"] != aid:
            return
        job["attempt_vigente"] = None
        state, job["attempts_count"], job["lost_count"] = scheduler.next_state_after_revocation(
            reason, job["attempts_count"], job["lost_count"], self.cfg.max_attempts, self.cfg.max_lost)
        job["estado"] = state
        if state in C.TERMINAL_STATES:
            job["completed_at"] = self._wall()
        delay = ""
        if reason == C.WORKER_LOST and state == C.PENDIENTE:
            wait = scheduler.backoff_ms(job["lost_count"], self.cfg.t_sync_ms)
            self._not_before_ms[job["job_id"]] = (self._clock() if lost_at_ms is None else lost_at_ms) + wait
            delay = f", se reasigna en {wait} ms"
        level = self.log.warning if state == C.FALLIDO or reason in (C.WORKER_LOST, C.WORKER_ERROR) else self.log.info
        level("attempt %s revocado (%s): job %s -> %s (errores %d/%d, pérdidas %d/%d%s)",
              short(aid), reason, short(job["job_id"]), state, job["attempts_count"], self.cfg.max_attempts,
              job["lost_count"], self.cfg.max_lost, delay)

    def _reap(self) -> None:
        """Revoca los attempts cuyo lease venció (plazo = último renew + lease + margen)."""
        now = self._clock()
        deadlines = {aid: scheduler.lease_deadline_ms(last, self.cfg.lease_duration_ms, self.cfg.epsilon_ms)
                     for aid, last in self._last_renew_ms.items()}
        expired = [aid for aid, deadline in deadlines.items() if now > deadline]
        for aid in expired:
            self._revoke(self._attempts[aid], scheduler.revoke_reason_for_expired(aid in self._started),
                         lost_at_ms=deadlines[aid])
        if expired:
            self._persist_state()

    # ------------------------------------------------------------------ cliente: datasets
    def create_dataset(self, client_id, client_request_id, name, sha256, size, data) -> dict:
        with self._lock:
            did = ids.derive_dataset_id(client_id, client_request_id)
            if ids.sha256_hex(data) != sha256 or len(data) != size:
                raise bad_request("hash_mismatch", "sha256/size no coinciden con el contenido")
            row = self._datasets.get(did)
            if row is not None and row["sha256"] != sha256:
                raise conflict("request_id_reused", "client_request_id ya se usó con otro contenido")
            if row is None:
                meta = dataset.analyze(data, self.cfg.max_dataset_bytes)       # 400 si el CSV es inválido
                self.store.put(sha256, data)
                row = {"dataset_id": did, "client_id": client_id, "name": name, "sha256": sha256, "size": size,
                       **meta.to_dict(), "created_at": self._wall(), "client_request_id": client_request_id}
                self.tables.datasets.append(encode("datasets", row))
                self._datasets[did] = row
                self.log.info("dataset %s guardado: %d filas, %d características, clases=%s",
                              short(did), row["n_rows"], row["n_features"], row["classes"])
            return self._response_for(did, "dataset", sha256, lambda entry: {
                "dataset_id": did, "n_rows": row["n_rows"], "n_features": row["n_features"],
                "classes": row["classes"], "entry_id": entry})

    def list_datasets(self, client_id, limit=100, cursor=None) -> list[dict]:
        with self._lock:
            mine = [r for r in self._datasets.values() if r["client_id"] == client_id]
            return [{"dataset_id": r["dataset_id"], "name": r["name"], "sha256": r["sha256"], "size": r["size"]}
                    for r in self._page(mine, "dataset_id", limit, cursor)]

    def get_dataset(self, client_id, dataset_id) -> dict:
        with self._lock:
            r = self._own(self._datasets, client_id, dataset_id)
            return {"dataset_id": r["dataset_id"], "name": r["name"], "sha256": r["sha256"],
                    "size": r["size"], "replicas": 1}

    # ------------------------------------------------------------------ cliente: jobs
    def create_job(self, client_id, client_request_id, dataset_id, model_def_id, hyperparams=None) -> dict:
        with self._lock:
            jid = ids.derive_job_id(client_id, client_request_id)
            md = catalog.get(model_def_id)                                      # 400
            if hyperparams is not None and not isinstance(hyperparams, dict):
                raise bad_request("invalid_hyperparams", "hyperparams debe ser un objeto")
            hp = catalog.merge_hyperparams(md, hyperparams)                     # 400
            fingerprint = ids.sha256_hex(dumps([dataset_id, model_def_id, hp]).encode("utf-8"))
            existing = self._jobs.get(jid)
            if existing is not None:
                if (existing["dataset_id"], existing["model_def_id"], existing["hyperparams"]) != (dataset_id, model_def_id, hp):
                    raise conflict("request_id_reused", "client_request_id ya se usó con otros parámetros")
                return self._response_for(jid, "job", fingerprint, lambda entry: {"job_id": jid, "entry_id": entry})
            ds = self._datasets.get(dataset_id) if isinstance(dataset_id, str) else None
            if ds is None or ds["client_id"] != client_id:
                raise bad_request("dataset_not_found", "el dataset no existe o no es tuyo")
            catalog.check_compatibility(md, DatasetMeta.from_dict(ds))          # 400
            if not self.store.has(ds["sha256"]):
                raise unavailable("dataset_unavailable", "el dataset aún no está disponible")
            job = {"job_id": jid, "client_id": client_id, "dataset_id": dataset_id, "model_def_id": model_def_id,
                   "hyperparams": hp, "total_batches": catalog.compute_total_batches(hp, ds["n_rows"]),
                   "estado": C.PENDIENTE, "attempts_count": 0, "lost_count": 0, "last_confirmed_batch": 0,
                   "attempt_vigente": None, "model_id": None, "canonical_version_id": None, "model_version": None,
                   "completed_at": None, "created_at": self._wall(), "client_request_id": client_request_id}
            self._jobs[jid] = job
            self._save_jobs()
            self.log.info("job %s creado: %s sobre dataset %s, %d batches", short(jid), md.algorithm,
                          short(dataset_id), job["total_batches"])
            return self._response_for(jid, "job", fingerprint, lambda entry: {"job_id": jid, "entry_id": entry})

    def _job_view(self, job: dict) -> dict:
        view = {k: job[k] for k in ("job_id", "dataset_id", "model_def_id", "estado", "hyperparams", "total_batches",
                                    "last_confirmed_batch", "attempt_vigente", "attempts_count", "lost_count",
                                    "model_id", "canonical_version_id", "model_version", "created_at", "completed_at")}
        view["hyperparams"] = dict(job["hyperparams"])
        view["attempts"] = [
            {"attempt_id": a["attempt_id"], "worker_id": a["worker_id"], "estado": a["estado"],
             "revoke_reason": a["revoke_reason"], "base_batch": a["base_batch"], "created_at": a["created_at"]}
            for a in self._attempts.values() if a["job_id"] == job["job_id"]]
        return view

    def get_job(self, client_id, job_id) -> dict:
        with self._lock:
            self._reap()
            return self._job_view(self._own(self._jobs, client_id, job_id))

    def list_jobs(self, client_id, estado=None, limit=100, cursor=None) -> list[dict]:
        with self._lock:
            self._reap()
            mine = [j for j in self._jobs.values() if j["client_id"] == client_id and (estado is None or j["estado"] == estado)]
            return [{k: j[k] for k in ("job_id", "dataset_id", "model_def_id", "estado", "last_confirmed_batch", "total_batches")}
                    for j in self._page(mine, "job_id", limit, cursor)]

    def cancel_job(self, client_id, client_request_id, job_id) -> dict:
        with self._lock:
            job = self._own(self._jobs, client_id, job_id)
            cid = ids.derive_cancel_id(client_id, client_request_id)
            prior = self._requests.get(cid)
            if prior is not None:
                if prior["payload_hash"] != job["job_id"]:
                    raise conflict("request_id_reused", "client_request_id ya se usó para cancelar otro job")
                return dict(prior["response"])
            self._reap()
            if job["estado"] in C.TERMINAL_STATES:
                raise conflict("job_terminal", f"el job ya está {job['estado']}")
            if job["attempt_vigente"]:
                self._revoke(self._attempts[job["attempt_vigente"]], C.CANCEL)
            if job["estado"] != C.CANCELADO:                 # PENDIENTE: no había attempt que revocar
                job.update(estado=C.CANCELADO, completed_at=self._wall())
                self.log.info("job %s CANCELADO (no tenía attempt en curso)", short(job["job_id"]))
            self._persist_state()
            return self._response_for(cid, "cancel", job["job_id"], lambda entry: {"estado": C.CANCELADO, "entry_id": entry})

    def get_job_metrics(self, client_id, job_id) -> dict:
        with self._lock:
            self._reap()
            job = self._own(self._jobs, client_id, job_id)
            rows = [{"batch_idx": k, "metrics": b["metrics"]}
                    for k, b in sorted(self._batches.get(job["job_id"], {}).items()) if b["estado"] == C.BATCH_CONFIRMED]
            final = None
            if job["estado"] == C.COMPLETADO:
                final = self._pick_version(job["model_id"], job["canonical_version_id"], None)["metrics"]
            return {"job_id": job["job_id"], "estado": job["estado"], "batches": rows, "final": final}

    # ------------------------------------------------------------------ cliente: modelos e inferencia
    @staticmethod
    def _version_view(model: dict, v: dict) -> dict:
        return {"model_id": model["model_id"], "version_id": v["version_id"], "version": v["version"],
                "job_id": model["job_id"], "model_def_id": model["model_def_id"], "estado": v["estado"],
                "canonical": v["canonical"], "sha256": v["sha256"], "metrics": v["metrics"]}

    def list_model_versions(self, client_id, model_id) -> list[dict]:
        with self._lock:
            m = self._own(self._models, client_id, model_id)
            return [self._version_view(m, v) for v in self._versions.get(m["model_id"], [])]

    def get_model_version(self, client_id, model_id, version_id) -> dict:
        with self._lock:
            m = self._own(self._models, client_id, model_id)
            for v in self._versions.get(m["model_id"], []):
                if v["version_id"] == version_id:
                    return self._version_view(m, v)
            raise not_found()

    def _pick_version(self, model_id: str, version_id, version) -> dict:
        """version_id manda; si no, `version` es el alias entero de la canónica; si no, la canónica."""
        vs = self._versions.get(model_id, [])
        if version_id is not None:
            found = [v for v in vs if v["version_id"] == version_id]
        elif version is not None:
            found = [v for v in vs if v["canonical"] and v["version"] == version]
        else:
            found = [v for v in vs if v["canonical"]]
        if not found:
            raise not_found()
        return found[0]

    def get_model_stats(self, client_id, model_id, version_id=None, from_minute=None, to_minute=None) -> dict:
        with self._lock:
            self._own(self._models, client_id, model_id)
            for name, value in (("from", from_minute), ("to", to_minute)):
                if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                    raise bad_request("invalid_range", f"{name} debe ser un entero (minutos desde epoch)")
            v = self._pick_version(model_id, version_id, None)
            return self.inference.stats(model_id, v["version_id"], from_minute, to_minute)

    def predict(self, client_id, model_id, input, version_id=None, version=None) -> dict:
        with self._lock:
            m = self._own(self._models, client_id, model_id)
            v = self._pick_version(m["model_id"], version_id, version)
            features = input.get("features") if isinstance(input, dict) else None
            if features is None:
                raise bad_request("invalid_input", "input.features es obligatorio")
            algorithm = catalog.get(m["model_def_id"]).algorithm
            return self.inference.predict(v, algorithm, features)

    # ------------------------------------------------------------------ worker
    def register_worker(self, worker_id) -> WorkerConfig:
        with self._lock:
            if not isinstance(worker_id, str) or not worker_id:
                raise bad_request("invalid_worker_id", "worker_id debe ser un texto no vacío")
            self._touch_worker(worker_id)
            return WorkerConfig(worker_id, self.cfg.renew_interval_ms, self.cfg.lease_duration_ms)

    def _startable(self, job: dict) -> bool:
        """El job puede arrancar si su dataset y el checkpoint de partida están en el almacén."""
        if not self.store.has(self._datasets[job["dataset_id"]]["sha256"]):
            return False
        last = job["last_confirmed_batch"]
        return last == 0 or self.store.has(self._batches[job["job_id"]][last]["checkpoint_sha256"])

    def request_job(self, worker_id) -> Assignment | None:
        with self._lock:
            self._touch_worker(worker_id)
            self._reap()
            pending = [j for j in self._jobs.values() if j["estado"] == C.PENDIENTE]
            job = scheduler.pick_next_job(pending, self._clock(), self._not_before_ms, self._startable)
            if job is None:
                return None
            ds = self._datasets[job["dataset_id"]]
            last = job["last_confirmed_batch"]
            base = self._batches[job["job_id"]][last] if last else None
            aid = ids.new_attempt_id()
            attempt = {"attempt_id": aid, "job_id": job["job_id"], "worker_id": worker_id,
                       "base_attempt": base["attempt_id"] if base else None, "base_batch": last,
                       "estado": C.ATTEMPT_ACTIVO, "revoke_reason": None, "created_at": self._wall()}
            self._attempts[aid] = attempt
            self.tables.attempts.append(encode("attempts", attempt))
            job["attempt_vigente"], job["estado"] = aid, C.EN_EJECUCION
            self._save_jobs()
            self._last_renew_ms[aid] = self._clock()
            self._not_before_ms.pop(job["job_id"], None)
            self.log.info("job %s asignado a %s: attempt %s desde el batch %d de %d", short(job["job_id"]),
                          worker_id, short(aid), last + 1, job["total_batches"])
            return Assignment(job["job_id"], aid, last + 1, job["total_batches"], ds["sha256"],
                              DatasetMeta.from_dict(ds).to_dict(), catalog.get(job["model_def_id"]).algorithm,
                              dict(job["hyperparams"]), self.cfg.lease_duration_ms,
                              base["checkpoint_sha256"] if base else None)

    def renew_lease(self, worker_id, attempt_id) -> str:
        with self._lock:
            self._touch_worker(worker_id)
            self._reap()
            a = self._vigente(attempt_id)
            if a is None:
                return C.RENEW_REVOKED
            self._last_renew_ms[attempt_id] = self._clock()
            self._started.add(attempt_id)
            return C.RENEW_CONTINUE

    def upload_checkpoint(self, worker_id, attempt_id, batch_idx, sha256, metrics) -> str:
        with self._lock:
            self._reap()
            a = self._vigente(attempt_id)
            if a is None:
                return C.CKPT_REJECTED
            job = self._jobs[a["job_id"]]
            existing = self._batches.get(job["job_id"], {}).get(batch_idx)
            if existing is not None:                       # reintento de un batch ya confirmado
                return C.CKPT_CONFIRMED if existing["attempt_id"] == attempt_id else C.CKPT_REJECTED
            if (not isinstance(batch_idx, int) or isinstance(batch_idx, bool)
                    or batch_idx != job["last_confirmed_batch"] + 1 or batch_idx > job["total_batches"]):
                return C.CKPT_REJECTED
            if not isinstance(metrics, dict):
                raise bad_request("invalid_metrics", "metrics debe ser un objeto")
            if not self.store.has(sha256):
                return C.CKPT_RETRY
            row = {"job_id": job["job_id"], "attempt_id": attempt_id, "batch_idx": batch_idx,
                   "checkpoint_sha256": sha256, "metrics": metrics, "estado": C.BATCH_CONFIRMED,
                   "entry_id": self._next_entry_id()}
            self.tables.job_batches.append(encode("job_batches", row))       # primero el batch, luego el job
            self._batches.setdefault(job["job_id"], {})[batch_idx] = row
            job["last_confirmed_batch"] = batch_idx
            self._save_jobs()
            self._last_renew_ms[attempt_id] = self._clock()
            self._started.add(attempt_id)
            return C.CKPT_CONFIRMED

    def complete_job(self, worker_id, attempt_id, model_sha256, metrics) -> str:
        with self._lock:
            self._reap()
            a = self._attempts.get(attempt_id) if isinstance(attempt_id, str) else None
            if a is not None and self._jobs[a["job_id"]]["canonical_version_id"] == attempt_id:
                return C.COMPLETE_OK                        # reintento de un cierre que ya se aplicó
            a = self._vigente(attempt_id)
            if a is None:
                return C.COMPLETE_REJECTED
            job = self._jobs[a["job_id"]]
            if job["last_confirmed_batch"] != job["total_batches"]:
                return C.COMPLETE_REJECTED
            if not isinstance(metrics, dict):
                raise bad_request("invalid_metrics", "metrics debe ser un objeto")
            if not self.store.has(model_sha256):
                raise unavailable("artifact_unavailable", "el modelo final no está en el almacén")
            mid = ids.derive_model_id(job["job_id"])
            model = {"model_id": mid, "job_id": job["job_id"], "client_id": job["client_id"],
                     "model_def_id": job["model_def_id"], "canonical_version_id": attempt_id,
                     "created_at": self._wall(), "training_metrics": metrics}
            version = {"model_id": mid, "version_id": attempt_id, "version": 1, "sha256": model_sha256,
                       "metrics": metrics, "estado": C.VERSION_ENTRENADO, "canonical": True,
                       "entry_id": self._next_entry_id()}
            self.tables.models.append(encode("models", model))                # primero modelo y versión,
            self.tables.model_versions.append(encode("model_versions", version))   # luego el cierre del job
            self._models[mid] = model
            self._versions[mid] = [version]
            job.update(estado=C.COMPLETADO, model_id=mid, canonical_version_id=attempt_id, model_version=1,
                       attempt_vigente=None, completed_at=self._wall())
            a["estado"], a["revoke_reason"] = C.ATTEMPT_REVOCADO, None
            self._last_renew_ms.pop(attempt_id, None)
            self._started.discard(attempt_id)
            self._persist_state()
            self.log.info("job %s COMPLETADO por el attempt %s: modelo %s, versión %s",
                          short(job["job_id"]), short(attempt_id), short(mid), short(attempt_id))
            return C.COMPLETE_OK

    def report_failure(self, worker_id, attempt_id, error) -> None:
        with self._lock:
            a = self._vigente(attempt_id)
            if a is not None:
                self.log.warning("worker %s reporta fallo en el attempt %s: %s", worker_id, short(attempt_id),
                                 short(str(error), 200))
                self._revoke(a, C.WORKER_ERROR)
                self._persist_state()

    def report_unavailable(self, worker_id, attempt_id, sha256) -> None:
        with self._lock:
            a = self._vigente(attempt_id)
            if a is not None:
                self.log.warning("worker %s no pudo obtener el artefacto %s (attempt %s)", worker_id,
                                 short(str(sha256)), short(attempt_id))
                self._revoke(a, C.ARTIFACT_UNAVAILABLE)
                self._persist_state()
