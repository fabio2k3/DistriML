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

Registro: cada llamada pública queda en el log (`distriml.db`) gracias al decorador `@traced`.
Además, cada cambio de estado de un job o attempt debe registrarse con `self.log.info(...)`.

Los métodos marcados TODO(DB-n) están por implementar (orden sugerido en docs/PLAN.md).
`tests/helpers/fake_db.py` es una implementación en memoria del mismo comportamiento, útil como referencia.
"""
from __future__ import annotations

import threading

from distriml.common import catalog
from distriml.common.artifacts import ArtifactStore
from distriml.common.config import Config
from distriml.common.interfaces import Assignment, WorkerConfig
from distriml.common.log import get_logger, setup_logging, traced
from distriml.db.tables import Tables


@traced("db")
class Database:
    def __init__(self, cfg: Config):
        setup_logging(cfg.log_level)          # el actor corre en su propio proceso: configura su log
        self.log = get_logger("db")
        self.cfg = cfg
        self.store = ArtifactStore(cfg.data_dir)      # artefactos en <data_dir>/artifacts/
        self.tables = Tables(cfg.data_dir)            # tablas CSV en <data_dir>/db/
        self._lock = threading.RLock()                # por si se añaden hilos auxiliares
        self._entry_counter = 0
        # TODO(DB-1): cargar las tablas CSV a estructuras en memoria (diccionarios por id) y recuperar
        #   `_entry_counter`. Si el actor se reinicia (Ray: max_restarts=-1) debe recargar todo aquí.
        # Estado volátil (no se guarda en CSV): último renew de cada attempt y workers registrados.
        self._last_renew_ms: dict[str, int] = {}
        self._workers: dict[str, int] = {}
        self.log.info("DB iniciada (data_dir=%s)", cfg.data_dir)

    # ------------------------------------------------------------------ ya implementado
    def ping(self) -> dict:
        return {"status": "ok"}

    def list_model_definitions(self) -> list[dict]:
        return catalog.list_definitions()

    def put_artifact(self, sha256: str, data: bytes) -> None:
        self.store.put(sha256, data)

    def get_artifact(self, sha256: str) -> bytes:
        return self.store.get(sha256)

    # ------------------------------------------------------------------ utilidades internas
    def _next_entry_id(self) -> str:
        """Identificador de una mutación ("db:<contador>"). Se devuelve al cliente en las escrituras.

        TODO(DB-1): persistir el contador (p. ej. derivarlo del número de mutaciones registradas)
        para que no se repita tras un reinicio del actor.
        """
        raise NotImplementedError("TODO(DB-1)")

    # ------------------------------------------------------------------ cliente: datasets
    def create_dataset(self, client_id, client_request_id, name, sha256, size, data) -> dict:
        """TODO(DB-2)

        1. dataset_id = ids.derive_dataset_id(client_id, client_request_id).
        2. Idempotencia: si ya existe devolver el mismo resultado (409 si el contenido es otro).
        3. meta = dataset.analyze(data, cfg.max_dataset_bytes)  (400 si el CSV es inválido).
        4. self.store.put(sha256, data) y guardar la fila en `datasets` (meta como JSON).
        5. Devolver {"dataset_id", "n_rows", "n_features", "classes", "entry_id"}.
        """
        raise NotImplementedError("TODO(DB-2)")

    def list_datasets(self, client_id, limit=100, cursor=None) -> list[dict]:
        """TODO(DB-2) Solo los del client_id, en orden de creación. `cursor` = id del último ya devuelto.
        Cada elemento: {"dataset_id", "name", "sha256", "size"}."""
        raise NotImplementedError("TODO(DB-2)")

    def get_dataset(self, client_id, dataset_id) -> dict:
        """TODO(DB-2) Ajeno o inexistente -> not_found().
        Devuelve {"dataset_id", "name", "sha256", "size", "replicas"} con replicas = 1."""
        raise NotImplementedError("TODO(DB-2)")

    # ------------------------------------------------------------------ cliente: jobs
    def create_job(self, client_id, client_request_id, dataset_id, model_def_id, hyperparams=None) -> dict:
        """TODO(DB-3)

        1. job_id = ids.derive_job_id(...); idempotencia como en create_dataset.
        2. md = catalog.get(model_def_id) (400); hp = catalog.merge_hyperparams(md, hyperparams) (400).
        3. Dataset propio y existente (400 si no); catalog.check_compatibility(md, meta) (400).
        4. Si el binario del dataset no está en el almacén: unavailable("dataset_unavailable").
        5. total_batches = catalog.compute_total_batches(hp, meta.n_rows). Estado PENDIENTE.
        6. Devolver {"job_id", "entry_id"}.
        """
        raise NotImplementedError("TODO(DB-3)")

    def get_job(self, client_id, job_id) -> dict:
        """TODO(DB-3) Job + lista `attempts` (attempt_id, worker_id, estado, revoke_reason, base_batch,
        created_at) + attempt_vigente, attempts_count, lost_count, model_id, canonical_version_id,
        model_version. Ajeno -> 404. Antes de responder, revocar los leases vencidos."""
        raise NotImplementedError("TODO(DB-3)")

    def list_jobs(self, client_id, estado=None, limit=100, cursor=None) -> list[dict]:
        """TODO(DB-3) Jobs del cliente, filtrables por estado, con paginación por cursor."""
        raise NotImplementedError("TODO(DB-3)")

    def cancel_job(self, client_id, client_request_id, job_id) -> dict:
        """TODO(DB-6) Solo el dueño. 409 si el job ya es terminal. Revoca el attempt vigente con
        motivo CANCEL. Idempotente por client_request_id (ids.derive_cancel_id)."""
        raise NotImplementedError("TODO(DB-6)")

    def get_job_metrics(self, client_id, job_id) -> dict:
        """TODO(DB-7) Métricas de los batches confirmados + `final` si el job está COMPLETADO."""
        raise NotImplementedError("TODO(DB-7)")

    # ------------------------------------------------------------------ cliente: modelos e inferencia
    def list_model_versions(self, client_id, model_id) -> list[dict]:
        """TODO(DB-7) Versiones del modelo (solo si es del cliente)."""
        raise NotImplementedError("TODO(DB-7)")

    def get_model_version(self, client_id, model_id, version_id) -> dict:
        """TODO(DB-7) Una versión concreta; inexistente o ajena -> 404."""
        raise NotImplementedError("TODO(DB-7)")

    def get_model_stats(self, client_id, model_id, version_id=None, from_minute=None, to_minute=None) -> dict:
        """TODO(DB-7) Ver db/inference.py."""
        raise NotImplementedError("TODO(DB-7)")

    def predict(self, client_id, model_id, input, version_id=None, version=None) -> dict:
        """TODO(DB-7) Ver db/inference.py."""
        raise NotImplementedError("TODO(DB-7)")

    # ------------------------------------------------------------------ worker
    def register_worker(self, worker_id) -> WorkerConfig:
        """TODO(DB-4) Anota al worker y devuelve WorkerConfig con renew_interval_ms y lease_duration_ms."""
        raise NotImplementedError("TODO(DB-4)")

    def request_job(self, worker_id) -> Assignment | None:
        """TODO(DB-4) Antes de elegir, revisar leases vencidos (db/scheduler.py). Elegir el job PENDIENTE
        más antiguo, crear un attempt nuevo (base = último checkpoint confirmado), marcar el job
        EN_EJECUCION y devolver el Assignment. None si no hay trabajo."""
        raise NotImplementedError("TODO(DB-4)")

    def renew_lease(self, worker_id, attempt_id) -> str:
        """TODO(DB-4) Actualiza el instante de renovación del attempt. RENEW_REVOKED si el attempt
        ya no es el vigente de su job."""
        raise NotImplementedError("TODO(DB-4)")

    def upload_checkpoint(self, worker_id, attempt_id, batch_idx, sha256, metrics) -> str:
        """TODO(DB-5) CKPT_REJECTED si el attempt no es el vigente; CKPT_RETRY si el binario aún no está;
        idempotente por (attempt_id, batch_idx). Si se confirma, avanza last_confirmed_batch."""
        raise NotImplementedError("TODO(DB-5)")

    def complete_job(self, worker_id, attempt_id, model_sha256, metrics) -> str:
        """TODO(DB-5) Crea `models` (model_id = ids.derive_model_id(job_id)) y `model_versions`
        (version_id = attempt_id, version = 1, canonical = True). El job pasa a COMPLETADO."""
        raise NotImplementedError("TODO(DB-5)")

    def report_failure(self, worker_id, attempt_id, error) -> None:
        """TODO(DB-6) Revoca con WORKER_ERROR; attempts_count += 1; el job vuelve a PENDIENTE o pasa a FALLIDO."""
        raise NotImplementedError("TODO(DB-6)")

    def report_unavailable(self, worker_id, attempt_id, sha256) -> None:
        """TODO(DB-6) Revoca con ARTIFACT_UNAVAILABLE (no cuenta); el job vuelve a PENDIENTE."""
        raise NotImplementedError("TODO(DB-6)")
