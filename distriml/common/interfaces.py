"""Interfaz entre el gateway HTTP, la DB y los workers.

La DB implementa `DBService`; el gateway, la consola y los workers solo conocen estos métodos
(no saben si hablan con un actor de Ray o con un objeto en memoria). Las firmas se cambian solo de
común acuerdo entre quienes desarrollan la DB y los workers.

Todos los métodos pueden levantar `ContractError` (400/404/409/503). Los métodos de worker devuelven
cadenas de resultado (ver constants.py) en lugar de levantar errores de protocolo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class WorkerConfig:
    """Respuesta a `register_worker`."""
    worker_id: str
    renew_interval_ms: int      # cada cuánto debe renovar el lease (sirve también de señal de vida)
    lease_duration_ms: int


@dataclass
class Assignment:
    """Respuesta a `request_job` cuando hay trabajo."""
    job_id: str
    attempt_id: str
    next_batch: int             # primer batch a entrenar (1 si parte de cero)
    total_batches: int
    dataset_sha256: str         # el worker lo descarga con db.get_artifact
    dataset_meta: dict          # DatasetMeta.to_dict(): n_rows, n_features, classes, feature_mean, feature_std
    algorithm: str              # model_def.algorithm
    hyperparams: dict           # YA mezclados con los por defecto (incluye seed, epochs, batch_size)
    lease_duration_ms: int
    checkpoint_sha256: str | None = None   # estado desde el que continuar (None si next_batch == 1)


class DBService(Protocol):
    # ------------------------------------------------------------------ común
    def ping(self) -> dict:
        """Comprueba que la DB responde. Devuelve {"status": "ok"}."""

    # ------------------------------------------------------------------ cliente (lo llama el gateway)
    def list_model_definitions(self) -> list[dict]:
        """Catálogo de modelos disponibles (ver catalog.list_definitions())."""

    def create_dataset(self, client_id: str, client_request_id: str, name: str,
                       sha256: str, size: int, data: bytes) -> dict:
        """Valida y guarda un dataset CSV. Repetir la misma petición devuelve el mismo resultado.
        Devuelve {"dataset_id", "n_rows", "n_features", "classes", "entry_id"}."""

    def list_datasets(self, client_id: str, limit: int = 100, cursor: str | None = None) -> list[dict]:
        """Datasets del cliente: [{"dataset_id", "name", "sha256", "size"}].
        `cursor` es el id del último elemento de la página anterior."""

    def get_dataset(self, client_id: str, dataset_id: str) -> dict:
        """Un dataset del cliente: {"dataset_id", "name", "sha256", "size", "replicas"}. 404 si es ajeno."""

    def create_job(self, client_id: str, client_request_id: str, dataset_id: str,
                   model_def_id: str, hyperparams: dict | None = None) -> dict:
        """Crea un job PENDIENTE tras validar modelo, hiperparámetros y dataset.
        Idempotente. Devuelve {"job_id", "entry_id"}."""

    def get_job(self, client_id: str, job_id: str) -> dict:
        """Estado completo del job, con la lista de sus attempts
        (attempt_id, worker_id, estado, revoke_reason, base_batch, created_at)."""

    def list_jobs(self, client_id: str, estado: str | None = None, limit: int = 100,
                  cursor: str | None = None) -> list[dict]:
        """Jobs del cliente: [{"job_id", "dataset_id", "model_def_id", "estado",
        "last_confirmed_batch", "total_batches"}], filtrables por estado."""

    def cancel_job(self, client_id: str, client_request_id: str, job_id: str) -> dict:
        """Cancela un job y revoca su attempt vigente. Devuelve {"estado": "CANCELADO", "entry_id"};
        409 si el job ya terminó."""

    def get_job_metrics(self, client_id: str, job_id: str) -> dict:
        """Métricas por batch confirmado y finales:
        {"job_id", "estado", "batches": [{"batch_idx", "metrics"}], "final": dict | None}."""

    def list_model_versions(self, client_id: str, model_id: str) -> list[dict]:
        """Versiones de un modelo del cliente (misma forma que get_model_version)."""

    def get_model_version(self, client_id: str, model_id: str, version_id: str) -> dict:
        """{"model_id", "version_id", "version", "job_id", "model_def_id", "estado", "canonical",
        "sha256", "metrics"}."""

    def get_model_stats(self, client_id: str, model_id: str, version_id: str | None = None,
                        from_minute: int | None = None, to_minute: int | None = None) -> dict:
        """Estadísticas de inferencia: {"model_id", "version_id", "total_calls", "mean_latency_ms",
        "por_minuto": [{"minute", "count", "mean_latency_ms"}]}."""

    def predict(self, client_id: str, model_id: str, input: dict,
                version_id: str | None = None, version: int | None = None) -> dict:
        """Predicción con un modelo del cliente. `input` = {"features": [...]}.
        Sin versión indicada usa la canónica. Devuelve {"prediction", "version_id", "sha256"}."""

    # ------------------------------------------------------------------ worker
    def register_worker(self, worker_id: str) -> WorkerConfig:
        """Registra al worker y le devuelve los tiempos de renovación de lease."""

    def request_job(self, worker_id: str) -> Assignment | None:
        """Asigna un job PENDIENTE: crea un attempt nuevo que continúa desde el último checkpoint
        confirmado. Devuelve None si no hay trabajo."""

    def renew_lease(self, worker_id: str, attempt_id: str) -> str:
        """RENEW_CONTINUE | RENEW_REVOKED. Renovar también indica a la DB que el worker sigue vivo."""

    def put_artifact(self, sha256: str, data: bytes) -> None:
        """Guarda un binario (checkpoint o modelo) por su hash. 400 si el hash no coincide."""

    def get_artifact(self, sha256: str) -> bytes:
        """Devuelve un binario por su hash. 503 artifact_unavailable si falta."""

    def upload_checkpoint(self, worker_id: str, attempt_id: str, batch_idx: int,
                          sha256: str, metrics: dict) -> str:
        """Confirma el batch `batch_idx` de un attempt. CKPT_CONFIRMED | CKPT_RETRY (el binario aún no
        está) | CKPT_REJECTED (el attempt ya no es el vigente). Idempotente por (attempt_id, batch_idx)."""

    def complete_job(self, worker_id: str, attempt_id: str, model_sha256: str, metrics: dict) -> str:
        """Cierra el job con el modelo final: crea el modelo y su versión (version_id = attempt_id).
        COMPLETE_OK | COMPLETE_REJECTED."""

    def report_failure(self, worker_id: str, attempt_id: str, error: str) -> None:
        """El entrenamiento falló: revoca el attempt con WORKER_ERROR (cuenta para max_attempts)."""

    def report_unavailable(self, worker_id: str, attempt_id: str, sha256: str) -> None:
        """El worker no pudo obtener un artefacto: revoca el attempt con ARTIFACT_UNAVAILABLE (no cuenta)."""
