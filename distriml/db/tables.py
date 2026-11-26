"""Esquema CSV de la BD: columnas de cada tabla y contenedor `Tables`.

Convenciones:
  - Columnas con estructura (hyperparams, metrics, classes, feature_mean, feature_std) se guardan
    como JSON en una celda (csvtable.dumps / csvtable.loads).
  - Tiempos en milisegundos desde epoch (informativos, no deciden nada salvo el lease).
  - Tablas de estado cambiante (jobs, attempts): `rewrite` (reescritura atómica completa).
  - Tablas de solo anexado (datasets, job_batches, models, model_versions, requests, inference_stats): `append`.
    `inference_stats` recibe una fila por predicción y se compacta (una fila por modelo, versión y minuto)
    al arrancar la DB.
  - Las filas se convierten de/hacia sus tipos en memoria con db/records.py.
"""
from __future__ import annotations

from pathlib import Path

from distriml.db.csvtable import CsvTable

DATASETS = ["dataset_id", "client_id", "name", "sha256", "size", "n_rows", "n_features", "classes",
            "feature_mean", "feature_std", "created_at", "client_request_id"]

JOBS = ["job_id", "client_id", "dataset_id", "model_def_id", "hyperparams", "total_batches", "estado",
        "attempts_count", "lost_count", "last_confirmed_batch", "attempt_vigente", "model_id",
        "canonical_version_id", "model_version", "completed_at", "created_at", "client_request_id"]

ATTEMPTS = ["attempt_id", "job_id", "worker_id", "base_attempt", "base_batch", "estado", "revoke_reason",
            "created_at"]   # el instante de la última renovación es VOLÁTIL: vive en memoria, no aquí

JOB_BATCHES = ["job_id", "attempt_id", "batch_idx", "checkpoint_sha256", "metrics", "estado", "entry_id"]

MODELS = ["model_id", "job_id", "client_id", "model_def_id", "canonical_version_id", "created_at",
          "training_metrics"]

MODEL_VERSIONS = ["model_id", "version_id", "version", "sha256", "metrics", "estado", "canonical", "entry_id"]

INFERENCE_STATS = ["model_id", "version_id", "minute", "count", "latency_sum"]

# Registro de peticiones de cliente ya atendidas: permite devolver el mismo resultado ante un
# reintento (idempotencia) y detectar el reuso de un client_request_id con otro contenido (409).
REQUESTS = ["request_key", "kind", "payload_hash", "response"]


class Tables:
    """Contenedor de las tablas CSV bajo <data_dir>/db/."""

    def __init__(self, data_dir: str | Path):
        root = Path(data_dir) / "db"
        self.datasets = CsvTable(root / "datasets.csv", DATASETS)
        self.jobs = CsvTable(root / "jobs.csv", JOBS)
        self.attempts = CsvTable(root / "attempts.csv", ATTEMPTS)
        self.job_batches = CsvTable(root / "job_batches.csv", JOB_BATCHES)
        self.models = CsvTable(root / "models.csv", MODELS)
        self.model_versions = CsvTable(root / "model_versions.csv", MODEL_VERSIONS)
        self.inference_stats = CsvTable(root / "inference_stats.csv", INFERENCE_STATS)
        self.requests = CsvTable(root / "requests.csv", REQUESTS)
