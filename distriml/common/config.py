"""Configuración común del sistema.

Se puede sobrescribir con variables de entorno DISTRIML_* o con `load_config(**overrides)`.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("./runtime")          # raíz de la BD (CSV) y de los artefactos
    ray_namespace: str = "distriml"
    db_actor_name: str = "db"                   # nombre del actor de la DB en Ray
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    log_level: str = "INFO"                     # INFO | DEBUG | WARNING (ver common/log.py)

    # Tiempos (ms)
    t_sync_ms: int = 2_000                      # base del retardo de reasignación tras perder un worker
    lease_duration_ms: int = 10_000             # tiempo que un attempt conserva su lease sin renovar
    renew_interval_ms: int = 3_333              # cada cuánto renueva el worker (un tercio del lease)
    epsilon_ms: int = 1_000                     # margen extra antes de dar un worker por perdido
    request_poll_interval_ms: int = 2_000       # cada cuánto pide trabajo un worker ocioso

    # Reintentos por job
    max_attempts: int = 5                       # máximo de errores de entrenamiento reportados por workers
    max_lost: int = 20                          # máximo de pérdidas de contacto (lease vencido)
    k_ckpt: int = 1                             # batches entre checkpoints

    max_dataset_bytes: int = 5 * 1024 * 1024    # tamaño máximo de un dataset subido


def load_config(**overrides) -> Config:
    env = os.environ
    values: dict = {}
    mapping = {
        "DISTRIML_DATA_DIR": ("data_dir", Path),
        "DISTRIML_RAY_NAMESPACE": ("ray_namespace", str),
        "DISTRIML_API_HOST": ("api_host", str),
        "DISTRIML_API_PORT": ("api_port", int),
        "DISTRIML_LOG_LEVEL": ("log_level", str),
        "DISTRIML_LEASE_MS": ("lease_duration_ms", int),
        "DISTRIML_RENEW_MS": ("renew_interval_ms", int),
        "DISTRIML_POLL_MS": ("request_poll_interval_ms", int),
        "DISTRIML_MAX_ATTEMPTS": ("max_attempts", int),
        "DISTRIML_MAX_LOST": ("max_lost", int),
        "DISTRIML_K_CKPT": ("k_ckpt", int),
    }
    for var, (field, cast) in mapping.items():
        if var in env:
            values[field] = cast(env[var])
    values.update({k: v for k, v in overrides.items() if v is not None})
    if "data_dir" in values:
        values["data_dir"] = Path(values["data_dir"])
    return dataclasses.replace(Config(), **values)
