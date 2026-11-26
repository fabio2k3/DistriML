"""Conversión entre filas CSV (todo cadenas) y filas en memoria (con sus tipos).

En memoria cada fila es un diccionario con las mismas claves que las columnas de la tabla; solo cambia
el tipo de los valores:
  - enteros          -> int (celda vacía = None en los que admiten ausencia)
  - columnas JSON    -> dict / list / None
  - booleanos        -> bool (se guardan como "1" / "0")
  - textos opcionales (p. ej. attempt_vigente) -> None cuando la celda está vacía
"""
from __future__ import annotations

from distriml.db.csvtable import dumps, loads

INT_COLS = {
    "datasets": {"size", "n_rows", "n_features", "created_at"},
    "jobs": {"total_batches", "attempts_count", "lost_count", "last_confirmed_batch", "completed_at",
             "created_at", "model_version"},
    "attempts": {"base_batch", "created_at"},
    "job_batches": {"batch_idx"},
    "models": {"created_at"},
    "model_versions": {"version"},
    "inference_stats": {"minute", "count"},
}
FLOAT_COLS = {"inference_stats": {"latency_sum"}}
JSON_COLS = {
    "datasets": {"classes", "feature_mean", "feature_std"},
    "jobs": {"hyperparams"},
    "job_batches": {"metrics"},
    "models": {"training_metrics"},
    "model_versions": {"metrics"},
    "requests": {"response"},
}
BOOL_COLS = {"model_versions": {"canonical"}}
NULLABLE_TEXT = {
    "jobs": {"attempt_vigente", "model_id", "canonical_version_id"},
    "attempts": {"base_attempt", "revoke_reason"},
    "job_batches": {"checkpoint_sha256"},
}


def decode(table: str, row: dict) -> dict:
    """Fila CSV -> fila en memoria."""
    out = dict(row)
    for col in INT_COLS.get(table, ()):
        out[col] = None if row[col] == "" else int(row[col])
    for col in FLOAT_COLS.get(table, ()):
        out[col] = float(row[col])
    for col in JSON_COLS.get(table, ()):
        out[col] = loads(row[col])
    for col in BOOL_COLS.get(table, ()):
        out[col] = row[col] == "1"
    for col in NULLABLE_TEXT.get(table, ()):
        out[col] = None if row[col] == "" else row[col]
    return out


def encode(table: str, row: dict) -> dict:
    """Fila en memoria -> fila lista para CsvTable (las claves ajenas a la tabla se ignoran)."""
    out = dict(row)
    for col in JSON_COLS.get(table, ()):
        out[col] = dumps(row.get(col))
    for col in BOOL_COLS.get(table, ()):
        out[col] = "1" if row.get(col) else "0"
    return out
