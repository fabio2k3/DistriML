"""Formato de dataset y metadatos.

CSV con cabecera, todas las columnas numéricas y la ÚLTIMA columna es la etiqueta.
`analyze` valida el CSV y calcula los metadatos que la DB guarda al subirlo; el worker usa
`load_for_training` para obtener (X estandarizada, y) con esos mismos metadatos.
"""
from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass

import numpy as np

from distriml.common.errors import bad_request

MAX_CLASSES = 50


@dataclass(frozen=True)
class DatasetMeta:
    n_rows: int
    n_features: int
    classes: list[int] | None      # None => solo sirve para regresión
    feature_mean: list[float]
    feature_std: list[float]       # std = 0 se sustituye por 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetMeta":
        return cls(
            n_rows=int(d["n_rows"]),
            n_features=int(d["n_features"]),
            classes=None if d["classes"] is None else [int(c) for c in d["classes"]],
            feature_mean=[float(v) for v in d["feature_mean"]],
            feature_std=[float(v) for v in d["feature_std"]],
        )


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise bad_request("invalid_dataset", "el CSV debe estar codificado en UTF-8") from None


def _read_matrix(data: bytes) -> np.ndarray:
    rows = [r for r in csv.reader(io.StringIO(_decode(data))) if r and any(c.strip() for c in r)]
    if not rows:
        raise bad_request("invalid_dataset", "el archivo está vacío")
    header, body = rows[0], rows[1:]
    if len(header) < 2:
        raise bad_request("invalid_dataset", "se necesita al menos una columna de características y la etiqueta")
    if len(body) < 2:
        raise bad_request("invalid_dataset", "se necesitan al menos 2 filas de datos")
    width = len(header)
    for line_no, row in enumerate(body, start=2):
        if len(row) != width:
            raise bad_request("invalid_dataset", f"línea {line_no}: {len(row)} columnas, se esperaban {width}")
    try:
        matrix = np.array(body, dtype=np.float64)
    except ValueError:
        raise bad_request("invalid_dataset", "hay valores no numéricos") from None
    if not np.all(np.isfinite(matrix)):
        raise bad_request("invalid_dataset", "hay valores no finitos (NaN o infinito)")
    return matrix


def analyze(data: bytes, max_bytes: int | None = None) -> DatasetMeta:
    """Valida el CSV (400 si es inválido) y calcula n_rows, n_features, classes, media y desviación."""
    if max_bytes is not None and len(data) > max_bytes:
        raise bad_request("dataset_too_large", f"el dataset supera el máximo de {max_bytes} bytes")
    m = _read_matrix(data)
    X, y = m[:, :-1], m[:, -1]
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    classes: list[int] | None = None
    if np.all(y == np.round(y)):
        unique = np.unique(y)
        if len(unique) <= MAX_CLASSES:
            classes = [int(c) for c in unique]
    return DatasetMeta(
        n_rows=int(m.shape[0]),
        n_features=int(X.shape[1]),
        classes=classes,
        feature_mean=mean.tolist(),
        feature_std=std.tolist(),
    )


def parse(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) sin estandarizar."""
    m = _read_matrix(data)
    return m[:, :-1], m[:, -1]


def standardize(X: np.ndarray, mean, std) -> np.ndarray:
    return (np.asarray(X, dtype=np.float64) - np.asarray(mean)) / np.asarray(std)


def load_for_training(data: bytes, meta: DatasetMeta, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(X estandarizada con los metadatos del dataset, y). En clasificación y es entera."""
    X, y = parse(data)
    if X.shape[0] != meta.n_rows or X.shape[1] != meta.n_features:
        raise bad_request("invalid_dataset", "el dataset no coincide con sus metadatos")
    X = standardize(X, meta.feature_mean, meta.feature_std)
    if task == "classification":
        y = y.astype(np.int64)
    return X, y
