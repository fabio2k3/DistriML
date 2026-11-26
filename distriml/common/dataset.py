"""Formato de dataset y metadatos.

CSV con cabecera. Las columnas de características son numéricas y la ÚLTIMA columna es la etiqueta.

La etiqueta puede ser:
  - numérica (0, 1, 2 ... o un valor real para regresión), o
  - TEXTO (setosa, versicolor ...): los nombres se ordenan alfabéticamente y se numeran 0..k-1. Ese orden
    es determinista, así que DB y workers obtienen siempre el mismo número para cada nombre, y el modelo
    guarda los nombres para responder con la clase por su nombre al predecir.

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


def _read(data: bytes) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """(X, y, nombres de clase). Si la etiqueta es texto, `y` son los índices en `nombres` (orden alfabético);
    si es numérica, `nombres` es None."""
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
        X = np.array([r[:-1] for r in body], dtype=np.float64)
    except ValueError:
        raise bad_request("invalid_dataset", "hay valores no numéricos en las columnas de características") from None
    if not np.all(np.isfinite(X)):
        raise bad_request("invalid_dataset", "hay valores no finitos (NaN o infinito)")
    raw = [r[-1].strip() for r in body]
    try:
        y = np.array(raw, dtype=np.float64)
        names = None
        if not np.all(np.isfinite(y)):
            raise bad_request("invalid_dataset", "hay valores no finitos (NaN o infinito)")
    except ValueError:                                  # alguna etiqueta no es un número -> etiquetas de texto
        if any(label == "" for label in raw):
            raise bad_request("invalid_dataset", "hay etiquetas vacías") from None
        names = sorted(set(raw))
        if len(names) > MAX_CLASSES:
            raise bad_request("invalid_dataset", f"demasiadas clases ({len(names)}; máximo {MAX_CLASSES})") from None
        index = {name: i for i, name in enumerate(names)}
        y = np.array([index[label] for label in raw], dtype=np.float64)
    return X, y, names


def class_names_of(data: bytes) -> list[str] | None:
    """Nombres de las clases (en el orden de sus números 0..k-1) si la etiqueta del CSV es texto; None si es numérica."""
    return _read(data)[2]


def analyze(data: bytes, max_bytes: int | None = None) -> DatasetMeta:
    """Valida el CSV (400 si es inválido) y calcula n_rows, n_features, classes, media y desviación."""
    if max_bytes is not None and len(data) > max_bytes:
        raise bad_request("dataset_too_large", f"el dataset supera el máximo de {max_bytes} bytes")
    X, y, names = _read(data)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    classes: list[int] | None = None
    if names is not None:
        classes = list(range(len(names)))
    elif np.all(y == np.round(y)):
        unique = np.unique(y)
        if len(unique) <= MAX_CLASSES:
            classes = [int(c) for c in unique]
    return DatasetMeta(
        n_rows=int(X.shape[0]),
        n_features=int(X.shape[1]),
        classes=classes,
        feature_mean=mean.tolist(),
        feature_std=std.tolist(),
    )


def parse(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) sin estandarizar."""
    X, y, _ = _read(data)
    return X, y


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
