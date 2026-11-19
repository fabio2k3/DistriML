"""Catálogo de modelos disponibles: predefinido, estático y común a todos los nodos.

No está en ningún log ni en la BD: cada nodo lo lleva en el código. Cambiarlo implica redesplegar.
Todos los nodos deben tener el mismo catálogo y los mismos `model_def_id` (derivados del algoritmo).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from distriml.common.batching import MAX_SEED, total_batches
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import bad_request
from distriml.common.ids import derive_model_def_id


@dataclass(frozen=True)
class ModelDef:
    model_def_id: str
    name: str
    algorithm: str
    task: str                                  # "classification" | "regression"
    version: int
    default_hyperparams: dict
    hyperparam_schema: dict
    input_format: dict
    output_format: dict
    fixed_hyperparams: dict = field(default_factory=dict)   # no configurables (p. ej. epochs=1 en gaussian_nb)


def _int(lo: int, hi: int) -> dict:
    return {"type": "int", "min": lo, "max": hi}


def _float(lo: float, hi: float, exclusive_min: bool = False) -> dict:
    return {"type": "float", "min": lo, "max": hi, "exclusive_min": exclusive_min}


_BATCH = _int(1, 100_000)
_EPOCHS = _int(1, 1_000)
_SEED = _int(0, MAX_SEED)
_INPUT = {"type": "tabular_csv", "label": "last_column", "min_features": 1}
_OUTPUT = {"type": "pickle", "protocol": 4, "keys": ["estimator", "feature_mean", "feature_std", "classes"]}


def _def(name, algorithm, task, defaults, schema, fixed=None) -> ModelDef:
    return ModelDef(
        model_def_id=derive_model_def_id(algorithm, 1),
        name=name, algorithm=algorithm, task=task, version=1,
        default_hyperparams=defaults, hyperparam_schema=schema,
        input_format=_INPUT, output_format=_OUTPUT, fixed_hyperparams=fixed or {},
    )


_DEFINITIONS = [
    _def("Linear Regression (SGD) v1", "sgd_regressor", "regression",
         {"batch_size": 32, "epochs": 10, "seed": 42, "alpha": 0.0001, "eta0": 0.01},
         {"batch_size": _BATCH, "epochs": _EPOCHS, "seed": _SEED,
          "alpha": _float(0.0, 1e6), "eta0": _float(0.0, 1e3, exclusive_min=True)}),
    _def("Logistic Regression (SGD) v1", "sgd_logistic", "classification",
         {"batch_size": 32, "epochs": 10, "seed": 42, "alpha": 0.0001, "eta0": 0.01},
         {"batch_size": _BATCH, "epochs": _EPOCHS, "seed": _SEED,
          "alpha": _float(0.0, 1e6), "eta0": _float(0.0, 1e3, exclusive_min=True)}),
    _def("Perceptron v1", "perceptron", "classification",
         {"batch_size": 32, "epochs": 10, "seed": 42, "eta0": 1.0},
         {"batch_size": _BATCH, "epochs": _EPOCHS, "seed": _SEED,
          "eta0": _float(0.0, 1e3, exclusive_min=True)}),
    _def("Gaussian Naive Bayes v1", "gaussian_nb", "classification",
         {"batch_size": 32, "seed": 42, "var_smoothing": 1e-9},
         {"batch_size": _BATCH, "seed": _SEED, "var_smoothing": _float(0.0, 1.0)},
         fixed={"epochs": 1}),   # repetir épocas duplicaría las estadísticas
]

CATALOG: dict[str, ModelDef] = {d.model_def_id: d for d in _DEFINITIONS}
BY_ALGORITHM: dict[str, ModelDef] = {d.algorithm: d for d in _DEFINITIONS}


def get(model_def_id: str) -> ModelDef:
    """400 si el model_def_id no está en el catálogo."""
    try:
        return CATALOG[model_def_id]
    except (KeyError, TypeError):
        raise bad_request("model_def_not_found", "model_def_id no está en el catálogo") from None


def public_dict(d: ModelDef) -> dict:
    """Forma que devuelve GET /model-definitions."""
    return {
        "model_def_id": d.model_def_id,
        "name": d.name,
        "algorithm": d.algorithm,
        "version": d.version,
        "default_hyperparams": dict(d.default_hyperparams),
        "hyperparam_schema": dict(d.hyperparam_schema),
    }


def list_definitions() -> list[dict]:
    return [public_dict(d) for d in _DEFINITIONS]


def _check(name: str, spec: dict, value) -> None:
    if spec["type"] == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    else:
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok:
        raise bad_request("invalid_hyperparams", f"{name}: se esperaba {spec['type']}")
    lo, hi = spec["min"], spec["max"]
    below = value <= lo if spec.get("exclusive_min") else value < lo
    if below or value > hi:
        raise bad_request("invalid_hyperparams", f"{name}: fuera de rango [{lo}, {hi}]")


def merge_hyperparams(model_def: ModelDef, user: dict | None) -> dict:
    """Mezcla los hiperparámetros del usuario con los por defecto y valida el esquema (400 si falla).

    El resultado (incluida la seed y los parámetros fijos) es lo que se guarda en el job.
    """
    user = dict(user or {})
    unknown = set(user) - set(model_def.hyperparam_schema)
    if unknown:
        raise bad_request("invalid_hyperparams", f"hiperparámetros desconocidos: {sorted(unknown)}")
    merged = {**model_def.default_hyperparams, **user}
    for key, spec in model_def.hyperparam_schema.items():
        _check(key, spec, merged[key])
    merged.update(model_def.fixed_hyperparams)
    return merged


def check_compatibility(model_def: ModelDef, meta: DatasetMeta) -> None:
    """400 si el dataset no sirve para el modelo."""
    if model_def.task == "classification" and meta.classes is None:
        raise bad_request("dataset_incompatible", "un modelo de clasificación exige etiquetas enteras (≤ 50 clases)")
    if meta.n_features < model_def.input_format["min_features"]:
        raise bad_request("dataset_incompatible", "el dataset no cumple el input_format del modelo")


def compute_total_batches(hyperparams: dict, n_rows: int) -> int:
    """total_batches = epochs × ⌈n_rows / batch_size⌉, a partir de los metadatos (sin leer el artefacto)."""
    return total_batches(n_rows, hyperparams["epochs"], hyperparams["batch_size"])
