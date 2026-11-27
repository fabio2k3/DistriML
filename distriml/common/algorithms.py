"""Algoritmos del catálogo y entrenamiento incremental.

Los cuatro primeros usan scikit-learn (`partial_fit`); kNN, centroide más cercano y mínimos cuadrados están
implementados a mano en `models.py` (solo numpy) con la misma interfaz.

Estado de un modelo (checkpoint y modelo final): un diccionario
    {"estimator", "feature_mean", "feature_std", "classes", "class_names"}
serializado con pickle (protocolo fijo). El estado interno de `partial_fit` viaja dentro del pickle,
así que reanudar desde un checkpoint produce el mismo resultado que continuar sin interrupción
(lo comprueba tests/test_algorithms.py).

SEGURIDAD: pickle ejecuta código al cargarse. Es aceptable aquí porque solo los workers del
sistema escriben artefactos de modelo y la red es de confianza. El sha256 garantiza
integridad, no autenticidad.
"""
from __future__ import annotations

import pickle

import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier, SGDRegressor
from sklearn.metrics import r2_score
from sklearn.naive_bayes import GaussianNB

from distriml.common import catalog, models
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import bad_request

PICKLE_PROTOCOL = 4


def task_of(algorithm: str) -> str:
    return catalog.BY_ALGORITHM[algorithm].task


def _build_estimator(algorithm: str, hp: dict):
    seed = hp.get("seed", 0)
    if algorithm == "sgd_regressor":
        return SGDRegressor(learning_rate="constant", shuffle=False, random_state=seed,
                            alpha=hp["alpha"], eta0=hp["eta0"])
    if algorithm == "sgd_logistic":
        return SGDClassifier(loss="log_loss", learning_rate="constant", shuffle=False, random_state=seed,
                             alpha=hp["alpha"], eta0=hp["eta0"])
    if algorithm == "perceptron":
        return Perceptron(shuffle=False, random_state=seed, eta0=hp["eta0"])
    if algorithm == "gaussian_nb":
        return GaussianNB(var_smoothing=hp["var_smoothing"])
    if algorithm == "knn_classifier":
        return models.KNNClassifier(k=hp["k"])
    if algorithm == "knn_regressor":
        return models.KNNRegressor(k=hp["k"])
    if algorithm == "nearest_centroid":
        return models.NearestCentroid()
    if algorithm == "least_squares":
        return models.LeastSquares(alpha=hp["alpha"])
    raise ValueError(f"algoritmo desconocido: {algorithm}")


def init_state(algorithm: str, hyperparams: dict, meta: DatasetMeta, class_names: list[str] | None = None) -> dict:
    """Estado inicial (batch 0): estimador sin entrenar + metadatos de estandarización.
    `class_names` (si las etiquetas del dataset eran texto) permite responder con el nombre de la clase."""
    return {
        "estimator": _build_estimator(algorithm, hyperparams),
        "feature_mean": list(meta.feature_mean),
        "feature_std": list(meta.feature_std),
        "classes": None if meta.classes is None else list(meta.classes),
        "class_names": None if class_names is None else list(class_names),
    }


def _score(estimator, task: str, X: np.ndarray, y: np.ndarray) -> float:
    if task == "classification":
        return float(np.mean(estimator.predict(X) == y))
    if len(y) < 2:
        return 0.0
    return float(r2_score(y, estimator.predict(X)))


def train_batch(state: dict, algorithm: str, X: np.ndarray, y: np.ndarray) -> dict:
    """Un `partial_fit` sobre el batch (X ya estandarizada). Devuelve las métricas del batch:
    {"n": filas, "score": exactitud (clasificación) o R² (regresión)}."""
    est = state["estimator"]
    task = task_of(algorithm)
    if task == "classification":
        est.partial_fit(X, y, classes=np.asarray(state["classes"]))
    else:
        est.partial_fit(X, y)
    return {"n": int(len(y)), "score": _score(est, task, X, y)}


def evaluate_full(state: dict, algorithm: str, X: np.ndarray, y: np.ndarray) -> dict:
    """Métricas finales sobre el dataset completo: exactitud (clasificación) o R², y error cuadrático medio."""
    est = state["estimator"]
    task = task_of(algorithm)
    pred = est.predict(X)
    mse = float(np.mean((pred - y) ** 2))
    if task == "classification":
        return {"accuracy": float(np.mean(pred == y)), "mse": mse, "n": int(len(y))}
    return {"r2": _score(est, task, X, y), "mse": mse, "n": int(len(y))}


def predict(state: dict, algorithm: str, features) -> int | float:
    """Predicción para un vector de características SIN estandarizar (se estandariza con los metadatos)."""
    try:
        x = np.asarray(features, dtype=np.float64)
    except (TypeError, ValueError):
        raise bad_request("invalid_input", "features debe ser una lista de números") from None
    mean = np.asarray(state["feature_mean"])
    if x.ndim != 1 or x.shape[0] != mean.shape[0] or not np.all(np.isfinite(x)):
        raise bad_request("invalid_input", f"se esperaban {mean.shape[0]} características numéricas")
    xs = ((x - mean) / np.asarray(state["feature_std"])).reshape(1, -1)
    value = state["estimator"].predict(xs)[0]
    return int(value) if task_of(algorithm) == "classification" else float(value)


def label_of(state: dict, prediction) -> str | None:
    """Nombre de la clase predicha, o None si el modelo se entrenó con etiquetas numéricas."""
    names, classes = state.get("class_names"), state.get("classes")
    if not names or classes is None or prediction not in classes:
        return None
    return names[classes.index(prediction)]


def serialize_state(state: dict) -> bytes:
    return pickle.dumps(state, protocol=PICKLE_PROTOCOL)


def deserialize_state(blob: bytes) -> dict:
    return pickle.loads(blob)  # noqa: S301 (artefactos de confianza, ver aviso al inicio)
