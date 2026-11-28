import pickle
import time

import numpy as np
import pytest

from distriml.common import algorithms, batching, catalog, constants as C, dataset, models
from distriml.common.config import load_config
from distriml.common.errors import ContractError
from distriml.common.ids import sha256_hex
from distriml.worker.trainer import run_attempt
from tests.conftest import SAMPLES
from tests.helpers.db_utils import CLIENT_A, make_db, model_def_id, new_dataset
from tests.helpers.fake_db import FakeDB
from tests.test_worker_wk3_wk4_wk5 import crash_after, final_model

IRIS = (SAMPLES / "iris.csv").read_bytes()
DIABETES = (SAMPLES / "diabetes.csv").read_bytes()
NEW = ["knn_classifier", "knn_regressor", "nearest_centroid", "least_squares"]
DATA = {"knn_classifier": IRIS, "nearest_centroid": IRIS, "knn_regressor": DIABETES, "least_squares": DIABETES}
NAME = {IRIS: "iris.csv", DIABETES: "diabetes.csv"}


def prepare(algo, user=None):
    data = DATA[algo]
    meta = dataset.analyze(data)
    X, y = dataset.load_for_training(data, meta, algorithms.task_of(algo))
    hp = catalog.merge_hyperparams(catalog.BY_ALGORITHM[algo], user)
    return meta, X, y, hp


def train(algo, user=None, upto=None):
    """Entrena `algo` batch a batch con el mismo procedimiento que el worker. Devuelve (estado, X, y, hp)."""
    meta, X, y, hp = prepare(algo, user)
    state = algorithms.init_state(algo, hp, meta)
    total = catalog.compute_total_batches(hp, meta.n_rows)
    for k in range(1, (upto or total) + 1):
        idx = batching.batch_indices(meta.n_rows, hp["batch_size"], hp["seed"], k)
        algorithms.train_batch(state, algo, X[idx], y[idx])
    return state, X, y, hp


# ---------------------------------------------------------------------------- igual que scikit-learn
@pytest.mark.parametrize("k", [1, 3, 5, 15, 149])
def test_knn_classifier_matches_reference(k):
    from sklearn.neighbors import KNeighborsClassifier
    st, X, y, _ = train("knn_classifier", {"k": k})
    ref = KNeighborsClassifier(k).fit(X, y)
    assert np.array_equal(st["estimator"].predict(X), ref.predict(X))


@pytest.mark.parametrize("k", [1, 5, 20])
def test_knn_regressor_matches_reference(k):
    from sklearn.neighbors import KNeighborsRegressor
    st, X, y, _ = train("knn_regressor", {"k": k})
    assert np.allclose(st["estimator"].predict(X), KNeighborsRegressor(k).fit(X, y).predict(X), atol=1e-12)


def test_nearest_centroid_matches_reference():
    from sklearn.neighbors import NearestCentroid
    st, X, y, _ = train("nearest_centroid")
    assert np.array_equal(st["estimator"].predict(X), NearestCentroid().fit(X, y).predict(X))


@pytest.mark.parametrize("alpha", [0.0, 5.0, 100.0])
def test_least_squares_matches_reference(alpha):
    from sklearn.linear_model import LinearRegression, Ridge
    st, X, y, _ = train("least_squares", {"alpha": alpha})
    ref = (LinearRegression() if alpha == 0 else Ridge(alpha=alpha)).fit(X, y)
    assert np.allclose(st["estimator"].coef_, ref.coef_, atol=1e-8)
    assert np.isclose(st["estimator"].intercept_, ref.intercept_, atol=1e-8)


# ---------------------------------------------------------------------------- el batching no cambia el modelo
@pytest.mark.parametrize("algo", NEW)
def test_batch_size_does_not_change_the_model(algo):
    small, X, _, _ = train(algo, {"batch_size": 7})
    big, _, _, _ = train(algo, {"batch_size": 100_000})            # un único batch con todo el dataset
    a, b = small["estimator"].predict(X), big["estimator"].predict(X)
    if algo == "least_squares":
        assert np.allclose(a, b, atol=1e-8)                        # sumas en otro orden: solo difiere el redondeo
    else:
        assert np.array_equal(a, b)


@pytest.mark.parametrize("algo", NEW)
def test_pickle_roundtrip_in_the_middle_keeps_the_model(algo):
    full, X, _, _ = train(algo, {"batch_size": 20})
    meta, Xs, ys, hp = prepare(algo, {"batch_size": 20})
    total = catalog.compute_total_batches(hp, meta.n_rows)
    half, _, _, _ = train(algo, {"batch_size": 20}, upto=total // 2)
    resumed = algorithms.deserialize_state(algorithms.serialize_state(half))
    for k in range(total // 2 + 1, total + 1):
        idx = batching.batch_indices(meta.n_rows, hp["batch_size"], hp["seed"], k)
        algorithms.train_batch(resumed, algo, Xs[idx], ys[idx])
    assert np.array_equal(full["estimator"].predict(X), resumed["estimator"].predict(X))
    assert algorithms.evaluate_full(full, algo, X, ys) == algorithms.evaluate_full(resumed, algo, X, ys)


# ---------------------------------------------------------------------------- tamaño del estado (checkpoint)
def test_knn_state_grows_with_data_but_centroid_and_least_squares_do_not():
    sizes = {}
    for algo in ("knn_classifier", "nearest_centroid"):
        s1, *_ = train(algo, {"batch_size": 25}, upto=2)
        s2, *_ = train(algo, {"batch_size": 25}, upto=6)
        sizes[algo] = (len(algorithms.serialize_state(s1)), len(algorithms.serialize_state(s2)))
    a1, a2 = sizes["knn_classifier"]
    c1, c2 = sizes["nearest_centroid"]
    assert a2 > 2 * a1                      # kNN guarda cada punto visto
    assert abs(c2 - c1) < 100               # el centroide solo guarda sumas y contadores


# ---------------------------------------------------------------------------- casos límite y validación
@pytest.mark.parametrize("est", [models.KNNClassifier(3), models.KNNRegressor(3), models.NearestCentroid(), models.LeastSquares()])
def test_predict_before_any_batch_raises(est):
    with pytest.raises(ValueError):
        est.predict(np.zeros((1, 2)))


def test_classifiers_need_classes_on_first_batch_and_reject_unknown_labels():
    for est in (models.KNNClassifier(3), models.NearestCentroid()):
        with pytest.raises(ValueError):
            est.partial_fit(np.zeros((2, 2)), np.array([0, 1]))
        est.partial_fit(np.zeros((2, 2)), np.array([0, 1]), classes=np.array([0, 1, 2]))
        with pytest.raises(ValueError):
            est.partial_fit(np.zeros((1, 2)), np.array([7]))


def test_invalid_constructor_arguments():
    with pytest.raises(ValueError):
        models.KNNClassifier(0)
    with pytest.raises(ValueError):
        models.LeastSquares(alpha=-1)


def test_knn_with_k_larger_than_stored_points_uses_all_of_them():
    est = models.KNNClassifier(k=1000).partial_fit(np.array([[0.0], [1.0], [2.0]]), np.array([1, 1, 0]), classes=np.array([0, 1]))
    assert est.predict(np.array([[0.0]])).tolist() == [1]            # 2 votos a la clase 1 frente a 1


def test_knn_tie_goes_to_the_smallest_label():
    est = models.KNNClassifier(k=2).partial_fit(np.array([[0.0], [1.0]]), np.array([5, 3]), classes=np.array([3, 5]))
    assert est.predict(np.array([[0.5]])).tolist() == [3]


def test_knn_blocked_search_gives_the_same_answer(monkeypatch):
    st, X, _, _ = train("knn_classifier", {"k": 7})
    whole = st["estimator"].predict(X)
    monkeypatch.setattr(models, "_BLOCK_ELEMENTS", 500)                   # fuerza muchos bloques pequeños
    assert np.array_equal(st["estimator"].predict(X), whole)
    st2, X2, _, _ = train("knn_regressor", {"k": 7})
    ref = st2["estimator"].predict(X2)
    monkeypatch.setattr(models, "_BLOCK_ELEMENTS", 10_000_000)
    assert np.array_equal(st2["estimator"].predict(X2), ref)


def test_knn_neighbour_ties_prefer_the_points_stored_first():
    est = models.KNNClassifier(k=2).partial_fit(np.zeros((6, 1)), np.array([1, 1, 0, 0, 0, 0]), classes=np.array([0, 1]))
    assert est.predict(np.zeros((1, 1))).tolist() == [1]                 # los 2 primeros guardados, todos a la misma distancia
    reg = models.KNNRegressor(k=3).partial_fit(np.zeros((5, 1)), np.array([10.0, 20.0, 30.0, 999.0, 999.0]))
    assert reg.predict(np.zeros((1, 1))).tolist() == [20.0]


def test_nearest_centroid_ignores_classes_not_seen_yet():
    # La clase 1 no tiene datos: si se contara como un centroide en 0, el punto 1.0 se le asignaría.
    est = models.NearestCentroid().partial_fit(np.array([[10.0], [20.0]]), np.array([0, 2]), classes=np.array([0, 1, 2]))
    assert est.predict(np.array([[1.0], [19.0]])).tolist() == [0, 2]


def test_least_squares_with_fewer_rows_than_columns_is_finite():
    est = models.LeastSquares().partial_fit(np.ones((2, 5)), np.array([1.0, 3.0]))
    assert np.all(np.isfinite(est.predict(np.ones((3, 5)))))


def test_least_squares_recovers_an_exact_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X @ np.array([2.0, -1.0, 0.5]) + 4.0
    est = models.LeastSquares()
    for i in range(0, 200, 13):
        est.partial_fit(X[i:i + 13], y[i:i + 13])
    assert np.allclose(est.coef_, [2.0, -1.0, 0.5]) and np.isclose(est.intercept_, 4.0)


@pytest.mark.parametrize("algo", NEW)
def test_predict_validates_input_like_the_other_models(algo):
    st, _, _, _ = train(algo)
    n = len(st["feature_mean"])
    assert isinstance(algorithms.predict(st, algo, [0.0] * n), int if algorithms.task_of(algo) == "classification" else float)
    for bad in ([1.0] * (n + 1), "x", [float("nan")] * n):
        with pytest.raises(ContractError):
            algorithms.predict(st, algo, bad)


# ---------------------------------------------------------------------------- catálogo
def test_catalog_has_the_new_models_with_stable_distinct_ids():
    defs = catalog.list_definitions()
    assert len(defs) == 8 and len({d["model_def_id"] for d in defs}) == 8
    assert {d["algorithm"] for d in defs} >= set(NEW)


@pytest.mark.parametrize("algo", NEW)
def test_new_models_have_a_single_epoch(algo):
    md = catalog.BY_ALGORITHM[algo]
    assert catalog.merge_hyperparams(md, None)["epochs"] == 1
    with pytest.raises(ContractError):
        catalog.merge_hyperparams(md, {"epochs": 5})


@pytest.mark.parametrize("algo", ["knn_classifier", "knn_regressor"])
@pytest.mark.parametrize("bad", [0, -1, 1001, 2.5, "3", True])
def test_k_is_validated(algo, bad):
    with pytest.raises(ContractError):
        catalog.merge_hyperparams(catalog.BY_ALGORITHM[algo], {"k": bad})


def test_alpha_of_least_squares_is_validated():
    md = catalog.BY_ALGORITHM["least_squares"]
    assert catalog.merge_hyperparams(md, {"alpha": 2})["alpha"] == 2
    for bad in (-0.1, float("nan"), 2e6):
        with pytest.raises(ContractError):
            catalog.merge_hyperparams(md, {"alpha": bad})


def test_task_compatibility_of_the_new_models():
    iris, dia = dataset.analyze(IRIS), dataset.analyze(DIABETES)
    for algo in ("knn_classifier", "nearest_centroid"):
        catalog.check_compatibility(catalog.BY_ALGORITHM[algo], iris)
        with pytest.raises(ContractError):
            catalog.check_compatibility(catalog.BY_ALGORITHM[algo], dia)          # etiquetas continuas
    for algo in ("knn_regressor", "least_squares"):
        catalog.check_compatibility(catalog.BY_ALGORITHM[algo], dia)


# ---------------------------------------------------------------------------- BD real + worker real
QUALITY = {"knn_classifier": ("accuracy", 0.9), "nearest_centroid": ("accuracy", 0.8),
           "knn_regressor": ("r2", 0.3), "least_squares": ("r2", 0.4)}


@pytest.mark.parametrize("algo", NEW)
def test_full_flow_with_real_db_and_real_worker(tmp_path, algo):
    """Subir dataset, lanzar el job, entrenar con el worker real y predecir con el modelo resultante."""
    db = make_db(tmp_path)
    data = DATA[algo]
    ds = new_dataset(db, data=data, name=NAME[data])
    job = db.create_job(CLIENT_A, "j1", ds["dataset_id"], model_def_id(algo), {"batch_size": 25})
    assert db.get_job(CLIENT_A, job["job_id"])["total_batches"] > 1
    a = db.request_job("w1")
    assert run_attempt(a, "w1", db, db.cfg) == "COMPLETED"
    done = db.get_job(CLIENT_A, job["job_id"])
    assert done["estado"] == C.COMPLETADO and done["last_confirmed_batch"] == done["total_batches"]
    (version,) = db.list_model_versions(CLIENT_A, done["model_id"])
    metric, floor = QUALITY[algo]
    assert version["metrics"][metric] > floor, version["metrics"]
    n_features = dataset.analyze(data).n_features
    out = db.predict(CLIENT_A, done["model_id"], {"features": [0.0] * n_features})
    assert out["version_id"] == version["version_id"] and out["sha256"] == version["sha256"]
    assert isinstance(out["prediction"], (int, float))


@pytest.mark.parametrize("algo", NEW)
def test_resume_after_worker_crash_gives_the_same_model(algo):
    """Un worker muere tras confirmar 3 batches; otro continúa desde el checkpoint y el modelo es idéntico."""
    cfg = load_config(lease_duration_ms=300, renew_interval_ms=100, epsilon_ms=50, request_poll_interval_ms=20)
    data = DATA[algo]
    meta = dataset.analyze(data)
    X, _ = dataset.load_for_training(data, meta, algorithms.task_of(algo))

    def seeded(rid):
        db = FakeDB(cfg)
        ds = db.create_dataset(CLIENT_A, rid, NAME[data], sha256_hex(data), len(data), data)
        db.create_job(CLIENT_A, rid, ds["dataset_id"], model_def_id(algo), {"batch_size": 25})
        return db

    clean = seeded("r1")
    run_attempt(clean.request_job("w0"), "w0", clean, cfg)

    db = seeded("r1")
    a1 = db.request_job("w0")
    with pytest.raises(BaseException):
        run_attempt(a1, "w0", crash_after(db, 3), cfg)
    time.sleep(0.45)                                              # vence el lease del attempt muerto
    a2 = db.request_job("w1")
    assert a2.next_batch == 4 and a2.checkpoint_sha256
    assert run_attempt(a2, "w1", db, cfg) == "COMPLETED"
    assert np.array_equal(final_model(clean)["estimator"].predict(X), final_model(db)["estimator"].predict(X))
