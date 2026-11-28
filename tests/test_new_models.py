"""Modelos base hechos a mano (distriml/common/models.py): kNN, centroide más cercano, mínimos cuadrados,
regresión softmax y regresión lineal por descenso de gradiente.

scikit-learn se usa AQUÍ solo como referencia para comprobar que nuestra implementación calcula lo mismo;
el código del sistema no depende de él para estos modelos.
"""
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
GD = ["softmax_regression", "linear_regression_gd"]       # descenso de gradiente: iteran por épocas
ALL = NEW + GD
DATA = {"knn_classifier": IRIS, "nearest_centroid": IRIS, "knn_regressor": DIABETES, "least_squares": DIABETES,
        "softmax_regression": IRIS, "linear_regression_gd": DIABETES}
# Hiperparámetros de las pruebas con BD y worker reales: pocas épocas para que sean rápidas.
FLOW_HP = {"softmax_regression": {"batch_size": 25, "epochs": 10}, "linear_regression_gd": {"batch_size": 50, "epochs": 10}}
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


@pytest.mark.parametrize("algo", ALL)
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
@pytest.mark.parametrize("est", [models.KNNClassifier(3), models.KNNRegressor(3), models.NearestCentroid(), models.LeastSquares(),
                                 models.SoftmaxRegression(), models.LinearRegressionGD()])
def test_predict_before_any_batch_raises(est):
    with pytest.raises(ValueError):
        est.predict(np.zeros((1, 2)))


def test_classifiers_need_classes_on_first_batch_and_reject_unknown_labels():
    for est in (models.KNNClassifier(3), models.NearestCentroid(), models.SoftmaxRegression()):
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
    for cls in (models.SoftmaxRegression, models.LinearRegressionGD):
        for bad in ({"lr": 0}, {"lr": -0.1}, {"lr": float("nan")}, {"l2": -1}):
            with pytest.raises(ValueError):
                cls(**bad)


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


@pytest.mark.parametrize("algo", ALL)
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
    assert len(defs) == 10 and len({d["model_def_id"] for d in defs}) == 10
    assert {d["algorithm"] for d in defs} >= set(ALL)


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
    for algo in ("knn_classifier", "nearest_centroid", "softmax_regression"):
        catalog.check_compatibility(catalog.BY_ALGORITHM[algo], iris)
        with pytest.raises(ContractError):
            catalog.check_compatibility(catalog.BY_ALGORITHM[algo], dia)          # etiquetas continuas
    for algo in ("knn_regressor", "least_squares", "linear_regression_gd"):
        catalog.check_compatibility(catalog.BY_ALGORITHM[algo], dia)


# ---------------------------------------------------------------------------- BD real + worker real
QUALITY = {"knn_classifier": ("accuracy", 0.9), "nearest_centroid": ("accuracy", 0.8),
           "knn_regressor": ("r2", 0.3), "least_squares": ("r2", 0.4),
           "softmax_regression": ("accuracy", 0.85), "linear_regression_gd": ("r2", 0.3)}


@pytest.mark.parametrize("algo", ALL)
def test_full_flow_with_real_db_and_real_worker(tmp_path, algo):
    """Subir dataset, lanzar el job, entrenar con el worker real y predecir con el modelo resultante."""
    db = make_db(tmp_path)
    data = DATA[algo]
    ds = new_dataset(db, data=data, name=NAME[data])
    job = db.create_job(CLIENT_A, "j1", ds["dataset_id"], model_def_id(algo), FLOW_HP.get(algo, {"batch_size": 25}))
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


@pytest.mark.parametrize("algo", ALL)
def test_resume_after_worker_crash_gives_the_same_model(algo):
    """Un worker muere tras confirmar 3 batches; otro continúa desde el checkpoint y el modelo es idéntico."""
    cfg = load_config(lease_duration_ms=300, renew_interval_ms=100, epsilon_ms=50, request_poll_interval_ms=20)
    data = DATA[algo]
    meta = dataset.analyze(data)
    X, _ = dataset.load_for_training(data, meta, algorithms.task_of(algo))

    def seeded(rid):
        db = FakeDB(cfg)
        ds = db.create_dataset(CLIENT_A, rid, NAME[data], sha256_hex(data), len(data), data)
        db.create_job(CLIENT_A, rid, ds["dataset_id"], model_def_id(algo), FLOW_HP.get(algo, {"batch_size": 25}))
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


# ---------------------------------------------------------------------------- descenso de gradiente
def _standardized(data, task):
    meta = dataset.analyze(data)
    return dataset.load_for_training(data, meta, task)


def _numeric_gradient(model, X, y, which, eps=1e-6):
    """Derivada numérica (diferencias centrales) de `model.loss` respecto a cada elemento de W_ o b_."""
    param = getattr(model, which)
    out = np.zeros_like(param)
    for i in np.ndindex(param.shape):
        old = param[i]
        param[i] = old + eps
        up = model.loss(X, y)
        param[i] = old - eps
        down = model.loss(X, y)
        param[i] = old
        out[i] = (up - down) / (2 * eps)
    return out


def test_softmax_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 4)), rng.integers(0, 3, 40)
    m = models.SoftmaxRegression(l2=0.3)
    m.classes_, m.W_, m.b_ = np.arange(3), rng.normal(size=(4, 3)), rng.normal(size=3)
    gW, gb = m._gradient(X, y)
    assert np.allclose(gW, _numeric_gradient(m, X, y, "W_"), atol=1e-6)
    assert np.allclose(gb, _numeric_gradient(m, X, y, "b_"), atol=1e-6)


def test_linear_gd_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40) * 10
    m = models.LinearRegressionGD(l2=0.3)
    m.W_, m.b_ = rng.normal(size=4), rng.normal(size=1)
    gW, gb = m._gradient(X, y)
    assert np.allclose(gW, _numeric_gradient(m, X, y, "W_"), atol=1e-5)
    assert np.allclose(gb, _numeric_gradient(m, X, y, "b_"), atol=1e-5)


def test_softmax_converges_to_the_same_solution_as_scikit_learn():
    from sklearn.linear_model import LogisticRegression
    X, y = _standardized(IRIS, "classification")
    l2 = 1e-2
    m = models.SoftmaxRegression(lr=1.0, l2=l2)
    for _ in range(4000):                                          # batch completo: descenso de gradiente clásico
        m.partial_fit(X, y, classes=np.unique(y))
    ref = LogisticRegression(C=1.0 / (len(X) * l2), max_iter=20_000, tol=1e-12).fit(X, y)   # misma pérdida
    assert np.allclose(m.W_.T, ref.coef_, atol=5e-3) and np.allclose(m.b_, ref.intercept_, atol=5e-3)
    assert np.array_equal(m.predict(X), ref.predict(X))


def test_linear_gd_converges_to_the_ridge_solution():
    from sklearn.linear_model import Ridge
    X, y = _standardized(DIABETES, "regression")
    l2 = 1e-2
    m = models.LinearRegressionGD(lr=0.2, l2=l2)      # con lr=0.5 diverge: el límite de este dataset es ≈ 0.50
    for _ in range(10_000):
        m.partial_fit(X, y)
    ref = Ridge(alpha=len(X) * l2).fit(X, y)                       # misma pérdida: alpha = n·l2
    assert np.allclose(m.coef_, ref.coef_, atol=1e-3) and abs(m.intercept_ - ref.intercept_) < 1e-3
    assert np.allclose(m.predict(X), ref.predict(X), atol=1e-2)


def test_linear_gd_reaches_the_exact_least_squares_fit():
    """Con l2=0 el descenso de gradiente tiende a la solución exacta que calcula `LeastSquares`."""
    X, y = _standardized(DIABETES, "regression")
    exact = models.LeastSquares().partial_fit(X, y)
    m = models.LinearRegressionGD(lr=0.2)
    for _ in range(20_000):
        m.partial_fit(X, y)
    assert np.allclose(m.predict(X), exact.predict(X), atol=0.05)


@pytest.mark.parametrize("cls,data,task", [(models.SoftmaxRegression, IRIS, "classification"),
                                           (models.LinearRegressionGD, DIABETES, "regression")])
def test_full_batch_loss_never_increases_with_a_small_step(cls, data, task):
    X, y = _standardized(data, task)
    m = cls(lr=0.05)
    kw = {"classes": np.unique(y)} if task == "classification" else {}
    m.partial_fit(X, y, **kw)
    losses = [m.loss(X, y)]
    for _ in range(60):
        m.partial_fit(X, y, **kw)
        losses.append(m.loss(X, y))
    assert all(b <= a + 1e-12 for a, b in zip(losses, losses[1:])) and losses[-1] < losses[0]


@pytest.mark.parametrize("algo", GD)
def test_gradient_descent_is_deterministic(algo):
    a, X, _, _ = train(algo, {"batch_size": 30, "epochs": 4})
    b, _, _, _ = train(algo, {"batch_size": 30, "epochs": 4})
    assert np.array_equal(a["estimator"].W_, b["estimator"].W_) and np.array_equal(a["estimator"].b_, b["estimator"].b_)
    assert a["estimator"].steps_ == b["estimator"].steps_ > 1


@pytest.mark.parametrize("algo", GD)
def test_gradient_descent_models_do_use_epochs(algo):
    md = catalog.BY_ALGORITHM[algo]
    assert "epochs" in md.default_hyperparams and "epochs" not in md.fixed_hyperparams
    one, _, _, hp1 = train(algo, {"batch_size": 50, "epochs": 1})
    three, _, _, hp3 = train(algo, {"batch_size": 50, "epochs": 3})
    assert three["estimator"].steps_ == 3 * one["estimator"].steps_
    assert not np.array_equal(one["estimator"].W_, three["estimator"].W_)


@pytest.mark.parametrize("algo", GD)
def test_default_hyperparameters_learn_something_useful(algo):
    state, X, y, _ = train(algo)
    metrics = algorithms.evaluate_full(state, algo, X, y)
    assert metrics["accuracy"] > 0.9 if algo == "softmax_regression" else metrics["r2"] > 0.45, metrics


@pytest.mark.parametrize("algo", GD)
def test_lr_and_l2_are_validated(algo):
    md = catalog.BY_ALGORITHM[algo]
    assert catalog.merge_hyperparams(md, {"lr": 0.2, "l2": 0.01})["lr"] == 0.2
    for bad in ({"lr": 0}, {"lr": -1}, {"lr": float("nan")}, {"lr": 2e3}, {"l2": -0.1}, {"l2": 2e6}, {"lr": "0.1"}):
        with pytest.raises(ContractError):
            catalog.merge_hyperparams(md, bad)


def test_softmax_does_not_overflow_with_huge_features():
    X, y = _standardized(IRIS, "classification")
    m = models.SoftmaxRegression(lr=0.5)
    m.partial_fit(X * 1e3, y, classes=np.unique(y))
    assert np.isfinite(m.W_).all() and np.isfinite(m._probabilities(X * 1e3)).all()


def test_softmax_handles_classes_not_seen_in_the_first_batches():
    X, y = _standardized(IRIS, "classification")
    order = np.argsort(y, kind="stable")                           # primero solo la clase 0, luego la 1...
    m = models.SoftmaxRegression(lr=0.5)
    for lo in range(0, len(X), 25):
        sel = order[lo:lo + 25]
        m.partial_fit(X[sel], y[sel], classes=np.unique(y))
    assert set(np.unique(m.predict(X))) <= set(np.unique(y)) and np.isfinite(m.W_).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")      # divergir a propósito desborda float64
def test_the_stable_step_limit_of_linear_gd_is_two_over_the_largest_eigenvalue():
    """Con batch completo, el descenso converge si lr < 2/λmax (λmax: mayor autovalor de XᵀX/n) y diverge si no."""
    X, y = _standardized(DIABETES, "regression")
    limit = 2.0 / np.linalg.eigvalsh(X.T @ X / len(X)).max()
    stable, unstable = models.LinearRegressionGD(lr=0.9 * limit), models.LinearRegressionGD(lr=1.1 * limit)
    for _ in range(300):
        stable.partial_fit(X, y)
    assert np.isfinite(stable.W_).all() and stable.loss(X, y) < 0.5 * np.var(y)
    with pytest.raises(ValueError, match="divergió"):
        for _ in range(20_000):
            unstable.partial_fit(X, y)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")      # divergir a propósito desborda float64
def test_linear_gd_divergence_is_detected_instead_of_saving_nans():
    X, y = _standardized(DIABETES, "regression")
    m = models.LinearRegressionGD(lr=1e3)
    with pytest.raises(ValueError, match="divergió"):
        for _ in range(500):
            m.partial_fit(X, y)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")      # divergir a propósito desborda float64
def test_a_diverging_job_is_reported_as_a_worker_error_not_saved():
    """lr exagerado -> el worker reporta fallo del entrenamiento (cuenta para max_attempts) y no sube ningún modelo."""
    cfg = load_config(lease_duration_ms=2000, renew_interval_ms=500, epsilon_ms=50, request_poll_interval_ms=20)
    db = FakeDB(cfg)
    ds = db.create_dataset(CLIENT_A, "r1", "diabetes.csv", sha256_hex(DIABETES), len(DIABETES), DIABETES)
    job = db.create_job(CLIENT_A, "r1", ds["dataset_id"], model_def_id("linear_regression_gd"),
                        {"lr": 1000.0, "epochs": 50, "batch_size": 20})
    a = db.request_job("w0")
    assert run_attempt(a, "w0", db, cfg) == "FAILED"
    view = db.get_job(CLIENT_A, job["job_id"])
    assert view["estado"] == C.PENDIENTE and view["attempts_count"] == 1 and not view.get("model_id")
    assert view["attempts"][0]["revoke_reason"] == C.WORKER_ERROR
