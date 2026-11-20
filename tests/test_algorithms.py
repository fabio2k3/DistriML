import numpy as np
import pytest

from tests.conftest import SAMPLES
from distriml.common import algorithms, batching, catalog, dataset

ALGOS = ["sgd_logistic", "perceptron", "gaussian_nb"]


def _train(algo, data, meta, hp, k_from, k_to, state=None):
    task = algorithms.task_of(algo)
    X, y = dataset.load_for_training(data, meta, task)
    state = state or algorithms.init_state(algo, hp, meta)
    for k in range(k_from, k_to + 1):
        idx = batching.batch_indices(meta.n_rows, hp["batch_size"], hp["seed"], k)
        algorithms.train_batch(state, algo, X[idx], y[idx])
    return state, X, y


@pytest.mark.parametrize("algo", ALGOS)
def test_resume_from_checkpoint_equals_continuous(algo):
    data = (SAMPLES / "iris.csv").read_bytes()
    meta = dataset.analyze(data)
    user = {"batch_size": 20} if algo == "gaussian_nb" else {"epochs": 3, "batch_size": 20}
    hp = catalog.merge_hyperparams(catalog.BY_ALGORITHM[algo], user)
    total = catalog.compute_total_batches(hp, meta.n_rows)
    full, X, y = _train(algo, data, meta, hp, 1, total)
    half, _, _ = _train(algo, data, meta, hp, 1, total // 2)
    resumed = algorithms.deserialize_state(algorithms.serialize_state(half))
    resumed, _, _ = _train(algo, data, meta, hp, total // 2 + 1, total, state=resumed)
    f, r = algorithms.evaluate_full(full, algo, X, y), algorithms.evaluate_full(resumed, algo, X, y)
    assert f == r
    for attr in ("coef_", "intercept_", "theta_", "var_", "class_count_"):
        if hasattr(full["estimator"], attr):
            assert np.array_equal(getattr(full["estimator"], attr), getattr(resumed["estimator"], attr))


def test_regressor_learns_and_predicts():
    data = (SAMPLES / "diabetes.csv").read_bytes()
    meta = dataset.analyze(data)
    hp = catalog.merge_hyperparams(catalog.BY_ALGORITHM["sgd_regressor"], {"epochs": 20})
    st, X, y = _train("sgd_regressor", data, meta, hp, 1, catalog.compute_total_batches(hp, meta.n_rows))
    assert algorithms.evaluate_full(st, "sgd_regressor", X, y)["r2"] > 0.2
    assert isinstance(algorithms.predict(st, "sgd_regressor", [0.0] * meta.n_features), float)


def test_classifier_accuracy_and_input_validation():
    data = (SAMPLES / "iris.csv").read_bytes()
    meta = dataset.analyze(data)
    hp = catalog.merge_hyperparams(catalog.BY_ALGORITHM["sgd_logistic"], {"epochs": 30})
    st, X, y = _train("sgd_logistic", data, meta, hp, 1, catalog.compute_total_batches(hp, meta.n_rows))
    assert algorithms.evaluate_full(st, "sgd_logistic", X, y)["accuracy"] > 0.8
    assert isinstance(algorithms.predict(st, "sgd_logistic", [5.1, 3.5, 1.4, 0.2]), int)
    from distriml.common.errors import ContractError
    for bad in ([1, 2], "x", [1, 2, 3, float("nan")]):
        with pytest.raises(ContractError):
            algorithms.predict(st, "sgd_logistic", bad)
