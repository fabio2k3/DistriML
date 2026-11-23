"""Pruebas de la fase 1 (hitos 1 y 2 del plan): bucle del worker (WK-1) y descarga/preparación (WK-2).

Todo contra FakeDB, sin Ray. Para ver los logs: pytest tests/test_worker_phase1.py -o log_cli=true --log-cli-level=DEBUG
"""
import threading
import time
from collections import Counter

import numpy as np
import pytest

from distriml.common import algorithms, batching, catalog, dataset
from distriml.common.config import load_config
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import ContractError, bad_request, unavailable
from distriml.common.ids import sha256_hex
from distriml.common.interfaces import Assignment
from distriml.worker import trainer
from distriml.worker.trainer import ArtifactUnavailable, AttemptAborted, download_artifact, prepare_attempt
from distriml.worker.worker import Worker
from tests.conftest import SAMPLES
from tests.helpers.fake_db import FakeDB

CLIENT = "a" * 64
IRIS = (SAMPLES / "iris.csv").read_bytes()
MODEL = catalog.BY_ALGORITHM["sgd_logistic"].model_def_id


# ---------------------------------------------------------------- utilidades
class FlakyDB:
    """Envuelve una DB y hace fallar las llamadas indicadas: plan = {"metodo": [excepcion, excepcion, ...]}."""

    def __init__(self, inner, plan=None):
        self.inner = inner
        self.plan = {k: list(v) for k, v in (plan or {}).items()}
        self.calls = Counter()

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self.calls[name] += 1
            queue = self.plan.get(name)
            if queue:
                raise queue.pop(0)
            return attr(*args, **kwargs)

        return call


def fast_cfg(**kw):
    return load_config(request_poll_interval_ms=20, **kw)


def seed_job(db, request_id="r1", hp=None):
    ds = db.create_dataset(CLIENT, request_id, "iris.csv", sha256_hex(IRIS), len(IRIS), IRIS)
    return db.create_job(CLIENT, request_id, ds["dataset_id"], MODEL, hp or {"epochs": 2})["job_id"]


def wait_until(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def make_assignment(db, next_batch=1, checkpoint_sha=None):
    """Asignación sintética sobre el dataset iris guardado en `db`."""
    meta = dataset.analyze(IRIS)
    sha = sha256_hex(IRIS)
    db.put_artifact(sha, IRIS)
    md = catalog.BY_ALGORITHM["sgd_logistic"]
    hp = dict(md.default_hyperparams)
    return Assignment("job", "attempt-1", next_batch, 10, sha, meta.to_dict(), md.algorithm, hp, 10_000, checkpoint_sha)


@pytest.fixture
def db():
    return FakeDB(fast_cfg())


@pytest.fixture
def workers():
    created = []
    yield created
    for w in created:
        w.stop()


# ---------------------------------------------------------------- WK-2: download_artifact
def test_download_ok(db):
    sha = sha256_hex(IRIS)
    db.put_artifact(sha, IRIS)
    assert download_artifact(db, sha, threading.Event()) == IRIS


def test_download_retries_on_503_then_succeeds(db):
    sha = sha256_hex(IRIS)
    db.put_artifact(sha, IRIS)
    flaky = FlakyDB(db, {"get_artifact": [unavailable("artifact_unavailable"), unavailable("no_ca")]})
    assert download_artifact(flaky, sha, threading.Event(), base_delay=0.01) == IRIS
    assert flaky.calls["get_artifact"] == 3


def test_download_gives_up_after_retries(db):
    sha = sha256_hex(IRIS)                       # nunca se guardó: la DB responde 503 artifact_unavailable
    flaky = FlakyDB(db)
    with pytest.raises(ArtifactUnavailable):
        download_artifact(flaky, sha, threading.Event(), retries=3, base_delay=0.01)
    assert flaky.calls["get_artifact"] == 4      # 1 intento + 3 reintentos


def test_download_rejects_wrong_hash(db):
    class Liar:
        calls = 0

        def get_artifact(self, sha):
            Liar.calls += 1
            return b"contenido distinto"

    with pytest.raises(ArtifactUnavailable):
        download_artifact(Liar(), sha256_hex(IRIS), threading.Event(), retries=2, base_delay=0.01)
    assert Liar.calls == 3                        # el hash erróneo también se reintenta


def test_download_non_retryable_error_fails_fast(db):
    class Bad:
        calls = 0

        def get_artifact(self, sha):
            Bad.calls += 1
            raise bad_request("invalid_sha256")

    with pytest.raises(ArtifactUnavailable):
        download_artifact(Bad(), "xx", threading.Event(), base_delay=0.01)
    assert Bad.calls == 1


def test_download_aborts_if_already_set(db):
    abort = threading.Event()
    abort.set()
    with pytest.raises(AttemptAborted):
        download_artifact(db, sha256_hex(IRIS), abort)


def test_download_abort_interrupts_backoff_wait(db):
    abort = threading.Event()
    threading.Timer(0.1, abort.set).start()
    t0 = time.monotonic()
    with pytest.raises(AttemptAborted):
        download_artifact(db, sha256_hex(IRIS), abort, retries=5, base_delay=30)   # esperaría 30 s sin el abort
    assert time.monotonic() - t0 < 2


# ---------------------------------------------------------------- WK-2: prepare_attempt
def test_prepare_from_scratch(db):
    a = make_assignment(db)
    p = prepare_attempt(a, db, threading.Event())
    assert p.X.shape == (150, 4) and p.y.shape == (150,)
    assert p.task == "classification"
    assert not hasattr(p.state["estimator"], "coef_")       # sin entrenar
    assert p.meta.n_rows == 150


def test_prepare_resumes_from_checkpoint(db):
    meta = dataset.analyze(IRIS)
    X, y = dataset.load_for_training(IRIS, meta, "classification")
    a0 = make_assignment(db)
    state = algorithms.init_state(a0.algorithm, a0.hyperparams, meta)
    hp = a0.hyperparams
    for k in (1, 2):
        idx = batching.batch_indices(meta.n_rows, hp["batch_size"], hp["seed"], k)
        algorithms.train_batch(state, a0.algorithm, X[idx], y[idx])
    blob = algorithms.serialize_state(state)
    db.put_artifact(sha256_hex(blob), blob)

    a = make_assignment(db, next_batch=3, checkpoint_sha=sha256_hex(blob))
    p = prepare_attempt(a, db, threading.Event())
    assert np.array_equal(p.state["estimator"].coef_, state["estimator"].coef_)


def test_prepare_needs_checkpoint_when_resuming(db):
    with pytest.raises(ValueError):
        prepare_attempt(make_assignment(db, next_batch=3, checkpoint_sha=None), db, threading.Event())


def test_prepare_missing_dataset_is_unavailable(db):
    a = make_assignment(db)
    a.dataset_sha256 = sha256_hex(b"no existe")
    with pytest.raises(ArtifactUnavailable):
        prepare_attempt(a, db, threading.Event(), retries=1, base_delay=0.01)


# ---------------------------------------------------------------- WK-1: bucle del worker
def test_worker_registers_and_requests_job(db, workers, monkeypatch):
    got = []
    monkeypatch.setattr(trainer, "run_attempt", lambda a, wid, db_, cfg: got.append((a, wid)) or "COMPLETED")
    job_id = seed_job(db)
    w = Worker("w0", db, fast_cfg())
    workers.append(w)
    assert wait_until(lambda: got)
    a, wid = got[0]
    assert wid == "w0" and a.job_id == job_id
    assert "w0" in db.workers                                  # se registró en la DB


def test_worker_idle_polling_and_status(db, workers):
    w = Worker("w0", db, fast_cfg())
    workers.append(w)
    time.sleep(0.15)
    st = w.status()
    assert st["running"] and st["current_attempt"] is None


def test_worker_current_attempt_visible_and_cleared(db, workers, monkeypatch):
    inside, release = threading.Event(), threading.Event()

    def fake_run(a, wid, db_, cfg):
        inside.set()
        release.wait(5)
        return "COMPLETED"

    monkeypatch.setattr(trainer, "run_attempt", fake_run)
    seed_job(db)
    w = Worker("w0", db, fast_cfg())
    workers.append(w)
    assert inside.wait(5)
    assert w.status()["current_attempt"] is not None
    release.set()
    assert wait_until(lambda: w.status()["current_attempt"] is None)


def test_worker_survives_db_unavailable_on_register_and_request(db, workers):
    flaky = FlakyDB(db, {"register_worker": [unavailable("no_ca")] * 2,
                         "request_job": [unavailable("no_ca")] * 2})
    w = Worker("w0", flaky, fast_cfg())
    workers.append(w)
    assert wait_until(lambda: flaky.calls["request_job"] >= 4)   # 2 fallos y luego sondeos normales
    assert w.status()["running"]
    assert flaky.calls["register_worker"] >= 4                   # 2 fallidos + 1 ok + 1 re-registro tras no_ca


def test_worker_survives_non_503_contract_error(db, workers):
    flaky = FlakyDB(db, {"request_job": [bad_request("x")]})
    w = Worker("w0", flaky, fast_cfg())
    workers.append(w)
    assert wait_until(lambda: flaky.calls["request_job"] >= 3)
    assert w.status()["running"]


def test_worker_survives_unexpected_error_and_takes_next_job(db, workers, monkeypatch):
    calls = []

    def fake_run(a, wid, db_, cfg):
        calls.append(a.job_id)
        if len(calls) == 1:
            raise RuntimeError("fallo inesperado")
        return "COMPLETED"

    monkeypatch.setattr(trainer, "run_attempt", fake_run)
    seed_job(db, "r1")
    seed_job(db, "r2")
    w = Worker("w0", db, fast_cfg())
    workers.append(w)
    assert wait_until(lambda: len(calls) == 2)
    assert w.status()["running"]
    assert wait_until(lambda: w.status()["current_attempt"] is None)


def test_worker_stop_ends_loop_and_start_resumes(db):
    w = Worker("w0", db, fast_cfg())
    assert w.status()["running"]
    w.stop()
    assert not w.status()["running"]
    w.start()
    assert w.status()["running"]
    w.stop()


def test_worker_stop_while_db_is_down(db):
    flaky = FlakyDB(db, {"register_worker": [unavailable("no_ca")] * 1000})
    w = Worker("w0", flaky, fast_cfg())
    time.sleep(0.1)
    t0 = time.monotonic()
    w.stop()                                                     # debe interrumpir los reintentos de registro
    assert time.monotonic() - t0 < 2 and not w.status()["running"]


def test_autostart_false_does_not_run(db):
    w = Worker("w0", db, fast_cfg(), autostart=False)
    assert not w.status()["running"]
