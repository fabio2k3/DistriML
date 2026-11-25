"""Hito 9 del plan: retirada ordenada. `Worker.stop()` aborta el attempt en curso.

Sin Ray (FakeDB). El caso con contenedores (`docker stop` / `docker kill` con --scale) se prueba al integrar.
"""
import threading
import time

import pytest

from distriml.common import algorithms, constants as C
from distriml.worker import trainer
from distriml.worker.trainer import run_attempt
from distriml.worker.worker import Worker
from tests.test_worker_wk1_wk2 import seed_job, wait_until
from tests.test_worker_wk3_wk4_wk5 import ScriptDB, assert_same_model, assignment, lease_threads, new_db
from tests.test_worker_wk6 import MissingArtifactDB


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(trainer, "RETRY_BASE_S", 0.01)
    monkeypatch.setattr(trainer, "RETRY_MAX_S", 0.02)
    yield
    assert wait_until(lambda: not lease_threads(), 2), "quedó un LeaseKeeper vivo"


def slow_training(monkeypatch, seconds=0.05):
    orig = algorithms.train_batch
    monkeypatch.setattr(algorithms, "train_batch", lambda *a, **k: (time.sleep(seconds), orig(*a, **k))[1])


def test_run_attempt_with_preset_abort_writes_nothing():
    db = new_db()
    a = assignment(db)
    abort = threading.Event()
    abort.set()
    spy = ScriptDB(db)
    assert run_attempt(a, "w0", spy, db.cfg, abort=abort) == "ABORTED"
    assert spy.calls["upload_checkpoint"] == 0 and spy.calls["complete_job"] == 0 and spy.calls["report_failure"] == 0


def test_stop_aborts_running_attempt_promptly(monkeypatch):
    slow_training(monkeypatch)
    db = new_db(lease=2_000, renew=200)
    job_id = seed_job(db, "r1", {"epochs": 100})
    w = Worker("w0", db, db.cfg)
    assert wait_until(lambda: db.jobs[job_id]["last_confirmed_batch"] >= 2, 5)
    t0 = time.monotonic()
    w.stop()
    assert time.monotonic() - t0 < 2                          # no espera a que acabe el job (varios segundos)
    assert not w.status()["running"] and w.status()["current_attempt"] is None
    job = db.jobs[job_id]
    assert job["estado"] == C.EN_EJECUCION                    # la DB aún no sabe nada: lo detectará el lease
    assert job["attempts_count"] == 0                         # retirarse no es un fallo


def test_stop_does_not_write_after_it_returns(monkeypatch):
    slow_training(monkeypatch)
    db = new_db(lease=2_000, renew=200)
    job_id = seed_job(db, "r1", {"epochs": 100})
    w = Worker("w0", db, db.cfg)
    assert wait_until(lambda: db.jobs[job_id]["last_confirmed_batch"] >= 2, 5)
    w.stop()
    confirmed = db.jobs[job_id]["last_confirmed_batch"]
    time.sleep(0.3)
    assert db.jobs[job_id]["last_confirmed_batch"] == confirmed   # ningún checkpoint tras stop()


def test_job_survives_worker_withdrawal_and_same_model(monkeypatch):
    """Criterio de entrega 4 (parte worker): retirar un worker no pierde el job; otro lo termina igual."""
    clean = new_db()
    run_attempt(assignment(clean, {"epochs": 30}), "w0", clean, clean.cfg)

    slow_training(monkeypatch, 0.02)
    db = new_db()
    job_id = seed_job(db, "r1", {"epochs": 30})
    w1 = Worker("w1", db, db.cfg)
    assert wait_until(lambda: db.jobs[job_id]["last_confirmed_batch"] >= 5, 5)
    w1.stop()                                                  # retirada
    done_before = db.jobs[job_id]["last_confirmed_batch"]
    assert done_before < db.jobs[job_id]["total_batches"]
    w2 = Worker("w2", db, db.cfg)
    try:
        assert wait_until(lambda: db.jobs[job_id]["estado"] == C.COMPLETADO, 15)
    finally:
        w2.stop()
    job = db.jobs[job_id]
    assert job["lost_count"] == 1 and job["attempts_count"] == 0
    first, second = sorted((x for x in db.attempts.values() if x["job_id"] == job_id), key=lambda x: x["created_at"])
    assert second["base_batch"] > 1                            # continuó desde el checkpoint, no desde cero
    assert_same_model(db, clean)


def test_stop_while_waiting_between_download_retries(monkeypatch):
    monkeypatch.setattr(trainer, "DOWNLOAD_BASE_S", 30)      # esperaría 30 s sin el aborto
    monkeypatch.setattr(trainer, "DOWNLOAD_RETRIES", 5)
    db = new_db()
    job_id = seed_job(db, "r1", {"epochs": 2})
    sha = db.datasets[db.jobs[job_id]["dataset_id"]]["sha256"]
    w = Worker("w0", MissingArtifactDB(db, {sha}), db.cfg)
    assert wait_until(lambda: w.status()["current_attempt"] is not None, 5)
    time.sleep(0.1)
    t0 = time.monotonic()
    w.stop()
    assert time.monotonic() - t0 < 2 and not w.status()["running"]


def test_stop_then_start_resumes_working():
    db = new_db()
    w = Worker("w0", db, db.cfg)
    w.stop()
    assert not w.status()["running"]
    job_id = seed_job(db, "r1", {"epochs": 2})
    w.start()
    try:
        assert wait_until(lambda: db.jobs[job_id]["estado"] == C.COMPLETADO, 10)
    finally:
        w.stop()
