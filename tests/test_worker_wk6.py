"""Pruebas de WK-6 (hitos 5 y 6 del plan): errores y abortos (WK-6) y pruebas de fallo.

Contra FakeDB, sin Ray. Las que dependen de la BD real (test_contract_live, chaos.py) se hacen al integrar.
"""
import threading
import time

import pytest

from distriml.common import algorithms, constants as C, dataset
from distriml.common.errors import bad_request, unavailable
from distriml.common.ids import sha256_hex
from distriml.worker import trainer
from distriml.worker.worker import Worker
from tests.test_worker_wk1_wk2 import CLIENT, seed_job, wait_until
from tests.test_worker_wk3_wk4_wk5 import Crash, ScriptDB, assignment, cfg_, lease_threads, new_db, assert_same_model  # noqa: F401
from distriml.worker.trainer import run_attempt


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    monkeypatch.setattr(trainer, "RETRY_BASE_S", 0.01)
    monkeypatch.setattr(trainer, "RETRY_MAX_S", 0.02)
    monkeypatch.setattr(trainer, "DOWNLOAD_RETRIES", 2)
    monkeypatch.setattr(trainer, "DOWNLOAD_BASE_S", 0.01)
    monkeypatch.setattr(trainer, "DOWNLOAD_MAX_S", 0.02)
    yield
    assert wait_until(lambda: not lease_threads(), 2), "quedó un LeaseKeeper vivo"


def break_training_at(monkeypatch, n, exc):
    """Hace que el batch n-ésimo (contando llamadas) lance `exc`."""
    orig, calls = algorithms.train_batch, []

    def broken(*a, **k):
        calls.append(1)
        if len(calls) == n:
            raise exc
        return orig(*a, **k)

    monkeypatch.setattr(algorithms, "train_batch", broken)


class MissingArtifactDB(ScriptDB):
    """DB cuyo get_artifact falla (503) para los hashes indicados. `reported` guarda los report_unavailable."""

    def __init__(self, inner, missing, wrong_bytes=False):
        super().__init__(inner)
        self.missing, self.wrong = set(missing), wrong_bytes
        self.reported = []

    def report_unavailable(self, worker_id, attempt_id, sha256):
        self.reported.append(sha256)
        return self.inner.report_unavailable(worker_id, attempt_id, sha256)

    def get_artifact(self, sha256):
        self.calls["get_artifact"] += 1
        if sha256 in self.missing:
            if self.wrong:
                return b"basura"
            raise unavailable("artifact_unavailable")
        return self.inner.get_artifact(sha256)


# ---------------------------------------------------------------- WK-6: fallo de entrenamiento
def test_training_error_is_reported_as_failure(monkeypatch):
    db = new_db()
    a = assignment(db)
    break_training_at(monkeypatch, 2, ValueError("batch corrupto"))
    spy = ScriptDB(db)
    assert run_attempt(a, "w0", spy, db.cfg) == "FAILED"
    assert spy.calls["report_failure"] == 1 and spy.calls["complete_job"] == 0
    att = db.attempts[a.attempt_id]
    assert att["revoke_reason"] == C.WORKER_ERROR
    job = db.jobs[a.job_id]
    assert job["attempts_count"] == 1 and job["estado"] == C.PENDIENTE and job["last_confirmed_batch"] == 1


def test_failure_message_is_one_line_and_bounded(monkeypatch):
    db = new_db()
    a = assignment(db)
    sent = []
    break_training_at(monkeypatch, 1, ValueError("línea 1\nlínea 2\n" + "x" * 1000))
    real = db.report_failure
    monkeypatch.setattr(db, "report_failure", lambda w, at, err: (sent.append(err), real(w, at, err))[1])
    assert run_attempt(a, "w0", db, db.cfg) == "FAILED"
    (msg,) = sent
    assert "\n" not in msg and len(msg) <= trainer.MAX_ERROR_LEN and msg.startswith("ValueError: línea 1 línea 2")


def test_corrupt_dataset_400_is_a_failure_not_infrastructure(monkeypatch):
    db = new_db()
    a = assignment(db)
    monkeypatch.setattr(dataset, "load_for_training", lambda *x, **k: (_ for _ in ()).throw(bad_request("invalid_dataset")))
    spy = ScriptDB(db)
    assert run_attempt(a, "w0", spy, db.cfg) == "FAILED"
    assert spy.calls["report_failure"] == 1


def test_non_503_contract_error_in_a_write_is_a_failure():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"put_artifact": [bad_request("hash_mismatch")]})
    assert run_attempt(a, "w0", spy, db.cfg) == "FAILED"
    assert spy.calls["report_failure"] == 1


def test_report_failure_retries_transient_503(monkeypatch):
    db = new_db()
    a = assignment(db)
    break_training_at(monkeypatch, 1, ValueError("x"))
    spy = ScriptDB(db, {"report_failure": [unavailable("no_ca")] * 2})
    assert run_attempt(a, "w0", spy, db.cfg) == "FAILED"
    assert db.attempts[a.attempt_id]["revoke_reason"] == C.WORKER_ERROR


def test_report_failure_best_effort_when_db_stays_down(monkeypatch):
    monkeypatch.setattr(trainer, "RETRY_COUNT", 2)
    db = new_db()
    a = assignment(db)
    break_training_at(monkeypatch, 1, ValueError("x"))
    spy = ScriptDB(db, {"report_failure": [unavailable("no_ca")] * 1000})
    assert run_attempt(a, "w0", spy, db.cfg) == "FAILED"          # no lanza: el lease cubrirá el hueco


# ---------------------------------------------------------------- WK-6: artefacto no disponible
def test_missing_dataset_reports_unavailable_without_counting():
    db = new_db()
    a = assignment(db)
    spy = MissingArtifactDB(db, {a.dataset_sha256})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert spy.reported == [a.dataset_sha256]                     # avisa de QUÉ artefacto falta
    att = db.attempts[a.attempt_id]
    assert att["revoke_reason"] == C.ARTIFACT_UNAVAILABLE
    job = db.jobs[a.job_id]
    assert job["attempts_count"] == 0 and job["lost_count"] == 0 and job["estado"] == C.PENDIENTE


def test_corrupt_download_reports_unavailable():
    db = new_db()
    a = assignment(db)
    spy = MissingArtifactDB(db, {a.dataset_sha256}, wrong_bytes=True)
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert db.attempts[a.attempt_id]["revoke_reason"] == C.ARTIFACT_UNAVAILABLE


def test_missing_checkpoint_on_resume_reports_unavailable():
    db = new_db()
    a1 = assignment(db)
    from tests.test_worker_wk3_wk4_wk5 import crash_after
    with pytest.raises(Crash):
        run_attempt(a1, "w0", crash_after(db, 2), db.cfg)
    time.sleep(0.45)
    a2 = db.request_job("w1")
    assert a2.next_batch == 3
    spy = MissingArtifactDB(db, {a2.checkpoint_sha256})
    assert run_attempt(a2, "w1", spy, db.cfg) == "ABORTED"
    assert spy.reported == [a2.checkpoint_sha256]
    assert db.attempts[a2.attempt_id]["revoke_reason"] == C.ARTIFACT_UNAVAILABLE
    assert db.jobs[a1.job_id]["last_confirmed_batch"] == 2         # el progreso confirmado se conserva


# ---------------------------------------------------------------- WK-6: la DB no responde / aborto prevalece
def test_db_down_after_retries_is_abandoned_and_recovered_after_lease(monkeypatch):
    monkeypatch.setattr(trainer, "RETRY_COUNT", 2)
    db = new_db()
    a1 = assignment(db)
    spy = ScriptDB(db, {"upload_checkpoint": [unavailable("no_ca")] * 1000})
    assert run_attempt(a1, "w0", spy, db.cfg) == "ABORTED"
    assert spy.calls["report_failure"] == 0                        # no se reporta lo que no se puede entregar
    time.sleep(0.45)                                               # vence el lease
    a2 = db.request_job("w1")
    assert a2.job_id == a1.job_id and a2.next_batch == 1
    assert run_attempt(a2, "w1", db, db.cfg) == "COMPLETED"        # el job no queda inconsistente
    assert db.jobs[a1.job_id]["lost_count"] == 1


def test_endless_ckpt_retry_is_infrastructure_not_a_failure(monkeypatch):
    monkeypatch.setattr(trainer, "CKPT_RETRY_COUNT", 2)
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"upload_checkpoint": [C.CKPT_RETRY] * 1000})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert spy.calls["upload_checkpoint"] == 3 and spy.calls["report_failure"] == 0


def test_error_after_revocation_is_not_reported(monkeypatch):
    db = new_db()
    a = assignment(db, {"epochs": 10})

    def slow_then_fail(*x, **k):
        time.sleep(0.1)
        raise ValueError("fallo tras ser revocado")

    monkeypatch.setattr(algorithms, "train_batch", slow_then_fail)
    spy = ScriptDB(db, {"renew_lease": [C.RENEW_REVOKED]})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert spy.calls["report_failure"] == 0


def test_cancel_running_job(monkeypatch):
    db = new_db(lease=2_000, renew=200)
    a = assignment(db, {"epochs": 50})
    orig = algorithms.train_batch
    monkeypatch.setattr(algorithms, "train_batch", lambda *x, **k: (time.sleep(0.02), orig(*x, **k))[1])
    result = []
    t = threading.Thread(target=lambda: result.append(run_attempt(a, "w0", db, db.cfg)))
    t.start()
    assert wait_until(lambda: db.jobs[a.job_id]["last_confirmed_batch"] >= 2, 5)
    db.cancel_job(CLIENT, "c1", a.job_id)
    t.join(5)
    assert not t.is_alive() and result == ["ABORTED"]
    job = db.jobs[a.job_id]
    assert job["estado"] == C.CANCELADO and job["last_confirmed_batch"] < a.total_batches
    assert job["attempts_count"] == 0                              # cancelar no es un fallo


# ---------------------------------------------------------------- el worker sigue vivo tras los fallos
def test_worker_retries_job_after_transient_training_error(monkeypatch):
    db = new_db()
    job_id = seed_job(db, "r1", {"epochs": 2})
    break_training_at(monkeypatch, 2, ValueError("fallo puntual"))
    w = Worker("w0", db, db.cfg)
    try:
        assert wait_until(lambda: db.jobs[job_id]["estado"] == C.COMPLETADO, 10)
    finally:
        w.stop()
    assert db.jobs[job_id]["attempts_count"] == 1                 # un fallo contado, y luego terminó bien


def test_persistent_error_ends_job_as_failed_after_max_attempts(monkeypatch):
    db = new_db(max_attempts=3)
    job_id = seed_job(db, "r1", {"epochs": 2})
    monkeypatch.setattr(algorithms, "train_batch", lambda *a, **k: (_ for _ in ()).throw(ValueError("siempre")))
    w = Worker("w0", db, db.cfg)
    try:
        assert wait_until(lambda: db.jobs[job_id]["estado"] == C.FALLIDO, 10)
    finally:
        w.stop()
    reasons = [x["revoke_reason"] for x in db.attempts.values() if x["job_id"] == job_id]
    assert reasons == [C.WORKER_ERROR] * 3                        # ni más ni menos intentos que max_attempts


def test_worker_with_unavailable_artifact_does_not_burn_attempts():
    db = new_db(max_attempts=2)
    job_id = seed_job(db, "r1", {"epochs": 2})
    sha = db.datasets[db.jobs[job_id]["dataset_id"]]["sha256"]
    flaky = MissingArtifactDB(db, {sha})
    w = Worker("w0", flaky, db.cfg)
    try:
        assert wait_until(lambda: sum(1 for x in db.attempts.values() if x["revoke_reason"] == C.ARTIFACT_UNAVAILABLE) >= 4, 10)
    finally:
        w.stop()
    assert db.jobs[job_id]["estado"] in (C.PENDIENTE, C.EN_EJECUCION) and db.jobs[job_id]["attempts_count"] == 0
