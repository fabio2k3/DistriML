"""Pruebas de WK-3, WK-4 y WK-5 (hitos 3 y 4 del plan): LeaseKeeper (WK-3), bucle de batches (WK-4) y cierre (WK-5).

Contra FakeDB, sin Ray. Los leases son cortos (300 ms) para que las pruebas de caída sean rápidas.
"""
import threading
import time
from collections import Counter

import pytest

from distriml.common import algorithms, constants as C
from distriml.common.config import load_config
from distriml.common.errors import ContractError, unavailable
from distriml.worker import trainer
from distriml.worker.trainer import LeaseKeeper, run_attempt
from distriml.worker.worker import Worker
from tests.helpers.fake_db import FakeDB
from tests.test_worker_wk1_wk2 import seed_job, wait_until


# ---------------------------------------------------------------- utilidades
class Crash(BaseException):
    """Simula la muerte brusca del worker: no la captura ningún `except Exception`."""


class ScriptDB:
    """Envuelve una DB. script = {"metodo": [item, ...]}: cada llamada consume un item
    (excepción -> se lanza; otro valor -> se devuelve sin llamar a la DB). Sin items, llama a la DB real.
    after = {"metodo": f(n_llamada, resultado)} se ejecuta tras la llamada real (puede lanzar)."""

    def __init__(self, inner, script=None, after=None):
        self.inner = inner
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.after = after or {}
        self.calls = Counter()

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            self.calls[name] += 1
            queue = self.script.get(name)
            if queue:
                item = queue.pop(0)
                if isinstance(item, BaseException):
                    raise item
                return item
            result = attr(*args, **kwargs)
            hook = self.after.get(name)
            if hook:
                hook(self.calls[name], result)
            return result

        return call


def cfg_(lease=300, renew=100, **kw):
    return load_config(lease_duration_ms=lease, renew_interval_ms=renew, epsilon_ms=50,
                       request_poll_interval_ms=20, **kw)


def new_db(**kw):
    return FakeDB(cfg_(**kw))


def assignment(db, hp=None, request_id="r1"):
    seed_job(db, request_id, hp or {"epochs": 3})
    return db.request_job("w0")


def lease_threads():
    return [t for t in threading.enumerate() if t.name.startswith("lease-") and t.is_alive()]


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    monkeypatch.setattr(trainer, "RETRY_BASE_S", 0.01)
    monkeypatch.setattr(trainer, "RETRY_MAX_S", 0.02)


@pytest.fixture(autouse=True)
def no_leaked_threads():
    yield
    assert wait_until(lambda: not lease_threads(), 2), "quedó un LeaseKeeper vivo"


def final_sha(db):
    (versions,) = db.versions.values()
    return versions[0]["sha256"]


def final_model(db):
    """Estado del modelo final de la única versión guardada."""
    return algorithms.deserialize_state(db.get_artifact(final_sha(db)))


def assert_same_model(db_a, db_b):
    """Mismo modelo = mismos parámetros y mismas predicciones.

    No se compara el sha256 del pickle: tras reanudar (des-serializar y volver a serializar) scikit-learn
    produce otros bytes para un atributo interno (`_loss_function_`) aunque lo aprendido sea idéntico.
    """
    import numpy as np
    from tests.test_worker_wk1_wk2 import IRIS
    from distriml.common import dataset
    ea, eb = final_model(db_a)["estimator"], final_model(db_b)["estimator"]
    assert np.array_equal(ea.coef_, eb.coef_) and np.array_equal(ea.intercept_, eb.intercept_)
    assert ea.t_ == eb.t_
    meta = dataset.analyze(IRIS)
    X, _ = dataset.load_for_training(IRIS, meta, "classification")
    assert np.array_equal(ea.predict(X), eb.predict(X))


# ---------------------------------------------------------------- WK-3: LeaseKeeper
def start_keeper(db, a, cfg):
    abort, done = threading.Event(), threading.Event()
    k = LeaseKeeper(db, "w0", a.attempt_id, cfg, abort, done)
    t0 = time.monotonic()
    k.start()
    return k, abort, done, t0


def test_keeper_renews_periodically_and_stops_on_done():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db)
    k, abort, done, _ = start_keeper(spy, a, cfg_())
    time.sleep(0.45)
    assert spy.calls["renew_lease"] >= 3 and not abort.is_set()
    done.set()
    k.join(1)
    assert not k.is_alive()


def test_keeper_stops_promptly_on_done_even_with_long_interval():
    db = new_db(lease=10_000, renew=5_000)
    a = assignment(db)
    k, abort, done, _ = start_keeper(db, a, db.cfg)
    time.sleep(0.05)
    done.set()
    k.join(1)
    assert not k.is_alive() and not abort.is_set()


def test_keeper_revoked_sets_abort_and_ends():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"renew_lease": [C.RENEW_REVOKED]})
    k, abort, done, _ = start_keeper(spy, a, cfg_())
    assert abort.wait(1)
    k.join(1)
    assert not k.is_alive()
    done.set()


def test_keeper_survives_transient_failures():
    db = new_db(lease=600)
    a = assignment(db)
    spy = ScriptDB(db, {"renew_lease": [unavailable("no_ca"), RuntimeError("fallo de red")]})
    k, abort, done, _ = start_keeper(spy, a, cfg_(lease=600))
    time.sleep(0.5)
    assert not abort.is_set() and k.is_alive() and spy.calls["renew_lease"] >= 4
    done.set()
    k.join(1)


def test_keeper_aborts_when_local_lease_expires():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"renew_lease": [unavailable("no_ca")] * 1000})
    k, abort, done, t0 = start_keeper(spy, a, cfg_())      # lease 300 ms
    assert abort.wait(2)
    elapsed = time.monotonic() - t0
    assert 0.28 <= elapsed <= 0.55, elapsed                 # ≈ lease, y antes de que la DB lo dé por perdido + margen
    k.join(1)
    done.set()


# ---------------------------------------------------------------- WK-4 + WK-5: camino feliz
def test_run_attempt_completes_job():
    db = new_db()
    a = assignment(db)
    assert run_attempt(a, "w0", db, db.cfg) == "COMPLETED"
    job = db.jobs[a.job_id]
    assert job["estado"] == C.COMPLETADO
    assert job["last_confirmed_batch"] == a.total_batches
    assert len(db.batches[a.job_id]) == a.total_batches
    (version,) = db.versions[job["model_id"]]
    assert version["version_id"] == a.attempt_id and version["metrics"]["accuracy"] > 0.5
    assert "mse" in version["metrics"] and version["metrics"]["n"] == 150


def test_run_attempt_checkpoint_metrics_have_n_and_score():
    db = new_db()
    a = assignment(db)
    run_attempt(a, "w0", db, db.cfg)
    for row in db.batches[a.job_id].values():
        assert set(row["metrics"]) == {"n", "score"}


def test_run_attempt_matches_reference_worker():
    from tests.helpers.reference_worker import run_one
    ref = new_db()
    seed_job(ref, "r1", {"epochs": 3})
    run_one(ref, "ref")
    db = new_db()
    run_attempt(assignment(db), "w0", db, db.cfg)
    assert final_sha(db) == final_sha(ref)            # mismo modelo, byte a byte


def test_run_attempt_stops_lease_keeper():
    db = new_db()
    run_attempt(assignment(db), "w0", db, db.cfg)
    assert wait_until(lambda: not lease_threads(), 1)


# ---------------------------------------------------------------- WK-4: abortos
def test_ckpt_rejected_aborts_without_reporting_error():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"upload_checkpoint": [C.CKPT_CONFIRMED, C.CKPT_REJECTED]})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert spy.calls["upload_checkpoint"] == 2            # tras el rechazo no se vuelve a escribir
    assert spy.calls["complete_job"] == 0 and spy.calls["report_failure"] == 0


def test_revoked_lease_stops_training_early(monkeypatch):
    orig = algorithms.train_batch
    monkeypatch.setattr(algorithms, "train_batch", lambda *a, **k: (time.sleep(0.03), orig(*a, **k))[1])
    db = new_db()
    a = assignment(db, {"epochs": 10})
    spy = ScriptDB(db, {"renew_lease": [C.RENEW_REVOKED]})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert spy.calls["upload_checkpoint"] < a.total_batches
    assert spy.calls["complete_job"] == 0


def test_abort_is_checked_every_batch_not_only_at_checkpoints(monkeypatch):
    """Con k_ckpt grande casi no hay checkpoints: el aborto debe detectarse igualmente entre batches."""
    trained = []
    orig = algorithms.train_batch
    monkeypatch.setattr(algorithms, "train_batch",
                        lambda *a, **k: (trained.append(1), time.sleep(0.02), orig(*a, **k))[2])
    db = new_db(k_ckpt=1000)
    a = assignment(db, {"epochs": 10})
    spy = ScriptDB(db, {"renew_lease": [C.RENEW_REVOKED]})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert len(trained) < a.total_batches / 2


def test_lost_lease_stops_training(monkeypatch):
    orig = algorithms.train_batch
    monkeypatch.setattr(algorithms, "train_batch", lambda *a, **k: (time.sleep(0.1), orig(*a, **k))[1])
    db = new_db()
    a = assignment(db, {"epochs": 10})
    assert a.total_batches >= 5
    spy = ScriptDB(db, {"renew_lease": [unavailable("no_ca")] * 1000})
    t0 = time.monotonic()
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert time.monotonic() - t0 < 1.5                    # no espera a terminar el entrenamiento
    assert spy.calls["upload_checkpoint"] < a.total_batches


# ---------------------------------------------------------------- WK-4/5: reintentos
def test_ckpt_retry_is_repeated():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"upload_checkpoint": [C.CKPT_RETRY, C.CKPT_RETRY]})
    assert run_attempt(a, "w0", spy, db.cfg) == "COMPLETED"
    assert spy.calls["put_artifact"] >= a.total_batches + 2     # cada RETRY vuelve a subir el binario


def test_transient_503_in_every_write_is_retried():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"put_artifact": [unavailable("no_ca")] * 2,
                        "upload_checkpoint": [unavailable("no_ca")] * 2,
                        "complete_job": [unavailable("no_ca")]})
    assert run_attempt(a, "w0", spy, db.cfg) == "COMPLETED"
    assert db.jobs[a.job_id]["estado"] == C.COMPLETADO


def test_persistent_503_gives_up_and_stops_keeper(monkeypatch):
    monkeypatch.setattr(trainer, "RETRY_COUNT", 2)
    db = new_db(lease=5_000, renew=1_000)
    a = assignment(db)
    spy = ScriptDB(db, {"put_artifact": [unavailable("no_ca")] * 1000})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"      # la DB no responde: se abandona (WK-6)
    assert spy.calls["report_failure"] == 0


def test_complete_job_rejected_is_aborted_not_failed():
    db = new_db()
    a = assignment(db)
    spy = ScriptDB(db, {"complete_job": [C.COMPLETE_REJECTED]})
    assert run_attempt(a, "w0", spy, db.cfg) == "ABORTED"
    assert spy.calls["report_failure"] == 0


# ---------------------------------------------------------------- reanudación (base del hito 7)
def crash_after(db, m):
    """DB que mata al worker justo después de confirmar el checkpoint m."""
    def hook(n, res):
        if n == m:
            raise Crash()
    return ScriptDB(db, after={"upload_checkpoint": hook})


def test_resume_after_crash_gives_same_model():
    clean = new_db()
    run_attempt(assignment(clean), "w0", clean, clean.cfg)

    db = new_db()
    a1 = assignment(db)
    assert a1.total_batches > 4
    with pytest.raises(Crash):
        run_attempt(a1, "w0", crash_after(db, 3), db.cfg)
    assert db.jobs[a1.job_id]["last_confirmed_batch"] == 3
    time.sleep(0.45)                                      # vence el lease (300 ms + 50 ms de margen)
    a2 = db.request_job("w1")
    assert a2.job_id == a1.job_id and a2.attempt_id != a1.attempt_id
    assert a2.next_batch == 4 and a2.checkpoint_sha256 == db.batches[a1.job_id][3]["sha256"]
    assert run_attempt(a2, "w1", db, db.cfg) == "COMPLETED"
    assert_same_model(db, clean)                          # mismo modelo que el entrenamiento sin fallo


def test_resume_when_all_batches_already_confirmed(monkeypatch):
    db = new_db()
    a1 = assignment(db)
    spy = ScriptDB(db, {"complete_job": [Crash()]})        # muere justo antes de cerrar
    with pytest.raises(Crash):
        run_attempt(a1, "w0", spy, db.cfg)
    assert db.jobs[a1.job_id]["last_confirmed_batch"] == a1.total_batches
    time.sleep(0.45)
    trained = []
    orig = algorithms.train_batch
    monkeypatch.setattr(algorithms, "train_batch", lambda *a, **k: trained.append(1) or orig(*a, **k))
    a2 = db.request_job("w1")
    assert a2.next_batch == a2.total_batches + 1
    assert run_attempt(a2, "w1", db, db.cfg) == "COMPLETED"
    assert not trained                                     # solo cierra: no reentrena nada
    assert db.jobs[a1.job_id]["estado"] == C.COMPLETADO


def test_stale_attempt_cannot_write_after_being_replaced():
    db = new_db()
    a1 = assignment(db)
    with pytest.raises(Crash):
        run_attempt(a1, "w0", crash_after(db, 2), db.cfg)
    time.sleep(0.45)
    a2 = db.request_job("w1")                              # el job pasa a otro attempt
    assert a2.attempt_id != a1.attempt_id
    # El attempt viejo «reaparece»: todo lo suyo se rechaza y aborta sin error.
    assert run_attempt(a1, "w0", db, db.cfg) == "ABORTED"
    assert db.jobs[a1.job_id]["attempt_vigente"] == a2.attempt_id


# ---------------------------------------------------------------- integración con el bucle del worker
def test_worker_loop_completes_jobs_end_to_end():
    db = new_db()
    ids_ = [seed_job(db, f"r{i}", {"epochs": 2}) for i in range(3)]
    w = Worker("w0", db, db.cfg)
    try:
        assert wait_until(lambda: all(db.jobs[j]["estado"] == C.COMPLETADO for j in ids_), 10)
    finally:
        w.stop()
