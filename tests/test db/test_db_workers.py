"""`Database` — lado worker: asignación, leases, retardo de reasignación, checkpoints, cierre y fallos.

Usan un reloj manual: el lease (10 s) + margen (1 s) = 11 s sin renovar vencen un attempt.
"""
import pytest

from distriml.common import constants as C
from distriml.common.errors import ContractError
from tests.helpers.db_utils import (CLIENT_A, IRIS, FakeClock, make_db, new_dataset, new_job, put_blob,
                                    start_job, train)
from distriml.common.ids import sha256_hex

LEASE = 10_000
EPS = 1_000
EXPIRY = LEASE + EPS


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def db(tmp_path, clock):
    return make_db(tmp_path, clock, t_sync_ms=2_000, max_attempts=3, max_lost=3)


def job_of(db, job):
    return db.get_job(CLIENT_A, job["job_id"])


def blob(db, text: str) -> str:
    return put_blob(db, text.encode())


# ---------------------------------------------------------------------------- registro y asignación
def test_register_worker_returns_lease_timings(db):
    cfg = db.register_worker("w1")
    assert (cfg.worker_id, cfg.renew_interval_ms, cfg.lease_duration_ms) == ("w1", 3_333, LEASE)


@pytest.mark.parametrize("bad", ["", None, 5])
def test_register_worker_rejects_empty_ids(db, bad):
    with pytest.raises(ContractError) as e:
        db.register_worker(bad)
    assert e.value.status == 400


def test_request_job_without_work_returns_none(db):
    assert db.request_job("w1") is None


def test_request_job_assigns_a_fresh_attempt_from_the_start(db):
    ds, job, a = start_job(db)
    assert (a.job_id, a.next_batch, a.total_batches, a.checkpoint_sha256) == (job["job_id"], 1, 6, None)
    assert a.dataset_sha256 == sha256_hex(IRIS) and a.algorithm == "sgd_logistic" and a.lease_duration_ms == LEASE
    assert a.hyperparams == {"batch_size": 50, "epochs": 2, "seed": 42, "alpha": 0.0001, "eta0": 0.01}
    assert (a.dataset_meta["n_rows"], a.dataset_meta["n_features"], a.dataset_meta["classes"]) == (150, 4, [0, 1, 2])
    assert len(a.dataset_meta["feature_mean"]) == 4
    v = job_of(db, job)
    assert v["estado"] == C.EN_EJECUCION and v["attempt_vigente"] == a.attempt_id
    assert [(x["attempt_id"], x["worker_id"], x["estado"], x["base_batch"]) for x in v["attempts"]] == [
        (a.attempt_id, "w1", C.ATTEMPT_ACTIVO, 0)]


def test_jobs_are_assigned_oldest_first_and_only_once(db):
    ds = new_dataset(db)
    first, second = new_job(db, ds["dataset_id"], rid="a"), new_job(db, ds["dataset_id"], rid="b")
    a1, a2 = db.request_job("w1"), db.request_job("w2")
    assert (a1.job_id, a2.job_id) == (first["job_id"], second["job_id"])
    assert db.request_job("w3") is None


def test_any_number_of_workers_can_pull_work(db):
    ds = new_dataset(db)
    jobs = [new_job(db, ds["dataset_id"], rid=f"j{i}")["job_id"] for i in range(5)]
    got = [db.request_job(f"w{i}").job_id for i in range(5)]
    assert got == jobs and db.request_job("w9") is None


def test_job_is_not_assigned_while_its_dataset_is_missing(db):
    ds = new_dataset(db)
    job = new_job(db, ds["dataset_id"])
    db.store.delete(sha256_hex(IRIS))
    assert db.request_job("w1") is None and job_of(db, job)["estado"] == C.PENDIENTE
    db.put_artifact(sha256_hex(IRIS), IRIS)
    assert db.request_job("w1") is not None


def test_a_job_that_cannot_start_does_not_block_the_next_one(db):
    other = IRIS + b"1,1,1,1,0\n"
    ds1, ds2 = new_dataset(db, rid="d1"), new_dataset(db, rid="d2", data=other)
    j1, j2 = new_job(db, ds1["dataset_id"], rid="a"), new_job(db, ds2["dataset_id"], rid="b")
    db.store.delete(sha256_hex(IRIS))
    assert db.request_job("w1").job_id == j2["job_id"]
    assert job_of(db, j1)["estado"] == C.PENDIENTE


# ---------------------------------------------------------------------------- leases
def test_lease_is_valid_until_last_renew_plus_lease_plus_margin(db, clock):
    _, job, a = start_job(db)
    clock.advance(EXPIRY)                                     # justo en el plazo: todavía vale
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_CONTINUE
    clock.advance(EXPIRY)
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_CONTINUE
    clock.advance(EXPIRY + 1)                                 # un milisegundo más: vencido
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_REVOKED


def test_unknown_attempt_renews_as_revoked(db):
    assert db.renew_lease("w1", "no-existe") == C.RENEW_REVOKED
    assert db.renew_lease("w1", None) == C.RENEW_REVOKED


def test_attempt_that_never_renewed_is_never_started_and_is_reassigned_at_once(db, clock):
    _, job, a = start_job(db)
    clock.advance(EXPIRY + 1)
    v = job_of(db, job)
    assert v["estado"] == C.PENDIENTE and v["attempt_vigente"] is None
    assert [(x["estado"], x["revoke_reason"]) for x in v["attempts"]] == [(C.ATTEMPT_REVOCADO, C.NEVER_STARTED)]
    assert (v["attempts_count"], v["lost_count"]) == (0, 0)   # no cuenta
    b = db.request_job("w2")                                  # sin retardo de reasignación
    assert b is not None and b.attempt_id != a.attempt_id and b.next_batch == 1


def test_lost_worker_is_counted_and_reassignment_waits_for_the_backoff(db, clock):
    _, job, a = start_job(db)
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_CONTINUE      # llegó a vivir: contará como pérdida
    clock.advance(EXPIRY + 1)
    v = job_of(db, job)
    assert v["estado"] == C.PENDIENTE and v["lost_count"] == 1 and v["attempts_count"] == 0
    assert v["attempts"][0]["revoke_reason"] == C.WORKER_LOST
    assert db.request_job("w2") is None                       # 2^1 × t_sync = 4 s, contados desde el vencimiento
    clock.advance(3_998)                                       # ahora: vencimiento + 3 999 ms
    assert db.request_job("w2") is None
    clock.advance(1)                                           # vencimiento + 4 000 ms
    assert db.request_job("w2") is not None


def test_backoff_counts_from_lease_expiry_not_from_detection(db, clock):
    _, job, a = start_job(db)
    db.renew_lease("w1", a.attempt_id)
    clock.advance(EXPIRY + 60_000)                             # nadie llamó a la DB en todo ese tiempo
    assert db.request_job("w2") is not None                    # la espera de 4 s ya había pasado


def test_backoff_grows_with_each_loss(db, clock):
    _, job, a = start_job(db)
    waits = []
    for _ in range(2):
        db.renew_lease("w1", a.attempt_id)
        clock.advance(EXPIRY + 1)
        waited = 0
        while (a := db.request_job("w1")) is None:
            clock.advance(500)
            waited += 500
        waits.append(waited)
    assert waits == [4_000, 8_000]


def test_exhausting_max_lost_fails_the_job(db, clock):
    _, job, a = start_job(db)                                 # max_lost = 3
    for n in range(3):
        db.renew_lease("w1", a.attempt_id)
        clock.advance(EXPIRY + 1)
        if n < 2:
            clock.advance(60_000)
            a = db.request_job("w1")
    v = job_of(db, job)
    assert v["estado"] == C.FALLIDO and v["lost_count"] == 3
    clock.advance(60_000)
    assert db.request_job("w1") is None


def test_reassignment_resumes_from_the_last_confirmed_checkpoint(db, clock):
    _, job, a = start_job(db)
    last_sha = train(db, "w1", a, upto=4, finish=False)
    clock.advance(EXPIRY + 1)                                  # el worker muere tras confirmar el batch 4
    clock.advance(60_000)
    b = db.request_job("w2")
    assert (b.job_id, b.next_batch, b.checkpoint_sha256) == (job["job_id"], 5, last_sha)
    v = job_of(db, job)
    assert [(x["estado"], x["base_batch"]) for x in v["attempts"]] == [(C.ATTEMPT_REVOCADO, 0), (C.ATTEMPT_ACTIVO, 4)]
    assert db.upload_checkpoint("w1", a.attempt_id, 5, last_sha, {}) == C.CKPT_REJECTED      # el viejo ya no escribe
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_REVOKED


def test_resumed_job_produces_the_same_model_as_an_uninterrupted_run(tmp_path, clock):
    straight = make_db(tmp_path / "a", FakeClock())
    _, j1, a1 = start_job(straight)
    train(straight, "w1", a1)
    resumed = make_db(tmp_path / "b", clock)
    _, j2, a2 = start_job(resumed)
    train(resumed, "w1", a2, upto=4, finish=False)
    clock.advance(EXPIRY + 60_001)
    train(resumed, "w2", resumed.request_job("w2"))
    s = job_of(straight, j1)
    r = job_of(resumed, j2)
    assert r["estado"] == C.COMPLETADO and r["lost_count"] == 1
    v1 = straight.list_model_versions(CLIENT_A, s["model_id"])[0]
    v2 = resumed.list_model_versions(CLIENT_A, r["model_id"])[0]
    assert v1["metrics"] == v2["metrics"]
    x = {"features": [6.1, 2.8, 4.7, 1.2]}
    assert straight.predict(CLIENT_A, s["model_id"], x)["prediction"] == resumed.predict(CLIENT_A, r["model_id"], x)["prediction"]


def test_expiry_is_checked_lazily_by_every_read(db, clock):
    _, job, a = start_job(db)
    db.renew_lease("w1", a.attempt_id)
    clock.advance(EXPIRY + 1)
    assert db.list_jobs(CLIENT_A, estado=C.PENDIENTE)[0]["job_id"] == job["job_id"]


# ---------------------------------------------------------------------------- checkpoints
def test_checkpoints_confirm_in_order_and_advance_the_job(db):
    _, job, a = start_job(db)
    for k in (1, 2, 3):
        assert db.upload_checkpoint("w1", a.attempt_id, k, blob(db, f"c{k}"), {"n": 50, "score": 0.5}) == C.CKPT_CONFIRMED
        assert job_of(db, job)["last_confirmed_batch"] == k


def test_checkpoint_upload_is_idempotent(db):
    _, job, a = start_job(db)
    sha = blob(db, "c1")
    assert db.upload_checkpoint("w1", a.attempt_id, 1, sha, {"n": 1}) == C.CKPT_CONFIRMED
    assert db.upload_checkpoint("w1", a.attempt_id, 1, sha, {"n": 1}) == C.CKPT_CONFIRMED
    assert job_of(db, job)["last_confirmed_batch"] == 1
    assert len(db.get_job_metrics(CLIENT_A, job["job_id"])["batches"]) == 1


def test_checkpoint_waits_until_its_binary_is_stored(db):
    _, job, a = start_job(db)
    data = b"estado"
    assert db.upload_checkpoint("w1", a.attempt_id, 1, sha256_hex(data), {}) == C.CKPT_RETRY
    assert job_of(db, job)["last_confirmed_batch"] == 0
    db.put_artifact(sha256_hex(data), data)
    assert db.upload_checkpoint("w1", a.attempt_id, 1, sha256_hex(data), {}) == C.CKPT_CONFIRMED


@pytest.mark.parametrize("batch", [2, 0, -1, 7, True, "1", None])
def test_out_of_sequence_checkpoints_are_rejected(db, batch):
    _, job, a = start_job(db)
    assert db.upload_checkpoint("w1", a.attempt_id, batch, blob(db, "x"), {}) == C.CKPT_REJECTED
    assert job_of(db, job)["last_confirmed_batch"] == 0


def test_checkpoint_past_the_last_batch_is_rejected(db):
    _, job, a = start_job(db)
    train(db, "w1", a, finish=False)
    assert db.upload_checkpoint("w1", a.attempt_id, 7, blob(db, "extra"), {}) == C.CKPT_REJECTED


def test_checkpoint_from_unknown_or_revoked_attempt_is_rejected(db, clock):
    _, job, a = start_job(db)
    assert db.upload_checkpoint("w1", "no-existe", 1, blob(db, "x"), {}) == C.CKPT_REJECTED
    clock.advance(EXPIRY + 1)
    assert db.upload_checkpoint("w1", a.attempt_id, 1, blob(db, "x"), {}) == C.CKPT_REJECTED


def test_a_zombie_cannot_confirm_batches_already_confirmed_by_its_successor(db, clock):
    _, job, a = start_job(db)
    db.upload_checkpoint("w1", a.attempt_id, 1, blob(db, "c1"), {})
    clock.advance(EXPIRY + 60_001)
    b = db.request_job("w2")
    assert db.upload_checkpoint("w2", b.attempt_id, 2, blob(db, "c2"), {}) == C.CKPT_CONFIRMED
    assert db.upload_checkpoint("w1", a.attempt_id, 2, blob(db, "otro"), {}) == C.CKPT_REJECTED
    assert db.upload_checkpoint("w1", a.attempt_id, 1, blob(db, "c1"), {}) == C.CKPT_REJECTED


def test_invalid_checkpoint_arguments_are_400(db):
    _, job, a = start_job(db)
    with pytest.raises(ContractError) as e:
        db.upload_checkpoint("w1", a.attempt_id, 1, "no-es-un-hash", {})
    assert e.value.status == 400
    with pytest.raises(ContractError) as e:
        db.upload_checkpoint("w1", a.attempt_id, 1, blob(db, "x"), "no-es-dict")
    assert e.value.status == 400


def test_confirming_a_batch_counts_as_a_sign_of_life(db, clock):
    _, job, a = start_job(db)
    for k in (1, 2, 3):
        clock.advance(EXPIRY - 1_000)                          # nunca llega a vencer entre confirmaciones
        assert db.upload_checkpoint("w1", a.attempt_id, k, blob(db, f"c{k}"), {}) == C.CKPT_CONFIRMED
    assert job_of(db, job)["estado"] == C.EN_EJECUCION


# ---------------------------------------------------------------------------- cierre del job
def test_complete_requires_every_batch(db):
    _, job, a = start_job(db)
    train(db, "w1", a, upto=5, finish=False)
    assert db.complete_job("w1", a.attempt_id, blob(db, "m"), {"accuracy": 1.0}) == C.COMPLETE_REJECTED
    assert job_of(db, job)["estado"] == C.EN_EJECUCION


def test_complete_creates_model_and_canonical_version(db):
    _, job, a = start_job(db)
    train(db, "w1", a)
    v = job_of(db, job)
    assert v["estado"] == C.COMPLETADO and v["attempt_vigente"] is None
    assert v["canonical_version_id"] == a.attempt_id and v["model_version"] == 1
    assert [(x["estado"], x["revoke_reason"]) for x in v["attempts"]] == [(C.ATTEMPT_REVOCADO, None)]
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_REVOKED
    assert db.request_job("w1") is None


def test_complete_retry_after_a_lost_response_is_ok_and_creates_nothing_new(db):
    _, job, a = start_job(db)
    sha = train(db, "w1", a)
    versions = db.list_model_versions(CLIENT_A, job_of(db, job)["model_id"])
    assert db.complete_job("w1", a.attempt_id, sha, {"accuracy": 0.0}) == C.COMPLETE_OK
    assert db.list_model_versions(CLIENT_A, job_of(db, job)["model_id"]) == versions


def test_complete_from_a_revoked_attempt_is_rejected(db, clock):
    _, job, a = start_job(db)
    train(db, "w1", a, finish=False)
    clock.advance(EXPIRY + 1)
    assert db.complete_job("w1", a.attempt_id, blob(db, "m"), {}) == C.COMPLETE_REJECTED
    assert job_of(db, job)["estado"] == C.PENDIENTE


def test_complete_with_missing_model_binary_is_503(db):
    _, job, a = start_job(db)
    train(db, "w1", a, finish=False)
    with pytest.raises(ContractError) as e:
        db.complete_job("w1", a.attempt_id, sha256_hex(b"nunca subido"), {"accuracy": 1.0})
    assert (e.value.status, e.value.reason) == (503, "artifact_unavailable")
    assert job_of(db, job)["estado"] == C.EN_EJECUCION


# ---------------------------------------------------------------------------- fallos reportados
def test_worker_error_counts_and_the_job_is_retried(db):
    _, job, a = start_job(db)
    db.report_failure("w1", a.attempt_id, "boom")
    v = job_of(db, job)
    assert v["estado"] == C.PENDIENTE and v["attempts_count"] == 1 and v["attempt_vigente"] is None
    assert v["attempts"][0]["revoke_reason"] == C.WORKER_ERROR
    assert db.request_job("w2") is not None                   # los errores del worker no esperan backoff


def test_exhausting_max_attempts_fails_the_job(db):
    _, job, a = start_job(db)                                 # max_attempts = 3
    for n in range(3):
        db.report_failure("w1", a.attempt_id, f"fallo {n}")
        a = db.request_job("w1")
    v = job_of(db, job)
    assert v["estado"] == C.FALLIDO and v["attempts_count"] == 3 and a is None


def test_report_failure_from_a_non_current_attempt_is_ignored(db, clock):
    _, job, a = start_job(db)
    clock.advance(EXPIRY + 1)                                  # a queda revocado
    b = db.request_job("w2")
    db.report_failure("w1", a.attempt_id, "tarde")
    db.report_failure("w1", "no-existe", "x")
    v = job_of(db, job)
    assert v["attempts_count"] == 0 and v["attempt_vigente"] == b.attempt_id


def test_unavailable_artifact_does_not_count_and_job_waits_for_it(db):
    _, job, a = start_job(db)
    db.store.delete(sha256_hex(IRIS))
    db.report_unavailable("w1", a.attempt_id, sha256_hex(IRIS))
    v = job_of(db, job)
    assert v["estado"] == C.PENDIENTE and (v["attempts_count"], v["lost_count"]) == (0, 0)
    assert v["attempts"][0]["revoke_reason"] == C.ARTIFACT_UNAVAILABLE
    assert db.request_job("w2") is None                        # sigue sin dataset
    db.put_artifact(sha256_hex(IRIS), IRIS)
    assert db.request_job("w2") is not None


def test_failure_after_cancel_does_not_resurrect_the_job(db):
    _, job, a = start_job(db)
    db.cancel_job(CLIENT_A, "c", job["job_id"])
    db.report_failure("w1", a.attempt_id, "tarde")
    assert job_of(db, job)["estado"] == C.CANCELADO and job_of(db, job)["attempts_count"] == 0


def test_concurrent_workers_never_receive_the_same_job(db):
    import threading
    ds = new_dataset(db)
    expected = {new_job(db, ds["dataset_id"], rid=f"j{i}")["job_id"] for i in range(20)}
    got, lock = [], threading.Lock()

    def pull(name):
        while (a := db.request_job(name)) is not None:
            with lock:
                got.append(a.job_id)

    threads = [threading.Thread(target=pull, args=(f"w{i}",)) for i in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(got) == sorted(expected) and len(got) == len(set(got)) == 20
