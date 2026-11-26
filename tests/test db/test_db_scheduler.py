"""Reglas puras de db/scheduler.py: plazos, motivos de revocación, presupuestos, retardo y elección de job."""
import pytest

from distriml.common import constants as C
from distriml.db import scheduler as S


def test_lease_deadline_is_renew_plus_lease_plus_margin():
    assert S.lease_deadline_ms(1_000, 10_000, 1_000) == 12_000


def test_expired_reason_depends_on_whether_attempt_started():
    assert S.revoke_reason_for_expired(True) == C.WORKER_LOST
    assert S.revoke_reason_for_expired(False) == C.NEVER_STARTED


@pytest.mark.parametrize("reason,attempts,lost,expected", [
    (C.WORKER_ERROR, 0, 0, (C.PENDIENTE, 1, 0)),
    (C.WORKER_ERROR, 3, 7, (C.PENDIENTE, 4, 7)),
    (C.WORKER_ERROR, 4, 0, (C.FALLIDO, 5, 0)),             # el quinto error agota max_attempts=5
    (C.WORKER_LOST, 0, 0, (C.PENDIENTE, 0, 1)),
    (C.WORKER_LOST, 2, 18, (C.PENDIENTE, 2, 19)),
    (C.WORKER_LOST, 2, 19, (C.FALLIDO, 2, 20)),            # la vigésima pérdida agota max_lost=20
    (C.NEVER_STARTED, 4, 19, (C.PENDIENTE, 4, 19)),        # no cuentan
    (C.ARTIFACT_UNAVAILABLE, 4, 19, (C.PENDIENTE, 4, 19)),
    (C.CANCEL, 1, 1, (C.CANCELADO, 1, 1)),
])
def test_next_state_after_revocation(reason, attempts, lost, expected):
    assert S.next_state_after_revocation(reason, attempts, lost, max_attempts=5, max_lost=20) == expected


def test_unknown_revocation_reason_is_a_programming_error():
    with pytest.raises(ValueError):
        S.next_state_after_revocation("OTRO", 0, 0, 5, 20)


def test_backoff_doubles_and_is_capped():
    assert [S.backoff_ms(n, 2_000) for n in (0, 1, 2, 3)] == [2_000, 4_000, 8_000, 16_000]
    assert S.backoff_ms(10, 2_000) == 60_000
    assert S.backoff_ms(-3, 2_000) == 2_000


def _job(job_id, created_at):
    return {"job_id": job_id, "created_at": created_at}


def test_pick_next_job_takes_the_oldest():
    jobs = [_job("b", 20), _job("a", 10), _job("c", 30)]
    assert S.pick_next_job(jobs, now_ms=0, not_before_ms={})["job_id"] == "a"


def test_pick_next_job_keeps_list_order_on_ties():
    assert S.pick_next_job([_job("x", 5), _job("y", 5)], 0, {})["job_id"] == "x"


def test_pick_next_job_skips_jobs_still_in_backoff():
    jobs = [_job("a", 10), _job("b", 20)]
    assert S.pick_next_job(jobs, now_ms=100, not_before_ms={"a": 101})["job_id"] == "b"
    assert S.pick_next_job(jobs, now_ms=101, not_before_ms={"a": 101})["job_id"] == "a"     # vence en ese instante


def test_pick_next_job_skips_jobs_that_cannot_start():
    jobs = [_job("a", 10), _job("b", 20)]
    assert S.pick_next_job(jobs, 0, {}, is_startable=lambda j: j["job_id"] != "a")["job_id"] == "b"


def test_pick_next_job_returns_none_when_nothing_is_eligible():
    assert S.pick_next_job([], 0, {}) is None
    assert S.pick_next_job([_job("a", 1)], 0, {"a": 10}) is None
    assert S.pick_next_job([_job("a", 1)], 0, {}, is_startable=lambda j: False) is None
