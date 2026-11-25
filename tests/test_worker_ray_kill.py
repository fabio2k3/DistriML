"""Hito 7 del plan (parte worker): matar un worker REAL (actor de Ray) a mitad de un job.

Cluster Ray local + FakeDB como actor (el mismo cableado que test_ray_proxy.py). Comprueba el criterio de
entrega 2: el job se reasigna, continúa desde el último checkpoint y termina COMPLETADO con el mismo modelo
que sin fallo. Se ejecuta con `pytest -m ray` (~30 s).
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.ray
ROOT = str(Path(__file__).resolve().parent.parent)
NS = "distriml-test"


@pytest.fixture(scope="module")
def ray_ctx():
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    import ray
    ray.init(namespace=NS, include_dashboard=False, logging_level="ERROR", num_cpus=4)
    yield ray
    ray.shutdown()


def make_cfg(tmp_path):
    from distriml.common.config import load_config
    return load_config(data_dir=tmp_path, ray_namespace=NS, lease_duration_ms=2000, renew_interval_ms=500,
                       epsilon_ms=200, request_poll_interval_ms=50)


def estimator(db, job):
    from distriml.common import algorithms
    from tests.test_worker_wk1_wk2 import CLIENT
    (v,) = db.list_model_versions(CLIENT, job["model_id"])
    return algorithms.deserialize_state(db.get_artifact(v["sha256"]))["estimator"]


def wait_job(db, job_id, pred, timeout):
    from tests.test_worker_wk1_wk2 import CLIENT
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = db.get_job(CLIENT, job_id)
        if pred(job):
            return job
        time.sleep(0.1)
    raise AssertionError(f"timeout esperando al job; último estado: {job['estado']} batch {job['last_confirmed_batch']}")


def test_kill_worker_mid_job_resumes_and_gives_same_model(ray_ctx, tmp_path):
    ray = ray_ctx
    from distriml.common import constants as C
    from distriml.common.rayproxy import RayDBProxy
    from distriml.worker.main import start_workers
    from distriml.worker.trainer import run_attempt
    from tests.helpers.fake_db import FakeDB
    from tests.test_worker_wk1_wk2 import CLIENT, seed_job

    cfg = make_cfg(tmp_path)
    hp = {"epochs": 60}

    # Referencia: el mismo job sin fallos, en local.
    clean = FakeDB(cfg, str(tmp_path / "clean"))
    seed_job(clean, "r1", hp)
    run_attempt(clean.request_job("ref"), "ref", clean, cfg)
    clean_job = next(iter(clean.jobs.values()))
    ref = estimator(clean, clean_job)

    # Sistema con Ray: DB como actor + un worker real.
    handle = ray.remote(FakeDB).options(name="fake-db-kill", namespace=NS).remote(cfg, str(tmp_path / "ray"))
    db = RayDBProxy(handle)
    job_id = seed_job(db, "r1", hp)
    workers = start_workers(cfg, handle, 1)                    # conservar los handles
    total = db.get_job(CLIENT, job_id)["total_batches"]
    try:
        before = wait_job(db, job_id, lambda j: j["last_confirmed_batch"] >= 20, 60)
        assert before["estado"] == C.EN_EJECUCION and before["last_confirmed_batch"] < total, "el job acabó antes de poder matarlo"
        ray.kill(workers[0], no_restart=False)                 # muerte brusca del worker; Ray lo reinicia
        killed_at = db.get_job(CLIENT, job_id)["last_confirmed_batch"]
        job = wait_job(db, job_id, lambda j: j["estado"] == C.COMPLETADO, 120)
    finally:
        for w in workers:
            ray.kill(w, no_restart=True)

    assert job["lost_count"] == 1 and job["attempts_count"] == 0     # se contó como pérdida, no como error
    first, second = sorted(job["attempts"], key=lambda x: x["created_at"])
    assert first["revoke_reason"] == C.WORKER_LOST
    assert before["last_confirmed_batch"] <= second["base_batch"] <= total   # reanudó desde un checkpoint, no desde cero
    assert second["base_batch"] >= killed_at - 1 or killed_at >= total        # y desde uno reciente, no uno antiguo
    print(f"\n[ray-kill] total={total} previo={before['last_confirmed_batch']} al_matar≈{killed_at} "
          f"reanuda_en={second['base_batch'] + 1} motivo={first['revoke_reason']}")
    got = estimator(db, job)
    assert np.array_equal(got.coef_, ref.coef_) and np.array_equal(got.intercept_, ref.intercept_)
    assert got.t_ == ref.t_
