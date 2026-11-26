"""La DB real como actor de Ray: sobrevive a que lo maten con un job a medias (marcada `ray`; pytest -m ray).

Es la prueba de «matar la DB»: Ray reinicia el actor (max_restarts=-1), el constructor recarga los CSV y
el trabajo continúa donde estaba.
"""
import os
import time
from pathlib import Path

import pytest

from tests.helpers.db_utils import CLIENT_A, IRIS, SMALL_HP, model_def_id, train
from distriml.common import constants as C
from distriml.common.ids import sha256_hex

pytestmark = pytest.mark.ray
ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="module")
def ray_ctx():
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")   # antes de ray.init
    import ray
    ray.init(namespace="distriml-db-test", include_dashboard=False, logging_level="ERROR", num_cpus=4,
             resources={"db": 1})
    yield ray
    ray.shutdown()


def _kill_and_wait_back(ray, handle, db, timeout=90):
    """Mata el actor y espera a verlo CAER y luego VOLVER. (`ray.kill` es asíncrono: un ping inmediato aún
    podría llegar al proceso viejo, así que primero se exige observar la caída.)"""
    from distriml.common.errors import ContractError
    ray.kill(handle, no_restart=False)
    deadline = time.time() + timeout
    went_down = False
    while time.time() < deadline:
        try:
            db.ping()
            if went_down:
                return
        except ContractError as exc:
            assert (exc.status, exc.reason) == (503, "no_ca")
            went_down = True
        time.sleep(0.2)
    pytest.fail("el actor de la DB " + ("no volvió tras el reinicio" if went_down else "no llegó a caer tras ray.kill"))


def test_killed_db_actor_comes_back_with_its_state_and_the_job_finishes(ray_ctx, tmp_path):
    from distriml.common.config import load_config
    from distriml.common.rayproxy import RayDBProxy
    from distriml.db.main import start_db

    cfg = load_config(data_dir=tmp_path, ray_namespace="distriml-db-test", db_actor_name="db-persist",
                      log_level="WARNING")
    handle = start_db(cfg)
    db = RayDBProxy(handle)
    ds = db.create_dataset(CLIENT_A, "ds", "iris.csv", sha256_hex(IRIS), len(IRIS), IRIS)
    job = db.create_job(CLIENT_A, "job", ds["dataset_id"], model_def_id("sgd_logistic"), SMALL_HP)
    a = db.request_job("w1")
    train(db, "w1", a, upto=3, finish=False)

    _kill_and_wait_back(ray_ctx, handle, db)

    v = db.get_job(CLIENT_A, job["job_id"])
    assert v["estado"] == C.EN_EJECUCION and v["last_confirmed_batch"] == 3 and v["attempt_vigente"] == a.attempt_id
    assert len(db.list_datasets(CLIENT_A)) == 1
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_CONTINUE          # el worker puede seguir con su attempt
    train(db, "w1", a)                                                      # batches 1-3 se aceptan como repetidos
    done = db.get_job(CLIENT_A, job["job_id"])
    assert done["estado"] == C.COMPLETADO and done["last_confirmed_batch"] == 6
    assert db.predict(CLIENT_A, done["model_id"], {"features": [5.1, 3.5, 1.4, 0.2]})["version_id"] == a.attempt_id

    _kill_and_wait_back(ray_ctx, handle, db)                                # y una segunda vez, ya con el modelo
    assert db.get_job(CLIENT_A, job["job_id"])["estado"] == C.COMPLETADO
    assert db.get_job_metrics(CLIENT_A, job["job_id"])["final"]["accuracy"] > 0.3
    assert db.get_model_stats(CLIENT_A, done["model_id"])["total_calls"] == 1
