"""Sistema completo, sin simulaciones: Ray real + DB real (actor) + 2 workers reales (actores) + gateway HTTP
+ cliente. Todo lo que hace el usuario pasa por HTTP, igual que con `python -m distriml.cluster`.

Incluye las dos averías que el sistema promete tolerar: matar un worker y matar la DB con un job a medias.
Marcada `ray` (pytest -m ray); tarda ~1 minuto.
"""
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from tests.conftest import SAMPLES

pytestmark = pytest.mark.ray
ROOT = str(Path(__file__).resolve().parent.parent)
NS = "distriml-e2e"
NAMES = ["setosa", "versicolor", "virginica"]
SHORT = {"epochs": 3, "batch_size": 50}                 # 9 batches
LONG = {"epochs": 12, "batch_size": 10}                 # 180 batches: da tiempo a provocar la avería


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    import ray
    import uvicorn

    from distriml.api.app import create_app
    from distriml.common.config import load_config
    from distriml.common.rayproxy import RayDBProxy
    from distriml.db.main import start_db
    from distriml.worker.main import start_workers

    ray.init(namespace=NS, include_dashboard=False, logging_level="ERROR", num_cpus=6, resources={"db": 1})
    cfg = load_config(data_dir=tmp_path_factory.mktemp("e2e"), ray_namespace=NS, log_level="WARNING",
                      lease_duration_ms=2000, renew_interval_ms=500, epsilon_ms=200, request_poll_interval_ms=50)
    db_handle = start_db(cfg)
    workers = start_workers(cfg, db_handle, 2)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(RayDBProxy(db_handle)), host="127.0.0.1", port=port,
                                           log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "ray": ray, "db": db_handle, "proxy": RayDBProxy(db_handle),
           "workers": workers}
    server.should_exit = True
    thread.join(timeout=5)
    ray.shutdown()


def new_client(system, user="a"):
    """`user` es un solo carácter hexadecimal: el client_id son 64 caracteres hex."""
    from distriml.client.http import ApiClient
    return ApiClient(system["url"], user * 64, timeout=60)


def model_def(client, algorithm="sgd_logistic"):
    return next(d["model_def_id"] for d in client.models() if d["algorithm"] == algorithm)


def iris_names():
    return (SAMPLES / "iris_names.csv").read_bytes()


def kill_and_wait_back(system, timeout=90):
    """Mata la DB y espera a verla CAER y luego VOLVER (`ray.kill` es asíncrono)."""
    from distriml.common.errors import ContractError
    proxy = system["proxy"]
    system["ray"].kill(system["db"], no_restart=False)
    went_down, end = False, time.time() + timeout
    while time.time() < end:
        try:
            proxy.ping()
            if went_down:
                return
        except ContractError as exc:
            assert (exc.status, exc.reason) == (503, "no_ca")
            went_down = True
        time.sleep(0.2)
    pytest.fail("la DB " + ("no volvió tras el reinicio" if went_down else "no llegó a caer"))


def wait_batches(client, job_id, at_least, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        job = client.job(job_id)
        if job["last_confirmed_batch"] >= at_least or job["estado"] in ("COMPLETADO", "FALLIDO", "CANCELADO"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"el job no avanzó hasta el batch {at_least}")


def train(client, hp, name="iris_names.csv", data=None, rid=None):
    ds = client.upload_dataset(name, data or iris_names())
    job = client.submit_job(ds["dataset_id"], model_def(client), hp, client_request_id=rid)
    return ds, job


# ---------------------------------------------------------------------------- camino normal
def test_full_user_journey_with_class_names(system):
    c = new_client(system, "a")
    try:
        ds, job = train(c, SHORT)
        assert ds["class_names"] == NAMES
        done = c.wait_job(job["job_id"], timeout=120)
        assert done["estado"] == "COMPLETADO" and done["last_confirmed_batch"] == done["total_batches"] == 9
        assert done["model_id"] and len(done["attempts"]) == 1

        (model,) = c.trained_models()                                         # el modelo aparece en el listado
        assert model["model_id"] == done["model_id"] and model["job_id"] == job["job_id"] and model["versions"] == 1
        assert model["metrics"]["accuracy"] > 0.5

        for features, expected in (([5.0, 3.4, 1.5, 0.2], "setosa"), ([6.9, 3.1, 5.4, 2.1], "virginica")):
            res = c.predict(model["model_id"], features)
            assert res["label"] == NAMES[res["prediction"]] and res["version_id"] == model["canonical_version_id"]
            assert res["label"] == expected
        assert c.stats(model["model_id"])["total_calls"] == 2
        (version,) = c.versions(model["model_id"])
        assert version["canonical"] and c.metrics(job["job_id"])["final"] == model["metrics"]
    finally:
        c.close()


def test_users_are_isolated_end_to_end(system):
    from distriml.client.http import ApiError
    a, b = new_client(system, "1"), new_client(system, "2")
    try:
        _, job = train(a, SHORT)
        done = a.wait_job(job["job_id"], timeout=120)
        assert b.jobs() == [] and b.trained_models() == [] and b.datasets() == []
        for call in (lambda: b.job(job["job_id"]), lambda: b.predict(done["model_id"], [1, 2, 3, 4]),
                     lambda: b.versions(done["model_id"])):
            with pytest.raises(ApiError) as exc:
                call()
            assert exc.value.status == 404
    finally:
        a.close()
        b.close()


def test_resubmitting_with_the_same_request_id_does_not_duplicate_work(system):
    c = new_client(system, "3")
    try:
        ds, first = train(c, SHORT, rid="mismo")
        again = c.submit_job(ds["dataset_id"], model_def(c), SHORT, client_request_id="mismo")
        assert again["job_id"] == first["job_id"]
        c.wait_job(first["job_id"], timeout=120)
        assert len(c.jobs()) == 1 and len(c.trained_models()) == 1
    finally:
        c.close()


@pytest.mark.parametrize("user,algo,sample,features,floor", [
    ("c", "knn_classifier", "iris_names.csv", [5.0, 3.4, 1.5, 0.2], ("accuracy", 0.9)),
    ("d", "nearest_centroid", "iris_names.csv", [6.9, 3.1, 5.4, 2.1], ("accuracy", 0.8)),
    ("e", "knn_regressor", "diabetes.csv", [0.0] * 10, ("r2", 0.3)),
    ("f", "least_squares", "diabetes.csv", [0.0] * 10, ("r2", 0.4)),
])
def test_hand_made_models_work_through_the_whole_stack(system, user, algo, sample, features, floor):
    """kNN, centroide y mínimos cuadrados: sus clases viajan por pickle entre procesos de Ray distintos."""
    c = new_client(system, user)
    try:
        ds = c.upload_dataset(sample, (SAMPLES / sample).read_bytes())
        job = c.submit_job(ds["dataset_id"], model_def(c, algo), {"batch_size": 50})
        done = c.wait_job(job["job_id"], timeout=120)
        assert done["estado"] == "COMPLETADO" and done["last_confirmed_batch"] == done["total_batches"] > 1
        (model,) = c.trained_models()
        metric, minimum = floor
        assert model["metrics"][metric] > minimum, model["metrics"]
        res = c.predict(model["model_id"], features)
        assert res["version_id"] == model["canonical_version_id"]
        if algo in ("knn_classifier", "nearest_centroid"):
            assert res["label"] == NAMES[res["prediction"]]
    finally:
        c.close()


# ---------------------------------------------------------------------------- averías
def test_killing_workers_mid_job_still_gives_the_same_model(system):
    c = new_client(system, "4")
    try:
        ds = c.upload_dataset("iris_names.csv", iris_names())
        control = c.submit_job(ds["dataset_id"], model_def(c), LONG, client_request_id="control")
        control_done = c.wait_job(control["job_id"], timeout=180)
        assert control_done["estado"] == "COMPLETADO"

        job = c.submit_job(ds["dataset_id"], model_def(c), LONG, client_request_id="con-averia")
        running = wait_batches(c, job["job_id"], 30)
        assert running["estado"] == "EN_EJECUCION", "el job terminó antes de provocar la avería: la prueba no probaría nada"
        for w in system["workers"]:                                            # se cae el worker que lo ejecuta
            system["ray"].kill(w, no_restart=False)
        done = c.wait_job(job["job_id"], timeout=180)

        assert done["estado"] == "COMPLETADO" and done["last_confirmed_batch"] == 180
        assert len(done["attempts"]) >= 2, "tendría que haberse reasignado tras la caída"
        m_control = c.metrics(control["job_id"])["final"]
        m_failed = c.metrics(job["job_id"])["final"]
        assert m_failed == m_control                                           # mismo entrenamiento, con o sin avería
        probe = [5.8, 2.7, 5.1, 1.9]
        assert (c.predict(done["model_id"], probe)["prediction"] ==
                c.predict(control_done["model_id"], probe)["prediction"])
    finally:
        c.close()


def test_killing_the_db_mid_job_loses_nothing(system):
    c = new_client(system, "5")
    try:
        ds, job = train(c, LONG, rid="db-cae")
        running = wait_batches(c, job["job_id"], 30)
        assert running["estado"] == "EN_EJECUCION", "el job terminó antes de provocar la avería: la prueba no probaría nada"
        kill_and_wait_back(system)
        done = c.wait_job(job["job_id"], timeout=180)
        assert done["estado"] == "COMPLETADO" and done["last_confirmed_batch"] == 180

        assert c.dataset(ds["dataset_id"])["name"] == "iris_names.csv"         # lo anterior sigue ahí
        (model,) = c.trained_models()
        assert model["model_id"] == done["model_id"]
        assert c.predict(model["model_id"], [5.0, 3.4, 1.5, 0.2])["label"] == "setosa"
        before = c.stats(model["model_id"])["total_calls"]

        kill_and_wait_back(system)                                             # y otra vez, ya con el modelo guardado
        assert c.trained_models()[0]["model_id"] == model["model_id"]
        assert c.stats(model["model_id"])["total_calls"] == before
        assert c.predict(model["model_id"], [5.0, 3.4, 1.5, 0.2])["label"] == "setosa"
    finally:
        c.close()


def test_duplicates_and_deletions_are_remembered_after_the_db_restarts(system):
    from distriml.client.http import ApiError
    c = new_client(system, "6")
    try:
        first = c.upload_dataset("iris_names.csv", iris_names())
        again = c.upload_dataset("copia.csv", iris_names())
        assert again["dataset_id"] == first["dataset_id"] and again["already_existed"] is True
        job = c.submit_job(first["dataset_id"], model_def(c), SHORT)
        done = c.wait_job(job["job_id"], timeout=120)
        (version,) = c.versions(done["model_id"])
        deleted = c.delete_version(done["model_id"], version["version_id"], client_request_id="borrar")
        assert deleted["model_deleted"] is True and c.trained_models() == []

        kill_and_wait_back(system)

        assert c.trained_models() == []                                           # el borrado sigue vigente
        with pytest.raises(ApiError) as exc:
            c.predict(done["model_id"], [5.0, 3.4, 1.5, 0.2])
        assert exc.value.status == 404
        assert c.delete_version(done["model_id"], version["version_id"], client_request_id="borrar") == deleted
        assert len(c.datasets()) == 1                                             # y el dataset no se duplicó
        assert c.upload_dataset("otra-vez.csv", iris_names())["already_existed"] is True
        assert len(c.datasets()) == 1 and c.job(job["job_id"])["estado"] == "COMPLETADO"
    finally:
        c.close()
