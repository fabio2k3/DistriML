"""Pruebas con un cluster Ray local real (marcadas `ray`; ~20-40 s). Ejecutar: pytest -m ray

Validan el pegamento de Ray, no la lógica de la BD:
  - el actor de la DB (esqueleto) arranca, responde y se REINICIA tras ser matado
  - ContractError cruza la frontera de Ray intacto (status/reason) y un actor caído => 503 no_ca
  - los workers se crean como actores nombrados y se reinician
  - un nodo worker crea su actor fijado a su nodo y lo retira al pararse
  - flujo completo con FakeDB como actor + worker de referencia + gateway (mismo cableado que cluster.py)
"""
import os
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.ray
ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture(scope="module")
def ray_ctx():
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")   # antes de ray.init
    import ray
    ray.init(namespace="distriml-test", include_dashboard=False, logging_level="ERROR", num_cpus=4,
             resources={"db": 1})                  # recurso que declara el nodo head en un cluster real
    yield ray
    ray.shutdown()


def test_db_actor_ping_errors_and_restart(ray_ctx, tmp_path):
    ray = ray_ctx
    from distriml.common.config import load_config
    from distriml.common.errors import ContractError
    from distriml.common.ids import sha256_hex
    from distriml.common.rayproxy import RayDBProxy
    from distriml.db.main import start_db

    cfg = load_config(data_dir=tmp_path, ray_namespace="distriml-test", db_actor_name="db-test")
    handle = start_db(cfg)
    db = RayDBProxy(handle)
    assert db.ping() == {"status": "ok"}
    from distriml.common import catalog
    assert len(db.list_model_definitions()) == len(catalog.CATALOG)
    with pytest.raises(ContractError) as e:                       # ContractError cruza Ray intacto
        db.get_artifact(sha256_hex(b"no existe"))
    assert (e.value.status, e.value.reason) == (503, "artifact_unavailable")
    data = b"hola"
    db.put_artifact(sha256_hex(data), data)
    ray.kill(handle, no_restart=False)
    for _ in range(80):                                            # Ray reinicia el actor (max_restarts=-1)
        try:                                                       # la lectura exige que ya esté de vuelta
            assert db.get_artifact(sha256_hex(data)) == data       # los artefactos siguen en disco
            break
        except ContractError as exc:
            assert (exc.status, exc.reason) == (503, "no_ca")
            time.sleep(0.5)
    else:
        pytest.fail("el actor de la DB no volvió tras el reinicio")


def test_workers_are_named_actors_and_restart(ray_ctx, tmp_path):
    ray = ray_ctx
    from distriml.common.config import load_config
    from distriml.db.main import start_db
    from distriml.worker.main import start_workers

    cfg = load_config(data_dir=tmp_path, ray_namespace="distriml-test", db_actor_name="db-test2")
    db = start_db(cfg)
    workers = start_workers(cfg, db, 2)          # IMPORTANTE: conservar los handles (si no, Ray mata los actores)
    w = ray.get_actor("worker-1", namespace="distriml-test")
    assert ray.get(w.status.remote())["worker_id"] == "worker-1"
    ray.kill(w, no_restart=False)                # Ray debe reiniciarlo con los mismos argumentos
    for _ in range(80):
        try:
            if ray.get(w.status.remote())["worker_id"] == "worker-1":
                break
        except Exception:
            time.sleep(0.5)
    else:
        pytest.fail("el worker no volvió tras el reinicio")
    assert len(workers) == 2


def test_worker_node_creates_and_retires_its_actor(ray_ctx, tmp_path):
    """worker/node.py: crea el actor del nodo, y al recibir la señal de parada lo retira."""
    import threading

    ray = ray_ctx
    from distriml.common.config import load_config
    from distriml.db.main import start_db
    from distriml.worker import node

    cfg = load_config(data_dir=tmp_path, ray_namespace="distriml-test", db_actor_name="db-test3")
    db = start_db(cfg)                                   # conservar el handle
    stop = threading.Event()
    t = threading.Thread(target=node.run, args=(cfg, "worker-nodo-test", stop, None), daemon=True)
    t.start()
    actor = None
    for _ in range(60):
        try:
            actor = ray.get_actor("worker-nodo-test", namespace="distriml-test")
            break
        except ValueError:
            time.sleep(0.5)
    assert actor is not None, "el nodo no creó su actor worker"
    assert ray.get(actor.status.remote())["worker_id"] == "worker-nodo-test"
    stop.set()
    t.join(timeout=30)
    assert not t.is_alive()
    for _ in range(40):                                  # el actor debe haber desaparecido
        try:
            ray.get_actor("worker-nodo-test", namespace="distriml-test")
            time.sleep(0.5)
        except ValueError:
            break
    else:
        pytest.fail("el actor no fue retirado")
    assert db is not None


def test_full_flow_with_fakedb_actor(ray_ctx, tmp_path):
    ray = ray_ctx
    from fastapi.testclient import TestClient

    from tests.conftest import SAMPLES
    from tests.helpers.fake_db import FakeDB
    from tests.helpers.reference_worker import ReferenceWorkerThread
    from distriml.api.app import create_app
    from distriml.client.http import ApiClient
    from distriml.common.ids import new_client_id
    from distriml.common.rayproxy import RayDBProxy

    handle = ray.remote(FakeDB).options(name="fake-db", namespace="distriml-test").remote(None, str(tmp_path))
    db = RayDBProxy(handle)
    worker = ReferenceWorkerThread(db)
    worker.start()
    try:
        client = TestClient(create_app(db))
        h = {"X-Client-Id": new_client_id("ray")}
        import base64
        from distriml.common.ids import sha256_hex
        raw = (SAMPLES / "iris.csv").read_bytes()
        ds = client.post("/datasets", headers=h, json={"client_request_id": "r", "name": "i", "sha256": sha256_hex(raw),
                                                       "size": len(raw), "data": base64.b64encode(raw).decode()}).json()
        mdef = next(d["model_def_id"] for d in client.get("/model-definitions", headers=h).json() if d["algorithm"] == "gaussian_nb")
        job = client.post("/jobs", headers=h, json={"client_request_id": "j", "dataset_id": ds["dataset_id"], "model_def_id": mdef}).json()
        for _ in range(100):
            st = client.get(f"/jobs/{job['job_id']}", headers=h).json()
            if st["estado"] == "COMPLETADO":
                break
            time.sleep(0.2)
        assert st["estado"] == "COMPLETADO"
        r = client.post("/predict", headers=h, json={"model_id": st["model_id"], "input": {"features": [5.1, 3.5, 1.4, 0.2]}})
        assert r.status_code == 200 and r.json()["prediction"] == 0
        assert client.get(f"/jobs/{job['job_id']}", headers={"X-Client-Id": new_client_id("otro")}).status_code == 404
    finally:
        worker.stop()
