"""PRUEBA DE CONTRATO EXTERNO de extremo a extremo (ver docs/CONTRATO.md).

Por defecto levanta en el propio proceso: gateway (uvicorn) + FakeDB + un worker de referencia.
Para probar OTRO stack (p. ej. tu BD real + workers reales en `python -m distriml.cluster`):

    DISTRIML_URL=http://127.0.0.1:8080 pytest tests/test_contract_live.py

Estas pruebas son la definición ejecutable del comportamiento visible para el usuario de la API.
"""
import os
import socket
import threading
import time

import pytest
import uvicorn

from tests.conftest import SAMPLES
from distriml.client.http import ApiClient, ApiError
from distriml.common.ids import new_client_id


@pytest.fixture(scope="module")
def base_url():
    url = os.environ.get("DISTRIML_URL")
    if url:
        yield url
        return
    from tests.helpers.fake_db import FakeDB
    from tests.helpers.reference_worker import ReferenceWorkerThread
    from distriml.api.app import create_app

    db = FakeDB()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(db), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    worker = ReferenceWorkerThread(db)
    worker.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    worker.stop()
    server.should_exit = True
    t.join(timeout=5)


@pytest.fixture()
def api(base_url):
    cl = ApiClient(base_url, new_client_id("tester"), retry_wait=0.1)
    yield cl
    cl.close()


@pytest.fixture(scope="module")
def iris():
    return (SAMPLES / "iris.csv").read_bytes()


def _model(api, algo):
    return next(d["model_def_id"] for d in api.models() if d["algorithm"] == algo)


def _trained(api, iris, epochs=5, rid=None):
    ds = api.upload_dataset("iris.csv", iris)
    job = api.submit_job(ds["dataset_id"], _model(api, "sgd_logistic"), {"epochs": epochs}, client_request_id=rid)
    return ds, job, api.wait_job(job["job_id"], timeout=60)


def test_catalog(api):
    defs = api.models()
    assert {d["algorithm"] for d in defs} >= {"sgd_regressor", "sgd_logistic", "perceptron", "gaussian_nb"}
    assert all({"model_def_id", "name", "default_hyperparams", "hyperparam_schema"} <= set(d) for d in defs)


def test_upload_idempotent(api, iris):
    a = api.upload_dataset("iris.csv", iris, client_request_id="same")
    b = api.upload_dataset("iris.csv", iris, client_request_id="same")
    assert a["dataset_id"] == b["dataset_id"] and (a["n_rows"], a["n_features"]) == (150, 4)
    assert any(d["dataset_id"] == a["dataset_id"] for d in api.datasets())
    assert api.dataset(a["dataset_id"])["sha256"] == __import__("hashlib").sha256(iris).hexdigest()


def test_full_flow_train_predict_stats_versions(api, iris):
    ds, job, done = _trained(api, iris)
    assert done["estado"] == "COMPLETADO" and done["last_confirmed_batch"] == done["total_batches"]
    assert done["model_id"] and done["canonical_version_id"] and done["model_version"] == 1
    m = api.metrics(job["job_id"])
    assert len(m["batches"]) == done["total_batches"] and m["final"]["accuracy"] > 0.5
    p = api.predict(done["model_id"], [5.1, 3.5, 1.4, 0.2])
    assert p["version_id"] == done["canonical_version_id"] and len(p["sha256"]) == 64
    assert api.predict(done["model_id"], [6.7, 3.0, 5.2, 2.3], version=1)["version_id"] == p["version_id"]
    st = api.stats(done["model_id"])
    assert st["total_calls"] == 2 and st["mean_latency_ms"] >= 0 and st["por_minuto"]
    vs = api.versions(done["model_id"])
    assert len(vs) == 1 and vs[0]["canonical"] and vs[0]["version_id"] == done["canonical_version_id"]
    assert api.version(done["model_id"], vs[0]["version_id"])["sha256"] == p["sha256"]


def test_list_trained_models(api, iris):
    _, _, done = _trained(api, iris)
    mine = api.trained_models()
    entry = next(m for m in mine if m["model_id"] == done["model_id"])
    assert entry["job_id"] == done["job_id"] and entry["versions"] == 1
    assert entry["canonical_version_id"] == done["canonical_version_id"] and entry["metrics"]["accuracy"] > 0.5
    assert api.predict(entry["model_id"], [5.1, 3.5, 1.4, 0.2])["version_id"] == entry["canonical_version_id"]
    assert [m["model_id"] for m in api.trained_models(limit=1)] == [mine[0]["model_id"]]


def test_text_labels_are_answered_with_the_class_name(api):
    data = (SAMPLES / "iris_names.csv").read_bytes()
    ds = api.upload_dataset("iris_names.csv", data)
    assert ds["class_names"] == ["setosa", "versicolor", "virginica"] and ds["classes"] == [0, 1, 2]
    job = api.submit_job(ds["dataset_id"], _model(api, "sgd_logistic"), {"epochs": 5})
    done = api.wait_job(job["job_id"], timeout=60)
    assert done["estado"] == "COMPLETADO"
    res = api.predict(done["model_id"], [5.1, 3.5, 1.4, 0.2])
    assert res["label"] == ds["class_names"][res["prediction"]]
    assert api.predict(done["model_id"], [5.1, 3.5, 1.4, 0.2], version=1)["label"] == res["label"]


def test_uploading_the_same_content_again_reuses_the_dataset(api, iris):
    first = api.upload_dataset("iris.csv", iris)
    again = api.upload_dataset("iris-copia.csv", iris)                   # otro nombre y otro request_id, mismo contenido
    assert again["dataset_id"] == first["dataset_id"] and again["already_existed"] is True
    assert "already_existed" not in first
    assert sum(1 for d in api.datasets() if d["dataset_id"] == first["dataset_id"]) == 1
    assert len({d["sha256"] for d in api.datasets()}) == len(api.datasets())           # ningún contenido repetido


def test_delete_a_model_version(api, iris):
    from distriml.client.http import ApiError
    _, job, done = _trained(api, iris)
    model_id = done["model_id"]
    (ver,) = api.versions(model_id)
    res = api.delete_version(model_id, ver["version_id"], client_request_id="borrar-1")
    assert res["deleted"] == ver["version_id"] and res["model_deleted"] is True and res["entry_id"]
    assert api.delete_version(model_id, ver["version_id"], client_request_id="borrar-1") == res       # idempotente
    assert all(m["model_id"] != model_id for m in api.trained_models())
    for call in (lambda: api.predict(model_id, [5.1, 3.5, 1.4, 0.2]), lambda: api.versions(model_id),
                 lambda: api.stats(model_id), lambda: api.delete_version(model_id, ver["version_id"])):
        with pytest.raises(ApiError) as exc:
            call()
        assert exc.value.status == 404
    assert api.job(job["job_id"])["estado"] == "COMPLETADO"


def test_delete_request_id_reuse_for_another_version_is_409(api, iris):
    from distriml.client.http import ApiError
    _, _, a = _trained(api, iris, rid="del-a")
    _, _, b = _trained(api, iris, rid="del-b")
    (va,), (vb,) = api.versions(a["model_id"]), api.versions(b["model_id"])
    api.delete_version(a["model_id"], va["version_id"], client_request_id="mismo-id")
    with pytest.raises(ApiError) as exc:
        api.delete_version(b["model_id"], vb["version_id"], client_request_id="mismo-id")
    assert exc.value.status == 409
    assert api.versions(b["model_id"])[0]["version_id"] == vb["version_id"]


def test_submit_idempotent(api, iris):
    ds = api.upload_dataset("iris.csv", iris)
    mid = _model(api, "perceptron")
    a = api.submit_job(ds["dataset_id"], mid, {"epochs": 1}, client_request_id="job-x")
    b = api.submit_job(ds["dataset_id"], mid, {"epochs": 1}, client_request_id="job-x")
    assert a["job_id"] == b["job_id"]
    with pytest.raises(ApiError) as e:
        api.submit_job(ds["dataset_id"], mid, {"epochs": 2}, client_request_id="job-x")
    assert e.value.status == 409


def test_isolation_between_clients(api, base_url, iris):
    _, job, done = _trained(api, iris, epochs=1)
    other = ApiClient(base_url, new_client_id("intruso"), retry_wait=0.1)
    try:
        for call in (lambda: other.job(job["job_id"]), lambda: other.metrics(job["job_id"]),
                     lambda: other.predict(done["model_id"], [1, 2, 3, 4]), lambda: other.versions(done["model_id"]),
                     lambda: other.cancel(job["job_id"])):
            with pytest.raises(ApiError) as e:
                call()
            assert e.value.status == 404
        assert other.jobs() == []
    finally:
        other.close()


def test_cancel_completed_is_409(api, iris):
    _, job, _ = _trained(api, iris, epochs=1)
    with pytest.raises(ApiError) as e:
        api.cancel(job["job_id"])
    assert e.value.status == 409


def test_cancel_running_job(api, iris):
    ds = api.upload_dataset("iris.csv", iris)
    job = api.submit_job(ds["dataset_id"], _model(api, "sgd_logistic"), {"epochs": 1000, "batch_size": 5})
    time.sleep(0.3)
    res = api.cancel(job["job_id"], client_request_id="c1")
    assert res["estado"] == "CANCELADO"
    assert api.cancel(job["job_id"], client_request_id="c1")["estado"] == "CANCELADO"   # idempotente
    time.sleep(0.3)
    final = api.job(job["job_id"])
    assert final["estado"] == "CANCELADO"
    assert final["last_confirmed_batch"] < final["total_batches"]      # no sigue avanzando ni se completa


@pytest.mark.parametrize("make", [
    lambda api, ds, md: api.submit_job(ds, "no-existe", None),
    lambda api, ds, md: api.submit_job(ds, md, {"epochs": 0}),
    lambda api, ds, md: api.submit_job(ds, md, {"inventado": 1}),
    lambda api, ds, md: api.submit_job("dataset-ajeno", md, None),
    lambda api, ds, md: api.upload_dataset("mal.csv", b"a,y\nx,1\n2,3\n"),
])
def test_validations_are_400(api, iris, make):
    ds = api.upload_dataset("iris.csv", iris)["dataset_id"]
    with pytest.raises(ApiError) as e:
        make(api, ds, _model(api, "sgd_logistic"))
    assert e.value.status == 400


def test_classification_rejected_on_regression_dataset(api):
    ds = api.upload_dataset("d.csv", (SAMPLES / "diabetes.csv").read_bytes())
    with pytest.raises(ApiError) as e:
        api.submit_job(ds["dataset_id"], _model(api, "sgd_logistic"), None)
    assert e.value.status == 400
    job = api.submit_job(ds["dataset_id"], _model(api, "sgd_regressor"), {"epochs": 2})
    assert api.wait_job(job["job_id"], timeout=60)["estado"] == "COMPLETADO"
