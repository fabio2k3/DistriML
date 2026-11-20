"""Gateway con FakeDB y TestClient: formas de error y validaciones, sin workers."""
import base64

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLES
from tests.helpers.fake_db import FakeDB
from distriml.api.app import create_app
from distriml.common.ids import new_client_id, sha256_hex


@pytest.fixture()
def c():
    return TestClient(create_app(FakeDB()))


@pytest.fixture()
def h():
    return {"X-Client-Id": new_client_id("ana")}


def _upload_body(raw: bytes, rid="r1", **over):
    body = {"client_request_id": rid, "name": "d.csv", "sha256": sha256_hex(raw), "size": len(raw),
            "data": base64.b64encode(raw).decode()}
    body.update(over)
    return body


def test_health_needs_no_identity(c):
    assert c.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("hdr", [{}, {"X-Client-Id": "zz"}])
def test_missing_or_bad_client_id_is_400(c, hdr):
    r = c.get("/jobs", headers=hdr)
    assert r.status_code == 400 and r.json()["error"]["reason"] == "invalid_client_id"


def test_error_body_shape_and_404(c, h):
    r = c.get("/jobs/nope", headers=h)
    assert r.status_code == 404 and set(r.json()["error"]) >= {"status", "reason", "message"}
    assert c.get("/no-such-route", headers=h).status_code == 404


def test_validation_errors_are_400(c, h):
    assert c.post("/datasets", headers=h, json={"name": "x"}).status_code == 400
    data = (SAMPLES / "iris.csv").read_bytes()
    assert c.post("/datasets", headers=h, json=_upload_body(data, sha256="0" * 64)).json()["error"]["reason"] == "hash_mismatch"
    assert c.post("/datasets", headers=h, json=_upload_body(data, size=1)).json()["error"]["reason"] == "size_mismatch"
    assert c.post("/datasets", headers=h, json=_upload_body(data, data="***")).status_code == 400


def test_upload_idempotent_and_conflict(c, h):
    data = (SAMPLES / "iris.csv").read_bytes()
    a = c.post("/datasets", headers=h, json=_upload_body(data)).json()
    b = c.post("/datasets", headers=h, json=_upload_body(data)).json()
    assert a["dataset_id"] == b["dataset_id"] and a["n_rows"] == 150 and a["classes"] == [0, 1, 2]
    other = b"a,y\n1,0\n2,1\n"
    assert c.post("/datasets", headers=h, json=_upload_body(other)).status_code == 409


def test_isolation_between_clients(c, h):
    data = (SAMPLES / "iris.csv").read_bytes()
    did = c.post("/datasets", headers=h, json=_upload_body(data)).json()["dataset_id"]
    other = {"X-Client-Id": new_client_id("bob")}
    assert c.get(f"/datasets/{did}", headers=other).status_code == 404
    assert c.get("/datasets", headers=other).json() == []


def test_predict_unknown_model_404_and_bad_body_400(c, h):
    assert c.post("/predict", headers=h, json={"model_id": "x", "input": {"features": [1]}}).status_code == 404
    assert c.post("/predict", headers=h, json={"model_id": "x"}).status_code == 400
