"""`ApiClient` (distriml/client/http.py) sin red: el transporte HTTP se sustituye por uno simulado.

Se comprueba lo que el cliente promete: cabeceras de identidad, cuerpos y rutas de cada operación, y la
política de reintentos (solo ante 503 y errores de conexión, SIEMPRE con el mismo client_request_id).
"""
import base64
import json

import httpx
import pytest

from distriml.client.http import ApiClient, ApiError
from distriml.common.ids import sha256_hex

CLIENT_ID = "c" * 64


@pytest.fixture()
def sleeps(monkeypatch):
    """Anota las esperas en vez de dormir de verdad."""
    waited = []
    monkeypatch.setattr("distriml.client.http.time.sleep", waited.append)
    return waited


def make_client(handler, **kw) -> ApiClient:
    kw.setdefault("retry_wait", 0.5)
    base = kw.pop("base_url", "http://gw")
    client = ApiClient(base, CLIENT_ID, **kw)
    client.close()
    client._http = httpx.Client(base_url=base.rstrip("/"), transport=httpx.MockTransport(handler))
    return client


def ok(body, status=200):
    return httpx.Response(status, json=body)


def error(status, reason, message="", retry_after=None):
    err = {"reason": reason, "message": message}
    if retry_after is not None:
        err["retry_after"] = retry_after
    return httpx.Response(status, json={"error": err})


class Recorder:
    """Handler que devuelve respuestas en orden y guarda las peticiones recibidas."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    def body(self, i=-1):
        return json.loads(self.requests[i].content)


# ---------------------------------------------------------------------------- identidad y rutas
def test_every_request_carries_the_client_id():
    rec = Recorder(ok([]))
    make_client(rec).models()
    assert rec.requests[0].headers["X-Client-Id"] == CLIENT_ID
    assert "X-Min-Entry" not in rec.requests[0].headers


def test_trailing_slash_in_base_url_is_ignored():
    rec = Recorder(ok([]))
    make_client(rec, base_url="http://gw/").models()
    assert str(rec.requests[0].url) == "http://gw/model-definitions"


def test_entry_id_of_a_write_is_remembered_sent_next_and_reported():
    seen = []
    rec = Recorder(ok({"job_id": "j", "entry_id": "db:7"}), ok([]))
    client = make_client(rec, on_entry=seen.append)
    client.submit_job("d", "m")
    assert client.last_entry == "db:7" and seen == ["db:7"]
    client.jobs()
    assert rec.requests[1].headers["X-Min-Entry"] == "db:7"


def test_initial_last_entry_is_sent_from_the_first_request():
    rec = Recorder(ok([]))
    make_client(rec, last_entry="db:3").models()
    assert rec.requests[0].headers["X-Min-Entry"] == "db:3"


def test_responses_without_entry_id_do_not_change_it():
    rec = Recorder(ok([{"entry_id": "no-cuenta"}]), ok({"x": 1}))
    client = make_client(rec, last_entry="db:1")
    client.jobs()
    client.job("j")
    assert client.last_entry == "db:1"


def test_none_query_parameters_are_dropped():
    rec = Recorder(ok([]))
    make_client(rec).jobs(limit=5)
    assert dict(rec.requests[0].url.params) == {"limit": "5"}


def test_each_operation_uses_its_route_and_parameters():
    rec = Recorder(ok({}))
    c = make_client(rec)
    c.models(); c.datasets(limit=3, cursor="x"); c.dataset("d1"); c.jobs(estado="COMPLETADO")
    c.job("j1"); c.metrics("j1"); c.versions("m1"); c.version("m1", "v1")
    c.stats("m1", version_id="v1", from_minute=10, to_minute=20)
    got = [(r.method, r.url.path, dict(r.url.params)) for r in rec.requests]
    assert got == [
        ("GET", "/model-definitions", {}),
        ("GET", "/datasets", {"limit": "3", "cursor": "x"}),
        ("GET", "/datasets/d1", {}),
        ("GET", "/jobs", {"estado": "COMPLETADO", "limit": "100"}),
        ("GET", "/jobs/j1", {}),
        ("GET", "/jobs/j1/metrics", {}),
        ("GET", "/models/m1/versions", {}),
        ("GET", "/models/m1/versions/v1", {}),
        ("GET", "/models/m1/stats", {"version_id": "v1", "from": "10", "to": "20"}),
    ]


# ---------------------------------------------------------------------------- cuerpos
def test_upload_sends_hash_size_and_base64_content():
    data = b"a,b,y\n1,2,0\n3,4,1\n"
    rec = Recorder(ok({"dataset_id": "d"}))
    make_client(rec).upload_dataset("x.csv", data, client_request_id="rid-1")
    body = rec.body()
    assert rec.requests[0].url.path == "/datasets"
    assert (body["client_request_id"], body["name"], body["sha256"], body["size"]) == ("rid-1", "x.csv", sha256_hex(data), len(data))
    assert base64.b64decode(body["data"]) == data


def test_generated_request_ids_are_different_on_each_call_but_explicit_ones_are_kept():
    rec = Recorder(ok({}))
    c = make_client(rec)
    c.submit_job("d", "m"); c.submit_job("d", "m"); c.submit_job("d", "m", client_request_id="fijo")
    ids = [rec.body(i)["client_request_id"] for i in range(3)]
    assert ids[0] != ids[1] and ids[2] == "fijo"


def test_submit_cancel_and_predict_bodies():
    rec = Recorder(ok({"entry_id": "db:9"}), ok({}), ok({}))
    c = make_client(rec)
    c.submit_job("d1", "m1", {"epochs": 3}, client_request_id="r1")
    assert rec.body(0) == {"client_request_id": "r1", "dataset_id": "d1", "model_def_id": "m1", "hyperparams": {"epochs": 3}}
    c.cancel("j1", client_request_id="r2")
    assert (rec.requests[1].method, rec.requests[1].url.path, rec.body(1)) == ("POST", "/jobs/j1/cancel", {"client_request_id": "r2"})
    c.predict("m1", [1.0, 2.0], version=1)
    assert rec.body(2) == {"model_id": "m1", "version_id": None, "version": 1, "min_entry": "db:9", "input": {"features": [1.0, 2.0]}}


def test_trained_models_hits_models_not_the_catalog():
    rec = Recorder(ok([{"model_id": "m1"}]), ok([]), ok([]))
    c = make_client(rec)
    assert c.trained_models() == [{"model_id": "m1"}]
    c.trained_models(limit=5, cursor="m1")
    c.models()
    assert [(r.method, r.url.path) for r in rec.requests] == [("GET", "/models"), ("GET", "/models"), ("GET", "/model-definitions")]
    assert dict(rec.requests[0].url.params) == {"limit": "100"}
    assert dict(rec.requests[1].url.params) == {"limit": "5", "cursor": "m1"}


def test_delete_version_sends_a_delete_with_its_request_id():
    rec = Recorder(ok({"deleted": "v1"}), ok({"deleted": "v1"}))
    c = make_client(rec)
    c.delete_version("m1", "v1", client_request_id="r1")
    c.delete_version("m1", "v1")
    first, second = rec.requests
    assert (first.method, first.url.path, dict(first.url.params)) == ("DELETE", "/models/m1/versions/v1", {"client_request_id": "r1"})
    assert second.url.params["client_request_id"] and second.url.params["client_request_id"] != "r1"


def test_delete_version_retries_with_the_same_request_id(sleeps):
    rec = Recorder(error(503, "no_ca"), ok({"deleted": "v1"}))
    make_client(rec).delete_version("m1", "v1")
    assert len(rec.requests) == 2
    assert rec.requests[0].url.params["client_request_id"] == rec.requests[1].url.params["client_request_id"]


# ---------------------------------------------------------------------------- reintentos
def test_503_is_retried_with_the_same_request_id_and_waits(sleeps):
    rec = Recorder(error(503, "no_ca", "sin DB", retry_after=2), error(503, "no_ca"), ok({"job_id": "j", "entry_id": "db:1"}))
    res = make_client(rec).submit_job("d", "m", client_request_id="mismo")
    assert res["job_id"] == "j" and len(rec.requests) == 3
    assert {rec.body(i)["client_request_id"] for i in range(3)} == {"mismo"}
    assert sleeps == [2, 1.0]                   # retry_after del servidor; si no hay, retry_wait × intento


def test_generated_request_id_is_also_kept_across_retries(sleeps):
    rec = Recorder(error(503, "no_ca"), ok({"job_id": "j"}))
    make_client(rec).submit_job("d", "m")
    assert rec.body(0)["client_request_id"] == rec.body(1)["client_request_id"]


def test_gives_up_after_max_retries_with_the_last_503(sleeps):
    rec = Recorder(error(503, "no_ca", "caída"))
    with pytest.raises(ApiError) as e:
        make_client(rec, max_retries=2).jobs()
    assert (e.value.status, e.value.reason, e.value.message) == (503, "no_ca", "caída")
    assert len(rec.requests) == 3 and len(sleeps) == 2


@pytest.mark.parametrize("status,reason", [(400, "invalid_hyperparams"), (404, "not_found"), (409, "job_terminal"), (500, "boom")])
def test_other_errors_are_not_retried(status, reason, sleeps):
    rec = Recorder(error(status, reason, "detalle"))
    with pytest.raises(ApiError) as e:
        make_client(rec).job("j")
    assert (e.value.status, e.value.reason, e.value.message) == (status, reason, "detalle")
    assert len(rec.requests) == 1 and sleeps == []


def test_connection_errors_are_retried_and_then_reported_as_unreachable(sleeps):
    rec = Recorder(httpx.ConnectError("sin ruta"))
    with pytest.raises(ApiError) as e:
        make_client(rec, max_retries=2).models()
    assert (e.value.status, e.value.reason) == (503, "unreachable")
    assert len(rec.requests) == 3 and sleeps == [0.5, 1.0, 1.5]


def test_recovers_if_the_connection_comes_back(sleeps):
    rec = Recorder(httpx.ConnectTimeout("lento"), ok([{"algorithm": "x"}]))
    assert make_client(rec).models() == [{"algorithm": "x"}]
    assert len(rec.requests) == 2


def test_non_json_error_body_becomes_http_error():
    rec = Recorder(httpx.Response(502, text="bad gateway"))
    with pytest.raises(ApiError) as e:
        make_client(rec).models()
    assert (e.value.status, e.value.reason, e.value.message) == (502, "http_error", "bad gateway")


# ---------------------------------------------------------------------------- wait_job
def test_wait_job_returns_the_first_terminal_state(sleeps):
    rec = Recorder(ok({"estado": "PENDIENTE"}), ok({"estado": "EN_EJECUCION"}), ok({"estado": "COMPLETADO", "n": 1}))
    assert make_client(rec).wait_job("j", poll=0.1)["n"] == 1
    assert len(rec.requests) == 3 and sleeps == [0.1, 0.1]


@pytest.mark.parametrize("state", ["COMPLETADO", "FALLIDO", "CANCELADO"])
def test_wait_job_stops_on_every_terminal_state(state, sleeps):
    assert make_client(Recorder(ok({"estado": state}))).wait_job("j")["estado"] == state


def test_wait_job_times_out(sleeps):
    with pytest.raises(TimeoutError, match="EN_EJECUCION"):
        make_client(Recorder(ok({"estado": "EN_EJECUCION"}))).wait_job("j", timeout=0, poll=0)
