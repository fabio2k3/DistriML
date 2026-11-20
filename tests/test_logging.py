"""El registro de operaciones: formato, niveles y cobertura de gateway y DB."""
import logging

import pytest
from fastapi.testclient import TestClient

from tests.helpers.fake_db import FakeDB
from distriml.api.app import create_app
from distriml.common.errors import ContractError
from distriml.common.ids import new_client_id, sha256_hex
from distriml.common.log import setup_logging, short, traced


@pytest.fixture()
def records():
    """Captura lo que escribe el logger `distriml` (que no propaga al logger raíz)."""
    logger = logging.getLogger("distriml")
    old_level = logger.level
    out: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record):
            out.append(record)

    handler = Capture()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield out
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def _lines(records, name=None):
    return [(r.levelno, r.name, r.getMessage()) for r in records if name is None or r.name == name]


def test_short_summaries():
    assert short(b"x" * 1000) == "<1000 bytes>"
    assert short("a" * 64) == "a" * 8
    assert short(list(range(10))) == "[10 elementos]"
    assert len(short("x" * 500)) <= 80


def test_setup_logging_is_idempotent_and_accepts_levels():
    setup_logging("DEBUG")
    setup_logging("INFO")
    handlers = [h for h in logging.getLogger("distriml").handlers if getattr(h, "_distriml", False)]
    assert len(handlers) == 1 and logging.getLogger("distriml").level == logging.INFO


def test_traced_logs_calls_results_and_errors(records):
    @traced("t")
    class Svc:
        def create_job(self, cid, rid):
            return {"job_id": "j1"}

        def ping(self):
            return {"status": "ok"}

        def request_job(self, worker):
            return None

        def boom(self):
            raise ContractError(404, "not_found", "no existe")

    s = Svc()
    s.create_job("c", "r")
    s.ping()
    s.request_job("w0")
    with pytest.raises(ContractError):
        s.boom()
    by_msg = {m.split("(")[0]: lvl for lvl, _, m in _lines(records, "distriml.t")}
    assert by_msg["create_job"] == logging.INFO
    assert by_msg["ping"] == logging.DEBUG and by_msg["request_job"] == logging.DEBUG
    assert by_msg["boom"] == logging.WARNING
    assert any("404 not_found" in m for _, _, m in _lines(records, "distriml.t"))


def test_gateway_logs_each_request(records):
    c = TestClient(create_app(FakeDB()))
    h = {"X-Client-Id": new_client_id("ana")}
    c.get("/health")
    c.get("/model-definitions", headers=h)
    c.get("/jobs/nope", headers=h)
    c.get("/jobs")                       # sin identidad -> 400
    api = _lines(records, "distriml.api")
    assert any(l == logging.DEBUG and "GET /health" in m for l, _, m in api)
    assert any(l == logging.INFO and "GET /model-definitions" in m and "200" in m for l, _, m in api)
    assert any(l == logging.WARNING and "GET /jobs/nope" in m and "404" in m for l, _, m in api)
    assert any(l == logging.WARNING and "GET /jobs " in m and "400" in m and "invalid_client_id" in m for l, _, m in api)


def test_db_operations_are_logged_without_payloads(records):
    db = FakeDB()
    data = b"a,y\n1,0\n2,1\n"
    db.create_dataset(new_client_id("ana"), "r1", "d.csv", sha256_hex(data), len(data), data)
    msgs = [m for _, n, m in _lines(records, "distriml.fakedb") if m.startswith("create_dataset")]
    assert msgs and "<12 bytes>" in msgs[0] and "a,y" not in msgs[0]
