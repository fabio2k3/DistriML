"""Contrato externo (docs/CONTRATO.md) contra la `Database` REAL: gateway + Database + worker de referencia.

Reutiliza las pruebas de `test_contract_live.py` (que por defecto usan FakeDB) cambiando solo el fixture
que levanta el sistema. Así el comportamiento visible de la BD real queda comprobado con las mismas
pruebas que definen el contrato.
"""
import socket
import threading
import time

import pytest
import uvicorn

from tests.test_contract_live import *  # noqa: F401,F403  (importa las pruebas y los fixtures `api`, `iris`)


@pytest.fixture(scope="module")
def base_url(tmp_path_factory):
    from distriml.api.app import create_app
    from distriml.common.config import load_config
    from distriml.db.service import Database
    from tests.helpers.reference_worker import ReferenceWorkerThread

    db = Database(load_config(data_dir=tmp_path_factory.mktemp("db-contract"), log_level="WARNING"))
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
