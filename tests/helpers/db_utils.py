"""Utilidades para probar el módulo `db` sin Ray ni HTTP: reloj controlable, fábrica de `Database`
y un entrenador mínimo que sube checkpoints reales (comportamiento de un worker, sin leases)."""
from __future__ import annotations

from tests.conftest import SAMPLES
from distriml.common import algorithms, batching, catalog, dataset
from distriml.common.config import load_config
from distriml.common.dataset import DatasetMeta
from distriml.common.ids import sha256_hex
from distriml.db.service import Database

CLIENT_A = "a" * 64
CLIENT_B = "b" * 64
IRIS = (SAMPLES / "iris.csv").read_bytes()
SMALL_HP = {"batch_size": 50, "epochs": 2}          # iris (150 filas) -> 3 batches por época, 6 en total


class FakeClock:
    """Reloj manual en milisegundos: `clock()` devuelve el instante; `advance(ms)` lo mueve."""

    def __init__(self, start: int = 1_000_000):
        self.now = start

    def __call__(self) -> int:
        return self.now

    def advance(self, ms: int) -> None:
        self.now += ms


def make_db(path, clock: FakeClock | None = None, wall: FakeClock | None = None, **cfg) -> Database:
    cfg.setdefault("log_level", "WARNING")
    return Database(load_config(data_dir=path, **cfg), clock=clock, wall_clock=wall)


def model_def_id(algorithm: str) -> str:
    return catalog.BY_ALGORITHM[algorithm].model_def_id


def put_blob(db: Database, data: bytes) -> str:
    sha = sha256_hex(data)
    db.put_artifact(sha, data)
    return sha


def new_dataset(db: Database, client: str = CLIENT_A, rid: str = "ds-1", data: bytes = IRIS, name: str = "iris.csv") -> dict:
    return db.create_dataset(client, rid, name, sha256_hex(data), len(data), data)


def new_job(db: Database, dataset_id: str, client: str = CLIENT_A, rid: str = "job-1",
            algorithm: str = "sgd_logistic", hyperparams: dict | None = None) -> dict:
    return db.create_job(client, rid, dataset_id, model_def_id(algorithm), SMALL_HP if hyperparams is None else hyperparams)


def start_job(db: Database, client: str = CLIENT_A, rid: str = "job-1", worker: str = "w1", **kw):
    """Crea dataset + job y los asigna a `worker`. Devuelve (dataset, job, assignment)."""
    ds = new_dataset(db, client, rid=f"ds-{rid}")
    job = new_job(db, ds["dataset_id"], client, rid, **kw)
    return ds, job, db.request_job(worker)


def train(db: Database, worker: str, a, upto: int | None = None, finish: bool = True) -> str | None:
    """Entrena de verdad desde `a.next_batch` hasta `upto` (por defecto, todos), sube cada checkpoint y, si
    `finish` y se llegó al final, completa el job. Devuelve el sha256 del último estado subido."""
    meta = DatasetMeta.from_dict(a.dataset_meta)
    X, y = dataset.load_for_training(db.get_artifact(a.dataset_sha256), meta, algorithms.task_of(a.algorithm))
    state = (algorithms.deserialize_state(db.get_artifact(a.checkpoint_sha256)) if a.next_batch > 1
             else algorithms.init_state(a.algorithm, a.hyperparams, meta))
    last = a.total_batches if upto is None else upto
    sha = None
    for k in range(a.next_batch, last + 1):
        idx = batching.batch_indices(meta.n_rows, a.hyperparams["batch_size"], a.hyperparams["seed"], k)
        metrics = algorithms.train_batch(state, a.algorithm, X[idx], y[idx])
        sha = put_blob(db, algorithms.serialize_state(state))
        assert db.upload_checkpoint(worker, a.attempt_id, k, sha, metrics) == "CONFIRMED"
    if finish and last == a.total_batches:
        final = algorithms.evaluate_full(state, a.algorithm, X, y)
        db.complete_job(worker, a.attempt_id, sha, final)
    return sha
