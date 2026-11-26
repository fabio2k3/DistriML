"""Worker de referencia: camino feliz, ~40 líneas, SIN leases, abortos ni manejo de errores.

Sirve para (1) que el gateway y los tests de contrato tengan quien entrene mientras la BD y el
worker reales se desarrollan y (2) como guía de qué llamadas hace un worker y en qué orden.
El worker real (trainer.py) añade lo difícil: lease, abortos, errores, reintentos de descarga.
"""
from __future__ import annotations

import threading
import time

from distriml.common import algorithms, batching, constants as C, dataset
from distriml.common.dataset import DatasetMeta
from distriml.common.ids import sha256_hex


def run_one(db, worker_id: str) -> bool:
    """Pide un job y lo entrena entero. Devuelve False si no había trabajo."""
    a = db.request_job(worker_id)
    if a is None:
        return False
    meta = DatasetMeta.from_dict(a.dataset_meta)
    task = algorithms.task_of(a.algorithm)
    data = db.get_artifact(a.dataset_sha256)
    X, y = dataset.load_for_training(data, meta, task)
    if a.next_batch > 1:
        state = algorithms.deserialize_state(db.get_artifact(a.checkpoint_sha256))
    else:
        state = algorithms.init_state(a.algorithm, a.hyperparams, meta, dataset.class_names_of(data))
    hp = a.hyperparams
    for k in range(a.next_batch, a.total_batches + 1):
        idx = batching.batch_indices(meta.n_rows, hp["batch_size"], hp["seed"], k)
        metrics = algorithms.train_batch(state, a.algorithm, X[idx], y[idx])
        blob = algorithms.serialize_state(state)
        sha = sha256_hex(blob)
        db.put_artifact(sha, blob)
        if db.upload_checkpoint(worker_id, a.attempt_id, k, sha, metrics) != C.CKPT_CONFIRMED:
            return True          # revocado: se abandona
    final = algorithms.evaluate_full(state, a.algorithm, X, y)
    blob = algorithms.serialize_state(state)
    sha = sha256_hex(blob)
    db.put_artifact(sha, blob)
    db.complete_job(worker_id, a.attempt_id, sha, final)
    return True


class ReferenceWorkerThread(threading.Thread):
    def __init__(self, db, worker_id: str = "ref-0", poll: float = 0.05):
        super().__init__(daemon=True, name=worker_id)
        self.db, self.worker_id, self.poll = db, worker_id, poll
        self._stop_evt = threading.Event()

    def run(self) -> None:
        self.db.register_worker(self.worker_id)
        while not self._stop_evt.is_set():
            try:
                if not run_one(self.db, self.worker_id):
                    time.sleep(self.poll)
            except Exception:    # el worker de referencia no debe morir en tests
                time.sleep(self.poll)

    def stop(self) -> None:
        self._stop_evt.set()
