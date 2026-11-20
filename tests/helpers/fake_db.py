"""FakeDB: implementación EN MEMORIA y COMPLETA del contrato `DBService` (common/interfaces.py).

Para qué sirve:
  - Probar el gateway, la consola y los workers sin Ray ni la BD real.
  - Ser la REFERENCIA DE COMPORTAMIENTO para quien implementa db/service.py: ante una duda sobre
    qué debe devolver un método, mira aquí (y en tests/test_contract_live.py).
No es la entrega: la BD real persiste en CSV y sobrevive a reinicios; esta no.
Simplificaciones: sin retardo de reasignación y sin persistencia.
Registra cada operación en el logger `distriml.fakedb` (visible con `pytest -o log_cli=true --log-cli-level=DEBUG`).
"""
from __future__ import annotations

import tempfile
import threading
import time
from collections import defaultdict

from distriml.common import algorithms, catalog, constants as C, dataset, ids
from distriml.common.artifacts import ArtifactStore
from distriml.common.config import Config, load_config
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import bad_request, conflict, not_found, unavailable
from distriml.common.interfaces import Assignment, WorkerConfig
from distriml.common.log import traced


@traced("fakedb")
class FakeDB:
    def __init__(self, cfg: Config | None = None, data_dir: str | None = None, clock=None):
        self.cfg = cfg or load_config()
        self.store = ArtifactStore(data_dir or tempfile.mkdtemp(prefix="fakedb-"))
        self._clock = clock or (lambda: int(time.monotonic() * 1000))
        self._lock = threading.RLock()
        self._n = 0
        self.datasets: dict[str, dict] = {}
        self.jobs: dict[str, dict] = {}
        self.attempts: dict[str, dict] = {}
        self.batches: dict[str, dict[int, dict]] = defaultdict(dict)   # job_id -> batch_idx -> fila
        self.models: dict[str, dict] = {}
        self.versions: dict[str, list[dict]] = defaultdict(list)
        self.stats: dict[tuple, list] = {}                              # (model, version, minute) -> [count, latency_sum]
        self.workers: dict[str, int] = {}
        self._cancels: set[str] = set()

    # ------------------------------------------------------------------ utilidades
    def _entry(self) -> str:
        self._n += 1
        return f"fake:1:{self._n}"

    def _own(self, table: dict, client_id: str, key: str) -> dict:
        row = table.get(key)
        if row is None or row["client_id"] != client_id:
            raise not_found()
        return row

    def _page(self, rows: list[dict], key: str, limit: int, cursor: str | None) -> list[dict]:
        rows = sorted(rows, key=lambda r: (r["created_at"], r[key]))
        if cursor:
            ids_ = [r[key] for r in rows]
            rows = rows[ids_.index(cursor) + 1:] if cursor in ids_ else []
        return rows[:limit]

    def _revoke(self, attempt: dict, reason: str) -> None:
        """Revoca el attempt y decide el estado del job."""
        if attempt["estado"] != C.ATTEMPT_ACTIVO:
            return
        attempt["estado"], attempt["revoke_reason"] = C.ATTEMPT_REVOCADO, reason
        job = self.jobs[attempt["job_id"]]
        if job["attempt_vigente"] == attempt["attempt_id"]:
            job["attempt_vigente"] = None
        if reason == C.CANCEL:
            job["estado"] = C.CANCELADO
        elif reason == C.WORKER_ERROR:
            job["attempts_count"] += 1
            job["estado"] = C.FALLIDO if job["attempts_count"] >= self.cfg.max_attempts else C.PENDIENTE
        elif reason == C.WORKER_LOST:
            job["lost_count"] += 1
            job["estado"] = C.FALLIDO if job["lost_count"] >= self.cfg.max_lost else C.PENDIENTE
        else:                                     # NEVER_STARTED, ARTIFACT_UNAVAILABLE: no cuentan
            job["estado"] = C.PENDIENTE

    def _reap(self) -> None:
        now = self._clock()
        for a in list(self.attempts.values()):
            if a["estado"] == C.ATTEMPT_ACTIVO and now > a["last_renew"] + self.cfg.lease_duration_ms + self.cfg.epsilon_ms:
                self._revoke(a, C.WORKER_LOST if a["started"] else C.NEVER_STARTED)

    def _vigente(self, attempt_id: str) -> dict | None:
        a = self.attempts.get(attempt_id)
        if a is None or a["estado"] != C.ATTEMPT_ACTIVO:
            return None
        job = self.jobs[a["job_id"]]
        return a if job["attempt_vigente"] == attempt_id else None

    # ------------------------------------------------------------------ común / artefactos
    def ping(self) -> dict:
        return {"status": "ok"}

    def list_model_definitions(self) -> list[dict]:
        return catalog.list_definitions()

    def put_artifact(self, sha256: str, data: bytes) -> None:
        self.store.put(sha256, data)

    def get_artifact(self, sha256: str) -> bytes:
        return self.store.get(sha256)

    # ------------------------------------------------------------------ datasets
    def create_dataset(self, client_id, client_request_id, name, sha256, size, data) -> dict:
        with self._lock:
            did = ids.derive_dataset_id(client_id, client_request_id)
            if ids.sha256_hex(data) != sha256 or len(data) != size:
                raise bad_request("hash_mismatch", "sha256/size no coinciden con el contenido")
            row = self.datasets.get(did)
            if row is None:
                meta = dataset.analyze(data, self.cfg.max_dataset_bytes)
                self.store.put(sha256, data)
                row = {"dataset_id": did, "client_id": client_id, "name": name, "sha256": sha256, "size": size,
                       "meta": meta, "created_at": self._clock(), "entry_id": self._entry()}
                self.datasets[did] = row
            elif row["sha256"] != sha256:
                raise conflict("request_id_reused", "client_request_id ya se usó con otro contenido")
            m = row["meta"]
            return {"dataset_id": did, "n_rows": m.n_rows, "n_features": m.n_features,
                    "classes": m.classes, "entry_id": row["entry_id"]}

    def list_datasets(self, client_id, limit=100, cursor=None) -> list[dict]:
        with self._lock:
            mine = [r for r in self.datasets.values() if r["client_id"] == client_id]
            return [{"dataset_id": r["dataset_id"], "name": r["name"], "sha256": r["sha256"], "size": r["size"]}
                    for r in self._page(mine, "dataset_id", limit, cursor)]

    def get_dataset(self, client_id, dataset_id) -> dict:
        with self._lock:
            r = self._own(self.datasets, client_id, dataset_id)
            return {"dataset_id": r["dataset_id"], "name": r["name"], "sha256": r["sha256"],
                    "size": r["size"], "replicas": 1}

    # ------------------------------------------------------------------ jobs
    def create_job(self, client_id, client_request_id, dataset_id, model_def_id, hyperparams=None) -> dict:
        with self._lock:
            jid = ids.derive_job_id(client_id, client_request_id)
            md = catalog.get(model_def_id)
            hp = catalog.merge_hyperparams(md, hyperparams)
            existing = self.jobs.get(jid)
            if existing is not None:
                if (existing["dataset_id"], existing["model_def_id"], existing["hyperparams"]) != (dataset_id, model_def_id, hp):
                    raise conflict("request_id_reused", "client_request_id ya se usó con otros parámetros")
                return {"job_id": jid, "entry_id": existing["entry_id"]}
            ds = self.datasets.get(dataset_id)
            if ds is None or ds["client_id"] != client_id:
                raise bad_request("dataset_not_found", "el dataset no existe o no es tuyo")
            catalog.check_compatibility(md, ds["meta"])
            if not self.store.has(ds["sha256"]):
                raise unavailable("dataset_unavailable", "el dataset aún no está disponible")
            self.jobs[jid] = {
                "job_id": jid, "client_id": client_id, "dataset_id": dataset_id, "model_def_id": model_def_id,
                "hyperparams": hp, "total_batches": catalog.compute_total_batches(hp, ds["meta"].n_rows),
                "estado": C.PENDIENTE, "attempts_count": 0, "lost_count": 0, "last_confirmed_batch": 0,
                "attempt_vigente": None, "model_id": None, "canonical_version_id": None, "model_version": None,
                "created_at": self._clock(), "entry_id": self._entry(),
            }
            return {"job_id": jid, "entry_id": self.jobs[jid]["entry_id"]}

    def _job_view(self, job: dict) -> dict:
        view = {k: v for k, v in job.items() if k != "client_id"}
        view["attempts"] = [
            {"attempt_id": a["attempt_id"], "worker_id": a["worker_id"], "estado": a["estado"],
             "revoke_reason": a["revoke_reason"], "base_batch": a["base_batch"], "created_at": a["created_at"]}
            for a in self.attempts.values() if a["job_id"] == job["job_id"]
        ]
        return view

    def get_job(self, client_id, job_id) -> dict:
        with self._lock:
            self._reap()
            return self._job_view(self._own(self.jobs, client_id, job_id))

    def list_jobs(self, client_id, estado=None, limit=100, cursor=None) -> list[dict]:
        with self._lock:
            self._reap()
            mine = [j for j in self.jobs.values() if j["client_id"] == client_id and (estado is None or j["estado"] == estado)]
            return [{k: j[k] for k in ("job_id", "dataset_id", "model_def_id", "estado", "last_confirmed_batch", "total_batches")}
                    for j in self._page(mine, "job_id", limit, cursor)]

    def cancel_job(self, client_id, client_request_id, job_id) -> dict:
        with self._lock:
            job = self._own(self.jobs, client_id, job_id)
            cid = ids.derive_cancel_id(client_id, client_request_id)
            if cid in self._cancels and job["estado"] == C.CANCELADO:
                return {"estado": C.CANCELADO, "entry_id": job["cancel_entry"]}
            if job["estado"] in C.TERMINAL_STATES:
                raise conflict("job_terminal", f"el job ya está {job['estado']}")
            self._reap()
            if job["attempt_vigente"]:
                self._revoke(self.attempts[job["attempt_vigente"]], C.CANCEL)
            job["estado"] = C.CANCELADO
            job["cancel_entry"] = self._entry()
            self._cancels.add(cid)
            return {"estado": C.CANCELADO, "entry_id": job["cancel_entry"]}

    def get_job_metrics(self, client_id, job_id) -> dict:
        with self._lock:
            job = self._own(self.jobs, client_id, job_id)
            rows = [{"batch_idx": k, "metrics": b["metrics"]} for k, b in sorted(self.batches[job_id].items())
                    if b["estado"] == C.BATCH_CONFIRMED]
            final = None
            if job["estado"] == C.COMPLETADO:
                final = self.versions[job["model_id"]][0]["metrics"]
            return {"job_id": job_id, "estado": job["estado"], "batches": rows, "final": final}

    # ------------------------------------------------------------------ modelos / inferencia
    def _version_view(self, m: dict, v: dict) -> dict:
        return {"model_id": m["model_id"], "version_id": v["version_id"], "version": v["version"],
                "job_id": m["job_id"], "model_def_id": m["model_def_id"], "estado": v["estado"],
                "canonical": v["canonical"], "sha256": v["sha256"], "metrics": v["metrics"]}

    def list_model_versions(self, client_id, model_id) -> list[dict]:
        with self._lock:
            m = self._own(self.models, client_id, model_id)
            return [self._version_view(m, v) for v in self.versions[model_id]]

    def get_model_version(self, client_id, model_id, version_id) -> dict:
        with self._lock:
            m = self._own(self.models, client_id, model_id)
            for v in self.versions[model_id]:
                if v["version_id"] == version_id:
                    return self._version_view(m, v)
            raise not_found()

    def _pick_version(self, model_id, version_id, version) -> dict:
        vs = self.versions[model_id]
        if version_id is not None:
            found = [v for v in vs if v["version_id"] == version_id]
        elif version is not None:
            found = [v for v in vs if v["canonical"] and v["version"] == version]
        else:
            found = [v for v in vs if v["canonical"]]
        if not found:
            raise not_found()
        return found[0]

    def predict(self, client_id, model_id, input, version_id=None, version=None) -> dict:
        t0 = time.perf_counter()
        with self._lock:
            m = self._own(self.models, client_id, model_id)
            v = self._pick_version(model_id, version_id, version)
        feats = (input or {}).get("features") if isinstance(input, dict) else None
        if feats is None:
            raise bad_request("invalid_input", "input.features es obligatorio")
        state = algorithms.deserialize_state(self.store.get(v["sha256"]))
        algo = catalog.get(m["model_def_id"]).algorithm
        pred = algorithms.predict(state, algo, feats)
        ms = (time.perf_counter() - t0) * 1000
        with self._lock:
            key = (model_id, v["version_id"], ids.now_ms() // 60000)
            c = self.stats.setdefault(key, [0, 0.0])
            c[0] += 1
            c[1] += ms
        return {"prediction": pred, "version_id": v["version_id"], "sha256": v["sha256"]}

    def get_model_stats(self, client_id, model_id, version_id=None, from_minute=None, to_minute=None) -> dict:
        with self._lock:
            self._own(self.models, client_id, model_id)
            v = self._pick_version(model_id, version_id, None)
            rows = [(k[2], c) for k, c in sorted(self.stats.items())
                    if k[0] == model_id and k[1] == v["version_id"]
                    and (from_minute is None or k[2] >= from_minute) and (to_minute is None or k[2] <= to_minute)]
            total = sum(c[0] for _, c in rows)
            lat = sum(c[1] for _, c in rows)
            return {"model_id": model_id, "version_id": v["version_id"], "total_calls": total,
                    "mean_latency_ms": (lat / total) if total else 0.0,
                    "por_minuto": [{"minute": mnt, "count": c[0], "mean_latency_ms": c[1] / c[0]} for mnt, c in rows]}

    # ------------------------------------------------------------------ worker
    def register_worker(self, worker_id) -> WorkerConfig:
        with self._lock:
            self.workers[worker_id] = self._clock()
            return WorkerConfig(worker_id, self.cfg.renew_interval_ms, self.cfg.lease_duration_ms)

    def request_job(self, worker_id) -> Assignment | None:
        with self._lock:
            self._reap()
            pending = sorted((j for j in self.jobs.values() if j["estado"] == C.PENDIENTE),
                             key=lambda j: (j["created_at"], j["job_id"]))
            for job in pending:
                ds = self.datasets[job["dataset_id"]]
                if not self.store.has(ds["sha256"]):
                    continue
                last = job["last_confirmed_batch"]
                ckpt = self.batches[job["job_id"]][last]["sha256"] if last else None
                aid = ids.new_attempt_id()
                now = self._clock()
                self.attempts[aid] = {"attempt_id": aid, "job_id": job["job_id"], "worker_id": worker_id,
                                      "base_batch": last, "estado": C.ATTEMPT_ACTIVO, "revoke_reason": None,
                                      "created_at": ids.now_ms(), "last_renew": now, "started": False}
                job["attempt_vigente"], job["estado"] = aid, C.EN_EJECUCION
                md = catalog.get(job["model_def_id"])
                return Assignment(job["job_id"], aid, last + 1, job["total_batches"], ds["sha256"],
                                  ds["meta"].to_dict(), md.algorithm, dict(job["hyperparams"]),
                                  self.cfg.lease_duration_ms, ckpt)
            return None

    def renew_lease(self, worker_id, attempt_id) -> str:
        with self._lock:
            self._reap()
            a = self._vigente(attempt_id)
            if a is None:
                return C.RENEW_REVOKED
            a["last_renew"], a["started"] = self._clock(), True
            return C.RENEW_CONTINUE

    def upload_checkpoint(self, worker_id, attempt_id, batch_idx, sha256, metrics) -> str:
        with self._lock:
            self._reap()
            a = self._vigente(attempt_id)
            if a is None:
                return C.CKPT_REJECTED
            job = self.jobs[a["job_id"]]
            existing = self.batches[job["job_id"]].get(batch_idx)
            if existing is not None:
                return C.CKPT_CONFIRMED if existing["attempt_id"] == attempt_id else C.CKPT_REJECTED
            if batch_idx != job["last_confirmed_batch"] + 1 or batch_idx > job["total_batches"]:
                return C.CKPT_REJECTED
            if not self.store.has(sha256):
                return C.CKPT_RETRY
            self.batches[job["job_id"]][batch_idx] = {"attempt_id": attempt_id, "sha256": sha256,
                                                      "metrics": metrics, "estado": C.BATCH_CONFIRMED}
            job["last_confirmed_batch"] = batch_idx
            a["last_renew"], a["started"] = self._clock(), True
            return C.CKPT_CONFIRMED

    def complete_job(self, worker_id, attempt_id, model_sha256, metrics) -> str:
        with self._lock:
            self._reap()
            a = self._vigente(attempt_id)
            if a is None:
                return C.COMPLETE_REJECTED
            job = self.jobs[a["job_id"]]
            if job["last_confirmed_batch"] != job["total_batches"]:
                return C.COMPLETE_REJECTED
            if not self.store.has(model_sha256):
                raise unavailable("artifact_unavailable", "el modelo final no está en el almacén")
            mid = ids.derive_model_id(job["job_id"])
            self.models[mid] = {"model_id": mid, "job_id": job["job_id"], "client_id": job["client_id"],
                                "model_def_id": job["model_def_id"]}
            self.versions[mid] = [{"version_id": attempt_id, "version": 1, "sha256": model_sha256,
                                   "metrics": metrics, "estado": C.VERSION_ENTRENADO, "canonical": True}]
            job.update(estado=C.COMPLETADO, model_id=mid, canonical_version_id=attempt_id, model_version=1,
                       attempt_vigente=None)
            a["estado"] = C.ATTEMPT_REVOCADO
            a["revoke_reason"] = None
            return C.COMPLETE_OK

    def report_failure(self, worker_id, attempt_id, error) -> None:
        with self._lock:
            a = self._vigente(attempt_id)
            if a is not None:
                self._revoke(a, C.WORKER_ERROR)

    def report_unavailable(self, worker_id, attempt_id, sha256) -> None:
        with self._lock:
            a = self._vigente(attempt_id)
            if a is not None:
                self._revoke(a, C.ARTIFACT_UNAVAILABLE)
