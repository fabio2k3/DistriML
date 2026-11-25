"""Mediciones de Ray para docs/RAY_CAPABILITIES.md (parte worker).

    python -m scripts.measure_ray                         # caso base (lease corto, 1 worker, 1 cliente)
    python -m scripts.measure_ray --lease-ms 10000 --renew-ms 3333 --epsilon-ms 1000   # configuración real
    python -m scripts.measure_ray --workers 4 --runs 3    # 4 workers muertos a la vez
    python -m scripts.measure_ray --clients 8 --payload-kb 1,64,1024,5120 --runs 0     # carga y binarios grandes

Levanta un Ray local con FakeDB como actor y mide:
  1. latencia de `ray.get` (ping y una operación pequeña), con `--clients` hilos llamando a la vez;
  2. latencia de subir/bajar artefactos de `--payload-kb` KB (el dataset máximo es de 5 MiB = 5120 KB);
  3. reinicio de los workers tras `ray.kill` (todos a la vez): desde la orden hasta que el worker NUEVO se
     registra en la DB (la DB apunta el instante del registro; sondear al actor con ray.get daba una medida
     con resolución de ~2 s, porque una llamada a un actor que reinicia tarda en fallar o en responder);
  4. hueco de cada job al matar a sus workers: desde la muerte hasta el primer checkpoint nuevo confirmado
     por otro attempt. Con `--workers N` hay N jobs en curso y se matan los N workers a la vez.
`--runs 0` omite 3 y 4. Para medir en Docker: copiar el script al contenedor head y ejecutarlo allí.
"""
from __future__ import annotations

import argparse
import os
import statistics as st
import tempfile
import threading
import time

import ray

from distriml.common.config import load_config
from distriml.common.ids import sha256_hex
from distriml.common.rayproxy import RayDBProxy
from distriml.worker.main import start_workers
from tests.helpers.fake_db import FakeDB
from tests.test_worker_wk1_wk2 import CLIENT, seed_job

NS = "distriml-measure"


class MeasuredDB(FakeDB):
    """FakeDB con un método extra para saber cuándo se registró por última vez un worker (ms, reloj de la DB)."""

    def registered_at(self, worker_id):
        return self.workers.get(worker_id)

    def clock_now(self):
        return self._clock()


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(label, ms, unit="ms"):
    if not ms:
        return
    print(f"{label:<52} n={len(ms):<4} media={st.mean(ms):9.2f}  p50={pct(ms, 50):9.2f}  "
          f"p95={pct(ms, 95):9.2f}  max={max(ms):9.2f}  ({unit})")


def bench(fn, calls: int, clients: int):
    """Ejecuta fn() `calls` veces en cada uno de `clients` hilos. Devuelve (latencias en ms, llamadas/s)."""
    lat, lock = [], threading.Lock()

    def worker():
        mine = []
        for _ in range(calls):
            t = time.perf_counter()
            fn()
            mine.append((time.perf_counter() - t) * 1000)
        with lock:
            lat.extend(mine)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return lat, len(lat) / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="repeticiones de reinicio y de hueco (0 = omitir)")
    ap.add_argument("--calls", type=int, default=300, help="llamadas por hilo en las medidas de latencia")
    ap.add_argument("--lease-ms", type=int, default=2000)
    ap.add_argument("--renew-ms", type=int, default=500)
    ap.add_argument("--epsilon-ms", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1, help="workers matados a la vez")
    ap.add_argument("--clients", type=int, default=1, help="hilos que llaman a la DB a la vez")
    ap.add_argument("--payload-kb", default="", help="tamaños de artefacto a medir, en KB, separados por comas")
    ap.add_argument("--epochs", type=int, default=200, help="épocas de los jobs (más = jobs más largos)")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp()
    cfg = load_config(data_dir=tmp, ray_namespace=NS, lease_duration_ms=args.lease_ms,
                      renew_interval_ms=args.renew_ms, epsilon_ms=args.epsilon_ms,
                      request_poll_interval_ms=50)
    ray.init(namespace=NS, include_dashboard=False, logging_level="ERROR",
             num_cpus=max(4, args.workers + 2), log_to_driver=False)
    handle = ray.remote(MeasuredDB).options(name="fake-db-m", namespace=NS).remote(cfg, tmp + "/db")
    db = RayDBProxy(handle)

    print(f"Ray {ray.__version__}, 1 nodo local, sin Docker | lease={args.lease_ms} ms, renovar={args.renew_ms} ms, "
          f"epsilon={args.epsilon_ms} ms, workers={args.workers}, clientes={args.clients}\n")

    # 1. latencia de ray.get (con carga si --clients > 1)
    ray.get(handle.ping.remote())
    lat, rate = bench(lambda: ray.get(handle.ping.remote()), args.calls, args.clients)
    summary(f"ray.get(ping)  [{args.clients} cliente(s)]", lat)
    print(f"{'':<52} rendimiento ≈ {rate:,.0f} llamadas/s")
    lat, rate = bench(db.list_model_definitions, args.calls, args.clients)
    summary(f"RayDBProxy.list_model_definitions()  [{args.clients} cliente(s)]", lat)

    # 2. artefactos de distinto tamaño
    for kb in [int(x) for x in args.payload_kb.split(",") if x.strip()]:
        data = os.urandom(kb * 1024)
        sha = sha256_hex(data)
        n = max(5, args.calls // 10)
        put, _ = bench(lambda: db.put_artifact(sha, data), n, args.clients)
        get, _ = bench(lambda: db.get_artifact(sha), n, args.clients)
        summary(f"put_artifact {kb} KB  [{args.clients} cliente(s)]", put)
        summary(f"get_artifact {kb} KB  [{args.clients} cliente(s)]", get)

    if args.runs > 0:
        workers = start_workers(cfg, handle, args.workers)
        N = len(workers)

        names = [f"worker-{i}" for i in range(N)]
        ray.get([handle.registered_at.remote(n) for n in names])
        restarts = []
        for _ in range(args.runs):
            old = {n: ray.get(handle.registered_at.remote(n)) for n in names}
            for w in workers:
                ray.kill(w, no_restart=False)
            killed_at = ray.get(handle.clock_now.remote())          # mismo reloj que usa la DB para anotar el registro
            pending = set(names)
            while pending:
                for n in list(pending):
                    reg = ray.get(handle.registered_at.remote(n))
                    if reg is not None and reg != old[n] and reg >= killed_at:
                        restarts.append(float(reg - killed_at))
                        pending.discard(n)
                time.sleep(0.02)
        summary(f"reinicio de un worker tras ray.kill  [{N} a la vez]", restarts)

        # 4. hueco de cada job al matar a sus workers
        gaps, all_done = [], []
        for r in range(args.runs):
            ids = [seed_job(db, f"gap{r}-{i}", {"epochs": args.epochs}) for i in range(N)]
            while any(db.get_job(CLIENT, j)["last_confirmed_batch"] < 15 for j in ids):
                time.sleep(0.02)
            at = {j: db.get_job(CLIENT, j)["last_confirmed_batch"] for j in ids}
            for w in workers:
                ray.kill(w, no_restart=False)
            t0 = time.perf_counter()
            pending = set(ids)
            while pending:
                for j in list(pending):
                    v = db.get_job(CLIENT, j)
                    if (len(v["attempts"]) >= 2 and v["last_confirmed_batch"] > at[j] + 1
                            and v["attempt_vigente"] != v["attempts"][0]["attempt_id"]):
                        gaps.append((time.perf_counter() - t0) * 1000)
                        pending.discard(j)
                time.sleep(0.02)
            all_done.append(max(gaps[-N:]))
            for i, j in enumerate(ids):
                db.cancel_job(CLIENT, f"c{r}-{i}", j)
            time.sleep(0.5 + 0.1 * N)
        summary(f"hueco de un job al matar su worker  [{N} a la vez]", gaps)
        if N > 1:
            summary(f"hueco hasta que continuaron TODOS los jobs", all_done)
        print(f"\nlease + epsilon = {args.lease_ms + args.epsilon_ms} ms: cota inferior del hueco "
              "(la DB no reasigna antes de que venza el lease).")
        for w in workers:
            ray.kill(w, no_restart=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
