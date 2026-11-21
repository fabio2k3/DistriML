"""Proceso de un nodo worker: se une al cluster y mantiene vivo el actor worker de este nodo.

    ray start --address=head:6379 && exec python -m distriml.worker.node --ray-address=auto

Es el proceso principal del contenedor worker. Crecer es levantar otro contenedor; decrecer es detenerlo
(`docker stop` envía SIGTERM, que retira el actor ordenadamente). Si el contenedor se mata de golpe,
la DB reasigna el job en curso al vencer su lease y continúa desde el último checkpoint confirmado.

El id del worker es `worker-<hostname>`: único por contenedor.
"""
from __future__ import annotations

import argparse
import logging
import signal
import socket
import threading
import time

from distriml.common.config import Config, load_config
from distriml.common.log import get_logger, setup_logging


def wait_for_db(cfg: Config, timeout_s: float = 120.0):
    """Espera a que exista el actor de la DB (el head puede tardar más en arrancar que este nodo)."""
    import ray

    log = get_logger("node")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            handle = ray.get_actor(cfg.db_actor_name, namespace=cfg.ray_namespace)
            ray.get(handle.ping.remote(), timeout=10)
            return handle
        except Exception as exc:  # noqa: BLE001  (el actor aún no existe o no responde)
            if time.monotonic() > deadline:
                raise TimeoutError(f"la DB no apareció en {timeout_s:.0f}s") from exc
            log.info("esperando a la DB (%s)", type(exc).__name__)
            time.sleep(2)


def run(cfg: Config, worker_id: str, stop: threading.Event | None = None, ray_address: str | None = "auto") -> None:
    import ray

    from distriml.worker.main import start_worker_on_this_node

    stop = stop or threading.Event()
    log = get_logger("node")
    ray.init(address=ray_address, namespace=cfg.ray_namespace, ignore_reinit_error=True, logging_level=logging.WARNING)
    db = wait_for_db(cfg)
    actor = start_worker_on_this_node(cfg, db, worker_id)
    log.info("worker %s en marcha en este nodo", worker_id)
    failures = 0
    try:
        while not stop.wait(5):
            try:
                ray.get(actor.status.remote(), timeout=30)
                failures = 0
            except Exception as exc:  # noqa: BLE001  (el actor puede estar reiniciándose)
                failures += 1
                log.warning("el actor %s no responde (%s), intento %d/6", worker_id, type(exc).__name__, failures)
                if failures >= 6:                              # ~30 s sin respuesta: el contenedor termina y se hace visible
                    raise
    finally:
        log.info("retirando worker %s", worker_id)
        try:
            ray.get(actor.stop.remote(), timeout=10)
        except Exception:  # noqa: BLE001  (se retira igualmente)
            pass
        ray.kill(actor, no_restart=True)
        log.info("worker %s retirado", worker_id)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="distriml-worker-node")
    p.add_argument("--ray-address", default="auto")
    p.add_argument("--worker-id", default=None, help="por defecto worker-<hostname>")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    cfg = load_config(log_level=args.log_level)
    setup_logging(cfg.log_level)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    run(cfg, args.worker_id or f"worker-{socket.gethostname()}", stop, args.ray_address)


if __name__ == "__main__":
    main()
