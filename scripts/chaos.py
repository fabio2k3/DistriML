"""Inyección de fallos sobre el cluster Ray (para probar la tolerancia a fallos).

    python scripts/chaos.py list                          # actores con nombre vivos (DB y workers)
    python scripts/chaos.py kill-worker 0                 # mata worker-0 (Ray lo reinicia: max_restarts=-1)
    python scripts/chaos.py kill-worker worker-ab12cd     # por nombre completo (workers de contenedores)
    python scripts/chaos.py kill-worker 0 --no-restart
    python scripts/chaos.py kill-db                       # mata el actor de la DB (se reinicia y recarga los CSV)

Requiere ejecutarse en una máquina del cluster Ray (p. ej. `docker compose exec head python scripts/chaos.py ...`).
Para matar un NODO entero (contenedor) se usa `docker kill`, no este script.
"""
from __future__ import annotations

import argparse
import logging

import ray
from ray.util import list_named_actors

from distriml.common.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["list", "kill-worker", "kill-db"])
    p.add_argument("worker", nargs="?", help="índice (0 -> worker-0) o nombre completo del worker")
    p.add_argument("--no-restart", action="store_true", help="no dejar que Ray reinicie el actor")
    args = p.parse_args()

    cfg = load_config()
    ray.init(address="auto", namespace=cfg.ray_namespace, logging_level=logging.ERROR)

    if args.cmd == "list":
        for name in sorted(list_named_actors()):
            print(name, "ALIVE")
        return
    if args.cmd == "kill-worker":
        if args.worker is None:
            p.error("indica el índice o el nombre del worker")
        name = f"worker-{args.worker}" if args.worker.isdigit() else args.worker
        actor = ray.get_actor(name, namespace=cfg.ray_namespace)
    else:
        actor = ray.get_actor(cfg.db_actor_name, namespace=cfg.ray_namespace)
    ray.kill(actor, no_restart=args.no_restart)
    print("actor eliminado" + (" (sin reinicio)" if args.no_restart else " (Ray lo reiniciará)"))


if __name__ == "__main__":
    main()
