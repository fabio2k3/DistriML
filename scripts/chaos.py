"""Inyección de fallos sobre el cluster Ray local (para probar la tolerancia a fallos).

    python scripts/chaos.py list [N]               # N = nº máximo de workers a buscar (por defecto 8)
    python scripts/chaos.py kill-worker 0          # mata worker-0 (Ray lo reinicia: max_restarts=-1)
    python scripts/chaos.py kill-worker 0 --no-restart
    python scripts/chaos.py kill-db                # mata el actor de la DB (se reinicia y recarga los CSV)

Requiere que `python -m distriml.cluster` esté corriendo en esta misma máquina.
"""
from __future__ import annotations

import argparse
import logging

import ray

from distriml.common.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["list", "kill-worker", "kill-db"])
    p.add_argument("index", nargs="?", type=int)
    p.add_argument("--no-restart", action="store_true", help="no dejar que Ray reinicie el actor")
    args = p.parse_args()

    cfg = load_config()
    ray.init(address="auto", namespace=cfg.ray_namespace, logging_level=logging.ERROR)

    if args.cmd == "list":      # sin dashboard: se comprueba qué nombres conocidos responden
        names = [cfg.db_actor_name] + [f"worker-{i}" for i in range(args.index if args.index is not None else 8)]
        for name in names:
            try:
                ray.get_actor(name, namespace=cfg.ray_namespace)
                print(name, "ALIVE")
            except ValueError:
                pass
        return
    if args.cmd == "kill-worker":
        if args.index is None:
            p.error("indica el índice del worker")
        actor = ray.get_actor(f"worker-{args.index}", namespace=cfg.ray_namespace)
    else:
        actor = ray.get_actor(cfg.db_actor_name, namespace=cfg.ray_namespace)
    ray.kill(actor, no_restart=args.no_restart)
    print("actor eliminado" + (" (sin reinicio)" if args.no_restart else " (Ray lo reiniciará)"))


if __name__ == "__main__":
    main()
