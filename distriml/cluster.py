"""Lanzador: arranca Ray, la DB, los workers y el gateway HTTP con un solo comando.

    python -m distriml.cluster --workers 3 --data-dir ./runtime

Deja el proceso en primer plano sirviendo la API.

Workers: `--workers N` crea N actores en este mismo nodo (desarrollo en una máquina). En un cluster con
contenedores se usa `--workers 0`: cada contenedor worker aporta su propio actor (ver worker/node.py). Para inyectar fallos desde otra terminal
(matar un worker o la DB) usa scripts/chaos.py: se conecta al mismo cluster Ray local.

Todas las operaciones se escriben en consola. Opciones de registro:
    --log-level INFO    operaciones relevantes (por defecto)
    --log-level DEBUG   además heartbeats, polling, artefactos y llamadas internas a la DB
"""
from __future__ import annotations

import argparse
import logging
import os

from distriml.common.config import load_config
from distriml.common.log import get_logger, setup_logging
from distriml.common.rayproxy import RayDBProxy


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="distriml-cluster")
    p.add_argument("--workers", type=int, default=2, help="actores worker locales; 0 si los workers son nodos aparte")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--ray-address", default=None, help="dirección de un cluster Ray existente (por defecto, uno local nuevo)")
    args = p.parse_args(argv)

    # Ray agrupa en una sola línea los mensajes repetidos de los workers ("[repeated 3x]"); para ver
    # cada operación hay que desactivarlo antes de iniciar Ray.
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")

    import ray
    import uvicorn

    from distriml.api.app import create_app
    from distriml.db.main import start_db
    from distriml.worker.main import start_workers

    cfg = load_config(data_dir=args.data_dir, api_host=args.host, api_port=args.port, log_level=args.log_level)
    setup_logging(cfg.log_level)
    log = get_logger("cluster")

    # Si se crea un Ray local, se declara el recurso `db` que usa la DB; un cluster ya existente
    # debe declararlo en su nodo head (`ray start --head --resources='{"db": 1}'`).
    local = {"resources": {"db": 1}} if args.ray_address is None else {}
    ray.init(address=args.ray_address, namespace=cfg.ray_namespace, include_dashboard=False,
             ignore_reinit_error=True, logging_level=logging.WARNING, **local)
    log.info("Ray iniciado (namespace=%s)", cfg.ray_namespace)
    db = start_db(cfg)
    log.info("DB lista (actor '%s')", cfg.db_actor_name)
    workers = start_workers(cfg, db, args.workers)   # conservar los handles: si se sueltan, Ray mata los actores
    log.info("%d workers locales iniciados", len(workers))
    app = create_app(RayDBProxy(db))
    log.info("API escuchando en http://%s:%s", cfg.api_host, cfg.api_port)
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port, log_level="warning")   # el gateway ya registra cada petición


if __name__ == "__main__":
    main()
