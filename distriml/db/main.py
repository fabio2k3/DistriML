"""Arranque de la DB como actor de Ray (pegamento ya hecho; no suele necesitar cambios)."""
from __future__ import annotations

from distriml.common.config import Config


def start_db(cfg: Config):
    """Crea (o recupera) el actor nombrado de la DB y espera a que responda.

    Capacidades de Ray usadas aquí (documentarlas en docs/RAY_CAPABILITIES.md):
      - actor con nombre en un namespace: descubrimiento de la DB sin DNS ni semillas
      - max_restarts=-1: si el proceso muere, Ray lo reinicia y el constructor recarga los CSV
      - max_concurrency=1 (por defecto): las llamadas se ejecutan de una en una
    """
    import ray

    from distriml.db.service import Database

    DBActor = ray.remote(max_restarts=-1)(Database)
    handle = DBActor.options(name=cfg.db_actor_name, namespace=cfg.ray_namespace,
                             get_if_exists=True).remote(cfg)
    ray.get(handle.ping.remote(), timeout=60)
    return handle
