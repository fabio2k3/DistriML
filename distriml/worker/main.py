"""Arranque de los workers como actores de Ray (pegamento ya hecho)."""
from __future__ import annotations

from distriml.common.config import Config
from distriml.common.rayproxy import RayDBProxy


def start_workers(cfg: Config, db_handle, n: int) -> list:
    """Crea `n` actores `worker-0 … worker-(n-1)`; cada uno arranca su bucle de trabajo solo.

    Capacidades de Ray usadas aquí (documentarlas en docs/RAY_CAPABILITIES.md):
      - actores con nombre (scripts/chaos.py los localiza para matarlos)
      - max_restarts=-1: un worker muerto se reinicia con los mismos argumentos y vuelve a registrarse
      - max_concurrency=2: el hilo de renovación de lease puede llamar mientras el bucle principal entrena
      - paso del handle de la DB como argumento (a través de RayDBProxy)
    """
    import ray

    from distriml.worker.worker import Worker

    WorkerActor = ray.remote(max_restarts=-1, max_concurrency=2)(Worker)
    db = RayDBProxy(db_handle)
    return [
        WorkerActor.options(name=f"worker-{i}", namespace=cfg.ray_namespace, get_if_exists=True)
        .remote(f"worker-{i}", db, cfg)
        for i in range(n)
    ]
