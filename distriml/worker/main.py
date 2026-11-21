"""Arranque de workers como actores de Ray (pegamento ya hecho).

Hay dos formas de tener workers:
  - `start_workers`: N actores en el nodo donde se llama (desarrollo en una sola máquina).
  - `start_worker_on_this_node`: un actor fijado al nodo actual. Lo usa `worker/node.py`, que corre
    como proceso principal de cada contenedor worker: añadir un contenedor añade un worker y
    detenerlo lo retira.
"""
from __future__ import annotations

from distriml.common.config import Config
from distriml.common.rayproxy import RayDBProxy


def _worker_actor_class():
    import ray

    from distriml.worker.worker import Worker

    # max_restarts=-1: un worker muerto se reinicia con los mismos argumentos y vuelve a registrarse.
    # max_concurrency=2: el hilo de renovación de lease puede llamar mientras el bucle principal entrena.
    return ray.remote(max_restarts=-1, max_concurrency=2)(Worker)


def start_workers(cfg: Config, db_handle, n: int) -> list:
    """Crea `n` actores `worker-0 … worker-(n-1)`; cada uno arranca su bucle de trabajo solo.

    Conservar los handles devueltos: si se sueltan, Ray mata los actores.
    """
    WorkerActor = _worker_actor_class()
    db = RayDBProxy(db_handle)
    return [
        WorkerActor.options(name=f"worker-{i}", namespace=cfg.ray_namespace, get_if_exists=True)
        .remote(f"worker-{i}", db, cfg)
        for i in range(n)
    ]


def start_worker_on_this_node(cfg: Config, db_handle, worker_id: str):
    """Crea un actor worker fijado al nodo desde el que se llama (afinidad estricta).

    Si ese nodo cae, el actor no se reinicia en otro: el job se reasigna cuando vence su lease.
    El actor vive mientras viva el proceso que lo crea.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    WorkerActor = _worker_actor_class()
    node_id = ray.get_runtime_context().get_node_id()
    strategy = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
    return WorkerActor.options(name=worker_id, namespace=cfg.ray_namespace, scheduling_strategy=strategy) \
        .remote(worker_id, RayDBProxy(db_handle), cfg)
