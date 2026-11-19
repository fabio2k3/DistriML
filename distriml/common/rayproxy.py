"""Adaptador entre el contrato `DBService` y un actor de Ray.

`RayDBProxy(handle)` expone los mismos métodos que `DBService`, pero síncronos: cada llamada es
`ray.get(handle.<metodo>.remote(...))`. Así el gateway HTTP y los workers no saben si hablan con
un actor de Ray o con un objeto en memoria (tests).

Traducciones de errores:
  - ContractError levantado dentro del actor  -> se relanza tal cual en el llamador.
  - Actor muerto / reiniciándose / inalcanzable -> ContractError(503, "no_ca"): no hay DB disponible.

Cada llamada queda registrada en el logger `distriml.proxy` (nivel DEBUG; ver common/log.py).
"""
from __future__ import annotations

import time

from distriml.common.errors import ContractError, unavailable
from distriml.common.log import describe_call, get_logger, short

log = get_logger("proxy")


def call_remote(method, *args, **kwargs):
    import ray
    from ray.exceptions import ActorDiedError, ActorUnavailableError, RayActorError, RayTaskError

    try:
        return ray.get(method.remote(*args, **kwargs))
    except RayTaskError as exc:
        cause = getattr(exc, "cause", None)
        if isinstance(cause, ContractError):
            raise ContractError(cause.status, cause.reason, cause.message, cause.retry_after) from None
        if isinstance(exc, ContractError):  # Ray crea una clase que hereda de la excepción original
            raise ContractError(exc.status, exc.reason, exc.message, exc.retry_after) from None
        raise
    except (ActorDiedError, ActorUnavailableError, RayActorError):
        raise unavailable("no_ca", "la DB no está disponible (actor caído o reiniciándose)") from None


class RayDBProxy:
    def __init__(self, handle):
        self._handle = handle

    def __getattr__(self, name: str):
        if name.startswith("_"):          # evita interferir con pickle/copy
            raise AttributeError(name)
        method = getattr(self._handle, name)

        def _call(*args, **kwargs):
            call = describe_call(name, args, kwargs)
            t0 = time.perf_counter()
            try:
                result = call_remote(method, *args, **kwargs)
            except ContractError as exc:
                ms = (time.perf_counter() - t0) * 1000
                # Los errores del contrato ya los registra la DB; aquí solo importa que no haya DB.
                level = log.warning if exc.reason == "no_ca" else log.debug
                level("llamada a DB %s -> %s %s  %.1fms", call, exc.status, exc.reason, ms)
                raise
            log.debug("llamada a DB %s -> %s  %.1fms", call, short(result, 80), (time.perf_counter() - t0) * 1000)
            return result

        _call.__name__ = name
        return _call
