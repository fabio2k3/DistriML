"""Registro de operaciones: todo lo que hace el sistema se puede seguir por consola.

Formato de cada línea:
    14:02:31.457 INFO  distriml.db         create_job(ab12cd34, 'r1', 9f3e...) -> {'job_id': ...}  4.2ms

Nombres de logger (columna central):
    distriml.api          peticiones HTTP que recibe el gateway
    distriml.proxy        llamadas del gateway/worker hacia la DB (lado que llama)
    distriml.db           operaciones que ejecuta la DB (lado que atiende)
    distriml.worker.<id>  eventos de cada worker (asignación, batches, leases, errores)
    distriml.client       peticiones de la consola (solo con --verbose)

Niveles:
    INFO     operaciones relevantes (crear/cancelar jobs, checkpoints, completar, fallos)
    DEBUG    además las rutinarias y repetitivas (ping, renovaciones de lease, polling vacío, artefactos)
    WARNING  errores esperados del contrato (400/404/409/503) y reintentos
    ERROR    fallos inesperados

Nivel por defecto: INFO. Se cambia con `--log-level` en el lanzador, con la variable de entorno
DISTRIML_LOG_LEVEL, o con `Config.log_level` (es lo que se propaga a los procesos de Ray).
"""
from __future__ import annotations

import dataclasses
import functools
import logging
import os
import sys
import time

from distriml.common.errors import ContractError

FORMAT = "%(asctime)s.%(msecs)03d %(levelname)-5s %(name)-18s %(message)s"
DATE_FORMAT = "%H:%M:%S"
ROOT = "distriml"

# Operaciones que solo se muestran con nivel DEBUG (son muy frecuentes y no cambian el estado).
ROUTINE = frozenset({"ping", "renew_lease", "get_artifact", "put_artifact", "list_model_definitions"})
# Operaciones que solo importan si devuelven algo (el polling de un worker ocioso devuelve None).
ROUTINE_WHEN_EMPTY = frozenset({"request_job"})


def setup_logging(level: str | int | None = None) -> None:
    """Configura el logger `distriml` (stderr). Se puede llamar muchas veces: solo configura una vez."""
    if level is None:
        level = os.environ.get("DISTRIML_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO
    root = logging.getLogger(ROOT)
    root.setLevel(level)
    if not any(getattr(h, "_distriml", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(FORMAT, DATE_FORMAT))
        handler._distriml = True  # type: ignore[attr-defined]
        root.addHandler(handler)
        root.propagate = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}")


def short(value, limit: int = 80) -> str:
    """Resumen de un valor para el log: ids abreviados, binarios sin contenido, textos recortados."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, str):
        if len(value) in (36, 64) and all(c in "0123456789abcdef-" for c in value):
            return value[:8]                      # uuid o sha256/client_id: los 8 primeros caracteres bastan
        text = repr(value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        text = f"{type(value).__name__}({_fields(value)})"
    elif isinstance(value, dict):
        text = "{" + ", ".join(f"{k!r}: {short(v, 30)}" for k, v in value.items()) + "}"
    elif isinstance(value, (list, tuple)):
        text = f"[{len(value)} elementos]" if len(value) > 3 else "[" + ", ".join(short(v, 30) for v in value) + "]"
    else:
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fields(obj) -> str:
    keys = ("job_id", "attempt_id", "worker_id", "next_batch", "total_batches")
    return ", ".join(f"{k}={short(getattr(obj, k))}" for k in keys if hasattr(obj, k))


def describe_call(name: str, args: tuple, kwargs: dict) -> str:
    parts = [short(a, 40) for a in args] + [f"{k}={short(v, 40)}" for k, v in kwargs.items()]
    return f"{name}({', '.join(parts)})"


def _level_for(name: str, result, failed: bool) -> int:
    if failed:
        return logging.WARNING
    if name in ROUTINE:
        return logging.DEBUG
    if name in ROUTINE_WHEN_EMPTY and result in (None, [], {}):
        return logging.DEBUG
    return logging.INFO


def log_call(logger: logging.Logger, name: str, args: tuple, kwargs: dict, func, *func_args, **func_kwargs):
    """Ejecuta `func`, registra la operación (argumentos, resultado o error, duración) y devuelve su resultado."""
    t0 = time.perf_counter()
    call = describe_call(name, args, kwargs)
    try:
        result = func(*func_args, **func_kwargs)
    except ContractError as exc:
        ms = (time.perf_counter() - t0) * 1000
        logger.warning("%s -> %s %s %s  %.1fms", call, exc.status, exc.reason, exc.message, ms)
        raise
    except Exception as exc:  # noqa: BLE001  (se registra y se relanza)
        ms = (time.perf_counter() - t0) * 1000
        logger.error("%s -> FALLO %s: %s  %.1fms", call, type(exc).__name__, short(str(exc), 120), ms)
        raise
    ms = (time.perf_counter() - t0) * 1000
    level = _level_for(name, result, failed=False)
    if logger.isEnabledFor(level):
        logger.log(level, "%s -> %s  %.1fms", call, short(result, 100), ms)
    return result


def traced(logger_name: str, operations: set[str] | frozenset[str] | None = None):
    """Decorador de CLASE: registra cada llamada a sus métodos públicos (o a `operations`, si se indica)."""

    def decorate(cls):
        logger = get_logger(logger_name)
        for attr, fn in list(vars(cls).items()):
            if attr.startswith("_") or not callable(fn) or (operations is not None and attr not in operations):
                continue

            def make(name, original):
                @functools.wraps(original)
                def wrapper(self, *args, **kwargs):
                    return log_call(logger, name, args, kwargs, original, self, *args, **kwargs)

                return wrapper

            setattr(cls, attr, make(attr, fn))
        return cls

    return decorate
