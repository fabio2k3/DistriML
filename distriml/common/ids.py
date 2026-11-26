"""Identificadores.

Las identidades de las entidades se DERIVAN de (client_id, client_request_id): un reintento
produce el mismo id y por tanto es idempotente. `model_id` se deriva del job.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import time
import uuid

NS_JOB = uuid.uuid5(uuid.NAMESPACE_DNS, "distriml.job")
NS_DATASET = uuid.uuid5(uuid.NAMESPACE_DNS, "distriml.dataset")
NS_CANCEL = uuid.uuid5(uuid.NAMESPACE_DNS, "distriml.cancel")
NS_DELETE = uuid.uuid5(uuid.NAMESPACE_DNS, "distriml.delete")
NS_MODEL = uuid.uuid5(uuid.NAMESPACE_DNS, "distriml.model")
NS_MODELDEF = uuid.uuid5(uuid.NAMESPACE_DNS, "distriml.modeldef")

_SEP = "\x1f"  # separador de campos, evita ambigüedades al concatenar
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_valid_sha256(value: str) -> bool:
    return isinstance(value, str) and _HEX64.match(value) is not None


def is_valid_client_id(value: str) -> bool:
    """Un client_id es un hexadecimal de 64 caracteres (SHA-256, )."""
    return isinstance(value, str) and _HEX64.match(value) is not None


def new_client_id(username: str) -> str:
    """client_id = hex(SHA-256(username ‖ salt)) con salt aleatorio de 256 bits que se descarta.

    El username es solo una etiqueta local: dos usuarios con el mismo username obtienen
    client_id distintos y no comparten nada.
    """
    salt = secrets.token_bytes(32)
    return hashlib.sha256(username.encode("utf-8") + salt).hexdigest()


def _derive(ns: uuid.UUID, *parts: str) -> str:
    return str(uuid.uuid5(ns, _SEP.join(parts)))


def derive_job_id(client_id: str, client_request_id: str) -> str:
    return _derive(NS_JOB, client_id, client_request_id)


def derive_dataset_id(client_id: str, client_request_id: str) -> str:
    return _derive(NS_DATASET, client_id, client_request_id)


def derive_cancel_id(client_id: str, client_request_id: str) -> str:
    """Clave de idempotencia de una cancelación."""
    return _derive(NS_CANCEL, client_id, client_request_id)


def derive_delete_id(client_id: str, client_request_id: str) -> str:
    """Clave de idempotencia del borrado de una versión de modelo."""
    return _derive(NS_DELETE, client_id, client_request_id)


def derive_model_id(job_id: str) -> str:
    """model_id = UUIDv5(NS_MODEL, job_id)."""
    return str(uuid.uuid5(NS_MODEL, job_id))


def derive_model_def_id(algorithm: str, version: int) -> str:
    """Id fijo y común a todos los nodos para una definición del catálogo."""
    return _derive(NS_MODELDEF, algorithm, str(version))


def new_attempt_id() -> str:
    return str(uuid.uuid4())


def new_request_id() -> str:
    return str(uuid.uuid4())


def now_ms() -> int:
    return int(time.time() * 1000)
