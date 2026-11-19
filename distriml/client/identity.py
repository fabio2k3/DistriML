"""Identidad local de la consola: username (etiqueta local) + client_id aleatorio.

Se guarda en ~/.distriml/identity.json (o en $DISTRIML_HOME). Si se pierde, se pierde el acceso
a los datos: no hay recuperación por username.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from distriml.common.ids import new_client_id


class NotInitialized(Exception):
    pass


def home_dir() -> Path:
    return Path(os.environ.get("DISTRIML_HOME", Path.home() / ".distriml"))


def _path() -> Path:
    return home_dir() / "identity.json"


def init_identity(username: str, force: bool = False) -> dict:
    path = _path()
    if path.exists() and not force:
        raise FileExistsError(f"ya existe una identidad en {path} (usa --force para reemplazarla; se perderá el acceso a lo anterior)")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"username": username, "client_id": new_client_id(username), "last_entry": None}
    _write(data)
    return data


def _write(data: dict) -> None:
    path = _path()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def load_identity() -> dict:
    try:
        return json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotInitialized("no hay identidad: ejecuta `distriml init <username>`") from None


def save_last_entry(entry_id: str) -> None:
    """Guarda el último entry_id devuelto por una escritura (se reenvía como X-Min-Entry, )."""
    data = load_identity()
    data["last_entry"] = entry_id
    _write(data)
