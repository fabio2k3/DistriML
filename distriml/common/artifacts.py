"""Almacén de artefactos direccionados por contenido.

Binarios (datasets, checkpoints, modelos) guardados como <root>/artifacts/<sha[:2]>/<sha>.
Escritura atómica (fichero temporal + fsync + rename): un artefacto nunca aparece incompleto.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from distriml.common.errors import bad_request, unavailable
from distriml.common.ids import is_valid_sha256, sha256_hex


class ArtifactStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root) / "artifacts"
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        if not is_valid_sha256(sha256):
            raise bad_request("invalid_sha256", "sha256 debe ser un hexadecimal de 64 caracteres")
        return self.root / sha256[:2] / sha256

    def has(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def put(self, sha256: str, data: bytes) -> bool:
        """Guarda `data` verificando su hash. Devuelve False si ya existía. 400 si el hash no coincide."""
        if sha256_hex(data) != sha256:
            raise bad_request("hash_mismatch", "el sha256 declarado no coincide con el contenido")
        path = self.path_for(sha256)
        if path.is_file():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        return True

    def get(self, sha256: str) -> bytes:
        """Devuelve los bytes verificando el hash. 503 artifact_unavailable si falta o está corrupto."""
        path = self.path_for(sha256)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            raise unavailable("artifact_unavailable", f"artefacto {sha256[:12]}… no disponible") from None
        if sha256_hex(data) != sha256:
            raise unavailable("artifact_unavailable", f"artefacto {sha256[:12]}… corrupto")
        return data

    def delete(self, sha256: str) -> None:
        try:
            self.path_for(sha256).unlink()
        except FileNotFoundError:
            pass

    def list_hashes(self) -> list[str]:
        return sorted(p.name for p in self.root.glob("*/*") if p.is_file() and not p.name.startswith(".tmp-"))
