"""Tabla CSV mínima y lista para usar (utilidad de la DB; adáptala si lo necesitas).

- `load()` devuelve las filas como diccionarios de cadenas (las columnas JSON las decodificas tú).
- `append()` añade una fila al final con fsync (para tablas de solo anexado, p. ej. job_batches).
- `rewrite()` reescribe la tabla completa de forma atómica (fichero temporal + fsync + rename),
  para tablas que cambian de estado (jobs, attempts).
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


def dumps(value) -> str:
    """Serializa un valor JSON para guardarlo en una celda."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def loads(cell: str):
    return json.loads(cell) if cell != "" else None


class CsvTable:
    def __init__(self, path: str | os.PathLike, columns: list[str]):
        self.path = Path(path)
        self.columns = list(columns)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.rewrite([])

    def load(self) -> list[dict]:
        with open(self.path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.columns).writerow(self._clean(row))
            fh.flush()
            os.fsync(fh.fileno())

    def rewrite(self, rows: Iterable[dict]) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.columns)
                writer.writeheader()
                for row in rows:
                    writer.writerow(self._clean(row))
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _clean(self, row: dict) -> dict:
        return {c: ("" if row.get(c) is None else row.get(c)) for c in self.columns}
