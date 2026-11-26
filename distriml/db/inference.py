"""Inferencia y estadísticas de uso de los modelos entrenados.

Reglas (las aplica `Database`, que resuelve el modelo y la versión antes de llamar aquí):
  - Solo se puede usar un modelo propio (client_id del job). Ajeno o inexistente -> 404.
  - Sin `version_id` se usa la versión canónica. `version` es el alias entero de la canónica (1);
    cualquier otro valor -> 404. Una versión ORPHANED solo se usa indicando su version_id.
  - Se carga el binario por su sha256 (ArtifactStore.get ya verifica el hash) y se guarda en caché.
  - La respuesta incluye siempre `version_id` y `sha256`, para saber exactamente qué modelo respondió.
  - Predecir no modifica nada salvo el contador de `inference_stats`.

Contadores: cada predicción añade una fila (modelo, versión, minuto, 1, latencia) al CSV `inference_stats`
(anexado con fsync, coste constante). Al arrancar se agregan por (modelo, versión, minuto) y, si el fichero
tenía filas repetidas, se compacta de forma atómica.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Callable

from distriml.common import algorithms
from distriml.common.ids import now_ms
from distriml.db.records import decode, encode

CACHE_SIZE = 32          # modelos deserializados que se conservan en memoria


class InferenceEngine:
    def __init__(self, store, tables, wall_ms: Callable[[], int] = now_ms):
        self.store = store
        self.tables = tables
        self._wall_ms = wall_ms
        self._cache: OrderedDict[str, dict] = OrderedDict()      # sha256 -> estado deserializado
        # (model_id, version_id, minute) -> [count, latency_sum_ms]
        self._counters: dict[tuple[str, str, int], list] = {}
        self._load_counters()

    # ------------------------------------------------------------------ persistencia de contadores
    def _load_counters(self) -> None:
        rows = [decode("inference_stats", r) for r in self.tables.inference_stats.load()]
        for r in rows:
            c = self._counters.setdefault((r["model_id"], r["version_id"], r["minute"]), [0, 0.0])
            c[0] += r["count"]
            c[1] += r["latency_sum"]
        if len(rows) > len(self._counters):                       # compactar filas repetidas
            self.tables.inference_stats.rewrite(
                {"model_id": m, "version_id": v, "minute": mn, "count": c[0], "latency_sum": c[1]}
                for (m, v, mn), c in sorted(self._counters.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])))

    # ------------------------------------------------------------------ modelo en caché
    def _state(self, sha256: str) -> dict:
        state = self._cache.get(sha256)
        if state is None:
            state = algorithms.deserialize_state(self.store.get(sha256))
            self._cache[sha256] = state
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(sha256)
        return state

    def forget(self, sha256: str) -> None:
        """Saca de la caché el modelo de una versión borrada."""
        self._cache.pop(sha256, None)

    # ------------------------------------------------------------------ API
    def predict(self, version_row: dict, algorithm: str, features) -> dict:
        """Devuelve {"prediction", "version_id", "sha256"} (más "label" si las etiquetas eran texto) y suma 1 al contador
        (model_id, version_id, minuto) con la latencia medida en ms."""
        t0 = time.perf_counter()
        state = self._state(version_row["sha256"])
        prediction = algorithms.predict(state, algorithm, features)      # 400 invalid_input si no encaja
        ms = (time.perf_counter() - t0) * 1000
        minute = self._wall_ms() // 60_000
        self.tables.inference_stats.append(encode("inference_stats", {
            "model_id": version_row["model_id"], "version_id": version_row["version_id"],
            "minute": minute, "count": 1, "latency_sum": ms}))
        c = self._counters.setdefault((version_row["model_id"], version_row["version_id"], minute), [0, 0.0])
        c[0] += 1
        c[1] += ms
        result = {"prediction": prediction, "version_id": version_row["version_id"], "sha256": version_row["sha256"]}
        label = algorithms.label_of(state, prediction)
        if label is not None:                      # solo si el dataset tenía etiquetas de texto
            result["label"] = label
        return result

    def stats(self, model_id: str, version_id: str, from_minute: int | None, to_minute: int | None) -> dict:
        """{"model_id","version_id","total_calls","mean_latency_ms",
        "por_minuto": [{"minute","count","mean_latency_ms"}]}. mean = latency_sum / count (0 si count = 0).
        `from_minute` y `to_minute` (minutos desde epoch) son inclusivos y opcionales."""
        rows = sorted((minute, c) for (m, v, minute), c in self._counters.items()
                      if m == model_id and v == version_id
                      and (from_minute is None or minute >= from_minute)
                      and (to_minute is None or minute <= to_minute))
        total = sum(c[0] for _, c in rows)
        latency = sum(c[1] for _, c in rows)
        return {"model_id": model_id, "version_id": version_id, "total_calls": total,
                "mean_latency_ms": (latency / total) if total else 0.0,
                "por_minuto": [{"minute": minute, "count": c[0], "mean_latency_ms": c[1] / c[0]} for minute, c in rows]}
