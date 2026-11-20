"""Inferencia y estadísticas de uso de los modelos entrenados.

Herramientas ya hechas en common/algorithms.py: deserialize_state, predict, task_of.
Reglas:
  - Solo se puede usar un modelo propio (client_id del job). Ajeno o inexistente -> 404.
  - Sin `version_id` se usa la versión canónica. `version` es el alias entero de la canónica (1);
    cualquier otro valor -> 404. Una versión ORPHANED solo se usa indicando su version_id.
  - Se carga el binario por su sha256 (ArtifactStore.get ya verifica el hash) y se guarda en caché.
  - La respuesta incluye siempre `version_id` y `sha256`, para saber exactamente qué modelo respondió.
  - Predecir no modifica nada salvo el contador de `inference_stats`.
"""
from __future__ import annotations


class InferenceEngine:
    def __init__(self, store, tables):
        self.store = store
        self.tables = tables
        self._cache: dict[str, dict] = {}     # sha256 -> estado deserializado

    def predict(self, version_row: dict, algorithm: str, features) -> dict:
        """TODO(DB-7): devolver {"prediction", "version_id", "sha256"} y sumar 1 al contador
        (model_id, version_id, minute) con la latencia medida en ms."""
        raise NotImplementedError("TODO(DB-7)")

    def stats(self, model_id: str, version_id: str, from_minute: int | None, to_minute: int | None) -> dict:
        """TODO(DB-7): {"model_id","version_id","total_calls","mean_latency_ms",
        "por_minuto": [{"minute","count","mean_latency_ms"}]}. mean = latency_sum / count (0 si count = 0)."""
        raise NotImplementedError("TODO(DB-7)")
