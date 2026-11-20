# Módulo `worker` — ejecución de entrenamientos

**Responsabilidad:** pedir trabajo a la DB, entrenar el modelo por batches y reportar el progreso. El worker no decide
nada sobre el estado de los jobs: la DB manda, y si la DB revoca su attempt, el worker abandona sin más.

## Archivos
| Archivo | Contenido |
|---|---|
| `worker.py` | Clase `Worker`: registro, bucle de trabajo en un hilo, estado consultable. |
| `trainer.py` | `run_attempt` (un attempt completo), `download_artifact` y `LeaseKeeper` (hilo que renueva el lease). |
| `main.py` | Crea los actores de Ray `worker-0 … worker-N` (ya hecho). |

## Funcionalidades
- **Bucle:** se registra en la DB y pide un job cada `request_poll_interval_ms`; si la DB no responde (503), espera y reintenta sin morir.
- **Preparación:** descarga el dataset y, si el attempt continúa un trabajo previo, el último checkpoint; verifica el hash de cada binario.
- **Entrenamiento:** para cada batch calcula sus filas de forma determinista (`batching.batch_indices`, depende de `seed`) y hace un `partial_fit`. Reanudar produce el mismo modelo que no interrumpir.
- **Checkpoints:** cada `k_ckpt` batches sube el estado del modelo (primero el binario, luego la confirmación). Solo continúa si la DB responde `CONFIRMED`.
- **Lease:** un hilo aparte renueva el lease aunque un batch tarde mucho. Si la DB responde `REVOKED`, o pasa el tiempo del lease sin poder renovar, el attempt se aborta.
- **Cierre:** evalúa el modelo sobre el dataset completo, sube el modelo final y llama a `complete_job`.
- **Errores:** una excepción de entrenamiento se reporta con `report_failure`; un binario irrecuperable con `report_unavailable`.

## Lo que ya viene hecho (módulo `common`)
Carga y estandarización del dataset, partición en batches, algoritmos con `partial_fit`, serialización de checkpoints y hash.

## Registro
Logger `distriml.worker.<id>`: una línea por evento (registro, job asignado, batch k de N, checkpoint
confirmado/rechazado, lease renovado/perdido, aborto, error, fin).

## Cómo probarlo
Sin la DB real: `Worker("w0", FakeDB(), cfg, autostart=False)`; `tests/helpers/reference_worker.py` muestra el flujo sin fallos.
Con Ray: `python -m distriml.cluster --workers 2` y `python scripts/chaos.py kill-worker 0` para matar un worker a mitad de un job.
