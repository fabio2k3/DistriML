# Módulo `db` — base de datos y coordinación

**Responsabilidad:** es la única fuente de verdad del sistema. Guarda datasets, jobs, attempts, checkpoints,
modelos y estadísticas; decide qué worker ejecuta cada job; y garantiza que solo el attempt vigente de un job
pueda escribir. Implementa `DBService` (`common/interfaces.py`), que es lo único que ven el gateway y los workers.

## Archivos
| Archivo | Contenido |
|---|---|
| `service.py` | Clase `Database` con todas las operaciones (cliente y worker). Registra cada llamada en el log. |
| `scheduler.py` | Reglas puras: plazo de un lease, motivo de revocación, estado del job tras revocar, retardo de reasignación, elección del siguiente job. |
| `inference.py` | Predicción con un modelo entrenado (caché del modelo) y estadísticas por minuto. |
| `tables.py` | Columnas de cada tabla CSV y contenedor `Tables`. |
| `csvtable.py` | Lectura, anexado y reescritura atómica de un CSV (ya hecho). |
| `main.py` | Crea el actor de Ray de la DB (ya hecho). |

## Funcionalidades
- **Datasets:** valida el CSV, calcula sus metadatos (filas, características, clases, media y desviación) y guarda el binario por hash.
- **Jobs:** valida modelo, hiperparámetros y compatibilidad con el dataset; calcula `total_batches`; estados `PENDIENTE → EN_EJECUCION → COMPLETADO | FALLIDO | CANCELADO`.
- **Idempotencia:** `dataset_id` y `job_id` se derivan de `(client_id, client_request_id)`; repetir una petición devuelve el mismo resultado, y reusar el id con otro contenido da 409.
- **Asignación:** `request_job` entrega el job más antiguo pendiente y crea un attempt que continúa desde el último checkpoint confirmado.
- **Leases:** cada attempt debe renovar su lease; si no lo hace en `lease + margen`, se revoca (`WORKER_LOST`, o `NEVER_STARTED` si nunca renovó) y el job vuelve a `PENDIENTE`.
- **Checkpoints:** solo el attempt vigente puede confirmar batches. Un attempt revocado recibe `REJECTED`.
- **Reintentos:** `max_attempts` cuenta errores reportados por workers; `max_lost` cuenta pérdidas de contacto; `NEVER_STARTED`, `ARTIFACT_UNAVAILABLE` y `CANCEL` no cuentan.
- **Modelos:** al completar un job se crea el modelo y su versión (`version_id` = `attempt_id`, alias `version = 1`).
- **Aislamiento:** un recurso de otro cliente se responde como inexistente (404).
- **Inferencia:** predicción con la versión canónica o una versión indicada, y contadores de uso por minuto.

## Persistencia y reinicio
Tablas CSV en `<data_dir>/db/` y binarios en `<data_dir>/artifacts/`. Al iniciar, `Database` recarga todo; el
estado volátil (último renew de cada lease, workers conectados) se reconstruye con las siguientes llamadas.

## Registro
Cada operación pública aparece en consola con logger `distriml.db`. Los cambios de estado internos
(job asignado, attempt revocado, job completado…) deben registrarse además con `self.log.info(...)`.

## Cómo probarlo
`pytest tests/test_contract_live.py` contra el sistema real (`DISTRIML_URL=...`), y `tests/helpers/fake_db.py`
como referencia del comportamiento esperado.
