# Módulo `db` — base de datos y coordinación

**Responsabilidad:** es la única fuente de verdad del sistema. Guarda datasets, jobs, attempts, checkpoints,
modelos y estadísticas; decide qué worker ejecuta cada job; y garantiza que solo el attempt vigente de un job
pueda escribir. Implementa `DBService` (`common/interfaces.py`), que es lo único que ven el gateway y los workers.

## Archivos
| Archivo | Contenido |
|---|---|
| `service.py` | Clase `Database` con todas las operaciones (cliente y worker), carga y reparación al arrancar, revisión de leases. Registra cada llamada en el log. |
| `scheduler.py` | Reglas puras: plazo de un lease, motivo de revocación, estado del job tras revocar, retardo de reasignación, elección del siguiente job. |
| `inference.py` | Predicción con un modelo entrenado (caché de modelos) y contadores de uso por minuto. |
| `records.py` | Conversión entre filas CSV (texto) y filas en memoria (enteros, JSON, booleanos, opcionales). |
| `tables.py` | Columnas de cada tabla CSV y contenedor `Tables`. |
| `csvtable.py` | Lectura, anexado y reescritura atómica de un CSV. |
| `main.py` | Crea el actor de Ray de la DB, fijado al nodo head con el recurso `db`. |

## Funcionalidades
- **Datasets:** valida el CSV, calcula sus metadatos (filas, características, clases, media y desviación) y guarda el binario por hash.
- **Jobs:** valida modelo, hiperparámetros y compatibilidad con el dataset; calcula `total_batches`; estados `PENDIENTE → EN_EJECUCION → COMPLETADO | FALLIDO | CANCELADO`.
- **Idempotencia:** `dataset_id` y `job_id` se derivan de `(client_id, client_request_id)`; repetir una petición devuelve exactamente la misma respuesta (incluido `entry_id`), y reusar el id con otro contenido da 409.
- **Asignación:** `request_job` entrega el job pendiente más antiguo cuyo dataset y checkpoint de partida estén en el almacén, y crea un attempt que continúa desde el último checkpoint confirmado. No asume un número fijo de workers: los workers aparecen y desaparecen (nodos que crecen y decrecen) y la DB reparte el trabajo a quien lo pida.
- **Leases:** cada attempt debe renovar su lease; confirmar un batch también cuenta como señal de vida. Si pasan `lease + margen` sin ninguna, se revoca (`WORKER_LOST`, o `NEVER_STARTED` si nunca renovó ni confirmó) y el job vuelve a `PENDIENTE`. La revisión es perezosa: se hace al inicio de las operaciones, sin hilos.
- **Retardo de reasignación:** tras `WORKER_LOST` el job espera `min(2^pérdidas × t_sync, 60 s)`, contado desde el vencimiento del lease (no desde que la DB lo detecta).
- **Checkpoints:** solo el attempt vigente puede confirmar batches, y en orden (`último + 1`). Un attempt revocado recibe `REJECTED`; si el binario aún no está, `RETRY`; repetir un batch ya confirmado por el mismo attempt devuelve `CONFIRMED`.
- **Reintentos:** `max_attempts` cuenta errores reportados por workers; `max_lost` cuenta pérdidas de contacto; `NEVER_STARTED`, `ARTIFACT_UNAVAILABLE` y `CANCEL` no cuentan. Al agotar un presupuesto el job pasa a `FALLIDO`.
- **Cierre:** `complete_job` exige todos los batches confirmados y crea el modelo y su versión (`version_id` = `attempt_id`, alias `version = 1`). Repetirlo con el mismo attempt devuelve `OK` sin crear nada nuevo.
- **Cancelación:** solo el dueño; revoca el attempt vigente con `CANCEL`; idempotente por `client_request_id`; 409 si el job ya terminó.
- **Aislamiento:** un recurso de otro cliente se responde como inexistente (404).
- **Inferencia:** predicción con la versión canónica o una indicada (`version_id`, o `version = 1` como alias de la canónica), y contadores de uso por minuto. Una predicción inválida no se cuenta.

## Persistencia y reinicio
Tablas CSV en `<data_dir>/db/` y binarios en `<data_dir>/artifacts/`.

- **Estado en memoria:** al iniciar, `Database` carga todas las tablas en diccionarios (en orden de creación) y cada cambio se escribe en su CSV antes de responder. Las tablas de estado cambiante (`jobs`, `attempts`) se reescriben de forma atómica; el resto solo se anexa.
- **Reinicio del actor:** Ray lo reinicia (`max_restarts=-1`) y el constructor recarga todo; los workers pueden seguir con sus attempts.
- **Cortes a mitad de operación:** las escrituras de cada operación siguen un orden fijo (primero lo que otras tablas referencian) y `_reconcile` repara lo que quede a medias: `last_confirmed_batch` se recalcula desde `job_batches`; un job con su modelo ya guardado pasa a `COMPLETADO`; un attempt activo que su job ya no considera se revoca sin contar en ningún presupuesto; un job `EN_EJECUCION` sin attempt activo vuelve a `PENDIENTE`.
- **Estado volátil (no se guarda):** instante del último renew de cada attempt, retardos de reasignación y workers vistos. Tras un reinicio, cada attempt activo recibe un plazo de lease completo nuevo (sus workers pudieron no poder renovar mientras la DB estaba caída) y los retardos de reasignación se reinician.
- **`entry_id`:** el contador se recupera al iniciar a partir de los `entry_id` ya entregados, así que no se repite.
- **Estadísticas de inferencia:** una fila anexada por predicción; al iniciar se agregan y el fichero se compacta.

## Límites conocidos
- Una sola DB, sin réplica: si el disco del head se pierde, se pierde el estado.
- Cada cambio de estado reescribe `jobs.csv`/`attempts.csv` completos: adecuado para el tamaño de este prototipo, no para millones de jobs.
- El texto del error que reporta un worker (`report_failure`) se registra en el log pero no se guarda en las tablas.
- Si el binario del último checkpoint de un job desaparece del disco, el job no se reasigna (queda `PENDIENTE`) hasta que reaparezca.

## Registro
Cada operación pública aparece en consola con logger `distriml.db`. Los cambios de estado internos (dataset
guardado, job creado, job asignado, attempt revocado con su motivo y contadores, job completado, reparaciones al
arrancar) se registran además con `self.log`.

## Cómo probarlo
```bash
pytest tests/test_db_scheduler.py tests/test_db_tables.py       # reglas puras y CSV
pytest tests/test_db_service.py tests/test_db_workers.py        # operaciones de cliente y de worker (reloj manual)
pytest tests/test_db_persistence.py                             # reinicio y cortes a mitad de operación
pytest tests/test_contract_db.py                                # contrato externo contra la DB real
pytest -m ray tests/test_db_ray.py                              # actor de Ray matado con un job a medias
```
`tests/helpers/fake_db.py` sigue siendo la implementación en memoria de referencia para probar gateway y workers sin la DB real.
