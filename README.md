# DistriML

Plataforma distribuida para entrenar y servir modelos de aprendizaje automático. Los usuarios suben datasets y envían trabajos de entrenamiento; un conjunto de *workers* los ejecuta en paralelo sobre [Ray](https://www.ray.io/), los resultados se guardan como modelos versionados y se consultan con una API de inferencia. El sistema tolera la caída de workers y de la base de datos sin perder trabajo ni cambiar el resultado del entrenamiento.

## Contenido

- [Funcionalidades](#funcionalidades)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Puesta en marcha](#puesta-en-marcha)
- [Uso del cliente](#uso-del-cliente)
- [Formato de los datasets](#formato-de-los-datasets)
- [Algoritmos disponibles](#algoritmos-disponibles)
- [Varios clientes en una misma máquina](#varios-clientes-en-una-misma-máquina)
- [Tolerancia a fallos](#tolerancia-a-fallos)
- [Configuración](#configuración)
- [API HTTP](#api-http)
- [Pruebas](#pruebas)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Limitaciones conocidas](#limitaciones-conocidas)

## Funcionalidades

**Para el usuario**
- Subir datasets CSV. Si el contenido ya estaba subido, no se guarda otra copia.
- Entrenar modelos eligiendo algoritmo e hiperparámetros, seguir el progreso de un job y cancelarlo.
- Consultar las métricas de cada job, las versiones de un modelo y sus estadísticas de uso.
- Hacer predicciones con un modelo propio; si el dataset tenía etiquetas de texto, la respuesta incluye el nombre de la clase.
- Listar los modelos entrenados y eliminar versiones de un modelo.
- Consola interactiva con un menú que muestra siempre todas las operaciones disponibles, o comandos sueltos para usarlos en scripts.
- Cada usuario ve únicamente sus propios datasets, jobs y modelos.

**Para el sistema**
- Entrenamiento por *batches* con *checkpoints*: si un worker cae, otro continúa desde el último checkpoint confirmado y el modelo final es el mismo que sin avería.
- *Leases*: un worker conserva su trabajo mientras renueva su plazo; si deja de hacerlo, el job se reasigna con un retardo creciente y presupuestos de reintentos.
- Base de datos propia sobre ficheros CSV, que se recarga y repara al reiniciarse; la base de datos corre como un actor de Ray que se reinicia solo.
- Operaciones del cliente idempotentes: repetir una petición con el mismo `client_request_id` no duplica datasets ni jobs, y el cliente reintenta solo ante errores de conexión.
- Workers que se pueden añadir o quitar sin cambiar nada más.

## Requisitos

| Requisito | Detalle |
|---|---|
| Sistema operativo | Linux, o Windows con WSL2 (recomendado). Ray en Windows nativo es beta y el modo multinodo solo existe en Linux. |
| Python | 3.10 o superior. Las versiones fijadas se probaron con 3.13 y 3.12. |
| Dependencias | `ray`, `numpy`, `scikit-learn`, `fastapi`, `uvicorn`, `httpx`, `pydantic` (para las pruebas, además `pytest` y `httpx2`). Las versiones exactas están en `requirements.txt`. |
| Variable de entorno | `RAY_AUTH_MODE=disabled` en **cada** terminal que use Ray o el sistema. |

Todos los nodos deben usar la misma versión de `scikit-learn` y `numpy`: los checkpoints son *pickles* y el generador de números aleatorios debe coincidir.

## Instalación

```bash
# Dependencias del sistema (Ubuntu/WSL)
sudo apt update && sudo apt install -y python3-venv python3-pip build-essential

# Entorno virtual y dependencias
cd DistriML-ray-version
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .          # opcional: instala los comandos `distriml` y `distriml-cluster`

# En cada terminal (o añádelo a ~/.bashrc)
export RAY_AUTH_MODE=disabled
```

Si tu Python es anterior a 3.12 y alguna versión de `requirements.txt` no se encuentra, instala una más reciente (por ejemplo con el PPA `deadsnakes`) y crea el entorno con ella.

## Puesta en marcha

Se usan dos terminales, ambas en la carpeta del proyecto, con el entorno activado y `RAY_AUTH_MODE=disabled`.

**Terminal 1: el cluster** (Ray, la base de datos, los workers y la API HTTP en un solo proceso)

```bash
python -m distriml.cluster --workers 3 --data-dir ./runtime
```

Cuando aparece `API escuchando en http://127.0.0.1:8080`, el sistema está listo. Esta terminal queda ocupada mostrando el registro de cada operación. Opciones:

| Opción | Significado |
|---|---|
| `--workers N` | Número de workers locales (por defecto 2). |
| `--data-dir RUTA` | Carpeta donde se guardan la base de datos y los artefactos (por defecto `./runtime`). |
| `--host`, `--port` | Dirección de la API (por defecto `127.0.0.1:8080`). |
| `--log-level` | `DEBUG`, `INFO`, `WARNING` o `ERROR`. Con `DEBUG` se ven también los latidos y las llamadas internas a la base de datos. |
| `--ray-address` | Dirección de un cluster Ray ya existente (por defecto se crea uno local). |

Para empezar de cero, detén el cluster con Ctrl+C y borra la carpeta `runtime`.

**Terminal 2: el cliente**

```bash
python -m distriml.client            # abre la consola interactiva
```

La primera vez pide un nombre de usuario. Si el cluster escucha en otra dirección, usa `--url http://HOST:PUERTO` o `export DISTRIML_URL=...`.

## Uso del cliente

### Consola interactiva

`python -m distriml.client` (o `python -m distriml.client app`) abre un menú que se vuelve a mostrar después de cada operación:

```
  1  Ver algoritmos disponibles
  2  Subir un dataset (CSV)
  3  Mis datasets
  4  Entrenar un modelo (enviar un job)
  5  Mis jobs
  6  Ver / seguir un job
  7  Cancelar un job
  8  Mis modelos entrenados
  9  Hacer una predicción
 10  Detalle de un modelo (versiones y estadísticas)
 11  Eliminar una versión de un modelo
  0  Salir
```

Los datasets, jobs y modelos se eligen por número de una lista, sin copiar identificadores. Un flujo típico es: **2** (subir), **4** (entrenar y esperar), **8** (ver el modelo), **9** (predecir). Las operaciones destructivas piden confirmación.

### Comandos

Los mismos servicios están disponibles como comandos (`python -m distriml.client <comando>`, o `distriml <comando>` si instalaste el paquete). Los comandos que reciben un identificador lo toman tal cual lo devuelve el comando anterior, sin comillas ni corchetes.

| Comando | Qué hace |
|---|---|
| `init USUARIO` | Crea tu identidad local (`--force` para sustituir una existente). |
| `whoami` | Muestra tu usuario (`--show-id` añade el `client_id`). |
| `models` | Catálogo de algoritmos con sus hiperparámetros. |
| `models --mine` | Tus modelos ya entrenados (los que se pueden usar con `predict`). |
| `upload RUTA [--name N]` | Sube un dataset CSV. |
| `datasets` | Lista tus datasets. |
| `submit --dataset ID --model ALGORITMO [--hp clave=valor ...]` | Envía un job de entrenamiento. |
| `jobs [--estado E]` | Lista tus jobs, opcionalmente filtrados por estado. |
| `job ID [--watch]` | Consulta un job; con `--watch` espera hasta que termine mostrando el avance. |
| `cancel ID` | Cancela un job. |
| `metrics ID` | Métricas por batch y finales de un job. |
| `versions MODEL_ID [--version-id V]` | Versiones de un modelo. |
| `delete-version MODEL_ID VERSION_ID` | Elimina una versión; si era la única, el modelo desaparece y ya no se puede usar. No se puede deshacer. |
| `stats MODEL_ID [--version-id V]` | Número de inferencias y latencia media. |
| `predict MODEL_ID --features a,b,c,d [--version-id V]` | Predicción con un modelo propio. |

Opciones generales, antes del comando: `--url URL` (dirección de la API) y `-v` (muestra cada petición HTTP y sus reintentos).

### Ejemplo completo

```bash
python -m distriml.client init ana
python -m distriml.client upload data/samples/iris_names.csv
python -m distriml.client submit --dataset <dataset_id> --model sgd_logistic --hp epochs=5 --hp batch_size=50
python -m distriml.client job <job_id> --watch          # el resultado trae el model_id
python -m distriml.client models --mine
python -m distriml.client predict <model_id> --features 5.1,3.5,1.4,0.2
```

Estados de un job: `PENDIENTE`, `EN_EJECUCION`, `COMPLETADO`, `FALLIDO` y `CANCELADO`.

## Formato de los datasets

- Archivo CSV en UTF-8, con cabecera y al menos 2 filas de datos. Tamaño máximo: 5 MB.
- Todas las columnas son características numéricas salvo la **última, que es la etiqueta**.
- La etiqueta puede ser:
  - **Numérica**: enteros (0, 1, 2…) para clasificación, o reales para regresión.
  - **Texto** (`setosa`, `versicolor`…): los nombres se ordenan alfabéticamente y se numeran 0, 1, 2… El modelo guarda los nombres y `predict` devuelve, además del número, el campo `label` con el nombre de la clase (la consola lo muestra como `Predicción: 1 = versicolor`). Máximo 50 clases.
- Subir dos veces el mismo contenido no crea otro dataset: se devuelve el existente con `already_existed: true`.

Ejemplos en `data/samples/`: `iris.csv` (clasificación, etiquetas numéricas), `iris_names.csv` (el mismo dataset con los nombres de las clases) y `diabetes.csv` (regresión).

## Algoritmos disponibles

| Algoritmo | Tarea | Hiperparámetros por defecto |
|---|---|---|
| `sgd_regressor` | Regresión | `batch_size=32`, `epochs=10`, `seed=42`, `alpha=0.0001`, `eta0=0.01` |
| `sgd_logistic` | Clasificación | `batch_size=32`, `epochs=10`, `seed=42`, `alpha=0.0001`, `eta0=0.01` |
| `perceptron` | Clasificación | `batch_size=32`, `epochs=10`, `seed=42`, `eta0=1.0` |
| `gaussian_nb` | Clasificación | `batch_size=32`, `seed=42`, `var_smoothing=1e-09` |

`models` muestra además los rangos permitidos; un valor fuera de rango se rechaza con un error 400. Cada job se divide en *batches* (filas del dataset ÷ `batch_size`, por `epochs`); tras cada batch se guarda un checkpoint. El orden de las filas de cada batch depende solo de `seed`, por lo que reanudar un entrenamiento produce el mismo modelo que no haberlo interrumpido.

## Varios clientes en una misma máquina

La identidad de cada cliente se guarda en una carpeta: `~/.distriml` por defecto, o la que indique la variable `DISTRIML_HOME`. Con una carpeta distinta por terminal, cada una es un usuario independiente:

```bash
export DISTRIML_HOME=~/.distriml-ana  && python -m distriml.client init ana
export DISTRIML_HOME=~/.distriml-beto && python -m distriml.client init beto     # en otra terminal
```

Cada usuario ve solo lo suyo; consultar un identificador ajeno devuelve un error 404. En PowerShell: `$env:DISTRIML_HOME = "$HOME\.distriml-ana"`.

## Tolerancia a fallos

Con el cluster en marcha, en otra terminal se pueden provocar averías con `scripts/chaos.py`:

```bash
python scripts/chaos.py list                          # actores vivos: db, worker-0, worker-1, ...
python scripts/chaos.py kill-worker 0                 # mata worker-0 (Ray lo reinicia)
python scripts/chaos.py kill-worker 0 --no-restart    # lo mata sin reinicio
python scripts/chaos.py kill-db                       # mata la base de datos (Ray la reinicia y recarga los CSV)
```

Lo que cabe esperar:
- **Cae un worker**: su job queda sin renovar el lease y, al vencer el plazo, la base de datos lo reasigna a otro worker, que continúa desde el último checkpoint. Para verlo, mata el worker mientras hay un job largo en marcha (por ejemplo `--hp epochs=1000 --hp batch_size=2`).
- **Cae la base de datos**: la API responde `503` mientras se reinicia y el cliente reintenta solo con el mismo `client_request_id`. Al volver, recarga los CSV, repara cualquier escritura que hubiera quedado a medias y los jobs en curso siguen.
- **Un job falla**: se reintenta hasta `max_attempts` errores de entrenamiento y `max_lost` pérdidas de contacto; pasado el presupuesto el job queda `FALLIDO`.

Los workers piden trabajo a la base de datos y esta se lo da al primero que lo pide: con jobs enviados de uno en uno, es normal que casi siempre responda el mismo worker. Para verlos repartirse, envía varios jobs largos a la vez.

## Configuración

Valores por defecto en `distriml/common/config.py`; se pueden sobrescribir con variables de entorno:

| Variable | Por defecto | Significado |
|---|---|---|
| `DISTRIML_DATA_DIR` | `./runtime` | Carpeta de la base de datos y los artefactos. |
| `DISTRIML_API_HOST` / `DISTRIML_API_PORT` | `127.0.0.1` / `8080` | Dirección de la API. |
| `DISTRIML_LOG_LEVEL` | `INFO` | Nivel de registro. |
| `DISTRIML_RAY_NAMESPACE` | `distriml` | Espacio de nombres de Ray. |
| `DISTRIML_LEASE_MS` | `10000` | Tiempo que un worker conserva su job sin renovar. |
| `DISTRIML_RENEW_MS` | `3333` | Cada cuánto renueva el worker su lease. |
| `DISTRIML_POLL_MS` | `2000` | Cada cuánto pide trabajo un worker libre. |
| `DISTRIML_MAX_ATTEMPTS` | `5` | Máximo de errores de entrenamiento por job. |
| `DISTRIML_MAX_LOST` | `20` | Máximo de pérdidas de contacto por job. |
| `DISTRIML_K_CKPT` | `1` | Batches entre checkpoints. |
| `DISTRIML_URL` | `http://127.0.0.1:8080` | (Cliente) dirección de la API. |
| `DISTRIML_HOME` | `~/.distriml` | (Cliente) carpeta de la identidad del usuario. |

Para pruebas de caída más rápidas, reduce el lease, por ejemplo `DISTRIML_LEASE_MS=3000 DISTRIML_RENEW_MS=1000`.

## API HTTP

Todas las rutas (salvo `/health`) exigen la cabecera `X-Client-Id` con el identificador del cliente (64 caracteres hexadecimales). Los errores devuelven un cuerpo JSON con `reason` y `message`.

| Método y ruta | Descripción |
|---|---|
| `GET /health` | Estado del servicio. |
| `GET /model-definitions` | Catálogo de algoritmos. |
| `POST /datasets`, `GET /datasets`, `GET /datasets/{id}` | Subir y consultar datasets. |
| `POST /jobs`, `GET /jobs`, `GET /jobs/{id}` | Enviar y consultar jobs. |
| `POST /jobs/{id}/cancel` | Cancelar un job. |
| `GET /jobs/{id}/metrics` | Métricas del job. |
| `GET /models` | Modelos entrenados del cliente. |
| `GET /models/{id}/versions`, `GET /models/{id}/versions/{version_id}` | Versiones de un modelo. |
| `DELETE /models/{id}/versions/{version_id}?client_request_id=...` | Eliminar una versión. |
| `GET /models/{id}/stats` | Estadísticas de inferencia. |
| `POST /predict` | Predicción. |

Las operaciones que modifican datos llevan un `client_request_id` y son idempotentes.

## Pruebas

```bash
python -m pytest -m "not ray"                  # pruebas rápidas, sin cluster Ray
python -m pytest -m ray                        # pruebas con Ray real (más lentas)
python -m pytest                               # todas
python -m pytest tests/test_system_e2e.py -m ray    # sistema completo: Ray + DB + workers + API + cliente
```

Las pruebas que usan Ray necesitan `RAY_AUTH_MODE=disabled`. Los grupos principales:

| Archivos | Qué comprueban |
|---|---|
| `tests/test db/` | La base de datos: planificador, tablas, servicio, workers, persistencia y recuperación, y el actor de Ray. |
| `tests/test_worker_*.py` | El worker: bucle, entrenamiento, leases, abortos, errores y caída de un worker real. |
| `tests/test_client_*.py` | El cliente: peticiones HTTP, identidad, comandos y consola interactiva. |
| `tests/test_contract_live.py` | El contrato de la API, contra una base de datos simulada y contra la real. |
| `tests/test_system_e2e.py` | Todo el sistema real, incluidas las caídas de workers y de la base de datos. |

## Estructura del proyecto

```
distriml/
  api/        API HTTP (FastAPI)
  client/     cliente: consola con menú, comandos, cliente HTTP e identidad
  common/     catálogo de algoritmos, datasets, interfaces, configuración, registro
  db/         base de datos: servicio, planificador, tablas CSV, inferencia
  worker/     workers: bucle de trabajo, entrenamiento, lease y arranque de actores
  cluster.py  lanzador del sistema completo
scripts/      chaos.py (provocar averías), make_sample_datasets.py, measure_ray.py
data/samples/ datasets de ejemplo
tests/        pruebas
```

Cada módulo (`distriml/db`, `distriml/worker`, `distriml/client`) tiene su propio `README.md` con el detalle.

## Limitaciones conocidas

- Los datasets no se pueden eliminar. Los duplicados ya guardados antes de la detección por contenido no se unen solos.
- Cada modelo tiene una única versión; las operaciones sobre varias versiones existen, pero el sistema aún no las crea.
- Los nombres de clase se obtienen de la etiqueta del CSV; para un dataset con etiquetas numéricas, el modelo responde solo con el número.
- Esta documentación cubre el uso en una sola máquina. El diseño admite workers en otros nodos (`--workers 0` y `--ray-address`), pero ese modo no está documentado aquí.
- Ray en Windows nativo es experimental; en Windows se recomienda WSL2.
