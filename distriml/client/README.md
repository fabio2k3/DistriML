# Módulo `client` — consola y cliente HTTP

**Responsabilidad:** es lo que usa el usuario final. Traduce comandos de consola a peticiones HTTP contra el gateway,
gestiona la identidad local y reintenta de forma segura. No contiene lógica de negocio ni conoce la topología del sistema:
solo necesita la URL del gateway (`--url` o `DISTRIML_URL`).

## Archivos
| Archivo | Contenido |
|---|---|
| `cli.py` | Comando `distriml` (argparse): un subcomando por operación de la API. |
| `http.py` | `ApiClient`: una petición por operación, reintentos y espera de jobs. |
| `identity.py` | Identidad local: `username` (etiqueta) + `client_id` aleatorio en `~/.distriml/identity.json`. |
| `__main__.py` | Permite `python -m distriml.client`. |

## Funcionalidades
- **Identidad:** `distriml init <username>` crea un `client_id` aleatorio (SHA-256 de username + sal descartada). Si se pierde el archivo se pierde el acceso a los datos.
- **Comandos:** `models`, `upload`, `datasets`, `submit --hp clave=valor`, `jobs`, `job --watch`, `cancel`, `metrics`, `versions`, `stats`, `predict`.
- **Reintentos seguros:** ante `503` o error de conexión repite la petición con el **mismo** `client_request_id`, de modo que no se duplican datasets ni jobs.
- **Seguimiento de escrituras:** guarda el último `entry_id` recibido y lo reenvía en `X-Min-Entry`.
- **Esperas:** `ApiClient.wait_job()` y `distriml job <id> --watch` consultan hasta que el job termina.

## Registro
`distriml -v <comando>` muestra cada petición HTTP (método, ruta, código, duración) y cada reintento en consola
(logger `distriml.client`).

## Cómo probarlo
`pytest tests/test_contract_live.py` usa `ApiClient` contra un gateway real o en memoria.
