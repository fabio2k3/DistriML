"""Gateway HTTP: expone la API pública del sistema (ver docs/CONTRATO.md).

Es una capa fina: valida la forma de la petición, extrae el `client_id` y delega en un `DBService`
(un actor de Ray a través de `RayDBProxy`, o un FakeDB en las pruebas). No contiene lógica de negocio.

Cabeceras:
  X-Client-Id   obligatoria (hex de 64 caracteres) -> 400 si falta o es inválida
  X-Min-Entry   opcional: identificador de la última escritura del cliente. Se acepta y se ignora,
                porque hay una única DB que siempre está al día.
Errores: {"error": {"status", "reason", "message", "retry_after"?}} con 400/404/409/503.

Registro: cada petición se escribe en el logger `distriml.api` con método, ruta, cliente, código y duración.
"""
from __future__ import annotations

import base64
import binascii
import time

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from distriml.common.errors import ContractError, bad_request
from distriml.common.ids import is_valid_client_id, sha256_hex
from distriml.common.interfaces import DBService
from distriml.common.log import get_logger, short

log = get_logger("api")

_RID = Field(min_length=1, max_length=128)


class DatasetCreate(BaseModel):
    client_request_id: str = _RID
    name: str = Field(min_length=1, max_length=200)
    sha256: str
    size: int = Field(ge=0)
    data: str                                  # base64 del CSV


class JobCreate(BaseModel):
    client_request_id: str = _RID
    dataset_id: str
    model_def_id: str
    hyperparams: dict | None = None


class CancelRequest(BaseModel):
    client_request_id: str = _RID


class PredictRequest(BaseModel):
    model_id: str
    version_id: str | None = None
    version: int | None = None                 # alias entero de la versión canónica
    min_entry: str | None = None
    input: dict


def client_id_dep(x_client_id: str | None = Header(default=None)) -> str:
    if not x_client_id or not is_valid_client_id(x_client_id):
        raise bad_request("invalid_client_id", "falta la cabecera X-Client-Id o no es un hexadecimal de 64 caracteres")
    return x_client_id


def create_app(db: DBService) -> FastAPI:
    app = FastAPI(title="DistriML API", version="1")

    @app.middleware("http")
    async def _log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        who = short(request.headers.get("x-client-id", "-"))
        route = f"{request.method} {request.url.path}"
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001  (se registra y se deja que el manejador general responda)
            log.error("%s cliente=%s -> 500 %s  %.1fms", route, who, type(exc).__name__, (time.perf_counter() - t0) * 1000)
            raise
        ms = (time.perf_counter() - t0) * 1000
        detail = getattr(request.state, "error_reason", "")
        if response.status_code >= 500:
            log.error("%s cliente=%s -> %s %s  %.1fms", route, who, response.status_code, detail, ms)
        elif response.status_code >= 400:
            log.warning("%s cliente=%s -> %s %s  %.1fms", route, who, response.status_code, detail, ms)
        else:
            level = log.debug if request.url.path == "/health" else log.info
            level("%s cliente=%s -> %s  %.1fms", route, who, response.status_code, ms)
        return response

    # ------------------------------------------------------------ errores -> contrato
    @app.exception_handler(ContractError)
    async def _contract_error(request: Request, exc: ContractError):
        request.state.error_reason = f"{exc.reason}: {exc.message}"[:160]
        headers = {"Retry-After": str(int(exc.retry_after) or 1)} if exc.retry_after else None
        return JSONResponse(status_code=exc.status, content=exc.to_body(), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())[:500]
        request.state.error_reason = f"invalid_request: {msg}"[:160]
        return JSONResponse(status_code=400, content=bad_request("invalid_request", msg).to_body())

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        reason = "not_found" if exc.status_code == 404 else "http_error"
        request.state.error_reason = reason
        return JSONResponse(status_code=exc.status_code,
                            content=ContractError(exc.status_code, reason, str(exc.detail)).to_body())

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        log.exception("error interno no controlado")
        return JSONResponse(status_code=500, content=ContractError(500, "internal", type(exc).__name__).to_body())

    # ------------------------------------------------------------ endpoints
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/model-definitions")
    def model_definitions(cid: str = Depends(client_id_dep)):
        return db.list_model_definitions()

    @app.post("/datasets")
    def post_dataset(body: DatasetCreate, cid: str = Depends(client_id_dep)):
        try:
            data = base64.b64decode(body.data, validate=True)
        except (binascii.Error, ValueError):
            raise bad_request("invalid_request", "data no es base64 válido") from None
        if len(data) != body.size:
            raise bad_request("size_mismatch", "size no coincide con el contenido")
        if sha256_hex(data) != body.sha256:
            raise bad_request("hash_mismatch", "sha256 no coincide con el contenido")
        return db.create_dataset(cid, body.client_request_id, body.name, body.sha256, body.size, data)

    @app.get("/datasets")
    def get_datasets(limit: int = Query(100, ge=1, le=500), cursor: str | None = None,
                     cid: str = Depends(client_id_dep)):
        return db.list_datasets(cid, limit, cursor)

    @app.get("/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, cid: str = Depends(client_id_dep)):
        return db.get_dataset(cid, dataset_id)

    @app.post("/jobs")
    def post_job(body: JobCreate, cid: str = Depends(client_id_dep)):
        return db.create_job(cid, body.client_request_id, body.dataset_id, body.model_def_id, body.hyperparams)

    @app.get("/jobs")
    def get_jobs(estado: str | None = None, limit: int = Query(100, ge=1, le=500), cursor: str | None = None,
                 cid: str = Depends(client_id_dep)):
        return db.list_jobs(cid, estado, limit, cursor)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str, consistent: bool = False, cid: str = Depends(client_id_dep)):
        return db.get_job(cid, job_id)       # `consistent` se acepta por compatibilidad: la lectura siempre es la más reciente

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str, body: CancelRequest, cid: str = Depends(client_id_dep)):
        return db.cancel_job(cid, body.client_request_id, job_id)

    @app.get("/jobs/{job_id}/metrics")
    def job_metrics(job_id: str, cid: str = Depends(client_id_dep)):
        return db.get_job_metrics(cid, job_id)

    @app.get("/models")
    def get_models(limit: int = Query(100, ge=1, le=500), cursor: str | None = None,
                   cid: str = Depends(client_id_dep)):
        return db.list_models(cid, limit, cursor)

    @app.get("/models/{model_id}/versions")
    def model_versions(model_id: str, cid: str = Depends(client_id_dep)):
        return db.list_model_versions(cid, model_id)

    @app.get("/models/{model_id}/versions/{version_id}")
    def model_version(model_id: str, version_id: str, cid: str = Depends(client_id_dep)):
        return db.get_model_version(cid, model_id, version_id)

    @app.delete("/models/{model_id}/versions/{version_id}")
    def delete_model_version(model_id: str, version_id: str, client_request_id: str = Query(..., min_length=1, max_length=128),
                             cid: str = Depends(client_id_dep)):
        return db.delete_model_version(cid, client_request_id, model_id, version_id)

    @app.get("/models/{model_id}/stats")
    def model_stats(model_id: str, version_id: str | None = None, from_: int | None = Query(None, alias="from"),
                    to: int | None = None, cid: str = Depends(client_id_dep)):
        return db.get_model_stats(cid, model_id, version_id, from_, to)

    @app.post("/predict")
    def predict(body: PredictRequest, cid: str = Depends(client_id_dep)):
        return db.predict(cid, body.model_id, body.input, body.version_id, body.version)

    return app
