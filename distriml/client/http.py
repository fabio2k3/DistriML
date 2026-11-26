"""Cliente HTTP de la plataforma (la consola y los tests de contrato lo usan).

Reintenta ante `503` y errores de conexión REUTILIZANDO el mismo `client_request_id`, que es lo que
hace seguros los reintentos (idempotencia por ids derivados). Un `503` no significa que la
operación no se aplicó: por eso el id no se cambia.
"""
from __future__ import annotations

import base64
import time
from typing import Callable

import httpx

from distriml.common.constants import TERMINAL_STATES
from distriml.common.ids import new_request_id, sha256_hex
from distriml.common.log import get_logger

log = get_logger("client")


class ApiError(Exception):
    def __init__(self, status: int, reason: str, message: str = "", retry_after: float | None = None):
        super().__init__(f"{status} {reason}: {message}")
        self.status, self.reason, self.message, self.retry_after = status, reason, message, retry_after


class ApiClient:
    def __init__(self, base_url: str, client_id: str, *, timeout: float = 30.0, max_retries: int = 5,
                 retry_wait: float = 1.0, last_entry: str | None = None,
                 on_entry: Callable[[str], None] | None = None):
        self.client_id = client_id
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.last_entry = last_entry
        self._on_entry = on_entry
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------ núcleo
    def request(self, method: str, path: str, *, json: dict | None = None, params: dict | None = None,
                headers: dict | None = None):
        hdrs = {"X-Client-Id": self.client_id, **(headers or {})}
        if self.last_entry:
            hdrs["X-Min-Entry"] = self.last_entry
        params = {k: v for k, v in (params or {}).items() if v is not None}
        last_error: ApiError | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._http.request(method, path, json=json, params=params, headers=hdrs)
            except httpx.TransportError as exc:
                last_error = ApiError(503, "unreachable", str(exc))
                log.warning("%s %s -> sin conexión (%s); reintento %d/%d", method, path,
                            type(exc).__name__, attempt + 1, self.max_retries)
                time.sleep(self.retry_wait * (attempt + 1))
                continue
            log.debug("%s %s -> %s  %.1fms", method, path, resp.status_code, (time.perf_counter() - t0) * 1000)
            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict) and body.get("entry_id"):
                    self.last_entry = body["entry_id"]
                    if self._on_entry:
                        self._on_entry(self.last_entry)
                return body
            err = _to_error(resp)
            if resp.status_code == 503 and attempt < self.max_retries:
                last_error = err
                log.warning("%s %s -> 503 %s; reintento %d/%d con el mismo client_request_id", method, path,
                            err.reason, attempt + 1, self.max_retries)
                time.sleep(err.retry_after or self.retry_wait * (attempt + 1))
                continue
            raise err
        raise last_error or ApiError(503, "unreachable", "sin respuesta")

    # ------------------------------------------------------------------ operaciones de la API
    def models(self) -> list:
        return self.request("GET", "/model-definitions")

    def upload_dataset(self, name: str, data: bytes, client_request_id: str | None = None) -> dict:
        return self.request("POST", "/datasets", json={
            "client_request_id": client_request_id or new_request_id(),
            "name": name, "sha256": sha256_hex(data), "size": len(data),
            "data": base64.b64encode(data).decode("ascii"),
        })

    def datasets(self, limit: int = 100, cursor: str | None = None) -> list:
        return self.request("GET", "/datasets", params={"limit": limit, "cursor": cursor})

    def dataset(self, dataset_id: str) -> dict:
        return self.request("GET", f"/datasets/{dataset_id}")

    def submit_job(self, dataset_id: str, model_def_id: str, hyperparams: dict | None = None,
                   client_request_id: str | None = None) -> dict:
        return self.request("POST", "/jobs", json={
            "client_request_id": client_request_id or new_request_id(),
            "dataset_id": dataset_id, "model_def_id": model_def_id, "hyperparams": hyperparams,
        })

    def job(self, job_id: str) -> dict:
        return self.request("GET", f"/jobs/{job_id}")

    def jobs(self, estado: str | None = None, limit: int = 100, cursor: str | None = None) -> list:
        return self.request("GET", "/jobs", params={"estado": estado, "limit": limit, "cursor": cursor})

    def cancel(self, job_id: str, client_request_id: str | None = None) -> dict:
        return self.request("POST", f"/jobs/{job_id}/cancel",
                            json={"client_request_id": client_request_id or new_request_id()})

    def metrics(self, job_id: str) -> dict:
        return self.request("GET", f"/jobs/{job_id}/metrics")

    def trained_models(self, limit: int = 100, cursor: str | None = None) -> list:
        return self.request("GET", "/models", params={"limit": limit, "cursor": cursor})

    def versions(self, model_id: str) -> list:
        return self.request("GET", f"/models/{model_id}/versions")

    def version(self, model_id: str, version_id: str) -> dict:
        return self.request("GET", f"/models/{model_id}/versions/{version_id}")

    def delete_version(self, model_id: str, version_id: str, client_request_id: str | None = None) -> dict:
        """Borra una versión. Si era la única, el modelo desaparece. Los reintentos reutilizan el mismo id."""
        return self.request("DELETE", f"/models/{model_id}/versions/{version_id}",
                            params={"client_request_id": client_request_id or new_request_id()})

    def stats(self, model_id: str, version_id: str | None = None,
              from_minute: int | None = None, to_minute: int | None = None) -> dict:
        return self.request("GET", f"/models/{model_id}/stats",
                            params={"version_id": version_id, "from": from_minute, "to": to_minute})

    def predict(self, model_id: str, features: list[float], version_id: str | None = None,
                version: int | None = None) -> dict:
        return self.request("POST", "/predict", json={
            "model_id": model_id, "version_id": version_id, "version": version,
            "min_entry": self.last_entry, "input": {"features": features},
        })

    # ------------------------------------------------------------------ utilidades
    def wait_job(self, job_id: str, timeout: float = 120.0, poll: float = 0.5) -> dict:
        """Espera a que el job llegue a un estado terminal y devuelve su última consulta."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.job(job_id)
            if job["estado"] in TERMINAL_STATES:
                return job
            if time.monotonic() > deadline:
                raise TimeoutError(f"el job {job_id} sigue en {job['estado']} tras {timeout}s")
            time.sleep(poll)


def _to_error(resp: httpx.Response) -> ApiError:
    try:
        err = resp.json()["error"]
        return ApiError(resp.status_code, err.get("reason", "error"), err.get("message", ""), err.get("retry_after"))
    except Exception:
        return ApiError(resp.status_code, "http_error", resp.text[:200])
