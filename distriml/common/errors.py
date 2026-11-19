"""Errores del contrato externo (ver docs/CONTRATO.md).

El cliente solo ve 200, 400, 404, 409 y 503 (y nunca redirecciones). Toda capa interna
(DB, workers, gateway) levanta `ContractError` para señalar uno de esos casos; el gateway
lo traduce a HTTP. Es picklable para que viaje a través de Ray sin perder sus campos.
"""
from __future__ import annotations


class ContractError(Exception):
    def __init__(self, status: int, reason: str, message: str = "", retry_after: float | None = None):
        super().__init__(f"{status} {reason}: {message}")
        self.status = status
        self.reason = reason
        self.message = message
        self.retry_after = retry_after

    def __reduce__(self):  # necesario para que sobreviva a pickle/Ray
        return (self.__class__, (self.status, self.reason, self.message, self.retry_after))

    def to_body(self) -> dict:
        err = {"status": self.status, "reason": self.reason, "message": self.message}
        if self.retry_after is not None:
            err["retry_after"] = self.retry_after
        return {"error": err}


def bad_request(reason: str, message: str = "") -> ContractError:
    return ContractError(400, reason, message)


def not_found(message: str = "not found") -> ContractError:
    # Un recurso ajeno se trata igual que uno inexistente: no revela su existencia.
    return ContractError(404, "not_found", message)


def conflict(reason: str, message: str = "") -> ContractError:
    return ContractError(409, reason, message)


def unavailable(reason: str, message: str = "", retry_after: float = 2.0) -> ContractError:
    # reason ∈ no_ca | no_confirm | dataset_unavailable | artifact_unavailable | stale_replica
    return ContractError(503, reason, message, retry_after)
