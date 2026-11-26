"""Política de asignación de jobs y de leases (reglas puras, sin Ray ni CSV).

Separarlo de `service.py` permite probar las reglas aisladamente. La revisión de leases no usa un hilo:
`Database` la ejecuta de forma perezosa al inicio de request_job, renew_lease, get_job y otras
operaciones (el actor ya serializa las llamadas, así que no hay carreras).
"""
from __future__ import annotations

from typing import Callable

from distriml.common import constants as C


def lease_deadline_ms(last_renew_ms: int, lease_duration_ms: int, epsilon_ms: int) -> int:
    """Instante a partir del cual la DB puede dar por perdido a un worker:
    última renovación + duración del lease + margen.

    La DB nunca revoca antes de ese plazo: así un worker vivo ve vencer su propio lease antes
    (o a la vez) que la DB lo revoque.
    """
    return last_renew_ms + lease_duration_ms + epsilon_ms


def revoke_reason_for_expired(attempt_started: bool) -> str:
    """Motivo de revocación de un lease vencido: WORKER_LOST si el worker llegó a renovar o confirmar
    al menos una vez; NEVER_STARTED si nunca lo hizo (no cuenta para ningún presupuesto)."""
    return C.WORKER_LOST if attempt_started else C.NEVER_STARTED


def next_state_after_revocation(reason: str, attempts_count: int, lost_count: int,
                                max_attempts: int, max_lost: int) -> tuple[str, int, int]:
    """Devuelve (nuevo_estado_del_job, attempts_count, lost_count) tras una revocación.

    WORKER_ERROR  -> attempts_count + 1;  FALLIDO si attempts_count >= max_attempts, si no PENDIENTE
    WORKER_LOST   -> lost_count + 1;      FALLIDO si lost_count >= max_lost,         si no PENDIENTE
    NEVER_STARTED / ARTIFACT_UNAVAILABLE -> PENDIENTE sin tocar contadores
    CANCEL        -> CANCELADO
    """
    if reason == C.CANCEL:
        return C.CANCELADO, attempts_count, lost_count
    if reason == C.WORKER_ERROR:
        attempts_count += 1
        return (C.FALLIDO if attempts_count >= max_attempts else C.PENDIENTE), attempts_count, lost_count
    if reason == C.WORKER_LOST:
        lost_count += 1
        return (C.FALLIDO if lost_count >= max_lost else C.PENDIENTE), attempts_count, lost_count
    if reason in (C.NEVER_STARTED, C.ARTIFACT_UNAVAILABLE):
        return C.PENDIENTE, attempts_count, lost_count
    raise ValueError(f"motivo de revocación desconocido: {reason!r}")


def backoff_ms(lost_count: int, t_sync_ms: int, cap_ms: int = 60_000) -> int:
    """Espera antes de reasignar un job tras perder a su worker: min(2^lost_count × t_sync, 60 s)."""
    return min((2 ** max(lost_count, 0)) * t_sync_ms, cap_ms)


def pick_next_job(pending_jobs: list[dict], now_ms: int, not_before_ms: dict[str, int],
                  is_startable: Callable[[dict], bool] | None = None) -> dict | None:
    """Elige el job PENDIENTE más antiguo (por `created_at`; a igualdad, el primero de la lista) cuyo
    retardo de reasignación ya venció (`not_before_ms[job_id] <= now_ms`).

    `is_startable(job)` permite descartar jobs que aún no pueden arrancar (p. ej. porque el dataset o el
    checkpoint de partida no están en el almacén); esos jobs se saltan y siguen pendientes.
    """
    for job in sorted(pending_jobs, key=lambda j: j.get("created_at") or 0):
        if not_before_ms.get(job["job_id"], 0) > now_ms:
            continue
        if is_startable is not None and not is_startable(job):
            continue
        return job
    return None
