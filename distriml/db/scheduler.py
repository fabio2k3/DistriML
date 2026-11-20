"""Política de asignación de jobs y de leases (reglas puras, sin Ray ni CSV).

Separarlo de `service.py` permite probar las reglas aisladamente. Sugerencia: no usar un hilo que
revise leases; llamar a la revisión de forma perezosa al inicio de request_job, renew_lease y get_job
(el actor ya serializa las llamadas, así que no hay carreras).
"""
from __future__ import annotations


def lease_deadline_ms(last_renew_ms: int, lease_duration_ms: int, epsilon_ms: int) -> int:
    """Instante a partir del cual la DB puede dar por perdido a un worker:
    última renovación + duración del lease + margen.

    La DB nunca revoca antes de ese plazo: así un worker vivo ve vencer su propio lease antes
    (o a la vez) que la DB lo revoque.
    """
    return last_renew_ms + lease_duration_ms + epsilon_ms


def revoke_reason_for_expired(attempt_started: bool) -> str:
    """TODO(DB-6) Motivo de revocación de un lease vencido: WORKER_LOST si el worker llegó a renovar o
    confirmar al menos una vez; NEVER_STARTED si nunca lo hizo (no cuenta para ningún presupuesto)."""
    raise NotImplementedError("TODO(DB-6)")


def next_state_after_revocation(reason: str, attempts_count: int, lost_count: int,
                                max_attempts: int, max_lost: int) -> tuple[str, int, int]:
    """TODO(DB-6) Devuelve (nuevo_estado_del_job, attempts_count, lost_count) tras una revocación.

    WORKER_ERROR  -> attempts_count + 1;  FALLIDO si attempts_count >= max_attempts, si no PENDIENTE
    WORKER_LOST   -> lost_count + 1;      FALLIDO si lost_count >= max_lost,         si no PENDIENTE
    NEVER_STARTED / ARTIFACT_UNAVAILABLE -> PENDIENTE sin tocar contadores
    CANCEL        -> CANCELADO
    """
    raise NotImplementedError("TODO(DB-6)")


def backoff_ms(lost_count: int, t_sync_ms: int, cap_ms: int = 60_000) -> int:
    """Espera antes de reasignar un job tras perder a su worker: min(2^lost_count × t_sync, 60 s)."""
    return min((2 ** max(lost_count, 0)) * t_sync_ms, cap_ms)


def pick_next_job(pending_jobs: list[dict], now_ms: int, not_before_ms: dict[str, int]) -> dict | None:
    """TODO(DB-4) Elige el job PENDIENTE más antiguo cuyo backoff ya venció (`not_before_ms[job_id]`).
    Solo se asigna si el dataset (y el checkpoint de partida) están disponibles en el almacén."""
    raise NotImplementedError("TODO(DB-4)")
