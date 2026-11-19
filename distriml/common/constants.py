"""Constantes compartidas: estados y motivos de revocación."""

# Estados de un job
PENDIENTE = "PENDIENTE"
EN_EJECUCION = "EN_EJECUCION"
COMPLETADO = "COMPLETADO"
FALLIDO = "FALLIDO"
CANCELADO = "CANCELADO"
JOB_STATES = (PENDIENTE, EN_EJECUCION, COMPLETADO, FALLIDO, CANCELADO)
TERMINAL_STATES = frozenset({COMPLETADO, FALLIDO, CANCELADO})

# Estados de un attempt
ATTEMPT_ACTIVO = "ACTIVO"
ATTEMPT_REVOCADO = "REVOCADO"

# Motivos de revocación
#   WORKER_ERROR         el worker reportó un fallo de entrenamiento       -> cuenta en attempts_count (max_attempts)
#   WORKER_LOST          lease vencido / sin contacto                      -> cuenta en lost_count (max_lost)
#   NEVER_STARTED        el worker nunca renovó ni confirmó un batch      -> no cuenta
#   CANCEL               el dueño canceló el job                          -> no cuenta
#   ARTIFACT_UNAVAILABLE el worker no pudo obtener un artefacto           -> no cuenta
WORKER_ERROR = "WORKER_ERROR"
WORKER_LOST = "WORKER_LOST"
NEVER_STARTED = "NEVER_STARTED"
CANCEL = "CANCEL"
ARTIFACT_UNAVAILABLE = "ARTIFACT_UNAVAILABLE"
REVOKE_REASONS = (WORKER_ERROR, WORKER_LOST, NEVER_STARTED, CANCEL, ARTIFACT_UNAVAILABLE)

# Estados de un batch y de una versión de modelo
BATCH_CONFIRMED = "CONFIRMED"
BATCH_DISCARDED = "DISCARDED"
VERSION_ENTRENADO = "ENTRENADO"
VERSION_ORPHANED = "ORPHANED"

# Respuestas de las operaciones de worker
RENEW_CONTINUE = "CONTINUE"
RENEW_REVOKED = "REVOKED"
CKPT_CONFIRMED = "CONFIRMED"
CKPT_RETRY = "RETRY"
CKPT_REJECTED = "REJECTED"
COMPLETE_OK = "OK"
COMPLETE_REJECTED = "REJECTED"
