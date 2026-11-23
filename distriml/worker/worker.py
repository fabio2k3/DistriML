"""Worker: proceso que pide jobs a la DB y los entrena.

Es una clase normal; worker/main.py la convierte en actor de Ray. `db` es cualquier objeto que cumpla
`DBService` (un RayDBProxy en el sistema real o un FakeDB en las pruebas):
`Worker("w0", FakeDB(), cfg, autostart=False)` se puede probar sin Ray.

Ciclo de vida:
    register_worker -> bucle: request_job (cada request_poll_interval_ms) -> trainer.run_attempt -> repetir

Concurrencia: el actor admite dos hilos a la vez. El bucle de trabajo corre en un hilo propio que se
lanza en `start()`; el hilo de renovación de lease lo crea `trainer.run_attempt` para cada attempt.

Registro: cada evento relevante se escribe con `self.log` (logger `distriml.worker.<id>`): registro en
la DB, job asignado, batch entrenado/confirmado, checkpoint rechazado, lease perdido, error, fin.

Los métodos marcados TODO(WK-n) están por implementar (orden sugerido en docs/PLAN.md).
"""
from __future__ import annotations

import threading

from distriml.common.config import Config
from distriml.common.errors import ContractError
from distriml.common.interfaces import DBService
from distriml.common.log import get_logger, setup_logging
from distriml.worker import trainer

class Worker:
    def __init__(self, worker_id: str, db: DBService, cfg: Config, autostart: bool = True):
        setup_logging(cfg.log_level)          # el actor corre en su propio proceso: configura su log
        self.log = get_logger(f"worker.{worker_id}")
        self.worker_id = worker_id
        self.db = db
        self.cfg = cfg
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_attempt: str | None = None
        # Un actor reiniciado por Ray vuelve a ejecutar __init__: con autostart reanuda el bucle solo.
        if autostart:
            self.start()

    # ------------------------------------------------------------------ ya implementado
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._main_loop, name=f"{self.worker_id}-loop", daemon=True)
        self._thread.start()
        self.log.info("worker iniciado")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.log.info("worker detenido")

    def status(self) -> dict:
        """Estado actual; se consulta con `ray.get(actor.status.remote())`."""
        return {"worker_id": self.worker_id,
                "running": bool(self._thread and self._thread.is_alive()),
                "current_attempt": self._current_attempt}

    # ------------------------------------------------------------------ por implementar
    def _main_loop(self) -> None:
        """TODO(WK-1) Bucle principal.

        1. Registrarse: wc = self.db.register_worker(self.worker_id). Si la DB responde 503
           (reiniciándose), reintentar con espera: el worker no debe morir por eso.
        2. Mientras no esté `_stop`:
             a = self.db.request_job(self.worker_id)
             si a es None: dormir cfg.request_poll_interval_ms y continuar
             si hay trabajo: self._current_attempt = a.attempt_id
                              trainer.run_attempt(a, self.worker_id, self.db, self.cfg)
                              self._current_attempt = None
        3. Capturar ContractError (503) y errores inesperados sin matar el hilo: registrar con
           self.log.warning / self.log.exception y seguir.
        """
        raise NotImplementedError("TODO(WK-1)")
# ------------------------------------------------------------------ WK-1
    def _register(self) -> bool:
        """Se registra en la DB reintentando mientras responda 503 (p. ej. la DB se está reiniciando).

        Devuelve True si quedó registrado y False si se pidió parar antes de conseguirlo.
        """
        poll = self.cfg.request_poll_interval_ms / 1000
        while not self._stop.is_set():
            try:
                wc = self.db.register_worker(self.worker_id)
                self.log.info("registrado en la DB (renovar cada %d ms, lease %d ms)",
                              wc.renew_interval_ms, wc.lease_duration_ms)
                return True
            except ContractError as exc:
                self.log.warning("no se pudo registrar (%s %s); reintento en %.1fs", exc.status, exc.reason, poll)
            except Exception:  # noqa: BLE001  (el bucle no debe morir por un error inesperado)
                self.log.exception("error inesperado al registrarse; reintento en %.1fs", poll)
            self._stop.wait(poll)
        return False

    def _main_loop(self) -> None:
        """Bucle principal: registro, y luego pedir un job, entrenarlo y repetir.

        - Si la DB responde 503 (caída o reinicio) espera y reintenta; si la DB estaba caída
          (`no_ca`) vuelve a registrarse, porque parte de su estado es volátil.
        - Ninguna excepción mata el hilo: se registra y se sigue.
        """
        poll = self.cfg.request_poll_interval_ms / 1000
        if not self._register():
            return
        while not self._stop.is_set():
            try:
                a = self.db.request_job(self.worker_id)
                if a is None:
                    self._stop.wait(poll)
                    continue
                self._current_attempt = a.attempt_id
                self.log.info("job %s asignado: attempt %s, batches %d..%d de %d",
                              a.job_id[:8], a.attempt_id[:8], a.next_batch, a.total_batches, a.total_batches)
                try:
                    result = trainer.run_attempt(a, self.worker_id, self.db, self.cfg)
                    self.log.info("attempt %s terminado: %s", a.attempt_id[:8], result)
                finally:
                    self._current_attempt = None
            except ContractError as exc:
                self.log.warning("la DB respondió %s %s; reintento en %.1fs", exc.status, exc.reason, poll)
                if exc.reason == "no_ca" and not self._register():
                    return
                self._stop.wait(poll)
            except Exception:  # noqa: BLE001
                self.log.exception("error inesperado en el bucle; sigo en %.1fs", poll)
                self._stop.wait(poll)
