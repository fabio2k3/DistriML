"""Consola interactiva con menú: `distriml app` (o `python -m distriml.client` sin argumentos).

El menú con todas las operaciones se muestra SIEMPRE; el usuario elige por número y escoge datasets, jobs y
modelos de listas numeradas, sin copiar identificadores. Es una capa fina sobre `ApiClient`: no añade
operaciones nuevas, solo las presenta de forma guiada.

Para poder probarla sin teclado, la entrada y la salida son inyectables (`ask` y `out`).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

from distriml.client import identity
from distriml.client.http import ApiClient, ApiError
from distriml.common.constants import TERMINAL_STATES

MENU = [
    ("1", "Ver algoritmos disponibles", "_catalog"),
    ("2", "Subir un dataset (CSV)", "_upload"),
    ("3", "Mis datasets", "_datasets"),
    ("4", "Entrenar un modelo (enviar un job)", "_train"),
    ("5", "Mis jobs", "_jobs"),
    ("6", "Ver / seguir un job", "_job"),
    ("7", "Cancelar un job", "_cancel"),
    ("8", "Mis modelos entrenados", "_models"),
    ("9", "Hacer una predicción", "_predict"),
    ("10", "Detalle de un modelo (versiones y estadísticas)", "_model_detail"),
    ("11", "Eliminar una versión de un modelo", "_delete_version"),
    ("0", "Salir", None),
]


class Cancelled(Exception):
    """El usuario dejó una pregunta en blanco o escribió 0: vuelve al menú."""


def _short(value: str, n: int = 8) -> str:
    return value[:n]


def _metrics_text(metrics: dict | None) -> str:
    if not metrics:
        return "-"
    return "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())


def parse_hyperparams(text: str) -> dict:
    """'epochs=5 batch_size=50' -> {'epochs': 5, 'batch_size': 50}. ValueError si algún trozo no es clave=valor."""
    out: dict = {}
    for item in text.replace(",", " ").split():
        if "=" not in item:
            raise ValueError(f"'{item}' no es clave=valor (ejemplo: epochs=5)")
        key, raw = item.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def parse_features(text: str) -> list[float]:
    try:
        values = [float(x) for x in text.replace(";", ",").replace(" ", ",").split(",") if x.strip()]
    except ValueError:
        raise ValueError("las características deben ser números separados por comas (ejemplo: 5.1,3.5,1.4,0.2)") from None
    if not values:
        raise ValueError("escribe al menos un número")
    return values


class Console:
    def __init__(self, client: ApiClient, username: str, url: str,
                 ask: Callable[[str], str] = input, out: Callable[[str], None] = print,
                 poll: float = 1.0):
        self.client, self.username, self.url = client, username, url
        self.ask, self.out, self.poll = ask, out, poll
        self._algorithms: dict[str, dict] | None = None

    # ------------------------------------------------------------------ bucle principal
    def run(self) -> int:
        while True:
            self._show_menu()
            try:
                choice = self.ask("Elige una opción: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.out("")
                return 0
            if choice in ("0", "q", "salir"):
                self.out("Hasta luego.")
                return 0
            action = next((a for key, _, a in MENU if key == choice and a), None)
            if action is None:
                self.out(f"\nOpción '{choice}' no válida: escribe un número del menú.")
                continue
            try:
                getattr(self, action)()
            except Cancelled:
                self.out("(cancelado)")
            except KeyboardInterrupt:
                self.out("\n(interrumpido)")
            except ApiError as exc:
                self.out(f"\nERROR {exc.status} {exc.reason}: {exc.message}")
                if exc.status == 503:
                    self.out("No hay conexión con el cluster. ¿Está levantado? (python -m distriml.cluster --workers 3)")
            except (ValueError, OSError, TimeoutError) as exc:
                self.out(f"\nERROR: {exc}")

    def _show_menu(self) -> None:
        self.out("")
        self.out("=" * 60)
        self.out(f"  DistriML   usuario: {self.username}   gateway: {self.url}")
        self.out("=" * 60)
        for key, label, _ in MENU:
            self.out(f"  {key:>2}  {label}")
        self.out("-" * 60)

    # ------------------------------------------------------------------ utilidades de presentación
    def _table(self, headers: list[str], rows: list[list]) -> None:
        cells = [[str(c) for c in r] for r in rows]
        widths = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
        self.out("  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        self.out("  " + "  ".join("-" * w for w in widths))
        for r in cells:
            self.out("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))

    def _pick(self, items: list, title: str, row: Callable, headers: list[str], empty: str):
        """Muestra una lista numerada y devuelve el elemento elegido. Enter o 0 cancelan."""
        if not items:
            self.out(f"\n{empty}")
            raise Cancelled
        self.out(f"\n{title}")
        self._table(["#"] + headers, [[i + 1] + row(it) for i, it in enumerate(items)])
        while True:
            raw = self.ask("Número (Enter para cancelar): ").strip()
            if raw in ("", "0"):
                raise Cancelled
            if raw.isdigit() and 1 <= int(raw) <= len(items):
                return items[int(raw) - 1]
            self.out(f"Escribe un número entre 1 y {len(items)}.")

    def _algorithm_names(self) -> dict[str, dict]:
        if self._algorithms is None:
            self._algorithms = {d["model_def_id"]: d for d in self.client.models()}
        return self._algorithms

    def _algo_of(self, model_def_id: str) -> str:
        d = self._algorithm_names().get(model_def_id)
        return d["algorithm"] if d else _short(model_def_id)

    def _progress(self, job: dict) -> str:
        return f"{job['last_confirmed_batch']}/{job['total_batches']}"

    # ------------------------------------------------------------------ operaciones
    def _catalog(self) -> None:
        defs = list(self._algorithm_names().values())
        self.out("\nAlgoritmos disponibles:")
        self._table(["algoritmo", "descripción", "hiperparámetros por defecto"],
                    [[d["algorithm"], d["name"], " ".join(f"{k}={v}" for k, v in d["default_hyperparams"].items())]
                     for d in defs])

    def _upload(self) -> None:
        raw = self.ask("Ruta del archivo CSV (la última columna es la etiqueta; Enter para cancelar): ").strip().strip("'\"")
        if not raw:
            raise Cancelled
        path = Path(raw).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no existe el archivo '{raw}'")
        res = self.client.upload_dataset(path.name, path.read_bytes())
        if res.get("already_existed"):
            self.out(f"\nEse dataset ya estaba subido ({_short(res['dataset_id'])}): no se guarda otra copia, se usará el que ya tienes.")
        else:
            self.out(f"\nDataset subido: {path.name}  ({res.get('n_rows', '?')} filas, {res.get('n_features', '?')} características)")
        if res.get("class_names"):
            self.out("Clases: " + ", ".join(f"{i} = {n}" for i, n in enumerate(res["class_names"])))
        self.out("Ya puedes entrenar con él (opción 4).")

    def _datasets(self) -> None:
        data = self.client.datasets()
        if not data:
            self.out("\nAún no has subido datasets (opción 2).")
            return
        self.out("\nTus datasets:")
        self._table(["#", "nombre", "bytes", "id"], [[i + 1, d["name"], d["size"], _short(d["dataset_id"])] for i, d in enumerate(data)])

    def _train(self) -> None:
        dataset = self._pick(self.client.datasets(), "¿Con qué dataset?", lambda d: [d["name"], d["size"]],
                             ["nombre", "bytes"], "No tienes datasets: sube uno primero (opción 2).")
        algo = self._pick(list(self._algorithm_names().values()), "¿Qué algoritmo?",
                          lambda d: [d["algorithm"], d["name"]], ["algoritmo", "descripción"], "No hay algoritmos disponibles.")
        defaults = " ".join(f"{k}={v}" for k, v in algo["default_hyperparams"].items())
        self.out(f"\nHiperparámetros por defecto: {defaults}")
        raw = self.ask("Cambia los que quieras (ej: epochs=5 batch_size=50) o Enter para usar los de defecto: ")
        hp = parse_hyperparams(raw) if raw.strip() else None
        res = self.client.submit_job(dataset["dataset_id"], algo["model_def_id"], hp)
        self.out(f"\nJob enviado: {_short(res['job_id'])}  ({algo['algorithm']} sobre {dataset['name']})")
        if self.ask("¿Esperar a que termine? [S/n]: ").strip().lower() in ("", "s", "si", "sí", "y"):
            self._follow(res["job_id"])
        else:
            self.out("Puedes seguirlo luego con la opción 6.")

    def _jobs(self) -> None:
        jobs = self.client.jobs()
        if not jobs:
            self.out("\nAún no has enviado jobs (opción 4).")
            return
        self.out("\nTus jobs:")
        self._table(["#", "estado", "algoritmo", "progreso", "id"],
                    [[i + 1, j["estado"], self._algo_of(j["model_def_id"]), self._progress(j), _short(j["job_id"])]
                     for i, j in enumerate(jobs)])

    def _pick_job(self, title: str, only_active: bool = False) -> dict:
        jobs = self.client.jobs()
        if only_active:
            jobs = [j for j in jobs if j["estado"] not in TERMINAL_STATES]
        return self._pick(jobs, title, lambda j: [j["estado"], self._algo_of(j["model_def_id"]), self._progress(j)],
                          ["estado", "algoritmo", "progreso"],
                          "No tienes jobs en curso." if only_active else "Aún no has enviado jobs (opción 4).")

    def _follow(self, job_id: str) -> dict:
        """Muestra el avance hasta que el job termina. Ctrl+C deja de mirar, pero NO cancela el job."""
        self.out("(Ctrl+C para dejar de esperar; el job sigue en el cluster)")
        last = None
        try:
            while True:
                job = self.client.job(job_id)
                state = (job["estado"], job["last_confirmed_batch"])
                if state != last:
                    self.out(f"  {job['estado']:<12} batches {self._progress(job)}")
                    last = state
                if job["estado"] in TERMINAL_STATES:
                    break
                time.sleep(self.poll)
        except KeyboardInterrupt:
            self.out("\nDejas de esperar. El job sigue; consúltalo con la opción 6.")
            raise Cancelled from None
        self._summarize(job)
        return job

    def _summarize(self, job: dict) -> None:
        self.out("")
        if job["estado"] == "COMPLETADO":
            metrics = self.client.metrics(job["job_id"]).get("final")
            self.out(f"Entrenamiento terminado. Métricas finales: {_metrics_text(metrics)}")
            self.out("Tu modelo ya está en 'Mis modelos entrenados' (opción 8) y listo para predecir (opción 9).")
        elif job["estado"] == "FALLIDO":
            reasons = [a["revoke_reason"] for a in job.get("attempts", []) if a.get("revoke_reason")]
            self.out(f"El job falló tras {len(job.get('attempts', []))} intentos. Último motivo: {reasons[-1] if reasons else 'desconocido'}.")
        else:
            self.out(f"El job terminó como {job['estado']}.")

    def _job(self) -> None:
        job = self._pick_job("¿Qué job quieres ver?")
        job = self.client.job(job["job_id"])
        self.out(f"\nJob {_short(job['job_id'])}: {job['estado']}  progreso {self._progress(job)}  "
                 f"algoritmo {self._algo_of(job['model_def_id'])}")
        if job["estado"] in TERMINAL_STATES:
            self._summarize(job)
        else:
            self._follow(job["job_id"])

    def _cancel(self) -> None:
        job = self._pick_job("¿Qué job quieres cancelar?", only_active=True)
        if self.ask(f"¿Cancelar el job {_short(job['job_id'])}? [s/N]: ").strip().lower() not in ("s", "si", "sí", "y"):
            raise Cancelled
        self.client.cancel(job["job_id"])
        self.out("Job cancelado.")

    def _models(self) -> None:
        models = self.client.trained_models()
        if not models:
            self.out("\nAún no tienes modelos entrenados: entrena uno con la opción 4.")
            return
        self.out("\nTus modelos entrenados:")
        self._table(["#", "algoritmo", "versiones", "métricas", "id"],
                    [[i + 1, self._algo_of(m["model_def_id"]), m["versions"], _metrics_text(m["metrics"]), _short(m["model_id"])]
                     for i, m in enumerate(models)])

    def _pick_model(self, title: str) -> dict:
        return self._pick(self.client.trained_models(), title,
                          lambda m: [self._algo_of(m["model_def_id"]), m["versions"], _metrics_text(m["metrics"])],
                          ["algoritmo", "versiones", "métricas"],
                          "Aún no tienes modelos entrenados: entrena uno con la opción 4.")

    def _predict(self) -> None:
        model = self._pick_model("¿Con qué modelo quieres predecir?")
        self.out("\nEscribe las características separadas por comas (ejemplo iris: 5.1,3.5,1.4,0.2).")
        while True:
            raw = self.ask("Características (Enter para terminar): ").strip()
            if not raw:
                return
            try:
                res = self.client.predict(model["model_id"], parse_features(raw))
            except ApiError as exc:
                self.out(f"  ERROR {exc.status} {exc.reason}: {exc.message}")
                continue
            except ValueError as exc:
                self.out(f"  {exc}")
                continue
            name = f" = {res['label']}" if res.get("label") else ""
            self.out(f"  Predicción: {res['prediction']}{name}   (versión {_short(res['version_id'])})")

    def _delete_version(self) -> None:
        model = self._pick_model("¿De qué modelo quieres eliminar una versión?")
        versions = self.client.versions(model["model_id"])
        version = (versions[0] if len(versions) == 1 else
                   self._pick(versions, "¿Qué versión?", lambda v: [v["version"], "sí" if v["canonical"] else "no", _metrics_text(v["metrics"])],
                              ["versión", "canónica", "métricas"], "Este modelo no tiene versiones."))
        name = self._algo_of(model["model_def_id"])
        aviso = " Es la única versión: el modelo dejará de existir y ya no podrás predecir con él." if len(versions) == 1 else ""
        self.out(f"\nVas a eliminar la versión {version['version']} del modelo {_short(model['model_id'])} ({name}).{aviso}")
        if self.ask("Esta acción no se puede deshacer. ¿Seguro? [s/N]: ").strip().lower() not in ("s", "si", "sí", "y"):
            raise Cancelled
        res = self.client.delete_version(model["model_id"], version["version_id"])
        self.out("Versión eliminada." + (" El modelo ya no existe." if res["model_deleted"] else ""))

    def _model_detail(self) -> None:
        model = self._pick_model("¿De qué modelo?")
        versions = self.client.versions(model["model_id"])
        self.out(f"\nModelo {_short(model['model_id'])} ({self._algo_of(model['model_def_id'])}), job {_short(model['job_id'])}")
        self._table(["versión", "canónica", "estado", "métricas"],
                    [[v["version"], "sí" if v["canonical"] else "no", v["estado"], _metrics_text(v["metrics"])] for v in versions])
        stats = self.client.stats(model["model_id"])
        self.out(f"\nInferencias: {stats['total_calls']}   latencia media: {stats['mean_latency_ms']:.2f} ms")


def run_console(url: str, ask: Callable[[str], str] = input, out: Callable[[str], None] = print) -> int:
    try:
        ident = identity.load_identity()
    except identity.NotInitialized:
        out("Primera vez: vamos a crear tu usuario.")
        try:
            name = ask("Elige un nombre de usuario: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 1
        if not name:
            out("El nombre no puede estar vacío.")
            return 2
        ident = identity.init_identity(name)
    client = ApiClient(url, ident["client_id"], last_entry=ident.get("last_entry"), on_entry=identity.save_last_entry)
    try:
        return Console(client, ident["username"], url, ask=ask, out=out).run()
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(run_console("http://127.0.0.1:8080"))
