"""Consola de usuario: `distriml <comando>` (o `python -m distriml.client <comando>`); sin comando abre el menú interactivo.

El usuario no conoce la topología: solo la URL del gateway (--url o $DISTRIML_URL).
La identidad (username local + client_id aleatorio) se crea una vez con `init`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from distriml.client import identity
from distriml.client.http import ApiClient, ApiError
from distriml.common.log import setup_logging

DEFAULT_URL = "http://127.0.0.1:8080"


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _parse_hp(items: list[str] | None) -> dict | None:
    if not items:
        return None
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"hiperparámetro inválido '{item}': usa clave=valor")
        key, raw = item.split("=", 1)
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            out[key] = raw
    return out


def _client(args) -> ApiClient:
    ident = identity.load_identity()
    return ApiClient(args.url, ident["client_id"], last_entry=ident.get("last_entry"),
                     on_entry=identity.save_last_entry)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="distriml", description="Consola de DistriML")
    p.add_argument("--url", default=os.environ.get("DISTRIML_URL", DEFAULT_URL), help="URL del gateway")
    p.add_argument("-v", "--verbose", action="store_true", help="muestra cada petición HTTP y sus reintentos")
    sub = p.add_subparsers(dest="cmd")           # sin comando se abre la consola interactiva

    sub.add_parser("app", help="consola interactiva con menú (es lo que se abre si no escribes ningún comando)")

    s = sub.add_parser("init", help="crea tu identidad local")
    s.add_argument("username")
    s.add_argument("--force", action="store_true")

    s = sub.add_parser("whoami", help="muestra tu username (y el client_id con --show-id)")
    s.add_argument("--show-id", action="store_true")

    s = sub.add_parser("models", help="lista el catálogo de modelos (con --mine, tus modelos entrenados)")
    s.add_argument("--mine", action="store_true", help="lista tus modelos ya entrenados (los que sirven para predict)")

    s = sub.add_parser("upload", help="sube un dataset CSV (última columna = etiqueta)")
    s.add_argument("path")
    s.add_argument("--name")
    s.add_argument("--request-id", help="reutiliza un client_request_id para reintentar")

    sub.add_parser("datasets", help="lista tus datasets")

    s = sub.add_parser("submit", help="envía un job de entrenamiento")
    s.add_argument("--dataset", required=True)
    s.add_argument("--model", required=True, help="model_def_id o algoritmo (p. ej. sgd_logistic)")
    s.add_argument("--hp", action="append", help="hiperparámetro clave=valor (repetible)")
    s.add_argument("--request-id")

    s = sub.add_parser("jobs", help="lista tus jobs")
    s.add_argument("--estado")

    s = sub.add_parser("job", help="consulta un job")
    s.add_argument("job_id")
    s.add_argument("--watch", action="store_true", help="espera a que termine")

    s = sub.add_parser("cancel", help="cancela un job")
    s.add_argument("job_id")
    s.add_argument("--request-id")

    s = sub.add_parser("metrics", help="métricas por batch y finales de un job")
    s.add_argument("job_id")

    s = sub.add_parser("versions", help="versiones de un modelo")
    s.add_argument("model_id")
    s.add_argument("--version-id")

    s = sub.add_parser("delete-version", help="borra una versión de un modelo (si es la única, el modelo desaparece)")
    s.add_argument("model_id")
    s.add_argument("version_id")
    s.add_argument("--request-id")

    s = sub.add_parser("stats", help="estadísticas de inferencia de un modelo")
    s.add_argument("model_id")
    s.add_argument("--version-id")

    s = sub.add_parser("predict", help="predicción con un modelo propio")
    s.add_argument("model_id")
    s.add_argument("--features", required=True, help="lista separada por comas: 5.1,3.5,1.4,0.2")
    s.add_argument("--version-id")
    return p


def _resolve_model(client: ApiClient, value: str) -> str:
    for d in client.models():
        if value in (d["model_def_id"], d["algorithm"]):
            return d["model_def_id"]
    return value  # que el servidor responda 400 si no existe


def run(args) -> int:
    if args.cmd in (None, "app"):
        from distriml.client.console import run_console
        return run_console(args.url)
    if args.cmd == "init":
        data = identity.init_identity(args.username, force=args.force)
        print(f"identidad creada para '{data['username']}'")
        return 0
    if args.cmd == "whoami":
        ident = identity.load_identity()
        _print({"username": ident["username"], **({"client_id": ident["client_id"]} if args.show_id else {})})
        return 0

    client = _client(args)
    try:
        if args.cmd == "models":
            _print(client.trained_models() if args.mine else client.models())
        elif args.cmd == "upload":
            path = Path(args.path)
            _print(client.upload_dataset(args.name or path.name, path.read_bytes(), args.request_id))
        elif args.cmd == "datasets":
            _print(client.datasets())
        elif args.cmd == "submit":
            _print(client.submit_job(args.dataset, _resolve_model(client, args.model),
                                     _parse_hp(args.hp), args.request_id))
        elif args.cmd == "jobs":
            _print(client.jobs(args.estado))
        elif args.cmd == "job":
            if args.watch:
                while True:
                    job = client.job(args.job_id)
                    print(f"{job['estado']:<13} {job['last_confirmed_batch']}/{job['total_batches']}", file=sys.stderr)
                    if job["estado"] in ("COMPLETADO", "FALLIDO", "CANCELADO"):
                        break
                    time.sleep(1.0)
                _print(job)
            else:
                _print(client.job(args.job_id))
        elif args.cmd == "cancel":
            _print(client.cancel(args.job_id, args.request_id))
        elif args.cmd == "metrics":
            _print(client.metrics(args.job_id))
        elif args.cmd == "versions":
            _print(client.version(args.model_id, args.version_id) if args.version_id else client.versions(args.model_id))
        elif args.cmd == "delete-version":
            _print(client.delete_version(args.model_id, args.version_id, args.request_id))
        elif args.cmd == "stats":
            _print(client.stats(args.model_id, args.version_id))
        elif args.cmd == "predict":
            feats = [float(x) for x in args.features.split(",")]
            _print(client.predict(args.model_id, feats, args.version_id))
        return 0
    except ApiError as exc:
        print(f"error {exc.status} {exc.reason}: {exc.message}", file=sys.stderr)
        return 1
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging("DEBUG" if args.verbose else "WARNING")
    try:
        code = run(args)
    except identity.NotInitialized as exc:
        print(str(exc), file=sys.stderr)
        code = 2
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
