"""Consola `distriml` (distriml/client/cli.py): comandos, códigos de salida y mensajes.

Los comandos se ejecutan en proceso con `main([...])` contra un gateway real (FakeDB + worker de
referencia en hilos). Cada prueba usa su propia carpeta de identidad ($DISTRIML_HOME).
"""
import json
import socket
import threading
import time

import pytest
import uvicorn

from distriml.client import cli, identity
from tests.conftest import SAMPLES

SMALL = ["--hp", "epochs=2", "--hp", "batch_size=50"]


@pytest.fixture(scope="module")
def gateway():
    from distriml.api.app import create_app
    from tests.helpers.fake_db import FakeDB
    from tests.helpers.reference_worker import ReferenceWorkerThread

    db = FakeDB()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(db), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    worker = ReferenceWorkerThread(db)
    worker.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    worker.stop()
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIML_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture()
def distriml(gateway, home, capsys, monkeypatch):
    """Ejecuta un comando de la consola y devuelve (código, stdout, stderr)."""
    monkeypatch.setattr("logging.raiseExceptions", False)       # `main` reinstala el log en cada llamada
    def run(*argv, url=gateway):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--url", url, *argv])
        captured = capsys.readouterr()
        return exc.value.code, captured.out, captured.err
    return run


def as_json(out):
    return json.loads(out)


def message(err):
    """Lo que imprime la consola: la última línea de stderr (el gateway corre en este mismo proceso y su
    registro también cae en stderr)."""
    return err.strip().splitlines()[-1]


@pytest.fixture()
def ana(distriml):
    assert distriml("init", "ana")[0] == 0
    return distriml


@pytest.fixture()
def trained(ana):
    """(dataset_id, job_id, model_id) de un job de iris ya completado."""
    _, out, _ = ana("upload", str(SAMPLES / "iris.csv"))
    dataset_id = as_json(out)["dataset_id"]
    _, out, _ = ana("submit", "--dataset", dataset_id, "--model", "sgd_logistic", *SMALL)
    job_id = as_json(out)["job_id"]
    code, out, _ = ana("job", job_id, "--watch")
    job = as_json(out)
    assert code == 0 and job["estado"] == "COMPLETADO"
    return dataset_id, job_id, job["model_id"]


# ---------------------------------------------------------------------------- hiperparámetros y parser
def test_parse_hp_converts_json_values_and_keeps_plain_text():
    assert cli._parse_hp(None) is None and cli._parse_hp([]) is None
    assert cli._parse_hp(["epochs=3", "alpha=0.5", "name=abc", "flag=true", "xs=[1,2]", "a=b=c"]) == {
        "epochs": 3, "alpha": 0.5, "name": "abc", "flag": True, "xs": [1, 2], "a": "b=c"}


def test_parse_hp_rejects_items_without_equals():
    with pytest.raises(SystemExit, match="clave=valor"):
        cli._parse_hp(["epochs"])


def test_without_a_command_the_interactive_console_opens(monkeypatch):
    opened = []
    monkeypatch.setattr("distriml.client.console.run_console", lambda url: opened.append(url) or 0)
    assert cli.build_parser().parse_args([]).cmd is None
    with pytest.raises(SystemExit) as exc:
        cli.main(["--url", "http://x:1"])
    assert exc.value.code == 0 and opened == ["http://x:1"]


@pytest.mark.parametrize("argv", [["submit", "--model", "m"], ["submit", "--dataset", "d"], ["predict", "m"], ["job"]])
def test_required_arguments_are_enforced(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_default_url_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("DISTRIML_URL", "http://otro:9999")
    assert cli.build_parser().parse_args(["models"]).url == "http://otro:9999"
    monkeypatch.delenv("DISTRIML_URL")
    assert cli.build_parser().parse_args(["models"]).url == cli.DEFAULT_URL


# ---------------------------------------------------------------------------- identidad
def test_init_and_whoami(distriml):
    code, out, _ = distriml("init", "ana")
    assert code == 0 and "ana" in out
    code, out, _ = distriml("whoami")
    assert code == 0 and as_json(out) == {"username": "ana"}                     # sin client_id por defecto
    code, out, _ = distriml("whoami", "--show-id")
    assert len(as_json(out)["client_id"]) == 64


def test_init_twice_is_refused_unless_forced(distriml):
    distriml("init", "ana")
    code, _, err = distriml("init", "otra")
    assert code == 2 and "--force" in err
    assert distriml("init", "otra", "--force")[0] == 0
    assert as_json(distriml("whoami")[1])["username"] == "otra"


@pytest.mark.parametrize("argv", [["whoami"], ["models"], ["datasets"], ["jobs"], ["job", "x"], ["predict", "m", "--features", "1"]])
def test_commands_need_an_identity(distriml, argv):
    code, _, err = distriml(*argv)
    assert code == 2 and "distriml init" in err


# ---------------------------------------------------------------------------- flujo completo
def test_models_lists_the_catalog(ana):
    code, out, _ = ana("models")
    assert code == 0 and {m["algorithm"] for m in as_json(out)} == {"sgd_regressor", "sgd_logistic", "perceptron", "gaussian_nb",
            "knn_classifier", "knn_regressor", "nearest_centroid", "least_squares"}


def test_upload_uses_the_file_name_and_lists_the_dataset(ana):
    code, out, _ = ana("upload", str(SAMPLES / "iris.csv"))
    meta = as_json(out)
    assert code == 0 and (meta["n_rows"], meta["n_features"], meta["classes"]) == (150, 4, [0, 1, 2])
    listed = as_json(ana("datasets")[1])
    assert [(d["dataset_id"], d["name"]) for d in listed] == [(meta["dataset_id"], "iris.csv")]


def test_upload_with_a_custom_name(ana):
    ana("upload", str(SAMPLES / "iris.csv"), "--name", "flores.csv")
    assert as_json(ana("datasets")[1])[0]["name"] == "flores.csv"


def test_repeating_an_upload_with_the_same_request_id_returns_the_same_dataset(ana):
    first = as_json(ana("upload", str(SAMPLES / "iris.csv"), "--request-id", "r1")[1])
    again = as_json(ana("upload", str(SAMPLES / "iris.csv"), "--request-id", "r1")[1])
    assert first == again and len(as_json(ana("datasets")[1])) == 1


def test_submit_accepts_the_algorithm_name_and_hyperparameters(ana):
    dataset_id = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["dataset_id"]
    code, out, _ = ana("submit", "--dataset", dataset_id, "--model", "perceptron", *SMALL)
    job_id = as_json(out)["job_id"]
    job = as_json(ana("job", job_id)[1])
    assert code == 0 and job["hyperparams"]["epochs"] == 2 and job["hyperparams"]["batch_size"] == 50
    assert job["total_batches"] == 6


def test_job_watch_prints_progress_on_stderr_and_the_final_job_on_stdout(ana):
    dataset_id = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["dataset_id"]
    job_id = as_json(ana("submit", "--dataset", dataset_id, "--model", "sgd_logistic", *SMALL)[1])["job_id"]
    code, out, err = ana("job", job_id, "--watch")
    assert code == 0 and as_json(out)["estado"] == "COMPLETADO"
    assert "6/6" in err


def test_jobs_can_be_filtered_by_state(trained, ana):
    _, job_id, _ = trained
    assert [j["job_id"] for j in as_json(ana("jobs")[1])] == [job_id]
    assert [j["job_id"] for j in as_json(ana("jobs", "--estado", "COMPLETADO")[1])] == [job_id]
    assert as_json(ana("jobs", "--estado", "CANCELADO")[1]) == []


def test_metrics_versions_stats_and_predict(trained, ana):
    _, job_id, model_id = trained
    metrics = as_json(ana("metrics", job_id)[1])
    assert len(metrics["batches"]) == 6 and "accuracy" in metrics["final"]
    versions = as_json(ana("versions", model_id)[1])
    version_id = versions[0]["version_id"]
    assert len(versions) == 1 and versions[0]["canonical"] is True
    assert as_json(ana("versions", model_id, "--version-id", version_id)[1])["version_id"] == version_id
    code, out, _ = ana("predict", model_id, "--features", "5.1,3.5,1.4,0.2")
    prediction = as_json(out)
    assert code == 0 and prediction["prediction"] in (0, 1, 2) and prediction["version_id"] == version_id
    assert as_json(ana("predict", model_id, "--features", "5.1,3.5,1.4,0.2", "--version-id", version_id)[1])["version_id"] == version_id
    assert as_json(ana("stats", model_id)[1])["total_calls"] == 2
    assert as_json(ana("stats", model_id, "--version-id", version_id)[1])["total_calls"] == 2


def test_models_mine_lists_trained_models_and_plain_models_lists_the_catalog(trained, ana):
    _, job_id, model_id = trained
    catalog = as_json(ana("models")[1])
    assert any(d["algorithm"] == "sgd_logistic" for d in catalog) and "model_id" not in catalog[0]
    mine = as_json(ana("models", "--mine")[1])
    entry = next(m for m in mine if m["model_id"] == model_id)
    assert entry["job_id"] == job_id and entry["versions"] == 1
    code, out, _ = ana("predict", entry["model_id"], "--features", "5.1,3.5,1.4,0.2")
    assert code == 0 and as_json(out)["version_id"] == entry["canonical_version_id"]


def test_models_mine_is_empty_for_a_new_user_and_private_to_each_user(trained, distriml, tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIML_HOME", str(tmp_path / "otro"))
    assert distriml("init", "bea")[0] == 0
    assert as_json(distriml("models", "--mine")[1]) == []


def test_predict_prints_the_class_name_when_the_dataset_had_text_labels(ana):
    code, out, _ = ana("upload", str(SAMPLES / "iris_names.csv"))
    up = as_json(out)
    assert code == 0 and up["class_names"] == ["setosa", "versicolor", "virginica"]
    job_id = as_json(ana("submit", "--dataset", up["dataset_id"], "--model", "sgd_logistic", *SMALL)[1])["job_id"]
    model_id = as_json(ana("job", job_id, "--watch")[1])["model_id"]
    res = as_json(ana("predict", model_id, "--features", "5.1,3.5,1.4,0.2")[1])
    assert res["label"] == up["class_names"][res["prediction"]]


def test_uploading_the_same_file_twice_does_not_duplicate_it(ana):
    first = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])
    second = as_json(ana("upload", str(SAMPLES / "iris.csv"), "--name", "otro.csv")[1])
    assert second["dataset_id"] == first["dataset_id"] and second["already_existed"] is True
    assert len(as_json(ana("datasets")[1])) == 1


def test_delete_version_removes_the_model(trained, ana):
    _, job_id, model_id = trained
    version_id = as_json(ana("versions", model_id)[1])[0]["version_id"]
    code, out, _ = ana("delete-version", model_id, version_id, "--request-id", "x1")
    res = as_json(out)
    assert code == 0 and res["deleted"] == version_id and res["model_deleted"] is True
    assert as_json(ana("models", "--mine")[1]) == []
    assert message(ana("predict", model_id, "--features", "5.1,3.5,1.4,0.2")[2]).startswith("error 404")
    assert as_json(ana("delete-version", model_id, version_id, "--request-id", "x1")[1]) == res
    assert message(ana("delete-version", model_id, version_id)[2]).startswith("error 404")


def test_cancel_is_idempotent_with_the_same_request_id(ana):
    dataset_id = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["dataset_id"]
    slow = ["--hp", "epochs=1000", "--hp", "batch_size=2"]          # 75 000 batches: sigue en curso al cancelar
    job_id = as_json(ana("submit", "--dataset", dataset_id, "--model", "sgd_logistic", *slow)[1])["job_id"]
    first = ana("cancel", job_id, "--request-id", "c1")
    again = ana("cancel", job_id, "--request-id", "c1")
    assert first[0] == again[0] == 0 and as_json(first[1])["estado"] == "CANCELADO" == as_json(again[1])["estado"]


def test_job_watch_also_stops_on_a_cancelled_job(ana):
    dataset_id = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["dataset_id"]
    slow = ["--hp", "epochs=1000", "--hp", "batch_size=2"]
    job_id = as_json(ana("submit", "--dataset", dataset_id, "--model", "sgd_logistic", *slow)[1])["job_id"]
    ana("cancel", job_id)
    code, out, _ = ana("job", job_id, "--watch")
    assert code == 0 and as_json(out)["estado"] == "CANCELADO"


def test_last_entry_is_saved_in_the_identity_after_a_write(ana, home):
    entry = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["entry_id"]
    assert json.loads((home / "identity.json").read_text(encoding="utf-8"))["last_entry"] == entry


# ---------------------------------------------------------------------------- errores del servidor
def test_unknown_job_is_reported_without_a_traceback(ana):
    code, out, err = ana("job", "no-existe")
    assert code == 1 and out == "" and message(err).startswith("error 404 not_found")


def test_unknown_model_is_a_400(ana):
    dataset_id = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["dataset_id"]
    code, _, err = ana("submit", "--dataset", dataset_id, "--model", "no-existe")
    assert code == 1 and message(err).startswith("error 400")


def test_invalid_hyperparameters_are_a_400(ana):
    dataset_id = as_json(ana("upload", str(SAMPLES / "iris.csv"))[1])["dataset_id"]
    code, _, err = ana("submit", "--dataset", dataset_id, "--model", "sgd_logistic", "--hp", "epochs=0")
    assert code == 1 and message(err).startswith("error 400 invalid_hyperparams")


def test_cancelling_a_finished_job_is_a_409(trained, ana):
    _, job_id, _ = trained
    code, _, err = ana("cancel", job_id)
    assert code == 1 and message(err).startswith("error 409 job_terminal")


def test_predict_with_the_wrong_number_of_features_is_a_400(trained, ana):
    _, _, model_id = trained
    code, _, err = ana("predict", model_id, "--features", "1,2")
    assert code == 1 and message(err).startswith("error 400")


def test_one_user_cannot_see_the_jobs_of_another(trained, ana, distriml, tmp_path, monkeypatch):
    _, job_id, model_id = trained
    monkeypatch.setenv("DISTRIML_HOME", str(tmp_path / "otro-usuario"))
    assert distriml("init", "intruso")[0] == 0
    assert message(distriml("job", job_id)[2]).startswith("error 404")
    assert message(distriml("predict", model_id, "--features", "1,2,3,4")[2]).startswith("error 404")
    assert as_json(distriml("jobs")[1]) == [] and as_json(distriml("datasets")[1]) == []


def test_unreachable_gateway_is_reported_as_503(ana, monkeypatch):
    monkeypatch.setattr("distriml.client.http.time.sleep", lambda s: None)
    code, _, err = ana("models", url="http://127.0.0.1:1")
    assert code == 1 and message(err).startswith("error 503 unreachable")


# ---------------------------------------------------------------------------- errores locales
@pytest.mark.xfail(strict=True, raises=FileNotFoundError,
                   reason="upload con un archivo inexistente muestra un traceback en vez de un mensaje")
def test_upload_of_a_missing_file_is_reported_cleanly(ana):
    code, _, err = ana("upload", "no_existe.csv")
    assert code == 2 and "no_existe.csv" in err


@pytest.mark.xfail(strict=True, raises=ValueError,
                   reason="predict con --features no numéricas muestra un traceback en vez de un mensaje")
def test_predict_with_non_numeric_features_is_reported_cleanly(ana):
    code, _, err = ana("predict", "m", "--features", "a,b")
    assert code == 2 and "features" in err
