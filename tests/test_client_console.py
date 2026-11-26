"""Consola interactiva (distriml/client/console.py) con entradas escritas de antemano.

Contra un gateway real en el mismo proceso (FakeDB + worker de referencia): lo que se comprueba es lo que ve
y puede hacer el usuario, no los detalles internos.
"""
import pytest

from distriml.client import identity
from distriml.client.console import Console, parse_features, parse_hyperparams, run_console
from distriml.client.http import ApiClient
from tests.conftest import SAMPLES
from tests.test_client_cli import gateway, home  # noqa: F401  (fixtures compartidos)

IRIS = str(SAMPLES / "iris.csv")
QUICK = "epochs=2 batch_size=50"


class Script:
    """Entrada y salida simuladas: `ask` devuelve las respuestas en orden y `out` guarda lo impreso."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.lines: list[str] = []
        self.prompts: list[str] = []

    def ask(self, prompt=""):
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def out(self, text=""):
        self.lines.append(str(text))

    @property
    def text(self):
        return "\n".join(self.lines)


@pytest.fixture()
def session(gateway, home):  # noqa: F811
    """Ejecuta la consola con las respuestas dadas y devuelve (código, salida)."""
    ident = identity.init_identity("ana")

    def go(*answers, poll=0.05):
        script = Script(*answers)
        client = ApiClient(gateway, ident["client_id"], last_entry=ident.get("last_entry"), on_entry=identity.save_last_entry)
        try:
            code = Console(client, "ana", gateway, ask=script.ask, out=script.out, poll=poll).run()
        finally:
            client.close()
        return code, script
    yield go
    # El worker de referencia es compartido por todo el módulo: no dejar jobs largos ocupándolo.
    cleaner = ApiClient(gateway, ident["client_id"])
    try:
        for job in cleaner.jobs():
            if job["estado"] not in ("COMPLETADO", "FALLIDO", "CANCELADO"):
                cleaner.cancel(job["job_id"])
    finally:
        cleaner.close()


# ---------------------------------------------------------------------------- analizadores
def test_parse_hyperparams_accepts_spaces_and_commas_and_converts_types():
    assert parse_hyperparams("epochs=5 alpha=0.01,name=x") == {"epochs": 5, "alpha": 0.01, "name": "x"}
    assert parse_hyperparams("") == {}


def test_parse_hyperparams_rejects_items_without_equals():
    with pytest.raises(ValueError, match="epochs"):
        parse_hyperparams("epochs")


@pytest.mark.parametrize("text, expected", [("5.1,3.5,1.4,0.2", [5.1, 3.5, 1.4, 0.2]), ("1 2 3", [1.0, 2.0, 3.0]),
                                            ("1; 2", [1.0, 2.0]), (" 7 ", [7.0])])
def test_parse_features(text, expected):
    assert parse_features(text) == expected


@pytest.mark.parametrize("text", ["", "a,b", "1,x"])
def test_parse_features_rejects_non_numbers(text):
    with pytest.raises(ValueError):
        parse_features(text)


# ---------------------------------------------------------------------------- menú
def test_menu_with_every_operation_is_shown_before_each_choice(session):
    code, s = session("1", "0")
    assert code == 0
    assert s.text.count("Ver algoritmos disponibles") == 2          # se vuelve a mostrar tras cada operación
    for label in ("Subir un dataset", "Mis datasets", "Entrenar un modelo", "Mis jobs", "Ver / seguir un job",
                  "Cancelar un job", "Mis modelos entrenados", "Hacer una predicción", "Detalle de un modelo",
                  "Eliminar una versión de un modelo", "Salir"):
        assert label in s.text
    assert "usuario: ana" in s.text and "sgd_logistic" in s.text


def test_invalid_option_is_reported_and_the_menu_returns(session):
    code, s = session("99", "abc", "0")
    assert code == 0 and s.text.count("no válida") == 2


def test_end_of_input_leaves_cleanly(session):
    assert session()[0] == 0


# ---------------------------------------------------------------------------- listas vacías con guía
def test_empty_lists_tell_the_user_what_to_do_next(session):
    _, s = session("3", "5", "8", "9", "4", "0")
    assert "Aún no has subido datasets (opción 2)" in s.text
    assert "Aún no has enviado jobs (opción 4)" in s.text
    assert "entrena uno con la opción 4" in s.text
    assert "sube uno primero (opción 2)" in s.text


# ---------------------------------------------------------------------------- flujo completo
def test_full_flow_upload_train_wait_list_and_predict(session):
    _, s = session(
        "2", IRIS,                      # subir
        "3",                            # ver datasets
        "4", "1", "2", QUICK, "s",      # entrenar: dataset 1, algoritmo 2, hiperparámetros, esperar
        "0")
    assert "Dataset subido: iris.csv" in s.text and "iris.csv" in s.text
    assert "Job enviado" in s.text and "COMPLETADO" in s.text and "Entrenamiento terminado" in s.text and "accuracy=" in s.text


def test_train_then_list_models_and_predict(session):
    algo = "sgd_logistic"
    _, probe = session("1", "0")
    names = [ln.split()[0] for ln in probe.lines if ln.startswith("  ") and ln.split() and ln.split()[0] in
             ("sgd_regressor", "sgd_logistic", "perceptron", "gaussian_nb")]
    idx = str(names.index(algo) + 1)
    _, s = session("2", IRIS, "4", "1", idx, QUICK, "s",
                   "8",
                   "9", "1", "5.1,3.5,1.4,0.2", "no,son,numeros", "1,2", "6.7,3.0,5.2,2.3", "",
                   "0")
    assert "Tus modelos entrenados" in s.text and algo in s.text
    assert s.text.count("Predicción:") == 2                           # las dos entradas válidas
    assert "números separados por comas" in s.text                    # la no numérica se explica
    assert "ERROR 400" in s.text                                      # nº de características incorrecto, sin salir


def test_model_detail_shows_versions_and_inference_stats(session):
    _, s = session("2", IRIS, "4", "1", "1", QUICK, "s", "9", "1", "5.1,3.5,1.4,0.2", "", "10", "1", "0")
    assert "versión" in s.text and "canónica" in s.text and "Inferencias: 1" in s.text


def test_train_without_waiting_then_follow_the_job(session):
    _, s = session("2", IRIS, "4", "1", "1", QUICK, "n", "5", "6", "1", "0")
    assert "opción 6" in s.text and "Tus jobs" in s.text


def test_cancel_a_running_job(session):
    slow = "epochs=1000 batch_size=2"
    _, s = session("2", IRIS, "4", "1", "1", slow, "n", "7", "1", "s", "5", "0")
    assert "Job cancelado." in s.text and "CANCELADO" in s.text


def test_cancel_asks_for_confirmation(session):
    slow = "epochs=1000 batch_size=2"
    _, s = session("2", IRIS, "4", "1", "1", slow, "n", "7", "1", "n", "5", "0")
    assert "(cancelado)" in s.text and "Job cancelado." not in s.text and "CANCELADO" not in s.text.split("(cancelado)")[1]


def test_cancel_with_no_active_jobs_says_so(session):
    _, s = session("7", "0")
    assert "No tienes jobs en curso." in s.text


# ---------------------------------------------------------------------------- errores sin salir del programa
def test_missing_file_is_explained_and_the_menu_returns(session):
    code, s = session("2", "no_existe.csv", "0")
    assert code == 0 and "ERROR: no existe el archivo 'no_existe.csv'" in s.text and "Traceback" not in s.text


def test_invalid_hyperparameters_are_explained_and_the_menu_returns(session):
    _, s = session("2", IRIS, "4", "1", "1", "epochs", "0")
    assert "no es clave=valor" in s.text


def test_server_validation_error_is_shown_and_the_menu_returns(session):
    _, s = session("2", IRIS, "4", "1", "1", "epochs=0", "0")
    assert "ERROR 400 invalid_hyperparams" in s.text and s.text.rstrip().endswith("Hasta luego.")


def test_blank_answer_cancels_a_list_pick(session):
    _, s = session("2", IRIS, "4", "", "0")
    assert "(cancelado)" in s.text


def test_out_of_range_pick_is_asked_again(session):
    _, s = session("2", IRIS, "4", "7", "1", "", "0")           # 7 fuera de rango; luego dataset 1; algoritmo vacío cancela
    assert "Escribe un número entre 1 y 1" in s.text


def test_unreachable_gateway_gives_a_hint(home):
    ident = identity.init_identity("ana")
    script = Script("3", "0")
    client = ApiClient("http://127.0.0.1:9", ident["client_id"], max_retries=0)
    Console(client, "ana", "http://127.0.0.1:9", ask=script.ask, out=script.out).run()
    assert "ERROR 503" in script.text and "cluster" in script.text


# ---------------------------------------------------------------------------- usuario nuevo
def test_first_run_creates_the_identity(gateway, home):  # noqa: F811
    script = Script("beto", "0")
    assert run_console(gateway, ask=script.ask, out=script.out) == 0
    assert identity.load_identity()["username"] == "beto" and "usuario: beto" in script.text


def test_first_run_with_an_empty_name_is_rejected(gateway, home):  # noqa: F811
    script = Script("  ")
    assert run_console(gateway, ask=script.ask, out=script.out) == 2
    with pytest.raises(identity.NotInitialized):
        identity.load_identity()


# ---------------------------------------------------------------------------- nombres de clase
def test_upload_shows_the_classes_and_predict_shows_the_name(session):
    names = str(SAMPLES / "iris_names.csv")
    _, s = session("2", names, "4", "1", "2", QUICK, "s", "9", "1", "5.1,3.5,1.4,0.2", "", "0")    # algoritmo 2 = sgd_logistic
    assert "Clases: 0 = setosa, 1 = versicolor, 2 = virginica" in s.text
    line = next(ln for ln in s.lines if "Predicción:" in ln)
    number, name = line.split("Predicción:")[1].split("(")[0].split("=")
    assert ["setosa", "versicolor", "virginica"][int(number)] == name.strip()


def test_numeric_labels_show_only_the_number(session):
    _, s = session("2", IRIS, "4", "1", "2", QUICK, "s", "9", "1", "5.1,3.5,1.4,0.2", "", "0")
    line = next(ln for ln in s.lines if "Predicción:" in ln)
    assert "=" not in line and "Clases:" not in s.text


# ---------------------------------------------------------------------------- eliminar y datasets repetidos
def test_uploading_the_same_dataset_twice_says_it_already_exists(session):
    _, s = session("2", IRIS, "2", IRIS, "3", "0")
    assert s.text.count("Dataset subido: iris.csv") == 1
    assert "ya estaba subido" in s.text and "no se guarda otra copia" in s.text
    listing = s.text.split("Tus datasets:")[1]
    assert listing.count("iris.csv") == 1


def test_delete_a_model_version_from_the_menu(session):
    _, s = session("2", IRIS, "4", "1", "2", QUICK, "s",
                   "11", "1", "s",                        # elimina, confirmando
                   "8", "0")
    assert "Es la única versión: el modelo dejará de existir" in s.text
    assert "Versión eliminada. El modelo ya no existe." in s.text
    assert s.text.rstrip().rsplit("Aún no tienes modelos entrenados", 1)[1].count("Hasta luego.") == 1


def test_delete_asks_for_confirmation_and_n_keeps_the_model(session):
    _, s = session("2", IRIS, "4", "1", "2", QUICK, "s", "11", "1", "n", "8", "0")
    assert "(cancelado)" in s.text and "Versión eliminada" not in s.text
    assert "Tus modelos entrenados" in s.text


def test_enter_at_the_confirmation_does_not_delete(session):
    _, s = session("2", IRIS, "4", "1", "2", QUICK, "s", "11", "1", "", "8", "0")
    assert "Versión eliminada" not in s.text and "Tus modelos entrenados" in s.text


def test_delete_with_no_models_says_what_to_do(session):
    _, s = session("11", "0")
    assert "entrena uno con la opción 4" in s.text


# ---------------------------------------------------------------------------- casos que los anteriores no distinguían
def _client_of(gateway):
    ident = identity.load_identity()
    return ApiClient(gateway, ident["client_id"])


def test_cancel_only_offers_jobs_that_are_still_active(session, gateway):  # noqa: F811
    slow = "epochs=1000 batch_size=2"
    _, s = session("2", IRIS, "4", "1", "1", QUICK, "s",            # un job que termina
                   "4", "1", "1", slow, "n",                          # y otro largo, sin esperar
                   "7", "1", "s", "0")
    client = _client_of(gateway)
    try:
        estados = sorted(j["estado"] for j in client.jobs())
    finally:
        client.close()
    assert estados == ["CANCELADO", "COMPLETADO"] and "Job cancelado." in s.text


def test_pressing_enter_at_the_wait_question_waits(session):
    _, s = session("2", IRIS, "4", "1", "1", QUICK, "", "0")
    assert "Entrenamiento terminado" in s.text


def test_predict_uses_the_model_that_was_picked(session, gateway):  # noqa: F811
    _, s = session("2", IRIS, "4", "1", "1", QUICK, "s", "4", "1", "2", QUICK, "s", "0")
    client = _client_of(gateway)
    try:
        models = client.trained_models()
    finally:
        client.close()
    assert len(models) >= 2
    first, second = models[0], models[1]
    _, s = session("9", "1", "5.1,3.5,1.4,0.2", "", "9", "2", "5.1,3.5,1.4,0.2", "", "0")
    assert f"(versión {first['canonical_version_id'][:8]})" in s.text
    assert f"(versión {second['canonical_version_id'][:8]})" in s.text


def test_identifiers_are_shown_shortened(session, gateway):  # noqa: F811
    _, s = session("2", IRIS, "4", "1", "1", QUICK, "s", "5", "8", "0")
    client = _client_of(gateway)
    try:
        job_id = client.jobs()[0]["job_id"]
    finally:
        client.close()
    assert job_id[:8] in s.text and job_id not in s.text


class _StubClient:
    """Cliente mínimo para probar el seguimiento de un job que termina mal."""
    def __init__(self, states):
        self.states = list(states)

    def job(self, job_id):
        estado = self.states.pop(0)
        return {"job_id": job_id, "estado": estado, "last_confirmed_batch": 1, "total_batches": 6,
                "attempts": [{"revoke_reason": "WORKER_ERROR"}, {"revoke_reason": "WORKER_ERROR"}]}


def test_follow_stops_and_explains_when_the_job_fails():
    script = Script()
    console = Console(_StubClient(["EN_EJECUCION", "FALLIDO"]), "ana", "http://x", ask=script.ask, out=script.out, poll=0)
    job = console._follow("j" * 36)
    assert job["estado"] == "FALLIDO"
    assert "El job falló tras 2 intentos. Último motivo: WORKER_ERROR." in script.text


def test_delete_removes_the_model_that_was_picked_and_keeps_the_other(session, gateway):  # noqa: F811
    session("2", IRIS, "4", "1", "2", QUICK, "s", "4", "1", "2", "epochs=3 batch_size=50", "s", "0")
    client = _client_of(gateway)
    try:
        first, second = [m["model_id"] for m in client.trained_models()]
        session("11", "1", "s", "0")                              # elige el modelo número 1 de la lista
        assert [m["model_id"] for m in client.trained_models()] == [second]
        session("11", "1", "s", "0")                              # y ahora el que queda
        assert client.trained_models() == []
    finally:
        client.close()
    assert first != second
