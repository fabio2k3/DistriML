"""`Database` — lado cliente: datasets, jobs, cancelación, modelos, inferencia, estadísticas y aislamiento."""
import pytest

from distriml.common import constants as C
from distriml.common.errors import ContractError
from distriml.common.ids import sha256_hex
from tests.helpers.db_utils import (CLIENT_A, CLIENT_B, IRIS, SMALL_HP, FakeClock, make_db, model_def_id,
                                    new_dataset, new_job, start_job, train)


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def db(tmp_path, clock):
    return make_db(tmp_path, clock, wall=FakeClock(10_000_000))


def expect(status, reason=None):
    """Contexto que exige un ContractError con ese código (y motivo)."""
    class _Ctx:
        def __enter__(self):
            self._inner = pytest.raises(ContractError)
            self.info = self._inner.__enter__()
            return self.info

        def __exit__(self, *exc):
            ok = self._inner.__exit__(*exc)
            if ok:
                assert self.info.value.status == status
                if reason:
                    assert self.info.value.reason == reason
            return ok
    return _Ctx()


# ---------------------------------------------------------------------------- datasets
def test_create_dataset_returns_metadata(db):
    r = new_dataset(db)
    assert (r["n_rows"], r["n_features"], r["classes"]) == (150, 4, [0, 1, 2])
    assert r["entry_id"].startswith("db:")
    d = db.get_dataset(CLIENT_A, r["dataset_id"])
    assert d == {"dataset_id": r["dataset_id"], "name": "iris.csv", "sha256": sha256_hex(IRIS),
                 "size": len(IRIS), "replicas": 1}
    assert db.get_artifact(sha256_hex(IRIS)) == IRIS          # el binario quedó en el almacén


def test_create_dataset_is_idempotent_and_returns_the_same_entry(db):
    a = new_dataset(db, rid="x")
    b = new_dataset(db, rid="x")
    assert a == b
    assert len(db.list_datasets(CLIENT_A)) == 1


def test_reusing_request_id_with_other_content_is_409(db):
    new_dataset(db, rid="x")
    other = IRIS + b"1,2,3,4,0\n"
    with expect(409, "request_id_reused"):
        new_dataset(db, rid="x", data=other)


def test_dataset_with_wrong_hash_or_size_is_400(db):
    with expect(400, "hash_mismatch"):
        db.create_dataset(CLIENT_A, "r", "n", "0" * 64, len(IRIS), IRIS)
    with expect(400, "hash_mismatch"):
        db.create_dataset(CLIENT_A, "r", "n", sha256_hex(IRIS), len(IRIS) + 1, IRIS)


@pytest.mark.parametrize("bad", [b"", b"a,y\n1,x\n2,3\n", b"a,y\n1,2\n", b"solo\n1\n2\n", b"\xff\xfe\x00"])
def test_invalid_csv_is_400_and_nothing_is_stored(db, bad):
    with expect(400):
        new_dataset(db, data=bad)
    assert db.list_datasets(CLIENT_A) == []
    assert db.store.list_hashes() == []


def test_oversized_dataset_is_400(tmp_path):
    small = make_db(tmp_path, max_dataset_bytes=100)
    with expect(400, "dataset_too_large"):
        new_dataset(small)


def test_list_datasets_pages_in_creation_order(db):
    made = [new_dataset(db, rid=f"r{i}", data=IRIS + b"1,1,1,1,0\n" * i)["dataset_id"] for i in range(5)]
    page1 = db.list_datasets(CLIENT_A, limit=2)
    page2 = db.list_datasets(CLIENT_A, limit=2, cursor=page1[-1]["dataset_id"])
    page3 = db.list_datasets(CLIENT_A, limit=2, cursor=page2[-1]["dataset_id"])
    assert [d["dataset_id"] for d in page1 + page2 + page3] == made
    assert set(page1[0]) == {"dataset_id", "name", "sha256", "size"}
    assert db.list_datasets(CLIENT_A, cursor="desconocido") == []


@pytest.mark.parametrize("limit", [0, -1, "5", None, True])
def test_invalid_limit_is_400(db, limit):
    with expect(400, "invalid_limit"):
        db.list_datasets(CLIENT_A, limit=limit)


def test_datasets_are_isolated_between_clients(db):
    a = new_dataset(db, CLIENT_A)
    with expect(404):
        db.get_dataset(CLIENT_B, a["dataset_id"])
    assert db.list_datasets(CLIENT_B) == []
    b = new_dataset(db, CLIENT_B)                              # mismo client_request_id, otro cliente: otro id
    assert b["dataset_id"] != a["dataset_id"]
    with expect(404):
        db.get_dataset(CLIENT_A, "no-existe")


# ---------------------------------------------------------------------------- jobs
def test_create_job_merges_defaults_and_computes_batches(db):
    ds = new_dataset(db)
    job = db.create_job(CLIENT_A, "j", ds["dataset_id"], model_def_id("sgd_logistic"), {"epochs": 10})
    v = db.get_job(CLIENT_A, job["job_id"])
    assert v["estado"] == C.PENDIENTE
    assert v["hyperparams"] == {"batch_size": 32, "epochs": 10, "seed": 42, "alpha": 0.0001, "eta0": 0.01}
    assert v["total_batches"] == 50 and v["last_confirmed_batch"] == 0     # 10 épocas × ceil(150/32)=5
    assert v["attempts"] == [] and v["attempt_vigente"] is None
    assert (v["attempts_count"], v["lost_count"], v["model_id"]) == (0, 0, None)


def test_gaussian_nb_has_a_single_epoch(db):
    ds = new_dataset(db)
    job = db.create_job(CLIENT_A, "j", ds["dataset_id"], model_def_id("gaussian_nb"), {"batch_size": 50})
    assert db.get_job(CLIENT_A, job["job_id"])["total_batches"] == 3


def test_create_job_is_idempotent_and_detects_reuse_with_other_params(db):
    ds = new_dataset(db)
    a = new_job(db, ds["dataset_id"], rid="x")
    assert new_job(db, ds["dataset_id"], rid="x") == a
    assert len(db.list_jobs(CLIENT_A)) == 1
    with expect(409, "request_id_reused"):
        new_job(db, ds["dataset_id"], rid="x", hyperparams={"epochs": 3})
    with expect(409, "request_id_reused"):
        new_job(db, ds["dataset_id"], rid="x", algorithm="perceptron")


@pytest.mark.parametrize("model,hp", [
    ("no-existe", None), ("sgd_logistic", {"epochs": 0}), ("sgd_logistic", {"inventado": 1}),
    ("sgd_logistic", {"epochs": "3"}), ("sgd_logistic", [1, 2]), ("gaussian_nb", {"epochs": 2}),
])
def test_create_job_validates_model_and_hyperparams(db, model, hp):
    ds = new_dataset(db)
    mid = model if model == "no-existe" else model_def_id(model)
    with expect(400):
        db.create_job(CLIENT_A, "j", ds["dataset_id"], mid, hp)
    assert db.list_jobs(CLIENT_A) == []


def test_create_job_requires_an_own_existing_dataset(db):
    foreign = new_dataset(db, CLIENT_B)
    for dataset_id in (foreign["dataset_id"], "no-existe", None, 7):
        with expect(400, "dataset_not_found"):
            db.create_job(CLIENT_A, "j", dataset_id, model_def_id("sgd_logistic"), None)


def test_classification_needs_integer_labels(db):
    diabetes = (__import__("tests.conftest", fromlist=["SAMPLES"]).SAMPLES / "diabetes.csv").read_bytes()
    ds = new_dataset(db, data=diabetes, name="d.csv")
    with expect(400, "dataset_incompatible"):
        db.create_job(CLIENT_A, "j", ds["dataset_id"], model_def_id("sgd_logistic"), None)
    assert db.create_job(CLIENT_A, "j2", ds["dataset_id"], model_def_id("sgd_regressor"), SMALL_HP)["job_id"]


def test_create_job_with_missing_dataset_binary_is_503(db):
    ds = new_dataset(db)
    db.store.delete(sha256_hex(IRIS))
    with expect(503, "dataset_unavailable"):
        new_job(db, ds["dataset_id"])
    assert db.list_jobs(CLIENT_A) == []


def test_list_jobs_filters_by_state_and_pages(db):
    ds = new_dataset(db)
    ids_ = [new_job(db, ds["dataset_id"], rid=f"j{i}")["job_id"] for i in range(4)]
    db.cancel_job(CLIENT_A, "c", ids_[1])
    assert [j["job_id"] for j in db.list_jobs(CLIENT_A)] == ids_
    assert [j["job_id"] for j in db.list_jobs(CLIENT_A, estado=C.CANCELADO)] == [ids_[1]]
    assert db.list_jobs(CLIENT_A, estado=C.COMPLETADO) == []
    first = db.list_jobs(CLIENT_A, limit=3)
    rest = db.list_jobs(CLIENT_A, limit=3, cursor=first[-1]["job_id"])
    assert [j["job_id"] for j in first + rest] == ids_
    assert set(first[0]) == {"job_id", "dataset_id", "model_def_id", "estado", "last_confirmed_batch", "total_batches"}


def test_jobs_are_isolated_between_clients(db):
    _, job, a = start_job(db)
    train(db, "w1", a)
    model_id = db.get_job(CLIENT_A, job["job_id"])["model_id"]
    calls = [lambda: db.get_job(CLIENT_B, job["job_id"]), lambda: db.get_job_metrics(CLIENT_B, job["job_id"]),
             lambda: db.cancel_job(CLIENT_B, "c", job["job_id"]),
             lambda: db.list_model_versions(CLIENT_B, model_id),
             lambda: db.get_model_version(CLIENT_B, model_id, a.attempt_id),
             lambda: db.get_model_stats(CLIENT_B, model_id),
             lambda: db.predict(CLIENT_B, model_id, {"features": [1, 2, 3, 4]})]
    for call in calls:
        with expect(404):
            call()
    assert db.list_jobs(CLIENT_B) == []


# ---------------------------------------------------------------------------- cancelación
def test_cancel_pending_job(db):
    ds = new_dataset(db)
    job = new_job(db, ds["dataset_id"])
    res = db.cancel_job(CLIENT_A, "c1", job["job_id"])
    assert res["estado"] == C.CANCELADO and res["entry_id"].startswith("db:")
    assert db.get_job(CLIENT_A, job["job_id"])["estado"] == C.CANCELADO
    assert db.request_job("w1") is None                       # un job cancelado no se asigna


def test_cancel_running_job_revokes_its_attempt(db):
    _, job, a = start_job(db)
    train(db, "w1", a, upto=2, finish=False)
    res = db.cancel_job(CLIENT_A, "c1", job["job_id"])
    assert res["estado"] == C.CANCELADO
    v = db.get_job(CLIENT_A, job["job_id"])
    assert v["estado"] == C.CANCELADO and v["attempt_vigente"] is None and v["last_confirmed_batch"] == 2
    assert [(x["estado"], x["revoke_reason"]) for x in v["attempts"]] == [(C.ATTEMPT_REVOCADO, C.CANCEL)]
    assert (v["attempts_count"], v["lost_count"]) == (0, 0)   # cancelar no cuenta en ningún presupuesto
    assert db.renew_lease("w1", a.attempt_id) == C.RENEW_REVOKED
    assert db.upload_checkpoint("w1", a.attempt_id, 3, "0" * 64, {}) == C.CKPT_REJECTED


def test_cancel_is_idempotent_by_request_id(db):
    _, job, _ = start_job(db)
    first = db.cancel_job(CLIENT_A, "c1", job["job_id"])
    assert db.cancel_job(CLIENT_A, "c1", job["job_id"]) == first
    with expect(409, "job_terminal"):                         # otro request_id sobre un job ya cancelado
        db.cancel_job(CLIENT_A, "c2", job["job_id"])


def test_cancel_completed_job_is_409(db):
    _, job, a = start_job(db)
    train(db, "w1", a)
    with expect(409, "job_terminal"):
        db.cancel_job(CLIENT_A, "c1", job["job_id"])
    assert db.get_job(CLIENT_A, job["job_id"])["estado"] == C.COMPLETADO


def test_cancel_request_id_cannot_be_reused_for_another_job(db):
    ds = new_dataset(db)
    j1, j2 = new_job(db, ds["dataset_id"], rid="a"), new_job(db, ds["dataset_id"], rid="b")
    db.cancel_job(CLIENT_A, "same", j1["job_id"])
    with expect(409, "request_id_reused"):
        db.cancel_job(CLIENT_A, "same", j2["job_id"])
    assert db.get_job(CLIENT_A, j2["job_id"])["estado"] == C.PENDIENTE


# ---------------------------------------------------------------------------- modelos, métricas y versiones
def _completed(db, algorithm="sgd_logistic"):
    ds, job, a = start_job(db, algorithm=algorithm)
    train(db, "w1", a)
    return job, a, db.get_job(CLIENT_A, job["job_id"])


def test_completed_job_exposes_model_version_and_metrics(db):
    job, a, v = _completed(db)
    assert v["estado"] == C.COMPLETADO and v["last_confirmed_batch"] == v["total_batches"] == 6
    assert v["canonical_version_id"] == a.attempt_id and v["model_version"] == 1 and v["model_id"]
    assert v["attempt_vigente"] is None
    m = db.get_job_metrics(CLIENT_A, job["job_id"])
    assert m["estado"] == C.COMPLETADO and [b["batch_idx"] for b in m["batches"]] == [1, 2, 3, 4, 5, 6]
    assert set(m["batches"][0]["metrics"]) == {"n", "score"}
    assert m["final"]["accuracy"] > 0.3 and set(m["final"]) == {"accuracy", "mse", "n"}
    versions = db.list_model_versions(CLIENT_A, v["model_id"])
    assert len(versions) == 1
    ver = versions[0]
    assert (ver["version_id"], ver["version"], ver["canonical"], ver["estado"]) == (a.attempt_id, 1, True, C.VERSION_ENTRENADO)
    assert ver["job_id"] == job["job_id"] and ver["model_def_id"] == model_def_id("sgd_logistic")
    assert ver["metrics"] == m["final"] and len(ver["sha256"]) == 64
    assert db.get_model_version(CLIENT_A, v["model_id"], a.attempt_id) == ver


def test_metrics_of_unfinished_job_have_no_final(db):
    _, job, a = start_job(db)
    train(db, "w1", a, upto=3, finish=False)
    m = db.get_job_metrics(CLIENT_A, job["job_id"])
    assert m["estado"] == C.EN_EJECUCION and len(m["batches"]) == 3 and m["final"] is None


def test_unknown_model_or_version_is_404(db):
    _, _, v = _completed(db)
    for call in (lambda: db.list_model_versions(CLIENT_A, "no-existe"),
                 lambda: db.get_model_version(CLIENT_A, v["model_id"], "no-existe"),
                 lambda: db.get_model_stats(CLIENT_A, "no-existe"),
                 lambda: db.get_model_stats(CLIENT_A, v["model_id"], version_id="no-existe")):
        with expect(404):
            call()


# ---------------------------------------------------------------------------- inferencia
def test_predict_uses_the_canonical_version_by_default(db):
    _, a, v = _completed(db)
    p = db.predict(CLIENT_A, v["model_id"], {"features": [5.1, 3.5, 1.4, 0.2]})
    ver = db.list_model_versions(CLIENT_A, v["model_id"])[0]
    assert p["version_id"] == a.attempt_id and p["sha256"] == ver["sha256"] and p["prediction"] in (0, 1, 2)
    assert db.predict(CLIENT_A, v["model_id"], {"features": [5.1, 3.5, 1.4, 0.2]}, version=1) == p
    assert db.predict(CLIENT_A, v["model_id"], {"features": [5.1, 3.5, 1.4, 0.2]}, version_id=a.attempt_id) == p


def test_predict_regression_returns_float(db):
    diabetes = (__import__("tests.conftest", fromlist=["SAMPLES"]).SAMPLES / "diabetes.csv").read_bytes()
    ds = new_dataset(db, data=diabetes, name="d.csv")
    job = new_job(db, ds["dataset_id"], algorithm="sgd_regressor")
    train(db, "w1", db.request_job("w1"))
    mid = db.get_job(CLIENT_A, job["job_id"])["model_id"]
    n = len(__import__("distriml.common.dataset", fromlist=["x"]).parse(diabetes)[0][0])
    assert isinstance(db.predict(CLIENT_A, mid, {"features": [0.1] * n})["prediction"], float)


@pytest.mark.parametrize("kwargs", [{"version": 2}, {"version": 0}, {"version_id": "no-existe"}, {"version": "1"}])
def test_predict_with_unknown_version_is_404(db, kwargs):
    _, _, v = _completed(db)
    with expect(404):
        db.predict(CLIENT_A, v["model_id"], {"features": [1, 2, 3, 4]}, **kwargs)


@pytest.mark.parametrize("bad", [None, {}, {"features": None}, {"features": [1, 2]}, {"features": "abc"},
                                 {"features": [1, 2, 3, "x"]}, {"features": [1, 2, 3, float("nan")]}, [1, 2, 3, 4]])
def test_predict_rejects_invalid_input_with_400(db, bad):
    _, _, v = _completed(db)
    with expect(400, "invalid_input"):
        db.predict(CLIENT_A, v["model_id"], bad)
    assert db.get_model_stats(CLIENT_A, v["model_id"])["total_calls"] == 0     # una predicción fallida no cuenta


def test_failed_predictions_do_not_change_stats_but_valid_ones_do(db):
    _, _, v = _completed(db)
    for _ in range(3):
        db.predict(CLIENT_A, v["model_id"], {"features": [6.0, 3.0, 4.5, 1.5]})
    assert db.get_model_stats(CLIENT_A, v["model_id"])["total_calls"] == 3


# ---------------------------------------------------------------------------- estadísticas
def test_stats_group_by_minute_and_filter_by_range(tmp_path, clock):
    wall = FakeClock(60_000 * 100)                           # minuto 100
    db = make_db(tmp_path, clock, wall=wall)
    _, _, v = _completed(db)
    x = {"features": [5.1, 3.5, 1.4, 0.2]}
    for minute, calls in ((100, 2), (101, 1), (105, 3)):
        wall.now = 60_000 * minute + 500
        for _ in range(calls):
            db.predict(CLIENT_A, v["model_id"], x)
    st = db.get_model_stats(CLIENT_A, v["model_id"])
    assert st["total_calls"] == 6 and [(r["minute"], r["count"]) for r in st["por_minuto"]] == [(100, 2), (101, 1), (105, 3)]
    assert st["version_id"] == v["canonical_version_id"] and st["model_id"] == v["model_id"]
    assert st["mean_latency_ms"] > 0
    assert st["mean_latency_ms"] == pytest.approx(sum(r["mean_latency_ms"] * r["count"] for r in st["por_minuto"]) / 6)
    sub = db.get_model_stats(CLIENT_A, v["model_id"], from_minute=101, to_minute=105)
    assert sub["total_calls"] == 4 and [r["minute"] for r in sub["por_minuto"]] == [101, 105]
    assert db.get_model_stats(CLIENT_A, v["model_id"], from_minute=102, to_minute=104)["por_minuto"] == []
    assert db.get_model_stats(CLIENT_A, v["model_id"], from_minute=200)["total_calls"] == 0


def test_stats_without_calls_are_zero(db):
    _, _, v = _completed(db)
    assert db.get_model_stats(CLIENT_A, v["model_id"]) == {
        "model_id": v["model_id"], "version_id": v["canonical_version_id"], "total_calls": 0,
        "mean_latency_ms": 0.0, "por_minuto": []}


@pytest.mark.parametrize("kw", [{"from_minute": "1"}, {"to_minute": 1.5}, {"from_minute": True}])
def test_stats_range_must_be_integers(db, kw):
    _, _, v = _completed(db)
    with expect(400, "invalid_range"):
        db.get_model_stats(CLIENT_A, v["model_id"], **kw)


# ---------------------------------------------------------------------------- misc
def test_catalog_and_ping(db):
    assert db.ping() == {"status": "ok"}
    assert {d["algorithm"] for d in db.list_model_definitions()} == {"sgd_regressor", "sgd_logistic", "perceptron", "gaussian_nb"}


def test_entry_ids_are_unique_and_increasing(db):
    entries = [new_dataset(db, rid="a")["entry_id"]]
    ds = new_dataset(db, rid="a")
    entries.append(new_job(db, ds["dataset_id"], rid="j1")["entry_id"])
    entries.append(db.cancel_job(CLIENT_A, "c", new_job(db, ds["dataset_id"], rid="j2")["job_id"])["entry_id"])
    numbers = [int(e.split(":")[1]) for e in entries]
    assert len(set(numbers)) == 3 and numbers == sorted(numbers)
