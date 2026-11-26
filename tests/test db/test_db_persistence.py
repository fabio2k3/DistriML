"""`Database` — persistencia: el estado sobrevive a un reinicio y las escrituras cortadas a mitad se reparan.

"Reiniciar" = construir otra `Database` sobre el mismo `data_dir` (es lo que hace Ray al reiniciar el actor).
Los cortes se simulan restaurando una copia anterior de un CSV: dejan exactamente el estado que dejaría
un fallo entre dos escrituras de una misma operación.
"""
import pytest

from distriml.common import constants as C
from distriml.common.errors import ContractError
from distriml.common.ids import sha256_hex
from tests.helpers.db_utils import CLIENT_A, CLIENT_B, IRIS, FakeClock, make_db, new_dataset, new_job, start_job, train

EXPIRY = 11_000                     # lease 10 s + margen 1 s


def reopen(db, clock=None, wall=None, **cfg):
    return make_db(db.cfg.data_dir, clock or FakeClock(5_000_000), wall=wall, **cfg)


def csv_path(db, name):
    return db.cfg.data_dir / "db" / f"{name}.csv"


def view(db, job):
    return db.get_job(CLIENT_A, job["job_id"])


# ---------------------------------------------------------------------------- reinicio normal
def test_everything_survives_a_restart(tmp_path):
    db = make_db(tmp_path, FakeClock(), wall=FakeClock(60_000 * 50))
    ds, done_job, a = start_job(db, rid="done")
    train(db, "w1", a)
    model_id = view(db, done_job)["model_id"]
    for _ in range(3):
        db.predict(CLIENT_A, model_id, {"features": [5.1, 3.5, 1.4, 0.2]})
    cancelled = new_job(db, ds["dataset_id"], rid="cancelled")
    db.cancel_job(CLIENT_A, "c1", cancelled["job_id"])
    pending = new_job(db, ds["dataset_id"], rid="pending")
    new_dataset(db, CLIENT_B, rid="ds-b")

    before = {
        "jobs": db.list_jobs(CLIENT_A), "done": view(db, done_job), "cancelled": view(db, cancelled),
        "pending": view(db, pending), "datasets_a": db.list_datasets(CLIENT_A), "datasets_b": db.list_datasets(CLIENT_B),
        "metrics": db.get_job_metrics(CLIENT_A, done_job["job_id"]), "versions": db.list_model_versions(CLIENT_A, model_id),
        "stats": db.get_model_stats(CLIENT_A, model_id), "dataset": db.get_dataset(CLIENT_A, ds["dataset_id"]),
    }
    again = reopen(db, wall=FakeClock(60_000 * 50))
    after = {
        "jobs": again.list_jobs(CLIENT_A), "done": view(again, done_job), "cancelled": view(again, cancelled),
        "pending": view(again, pending), "datasets_a": again.list_datasets(CLIENT_A), "datasets_b": again.list_datasets(CLIENT_B),
        "metrics": again.get_job_metrics(CLIENT_A, done_job["job_id"]), "versions": again.list_model_versions(CLIENT_A, model_id),
        "stats": again.get_model_stats(CLIENT_A, model_id), "dataset": again.get_dataset(CLIENT_A, ds["dataset_id"]),
    }
    assert after == before
    assert again.predict(CLIENT_A, model_id, {"features": [5.1, 3.5, 1.4, 0.2]})["version_id"] == a.attempt_id
    assert again.get_model_stats(CLIENT_A, model_id)["total_calls"] == 4
    assert again.request_job("w1").job_id == pending["job_id"]            # el pendiente sigue asignable


def test_restart_keeps_creation_order_and_isolation(tmp_path):
    db = make_db(tmp_path)
    ds = new_dataset(db)
    ids_ = [new_job(db, ds["dataset_id"], rid=f"j{i}")["job_id"] for i in range(4)]
    again = reopen(db)
    assert [j["job_id"] for j in again.list_jobs(CLIENT_A)] == ids_
    assert again.list_jobs(CLIENT_B) == []
    assert again.request_job("w1").job_id == ids_[0]


def test_special_characters_in_names_survive(tmp_path):
    db = make_db(tmp_path)
    name = 'datos, "raros"\ny ñandú ✓'
    ds = new_dataset(db, name=name)
    empty = new_dataset(db, rid="vacio", data=IRIS + b"1,1,1,1,0\n", name="")
    again = reopen(db)
    assert again.get_dataset(CLIENT_A, ds["dataset_id"])["name"] == name
    assert again.get_dataset(CLIENT_A, empty["dataset_id"])["name"] == ""


def test_responses_to_repeated_requests_survive_a_restart(tmp_path):
    db = make_db(tmp_path)
    ds = new_dataset(db, rid="x")
    job = new_job(db, ds["dataset_id"], rid="y")
    cancel = db.cancel_job(CLIENT_A, "z", job["job_id"])
    again = reopen(db)
    assert new_dataset(again, rid="x") == ds
    assert new_job(again, ds["dataset_id"], rid="y") == job
    assert again.cancel_job(CLIENT_A, "z", job["job_id"]) == cancel
    with pytest.raises(ContractError) as e:
        new_job(again, ds["dataset_id"], rid="y", hyperparams={"epochs": 9})
    assert e.value.status == 409
    assert len(again.list_datasets(CLIENT_A)) == 1 and len(again.list_jobs(CLIENT_A)) == 1


def test_entry_ids_keep_increasing_after_restart(tmp_path):
    db = make_db(tmp_path)
    _, job, a = start_job(db)
    train(db, "w1", a)                                          # checkpoints y cierre también consumen entry_id
    last = int(db.cancel_job(CLIENT_A, "c", new_job(db, db.list_datasets(CLIENT_A)[0]["dataset_id"], rid="n")["job_id"])["entry_id"].split(":")[1])
    again = reopen(db)
    nxt = int(new_dataset(again, rid="otro", data=IRIS + b"1,1,1,1,0\n")["entry_id"].split(":")[1])
    assert nxt > last


# ---------------------------------------------------------------------------- jobs en curso
def test_running_attempt_survives_restart_and_workers_can_keep_going(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    train(db, "w1", a, upto=3, finish=False)
    clock = FakeClock(9_000_000)
    again = reopen(db, clock)
    v = view(again, job)
    assert v["estado"] == C.EN_EJECUCION and v["attempt_vigente"] == a.attempt_id and v["last_confirmed_batch"] == 3
    assert again.renew_lease("w1", a.attempt_id) == C.RENEW_CONTINUE
    train(again, "w1", a)                                       # los batches 1-3 ya confirmados se aceptan como repetidos
    assert view(again, job)["estado"] == C.COMPLETADO


def test_after_a_restart_live_attempts_get_a_full_new_lease(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    train(db, "w1", a, upto=1, finish=False)
    clock = FakeClock(9_000_000)
    again = reopen(db, clock)
    clock.advance(EXPIRY)                                       # el plazo cuenta desde el arranque de la DB
    assert view(again, job)["estado"] == C.EN_EJECUCION
    clock.advance(1)
    v = view(again, job)
    assert v["estado"] == C.PENDIENTE and v["attempts"][0]["revoke_reason"] == C.WORKER_LOST     # ya había confirmado un batch


def test_attempt_without_progress_expires_as_never_started_after_restart(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    clock = FakeClock(9_000_000)
    again = reopen(db, clock)
    clock.advance(EXPIRY + 1)
    v = view(again, job)
    assert v["attempts"][0]["revoke_reason"] == C.NEVER_STARTED and (v["lost_count"], v["attempts_count"]) == (0, 0)


def test_counters_survive_restart(tmp_path):
    db = make_db(tmp_path, FakeClock(), max_attempts=5)
    _, job, a = start_job(db)
    db.report_failure("w1", a.attempt_id, "x")
    b = db.request_job("w1")
    db.report_failure("w1", b.attempt_id, "y")
    again = reopen(db, max_attempts=5)
    v = view(again, job)
    assert (v["attempts_count"], v["lost_count"], v["estado"]) == (2, 0, C.PENDIENTE)
    assert [x["revoke_reason"] for x in v["attempts"]] == [C.WORKER_ERROR, C.WORKER_ERROR]


def test_reassignment_delay_is_volatile_and_resets_on_restart(tmp_path):
    clock = FakeClock()
    db = make_db(tmp_path, clock)
    _, job, a = start_job(db)
    db.renew_lease("w1", a.attempt_id)
    clock.advance(EXPIRY + 1)
    assert db.request_job("w2") is None                         # en retardo
    again = reopen(db)
    assert again.request_job("w2") is not None                  # el retardo no se guarda: tras reiniciar se reasigna ya


def test_resume_after_restart_continues_from_checkpoint(tmp_path):
    clock = FakeClock()
    db = make_db(tmp_path, clock)
    _, job, a = start_job(db)
    sha = train(db, "w1", a, upto=4, finish=False)
    clock2 = FakeClock()
    again = reopen(db, clock2)
    clock2.advance(EXPIRY + 70_000)                             # el worker no volvió
    b = again.request_job("w2")
    assert (b.next_batch, b.checkpoint_sha256) == (5, sha)
    train(again, "w2", b)
    assert view(again, job)["estado"] == C.COMPLETADO


# ---------------------------------------------------------------------------- cortes a mitad de operación
def test_batches_confirmed_but_not_reflected_in_jobs_are_recovered(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    snapshot = csv_path(db, "jobs").read_bytes()               # jobs.csv antes de confirmar nada
    train(db, "w1", a, upto=3, finish=False)
    csv_path(db, "jobs").write_bytes(snapshot)                  # corte: job_batches se escribió, jobs no
    again = reopen(db)
    assert view(again, job)["last_confirmed_batch"] == 3
    assert again.tables.jobs.load()[0]["last_confirmed_batch"] == "3"        # y quedó reparado en disco
    assert view(reopen(again), job)["last_confirmed_batch"] == 3


def test_job_with_its_model_already_stored_is_closed_on_restart(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    last_sha = train(db, "w1", a, finish=False)
    jobs_before, attempts_before = csv_path(db, "jobs").read_bytes(), csv_path(db, "attempts").read_bytes()
    assert db.complete_job("w1", a.attempt_id, last_sha, {"accuracy": 0.9, "mse": 0.1, "n": 150}) == C.COMPLETE_OK
    csv_path(db, "jobs").write_bytes(jobs_before)               # corte: models/model_versions escritos, job sin cerrar
    csv_path(db, "attempts").write_bytes(attempts_before)
    again = reopen(db)
    v = view(again, job)
    assert v["estado"] == C.COMPLETADO and v["canonical_version_id"] == a.attempt_id and v["model_version"] == 1
    assert v["attempt_vigente"] is None and v["attempts"][0]["estado"] == C.ATTEMPT_REVOCADO
    assert v["attempts"][0]["revoke_reason"] is None
    assert again.predict(CLIENT_A, v["model_id"], {"features": [5.1, 3.5, 1.4, 0.2]})["version_id"] == a.attempt_id
    assert again.request_job("w1") is None


def test_attempt_written_but_job_not_updated_is_revoked_and_job_stays_assignable(tmp_path):
    db = make_db(tmp_path, FakeClock())
    ds = new_dataset(db)
    job = new_job(db, ds["dataset_id"])
    snapshot = csv_path(db, "jobs").read_bytes()
    a = db.request_job("w1")
    csv_path(db, "jobs").write_bytes(snapshot)                  # corte: attempt escrito, job sin actualizar
    again = reopen(db)
    v = view(again, job)
    assert v["estado"] == C.PENDIENTE and v["attempt_vigente"] is None
    assert [(x["estado"], x["revoke_reason"]) for x in v["attempts"]] == [(C.ATTEMPT_REVOCADO, C.NEVER_STARTED)]
    assert (v["attempts_count"], v["lost_count"]) == (0, 0)
    assert again.renew_lease("w1", a.attempt_id) == C.RENEW_REVOKED
    b = again.request_job("w2")
    assert b is not None and b.attempt_id != a.attempt_id


def test_cancelled_job_with_leftover_active_attempt_is_cleaned_up(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    attempts_before = csv_path(db, "attempts").read_bytes()
    db.cancel_job(CLIENT_A, "c", job["job_id"])
    csv_path(db, "attempts").write_bytes(attempts_before)       # corte: job guardado como CANCELADO, attempt aún ACTIVO
    v = view(reopen(db), job)
    assert v["estado"] == C.CANCELADO
    assert [(x["estado"], x["revoke_reason"]) for x in v["attempts"]] == [(C.ATTEMPT_REVOCADO, C.CANCEL)]


def test_running_job_whose_attempt_row_is_missing_goes_back_to_pending(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    csv_path(db, "attempts").write_text(",".join(db.tables.attempts.columns) + "\n", encoding="utf-8")
    again = reopen(db)
    v = view(again, job)
    assert v["estado"] == C.PENDIENTE and v["attempt_vigente"] is None
    assert again.request_job("w1") is not None


def test_dataset_stored_without_its_response_record_answers_a_retry(tmp_path):
    db = make_db(tmp_path)
    first = new_dataset(db, rid="x")
    csv_path(db, "requests").write_text(",".join(db.tables.requests.columns) + "\n", encoding="utf-8")
    again = reopen(db)                                          # corte: dataset guardado, respuesta no
    retry = new_dataset(again, rid="x")
    assert retry["dataset_id"] == first["dataset_id"] and (retry["n_rows"], retry["classes"]) == (150, [0, 1, 2])
    assert new_dataset(again, rid="x") == retry                 # y a partir de ahí la respuesta queda fija
    assert len(again.list_datasets(CLIENT_A)) == 1


# ---------------------------------------------------------------------------- estadísticas
def test_inference_counters_are_compacted_on_startup(tmp_path):
    wall = FakeClock(60_000 * 7)
    db = make_db(tmp_path, FakeClock(), wall=wall)
    _, job, a = start_job(db)
    train(db, "w1", a)
    mid = view(db, job)["model_id"]
    for _ in range(5):
        db.predict(CLIENT_A, mid, {"features": [5.1, 3.5, 1.4, 0.2]})
    total = db.get_model_stats(CLIENT_A, mid)
    raw = csv_path(db, "inference_stats").read_text(encoding="utf-8").strip().splitlines()
    assert len(raw) == 1 + 5                                    # cabecera + una fila por predicción
    again = reopen(db, wall=wall)
    compact = csv_path(again, "inference_stats").read_text(encoding="utf-8").strip().splitlines()
    assert len(compact) == 1 + 1
    after = again.get_model_stats(CLIENT_A, mid)
    assert after["total_calls"] == 5 == total["total_calls"]
    assert after["mean_latency_ms"] == pytest.approx(total["mean_latency_ms"])
    again.predict(CLIENT_A, mid, {"features": [5.1, 3.5, 1.4, 0.2]})
    assert reopen(again, wall=wall).get_model_stats(CLIENT_A, mid)["total_calls"] == 6


def test_model_binary_is_not_trusted_if_corrupted(tmp_path):
    db = make_db(tmp_path, FakeClock())
    _, job, a = start_job(db)
    train(db, "w1", a)
    v = view(db, job)
    sha = db.list_model_versions(CLIENT_A, v["model_id"])[0]["sha256"]
    path = db.store.path_for(sha)
    path.write_bytes(path.read_bytes() + b"x")                  # corrupción en disco
    again = reopen(db)
    with pytest.raises(ContractError) as e:
        again.predict(CLIENT_A, v["model_id"], {"features": [1, 2, 3, 4]})
    assert (e.value.status, e.value.reason) == (503, "artifact_unavailable")
    assert again.get_model_stats(CLIENT_A, v["model_id"])["total_calls"] == 0


def test_trained_models_survive_a_restart(tmp_path):
    db = make_db(tmp_path)
    models = []
    for i in range(2):
        _, job, a = start_job(db, rid=f"m{i}")
        train(db, "w1", a)
        models.append(view(db, job)["model_id"])
    before = db.list_models(CLIENT_A)
    assert [m["model_id"] for m in before] == models
    again = reopen(db)
    assert again.list_models(CLIENT_A) == before
    assert again.list_models(CLIENT_B) == []
    assert again.predict(CLIENT_A, models[1], {"features": [5.1, 3.5, 1.4, 0.2]})["version_id"] == before[1]["canonical_version_id"]


# ---------------------------------------------------------------------------- borrados y datasets repetidos
def test_deleted_model_stays_deleted_after_a_restart(tmp_path):
    db = make_db(tmp_path)
    _, job, a = start_job(db, rid="m")
    train(db, "w1", a)
    model_id = view(db, job)["model_id"]
    ver = db.list_model_versions(CLIENT_A, model_id)[0]
    deleted = db.delete_model_version(CLIENT_A, "d1", model_id, ver["version_id"])
    again = reopen(db)
    assert again.list_models(CLIENT_A) == []
    with pytest.raises(ContractError) as exc:
        again.predict(CLIENT_A, model_id, {"features": [5.1, 3.5, 1.4, 0.2]})
    assert exc.value.status == 404
    assert again.delete_model_version(CLIENT_A, "d1", model_id, ver["version_id"]) == deleted      # el reintento sigue siendo idempotente
    assert view(again, job)["estado"] == C.COMPLETADO and csv_path(db, "model_versions").read_text().count("\n") == 1


def test_entry_ids_keep_growing_after_deleting_the_newest_model(tmp_path):
    db = make_db(tmp_path)
    _, job, a = start_job(db, rid="m")
    train(db, "w1", a)
    model_id = view(db, job)["model_id"]
    ver = db.list_model_versions(CLIENT_A, model_id)[0]
    deleted = db.delete_model_version(CLIENT_A, "d1", model_id, ver["version_id"])
    nxt = reopen(db).create_dataset(CLIENT_A, "otro", "n.csv", sha256_hex(b"a,y\n1,0\n2,1\n"), 12, b"a,y\n1,0\n2,1\n")
    assert int(nxt["entry_id"].split(":")[1]) > int(deleted["entry_id"].split(":")[1])


def test_a_model_left_without_versions_by_a_cut_is_discarded_on_startup(tmp_path):
    db = make_db(tmp_path)
    _, job, a = start_job(db, rid="m")
    train(db, "w1", a)
    assert len(db.list_models(CLIENT_A)) == 1
    db.tables.model_versions.rewrite([])                         # corte entre quitar la versión y quitar el modelo
    again = reopen(db)
    assert again.list_models(CLIENT_A) == []
    assert csv_path(db, "models").read_text().count("\n") == 1   # solo la cabecera: el reinicio dejó el fichero limpio


def test_duplicate_dataset_detection_survives_a_restart(tmp_path):
    db = make_db(tmp_path)
    first = new_dataset(db, rid="a")
    again = reopen(db)
    dup = new_dataset(again, rid="b")
    assert dup["dataset_id"] == first["dataset_id"] and dup["already_existed"] is True
    assert len(again.list_datasets(CLIENT_A)) == 1


def test_promotion_of_another_version_survives_a_restart(tmp_path):
    from distriml.db.records import encode
    db = make_db(tmp_path)
    _, job, a = start_job(db, rid="m")
    train(db, "w1", a)
    model_id = view(db, job)["model_id"]
    v1 = db._versions[model_id][0]
    v2 = {**v1, "version_id": "version-2", "version": 2, "canonical": False, "entry_id": db._next_entry_id()}
    db.tables.model_versions.append(encode("model_versions", v2))
    db._versions[model_id].append(v2)
    db.delete_model_version(CLIENT_A, "d1", model_id, v1["version_id"])
    again = reopen(db)
    (listed,) = again.list_models(CLIENT_A)
    assert listed["canonical_version_id"] == "version-2" and listed["versions"] == 1
    (ver,) = again.list_model_versions(CLIENT_A, model_id)
    assert ver["version_id"] == "version-2" and ver["canonical"] is True
