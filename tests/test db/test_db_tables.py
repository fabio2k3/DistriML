"""CSV y codificación de filas: db/csvtable.py, db/records.py y db/tables.py."""
import csv

import pytest

from distriml.db import tables as T
from distriml.db.csvtable import CsvTable, dumps, loads
from distriml.db.records import decode, encode


def test_json_cells_roundtrip():
    value = {"a": [1, 2.5, None], "ñ": "texto, con \"comillas\"\ny salto"}
    assert loads(dumps(value)) == value
    assert loads("") is None


def test_csv_table_append_load_and_rewrite(tmp_path):
    t = CsvTable(tmp_path / "x.csv", ["id", "texto"])
    assert t.load() == []
    t.append({"id": "1", "texto": "hola, mundo"})
    t.append({"id": "2", "texto": "línea1\nlínea2"})
    assert t.load() == [{"id": "1", "texto": "hola, mundo"}, {"id": "2", "texto": "línea1\nlínea2"}]
    t.rewrite([{"id": "9", "texto": None}])
    assert t.load() == [{"id": "9", "texto": ""}]
    assert [p.name for p in tmp_path.iterdir()] == ["x.csv"]          # sin ficheros temporales


def test_csv_table_has_header_and_ignores_unknown_keys(tmp_path):
    t = CsvTable(tmp_path / "x.csv", ["id"])
    t.append({"id": "1", "extra": "no se guarda"})
    with open(t.path, newline="", encoding="utf-8") as fh:
        assert list(csv.reader(fh)) == [["id"], ["1"]]


def test_tables_container_creates_every_file(tmp_path):
    T.Tables(tmp_path)
    names = sorted(p.name for p in (tmp_path / "db").iterdir())
    assert names == ["attempts.csv", "datasets.csv", "inference_stats.csv", "job_batches.csv", "jobs.csv",
                     "model_versions.csv", "models.csv", "requests.csv"]


def test_job_row_roundtrip_through_csv(tmp_path):
    t = T.Tables(tmp_path)
    job = {"job_id": "j", "client_id": "c", "dataset_id": "d", "model_def_id": "m",
           "hyperparams": {"alpha": 1e-9, "epochs": 3, "eta0": 0.01}, "total_batches": 30, "estado": "PENDIENTE",
           "attempts_count": 1, "lost_count": 2, "last_confirmed_batch": 7, "attempt_vigente": None,
           "model_id": None, "canonical_version_id": None, "model_version": None, "completed_at": None,
           "created_at": 123, "client_request_id": "r"}
    t.jobs.rewrite([encode("jobs", job)])
    assert decode("jobs", t.jobs.load()[0]) == job


def test_empty_text_stays_empty_but_optional_text_becomes_none(tmp_path):
    t = T.Tables(tmp_path)
    ds = {"dataset_id": "d", "client_id": "c", "name": "", "sha256": "s", "size": 1, "n_rows": 2, "n_features": 1,
          "classes": None, "feature_mean": [0.5], "feature_std": [1.0], "created_at": 1, "client_request_id": "r"}
    t.datasets.append(encode("datasets", ds))
    back = decode("datasets", t.datasets.load()[0])
    assert back["name"] == "" and back["classes"] is None and back == ds
    attempt = {"attempt_id": "a", "job_id": "j", "worker_id": "w", "base_attempt": None, "base_batch": 0,
               "estado": "ACTIVO", "revoke_reason": None, "created_at": 5}
    t.attempts.append(encode("attempts", attempt))
    assert decode("attempts", t.attempts.load()[0]) == attempt


def test_booleans_and_floats_roundtrip(tmp_path):
    t = T.Tables(tmp_path)
    v = {"model_id": "m", "version_id": "v", "version": 1, "sha256": "s", "metrics": {"accuracy": 0.9},
         "estado": "ENTRENADO", "canonical": True, "entry_id": "db:1"}
    t.model_versions.append(encode("model_versions", {**v, "canonical": False}))
    t.model_versions.append(encode("model_versions", v))
    rows = [decode("model_versions", r) for r in t.model_versions.load()]
    assert [r["canonical"] for r in rows] == [False, True]
    s = {"model_id": "m", "version_id": "v", "minute": 7, "count": 3, "latency_sum": 1.25}
    t.inference_stats.append(encode("inference_stats", s))
    assert decode("inference_stats", t.inference_stats.load()[0]) == s


def test_rewrite_failure_keeps_previous_file(tmp_path, monkeypatch):
    t = CsvTable(tmp_path / "x.csv", ["id"])
    t.append({"id": "1"})
    monkeypatch.setattr("os.replace", lambda *a: (_ for _ in ()).throw(OSError("disco lleno")))
    with pytest.raises(OSError):
        t.rewrite([{"id": "2"}])
    monkeypatch.undo()
    assert t.load() == [{"id": "1"}]
    assert [p.name for p in tmp_path.iterdir()] == ["x.csv"]
