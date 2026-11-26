"""Identidad local de la consola (distriml/client/identity.py): se guarda en $DISTRIML_HOME/identity.json."""
import json
import os
import re
from pathlib import Path

import pytest

from distriml.client import identity


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIML_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_home_defaults_to_dot_distriml_in_the_user_home(monkeypatch, tmp_path):
    monkeypatch.delenv("DISTRIML_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert identity.home_dir() == tmp_path / ".distriml"


def test_env_variable_overrides_the_home(home):
    assert identity.home_dir() == home


def test_init_creates_username_random_client_id_and_no_last_entry(home):
    data = identity.init_identity("ana")
    assert data["username"] == "ana" and data["last_entry"] is None
    assert re.fullmatch(r"[0-9a-f]{64}", data["client_id"])
    assert json.loads((home / "identity.json").read_text(encoding="utf-8")) == data
    assert identity.load_identity() == data


def test_two_identities_with_the_same_username_get_different_client_ids(home):
    first = identity.init_identity("ana")["client_id"]
    second = identity.init_identity("ana", force=True)["client_id"]
    assert first != second


def test_init_refuses_to_overwrite_without_force_and_keeps_the_old_identity(home):
    original = identity.init_identity("ana")
    with pytest.raises(FileExistsError, match="--force"):
        identity.init_identity("otra")
    assert identity.load_identity() == original


def test_force_replaces_the_identity(home):
    identity.init_identity("ana")
    new = identity.init_identity("otra", force=True)
    assert identity.load_identity()["username"] == "otra" and identity.load_identity() == new


def test_load_without_init_explains_what_to_do(home):
    with pytest.raises(identity.NotInitialized, match="distriml init"):
        identity.load_identity()


def test_last_entry_is_saved_and_the_rest_is_kept(home):
    original = identity.init_identity("ana")
    identity.save_last_entry("db:42")
    saved = identity.load_identity()
    assert saved["last_entry"] == "db:42"
    assert (saved["username"], saved["client_id"]) == (original["username"], original["client_id"])
    identity.save_last_entry("db:43")
    assert identity.load_identity()["last_entry"] == "db:43"


def test_saving_an_entry_without_identity_fails_cleanly(home):
    with pytest.raises(identity.NotInitialized):
        identity.save_last_entry("db:1")


def test_no_temporary_file_is_left_behind(home):
    identity.init_identity("ana")
    identity.save_last_entry("db:1")
    assert [p.name for p in home.iterdir()] == ["identity.json"]


@pytest.mark.skipif(os.name == "nt", reason="los permisos POSIX no se aplican en Windows")
def test_identity_file_is_private_to_the_user(home):
    identity.init_identity("ana")
    assert (home / "identity.json").stat().st_mode & 0o777 == 0o600
