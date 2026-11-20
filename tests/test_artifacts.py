import pytest

from distriml.common.artifacts import ArtifactStore
from distriml.common.errors import ContractError
from distriml.common.ids import sha256_hex


def test_put_get_roundtrip_and_idempotent(tmp_path):
    s = ArtifactStore(tmp_path)
    d = b"hola"
    assert s.put(sha256_hex(d), d) is True and s.put(sha256_hex(d), d) is False
    assert s.get(sha256_hex(d)) == d and s.has(sha256_hex(d))
    assert s.list_hashes() == [sha256_hex(d)]


def test_hash_mismatch_missing_and_corrupt(tmp_path):
    s = ArtifactStore(tmp_path)
    with pytest.raises(ContractError) as e:
        s.put(sha256_hex(b"a"), b"b")
    assert e.value.status == 400
    with pytest.raises(ContractError) as e:
        s.get(sha256_hex(b"zz"))
    assert (e.value.status, e.value.reason) == (503, "artifact_unavailable")
    s.put(sha256_hex(b"a"), b"a")
    s.path_for(sha256_hex(b"a")).write_bytes(b"corrupto")
    with pytest.raises(ContractError):
        s.get(sha256_hex(b"a"))


def test_invalid_sha(tmp_path):
    with pytest.raises(ContractError):
        ArtifactStore(tmp_path).has("xyz")
