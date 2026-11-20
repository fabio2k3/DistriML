from distriml.common import ids


def test_derivation_is_deterministic_and_separated():
    a = ids.derive_job_id("c1", "r1")
    assert a == ids.derive_job_id("c1", "r1")
    assert a != ids.derive_job_id("c1", "r2")
    assert a != ids.derive_job_id("c2", "r1")
    assert ids.derive_job_id("ab", "c") != ids.derive_job_id("a", "bc")
    assert ids.derive_dataset_id("c1", "r1") != a      # espacios de nombres distintos


def test_client_id_is_hex64_and_random():
    x, y = ids.new_client_id("ana"), ids.new_client_id("ana")
    assert ids.is_valid_client_id(x) and x != y
    assert not ids.is_valid_client_id("abc") and not ids.is_valid_client_id(None)


def test_model_def_id_is_stable():
    assert ids.derive_model_def_id("sgd_logistic", 1) == ids.derive_model_def_id("sgd_logistic", 1)
