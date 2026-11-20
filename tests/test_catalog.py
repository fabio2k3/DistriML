import pytest

from distriml.common import catalog, dataset
from distriml.common.errors import ContractError


def test_four_models_with_stable_ids():
    defs = catalog.list_definitions()
    assert len(defs) == 4 and len({d["model_def_id"] for d in defs}) == 4


def test_merge_defaults_and_overrides():
    md = catalog.BY_ALGORITHM["sgd_logistic"]
    hp = catalog.merge_hyperparams(md, {"epochs": 3})
    assert hp["epochs"] == 3 and hp["seed"] == 42 and hp["batch_size"] == 32


def test_gaussian_nb_epochs_fixed():
    assert catalog.merge_hyperparams(catalog.BY_ALGORITHM["gaussian_nb"], None)["epochs"] == 1
    with pytest.raises(ContractError):
        catalog.merge_hyperparams(catalog.BY_ALGORITHM["gaussian_nb"], {"epochs": 5})


@pytest.mark.parametrize("hp", [{"epochs": 0}, {"epochs": True}, {"foo": 1}, {"eta0": 0}, {"batch_size": 1.5}, {"alpha": float("nan")}])
def test_invalid_hyperparams(hp):
    with pytest.raises(ContractError) as e:
        catalog.merge_hyperparams(catalog.BY_ALGORITHM["sgd_logistic"], hp)
    assert e.value.status == 400


def test_unknown_model_and_compat():
    with pytest.raises(ContractError):
        catalog.get("nope")
    meta = dataset.analyze(b"a,y\n1,0.5\n2,1.5\n3,2.5\n")
    with pytest.raises(ContractError):
        catalog.check_compatibility(catalog.BY_ALGORITHM["sgd_logistic"], meta)
    catalog.check_compatibility(catalog.BY_ALGORITHM["sgd_regressor"], meta)
