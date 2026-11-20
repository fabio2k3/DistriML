import pytest

from distriml.common import dataset
from distriml.common.errors import ContractError

OK = b"a,b,y\n1,2,0\n3,4,1\n5,6,0\n"


def test_analyze_ok():
    m = dataset.analyze(OK)
    assert (m.n_rows, m.n_features, m.classes) == (3, 2, [0, 1])
    assert m.feature_mean == pytest.approx([3.0, 4.0])


def test_regression_labels_have_no_classes():
    assert dataset.analyze(b"a,y\n1,0.5\n2,1.5\n3,2.5\n").classes is None


def test_constant_feature_std_replaced_by_one():
    assert dataset.analyze(b"a,b,y\n1,2,0\n1,3,1\n").feature_std[0] == 1.0


@pytest.mark.parametrize("bad", [b"", b"a,y\n1,2\n", b"a,y\n1,x\n2,3\n", b"a,y\n1,2\n3\n", b"a,y\n1,nan\n2,3\n", b"\xff\xfe"])
def test_invalid_rejected_with_400(bad):
    with pytest.raises(ContractError) as e:
        dataset.analyze(bad)
    assert e.value.status == 400


def test_size_limit():
    with pytest.raises(ContractError):
        dataset.analyze(OK, max_bytes=5)


def test_load_for_training_standardizes():
    m = dataset.analyze(OK)
    X, y = dataset.load_for_training(OK, m, "classification")
    assert abs(X.mean()) < 1e-9 and y.dtype.kind == "i"
