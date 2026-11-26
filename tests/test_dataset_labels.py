"""Etiquetas de texto en el CSV (setosa, versicolor...): parseo, modelo y respuesta con el nombre de la clase."""
import numpy as np
import pytest

from distriml.common import algorithms, dataset
from distriml.common.dataset import DatasetMeta
from distriml.common.errors import ContractError
from tests.conftest import SAMPLES

NAMED = (
    b"a,b,especie\n"
    b"1,1,zorro\n2,1,gato\n3,2,perro\n4,2,gato\n5,3,zorro\n6,3,perro\n"
)


def test_text_labels_become_alphabetical_indices_and_names():
    X, y, names = dataset._read(NAMED)
    assert names == ["gato", "perro", "zorro"]
    assert y.tolist() == [2, 0, 1, 0, 2, 1]
    assert X.shape == (6, 2)
    assert dataset.class_names_of(NAMED) == names


def test_text_labels_give_classes_zero_to_k_minus_one():
    meta = dataset.analyze(NAMED)
    assert meta.classes == [0, 1, 2] and meta.n_rows == 6 and meta.n_features == 2


def test_numeric_labels_keep_working_and_have_no_names():
    data = b"a,y\n1,0\n2,1\n3,1\n4,0\n"
    assert dataset.class_names_of(data) is None
    assert dataset.analyze(data).classes == [0, 1]
    assert dataset.analyze(b"a,y\n1,0.5\n2,1.5\n3,2.5\n").classes is None        # regresión


def test_labels_are_trimmed_and_case_sensitive():
    data = b"a,y\n1, si\n2,si \n3,No\n4,no\n"
    assert dataset.class_names_of(data) == ["No", "no", "si"]


def test_numbers_and_text_mixed_in_the_label_are_text():
    assert dataset.class_names_of(b"a,y\n1,1\n2,uno\n3,1\n") == ["1", "uno"]


@pytest.mark.parametrize("data, fragment", [
    (b"a,y\n1,x\n2,\n3,x\n", "etiquetas vacías"),
    (b"a,y\nuno,x\n2,y\n", "características"),                       # el texto sigue prohibido en las características
    (b"a,y\n1,nan\n2,1\n", "no finitos"),
    ("a,y\n" + "".join(f"{i},c{i}\n" for i in range(60)), "demasiadas clases"),
])
def test_invalid_label_columns_are_400(data, fragment):
    data = data if isinstance(data, bytes) else data.encode()
    with pytest.raises(ContractError) as exc:
        dataset.analyze(data)
    assert exc.value.status == 400 and fragment in exc.value.message


def test_training_uses_the_same_indices_in_every_process():
    meta = dataset.analyze(NAMED)
    X1, y1 = dataset.load_for_training(NAMED, meta, "classification")
    X2, y2 = dataset.load_for_training(NAMED, meta, "classification")
    assert y1.tolist() == y2.tolist() == [2, 0, 1, 0, 2, 1] and y1.dtype == np.int64


def test_iris_with_names_is_the_same_data_as_numeric_iris():
    numeric = dataset.parse((SAMPLES / "iris.csv").read_bytes())
    named = dataset.parse((SAMPLES / "iris_names.csv").read_bytes())
    assert np.array_equal(numeric[0], named[0]) and np.array_equal(numeric[1], named[1])
    assert dataset.class_names_of((SAMPLES / "iris_names.csv").read_bytes()) == ["setosa", "versicolor", "virginica"]


# ---------------------------------------------------------------------------- el modelo recuerda los nombres
def _train(data: bytes, with_names=True):
    meta = dataset.analyze(data)
    X, y = dataset.load_for_training(data, meta, "classification")
    state = algorithms.init_state("sgd_logistic", {"seed": 1, "alpha": 1e-4, "eta0": 0.1}, meta,
                                  dataset.class_names_of(data) if with_names else None)
    for _ in range(30):
        algorithms.train_batch(state, "sgd_logistic", X, y)
    return state


def test_label_of_returns_the_name_of_the_predicted_class():
    iris = (SAMPLES / "iris_names.csv").read_bytes()
    state = _train(iris)
    for features, name in (([5.0, 3.4, 1.5, 0.2], "setosa"), ([6.7, 3.0, 5.2, 2.3], "virginica")):
        pred = algorithms.predict(state, "sgd_logistic", features)
        assert algorithms.label_of(state, pred) == name


def test_label_of_is_none_without_names_or_for_unknown_class():
    state = _train((SAMPLES / "iris.csv").read_bytes(), with_names=False)
    assert algorithms.label_of(state, 1) is None
    named = _train((SAMPLES / "iris_names.csv").read_bytes())
    assert algorithms.label_of(named, 7) is None
    old_state = {k: v for k, v in named.items() if k != "class_names"}              # checkpoints de antes de esta función
    assert algorithms.label_of(old_state, 1) is None


def test_state_with_names_survives_serialization():
    state = _train((SAMPLES / "iris_names.csv").read_bytes())
    again = algorithms.deserialize_state(algorithms.serialize_state(state))
    assert again["class_names"] == ["setosa", "versicolor", "virginica"]
