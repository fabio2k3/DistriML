"""Genera datasets de ejemplo en data/samples/ con el formato del contrato (cabecera + etiqueta al final)."""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.datasets import load_diabetes, load_iris

OUT = Path(__file__).resolve().parent.parent / "data" / "samples"


def write(name: str, bunch, label_name: str, integer_label: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cols = [c.replace(" (cm)", "").replace(" ", "_") for c in bunch.feature_names] + [label_name]
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row, y in zip(bunch.data, bunch.target):
            w.writerow([f"{v:.6g}" for v in row] + [int(y) if integer_label else f"{y:.6g}"])
    print("escrito", OUT / name)


def write_named(name: str, bunch, label_name: str) -> None:
    """Igual que `write` para clasificación, pero con el NOMBRE de la clase como etiqueta (texto)."""
    cols = [c.replace(" (cm)", "").replace(" ", "_") for c in bunch.feature_names] + [label_name]
    with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for row, y in zip(bunch.data, bunch.target):
            w.writerow([f"{v:.6g}" for v in row] + [bunch.target_names[int(y)]])
    print("escrito", OUT / name)


if __name__ == "__main__":
    write("iris.csv", load_iris(), "species", True)          # clasificación, 3 clases
    write("diabetes.csv", load_diabetes(), "progression", False)  # regresión
    write_named("iris_names.csv", load_iris(), "species")  # clasificación con la etiqueta en texto (setosa, versicolor, virginica)
