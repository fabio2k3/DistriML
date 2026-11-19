"""Partición determinista en batches.

Reanudar un job, o repetir un batch tras un fallo, ve EXACTAMENTE las mismas filas porque la
partición depende solo de (n_rows, batch_size, seed, k). Los batches se numeran desde 1.
"""
from __future__ import annotations

import math

import numpy as np

MAX_SEED = 2**32 - 1  # RandomState exige semillas en [0, 2**32 - 1]


def batches_per_epoch(n_rows: int, batch_size: int) -> int:
    return math.ceil(n_rows / batch_size)


def total_batches(n_rows: int, epochs: int, batch_size: int) -> int:
    """total_batches = epochs × ⌈n_rows / batch_size⌉."""
    return epochs * batches_per_epoch(n_rows, batch_size)


def batch_indices(n_rows: int, batch_size: int, seed: int, k: int) -> np.ndarray:
    """Índices de las filas del batch `k` (k ≥ 1).

    epoca = (k-1) // n_batches_epoca ; j = (k-1) % n_batches_epoca
    perm  = RandomState([seed, epoca]).permutation(n_rows)
    filas = perm[j·batch_size : (j+1)·batch_size]   (el último batch de cada época puede ser menor)
    """
    if k < 1:
        raise ValueError("los batches se numeran desde 1")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError("seed fuera de rango")
    per_epoch = batches_per_epoch(n_rows, batch_size)
    epoch = (k - 1) // per_epoch
    j = (k - 1) % per_epoch
    perm = np.random.RandomState([seed, epoch]).permutation(n_rows)
    return perm[j * batch_size:(j + 1) * batch_size]
