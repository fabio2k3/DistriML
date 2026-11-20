import numpy as np
import pytest

from distriml.common import batching


def test_total_batches():
    assert batching.total_batches(150, 10, 32) == 10 * 5
    assert batching.total_batches(64, 1, 32) == 2


def test_deterministic_and_covers_each_epoch():
    n, bs, seed = 100, 32, 7
    per = batching.batches_per_epoch(n, bs)
    for epoch in range(3):
        rows = np.concatenate([batching.batch_indices(n, bs, seed, epoch * per + j + 1) for j in range(per)])
        assert sorted(rows) == list(range(n))          # cada fila exactamente una vez por época
    assert np.array_equal(batching.batch_indices(n, bs, seed, 5), batching.batch_indices(n, bs, seed, 5))
    assert not np.array_equal(batching.batch_indices(n, bs, 7, 1), batching.batch_indices(n, bs, 8, 1))


def test_epochs_differ_and_invalid_args():
    assert not np.array_equal(batching.batch_indices(100, 100, 1, 1), batching.batch_indices(100, 100, 1, 2))
    with pytest.raises(ValueError):
        batching.batch_indices(10, 2, 1, 0)
