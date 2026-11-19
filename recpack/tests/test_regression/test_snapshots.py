# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE

"""Snapshot (golden output) tests: outputs must match those stored in ``snapshots.npz``.

These tests guard refactorings that should not change behaviour. If an output changes on
purpose, regenerate the snapshots (see ``generate_snapshots.py``) and explain why in the PR.
"""

import numpy as np
import pytest

from recpack.tests.test_regression.cases import CASES
from recpack.tests.test_regression.generate_snapshots import SNAPSHOT_FILE

RTOL = 1e-6
ATOL = 1e-9


@pytest.fixture(scope="module")
def snapshots():
    with np.load(SNAPSHOT_FILE) as f:
        return dict(f)


def test_every_snapshot_has_a_case(snapshots):
    stored_cases = {key.rsplit("/", 1)[0] for key in snapshots}
    assert stored_cases == set(CASES)


@pytest.mark.parametrize("case", list(CASES))
def test_snapshot(case, snapshots):
    outputs = CASES[case]()
    expected_keys = {key.rsplit("/", 1)[1] for key in snapshots if key.rsplit("/", 1)[0] == case}
    assert set(outputs) == expected_keys

    for key, value in outputs.items():
        expected = snapshots[f"{case}/{key}"]
        assert value.shape == expected.shape, f"{case}/{key}: shape changed"
        np.testing.assert_allclose(value, expected, rtol=RTOL, atol=ATOL, equal_nan=True, err_msg=f"{case}/{key}")
