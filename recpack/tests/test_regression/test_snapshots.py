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

# Results that legitimately differ between dependency versions (e.g. the minimum versions in
# pyproject.toml vs the latest). Tested over that range: torch based models differ by up to
# ~4e-5 with the same top-10 per user, so a looser tolerance still catches real changes.
TORCH_TOLERANCE = dict(rtol=1e-4, atol=1e-4)
TOLERANCES = {
    f"algorithm/{name}": TORCH_TOLERANCE
    for name in [
        "WeightedMatrixFactorization",
        "BPRMF",
        "MultVAE",
        "RecVAE",
        "Prod2Vec",
        "Prod2VecClustered",
        "GRU4RecNegSampling",
        "GRU4RecCrossEntropy",
    ]
}
# SLIM's SGDRegressor gives a different model with older scikit-learn/scipy versions,
# so only its shape is compared.
SHAPE_ONLY = {"algorithm/SLIM"}


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
        if case in SHAPE_ONLY:
            continue
        tolerance = TOLERANCES.get(case, dict(rtol=RTOL, atol=ATOL))
        np.testing.assert_allclose(value, expected, **tolerance, equal_nan=True, err_msg=f"{case}/{key}")
