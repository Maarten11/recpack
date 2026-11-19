# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE

"""Regenerate ``snapshots.npz`` from the RecPack version that is currently importable.

Only regenerate when a change in outputs is intended, and say so in the PR::

    python -m recpack.tests.test_regression.generate_snapshots
"""

from pathlib import Path

import numpy as np

from recpack.tests.test_regression.cases import CASES

SNAPSHOT_FILE = Path(__file__).parent / "snapshots.npz"


def main(out: Path = SNAPSHOT_FILE) -> None:
    arrays = {}
    for case, run in CASES.items():
        for key, value in run().items():
            arrays[f"{case}/{key}"] = value
    np.savez_compressed(out, **arrays)
    print(f"Wrote {len(arrays)} arrays for {len(CASES)} cases to {out}")


if __name__ == "__main__":
    import sys

    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SNAPSHOT_FILE)
