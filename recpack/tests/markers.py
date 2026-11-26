# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert


"""Shared pytest markers."""

import pytest

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# torch is an optional extra: skip tests that need it when it is not installed.
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="requires torch (pip install 'recpack[torch]')")
