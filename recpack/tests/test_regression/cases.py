# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE

"""Cases for the snapshot (golden output) regression tests.

Every case runs a piece of RecPack on a fixed, seeded dataset and returns its outputs as
dense numpy arrays. ``generate_snapshots.py`` stores these outputs, ``test_snapshots.py``
recomputes them and checks that they did not change.

Outputs are converted to dense arrays, so the comparison does not depend on the container
type (``csr_matrix``, ``csr_array``, ``np.matrix``, ...), only on the values.
"""

import os
import shutil
import tempfile
from functools import partial
from typing import Callable, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # torch is an optional extra since RecPack 0.5
    torch = None
from scipy.sparse import csr_array, csr_matrix, issparse

from recpack import algorithms, datasets, metrics, scenarios
from recpack.algorithms.experimental import DAMIBCover, EASE_XY
from recpack.matrix import InteractionMatrix
from recpack.matrix.util import Matrix
from recpack.postprocessing.filters import ExcludeItems, SelectItems

NUM_USERS = 60
NUM_ITEMS = 40
NUM_INTERACTIONS = 700
SECONDS_PER_DAY = 86400
NUM_DAYS = 30
SEED = 42

Outputs = Dict[str, np.ndarray]

# Inputs are built with the sparse type the installed RecPack works with,
# so the same cases run on RecPack < 0.4 (csr_matrix) to generate the snapshots.
SPARSE = csr_array if csr_array in Matrix.__args__ else csr_matrix


def make_data() -> InteractionMatrix:
    """Seeded interaction data with timestamps and some repeated user-item pairs."""
    rng = np.random.default_rng(SEED)
    item_popularity = 1 / np.arange(1, NUM_ITEMS + 1)
    df = pd.DataFrame(
        {
            InteractionMatrix.USER_IX: rng.integers(0, NUM_USERS, NUM_INTERACTIONS),
            InteractionMatrix.ITEM_IX: rng.choice(
                NUM_ITEMS, NUM_INTERACTIONS, p=item_popularity / item_popularity.sum()
            ),
            InteractionMatrix.TIMESTAMP_IX: rng.integers(0, NUM_DAYS * SECONDS_PER_DAY, NUM_INTERACTIONS),
        }
    )
    return InteractionMatrix(
        df,
        InteractionMatrix.ITEM_IX,
        InteractionMatrix.USER_IX,
        timestamp_ix=InteractionMatrix.TIMESTAMP_IX,
        shape=(NUM_USERS, NUM_ITEMS),
    )


def make_scores() -> csr_matrix:
    """Seeded sparse prediction scores, about 30% dense."""
    rng = np.random.default_rng(SEED + 1)
    scores = rng.random((NUM_USERS, NUM_ITEMS))
    mask = rng.random((NUM_USERS, NUM_ITEMS)) < 0.3
    return SPARSE(scores * mask)


def to_array(x) -> np.ndarray:
    if issparse(x):
        return np.asarray(x.toarray(), dtype=float)
    if isinstance(x, pd.DataFrame):
        return x.sort_values(list(x.columns)).to_numpy(dtype=float)
    return np.asarray(x, dtype=float)


# --- Algorithms -------------------------------------------------------------------------

K = 10
TORCH_KWARGS = dict(max_epochs=2, seed=SEED, stop_early=False)
# Algorithm cases that need torch; skipped when it is not installed.
REQUIRES_TORCH = {
    "WeightedMatrixFactorization",
    "BPRMF",
    "MultVAE",
    "RecVAE",
    "Prod2Vec",
    "Prod2VecClustered",
    "GRU4RecNegSampling",
    "GRU4RecCrossEntropy",
}

ALGORITHMS: Dict[str, Callable[[], algorithms.Algorithm]] = {
    # K=11: at K=10 two items tie for the last place, which Popularity breaks differently per platform.
    "Popularity": lambda: algorithms.Popularity(K=11),
    "Random": lambda: algorithms.Random(K=K, seed=SEED),
    "NMF": lambda: algorithms.NMF(num_components=8, seed=SEED),
    "SVD": lambda: algorithms.SVD(num_components=8, seed=SEED),
    "NMFItemToItem": lambda: algorithms.NMFItemToItem(num_components=8, seed=SEED),
    "SVDItemToItem": lambda: algorithms.SVDItemToItem(num_components=8, seed=SEED),
    "EASE": lambda: algorithms.EASE(l2=10.0),
    "EASE-alpha-density": lambda: algorithms.EASE(l2=10.0, alpha=0.5, density=0.3),
    "SLIM": lambda: algorithms.SLIM(),
    "ItemKNN-cosine": lambda: algorithms.ItemKNN(K=K),
    "ItemKNN-cosine-normalized": lambda: algorithms.ItemKNN(K=K, normalize_X=True, normalize_sim=True),
    "ItemKNN-conditional_probability": lambda: algorithms.ItemKNN(K=K, similarity="conditional_probability"),
    "ItemKNN-conditional_probability-pop_discount": lambda: algorithms.ItemKNN(
        K=K, similarity="conditional_probability", pop_discount=0.5
    ),
    # ItemPNN with the empirical pdf needs at least K neighbours per item, hence the small K.
    "ItemPNN-empirical": lambda: algorithms.ItemPNN(K=2, pdf="empirical", seed=SEED),
    "ItemPNN-uniform": lambda: algorithms.ItemPNN(K=K, pdf="uniform", seed=SEED),
    "ItemPNN-softmax_empirical": lambda: algorithms.ItemPNN(K=K, pdf="softmax_empirical", seed=SEED),
    "KUNN": lambda: algorithms.KUNN(Ku=K, Ki=K),
    "WeightedMatrixFactorization": lambda: algorithms.WeightedMatrixFactorization(num_components=8, iterations=3),
    "STAN": lambda: algorithms.STAN(K=K),
    "SequentialRules": lambda: algorithms.SequentialRules(K=K),
    "TARSItemKNN-cosine": lambda: algorithms.TARSItemKNN(K=K),
    "TARSItemKNN-conditional_probability": lambda: algorithms.TARSItemKNN(K=K, similarity="conditional_probability"),
    "TARSItemKNN-pearson": lambda: algorithms.TARSItemKNN(K=K, similarity="pearson"),
    "TARSItemKNNCoocDistance-cooc": lambda: algorithms.TARSItemKNNCoocDistance(K=K, similarity="cooc"),
    "TARSItemKNNCoocDistance-conditional_probability": lambda: algorithms.TARSItemKNNCoocDistance(
        K=K, similarity="conditional_probability"
    ),
    "TARSItemKNNDing": lambda: algorithms.TARSItemKNNDing(K=K),
    "TARSItemKNNHermann": lambda: algorithms.TARSItemKNNHermann(K=K, decay_interval=SECONDS_PER_DAY),
    "TARSItemKNNLee": lambda: algorithms.TARSItemKNNLee(K=K),
    "TARSItemKNNLiu": lambda: algorithms.TARSItemKNNLiu(K=K),
    "TARSItemKNNLiu2012": lambda: algorithms.TARSItemKNNLiu2012(K=K),
    "TARSItemKNNXia": lambda: algorithms.TARSItemKNNXia(K=K),
    "TARSItemKNNVaz": lambda: algorithms.TARSItemKNNVaz(K=K),
    "DAMIBCover": lambda: DAMIBCover(algo=algorithms.ItemKNN(K=K)),
    "EASE_XY": lambda: EASE_XY(l2=10.0),
    # Torch based algorithms: few epochs, small models, fixed seed.
    "BPRMF": lambda: algorithms.BPRMF(num_components=8, batch_size=64, **TORCH_KWARGS),
    "MultVAE": lambda: algorithms.MultVAE(batch_size=16, dim_bottleneck_layer=8, dim_hidden_layer=16, **TORCH_KWARGS),
    "RecVAE": lambda: algorithms.RecVAE(batch_size=16, dim_bottleneck_layer=8, dim_hidden_layer=16, **TORCH_KWARGS),
    "Prod2Vec": lambda: algorithms.Prod2Vec(num_components=8, K=K, batch_size=64, **TORCH_KWARGS),
    "Prod2VecClustered": lambda: algorithms.Prod2VecClustered(
        num_components=8, K=K, num_clusters=3, Kcl=2, batch_size=64, **TORCH_KWARGS
    ),
    "GRU4RecNegSampling": lambda: algorithms.GRU4RecNegSampling(
        hidden_size=8, num_components=8, num_negatives=5, batch_size=16, **TORCH_KWARGS
    ),
    "GRU4RecCrossEntropy": lambda: algorithms.GRU4RecCrossEntropy(
        hidden_size=8, num_components=8, batch_size=16, **TORCH_KWARGS
    ),
}


def run_algorithm(name: str) -> Outputs:
    # SLIM (SGDRegressor) and WeightedMatrixFactorization have no seed parameter
    # and use the global random state.
    np.random.seed(SEED)
    if torch is not None:
        torch.manual_seed(SEED)
    data = make_data()
    algo = ALGORITHMS[name]()
    if torch is not None and isinstance(algo, algorithms.TorchMLAlgorithm):
        algo.fit(data, (data, data))
    elif isinstance(algo, EASE_XY):
        algo.fit(data, data)
    else:
        algo.fit(data)
    return {"prediction": to_array(algo.predict(data))}


# --- Metrics ----------------------------------------------------------------------------

METRICS: Dict[str, Callable[[], metrics.base.Metric]] = {
    "CoverageK": lambda: metrics.CoverageK(5),
    "DCGK": lambda: metrics.DCGK(5),
    "NDCGK": lambda: metrics.NDCGK(5),
    "IntraListDiversityK": lambda: metrics.IntraListDiversityK(5),
    "HitK": lambda: metrics.HitK(5),
    "DiscountedGainK": lambda: metrics.DiscountedGainK(5),
    "IPSHitRateK": lambda: metrics.IPSHitRateK(5),
    "PrecisionK": lambda: metrics.PrecisionK(5),
    "RecallK": lambda: metrics.RecallK(5),
    "CalibratedRecallK": lambda: metrics.CalibratedRecallK(5),
    "ReciprocalRankK": lambda: metrics.ReciprocalRankK(5),
    "PercentileRanking": lambda: metrics.PercentileRanking(),
}


def run_metric(name: str) -> Outputs:
    y_true = make_data().binary_values
    metric = METRICS[name]()
    if hasattr(metric, "fit"):
        metric.fit(y_true)
    metric.calculate(y_true, make_scores())
    return {"value": to_array(metric.value), "results": to_array(metric.results)}


# --- Scenarios --------------------------------------------------------------------------

T = 20 * SECONDS_PER_DAY
T_VALIDATION = 15 * SECONDS_PER_DAY

SCENARIOS: Dict[str, Callable[[], scenarios.Scenario]] = {
    "WeakGeneralization": lambda: scenarios.WeakGeneralization(0.7, validation=True, seed=SEED),
    "StrongGeneralization": lambda: scenarios.StrongGeneralization(0.7, 0.7, validation=True, seed=SEED),
    "Timed": lambda: scenarios.Timed(T, T_VALIDATION, validation=True, seed=SEED),
    "StrongGeneralizationTimed": lambda: scenarios.StrongGeneralizationTimed(
        0.7, T, T_VALIDATION, validation=True, seed=SEED
    ),
    "StrongGeneralizationTimedMostRecent": lambda: scenarios.StrongGeneralizationTimedMostRecent(
        T, T_VALIDATION, n_most_recent_out=2, validation=True, seed=SEED
    ),
    "LastItemPrediction": lambda: scenarios.LastItemPrediction(validation=True, seed=SEED, n_most_recent_in=3),
    "TimedLastItemPrediction": lambda: scenarios.TimedLastItemPrediction(
        T, T_VALIDATION, n_most_recent_in=3, validation=True, seed=SEED
    ),
}


def run_scenario(name: str) -> Outputs:
    scenario = SCENARIOS[name]()
    scenario.split(make_data())
    return {
        "full_training_data": to_array(scenario.full_training_data.values),
        "validation_training_data": to_array(scenario.validation_training_data.values),
        "validation_data_in": to_array(scenario.validation_data_in.values),
        "validation_data_out": to_array(scenario.validation_data_out.values),
        "test_data_in": to_array(scenario.test_data_in.values),
        "test_data_out": to_array(scenario.test_data_out.values),
    }


# --- InteractionMatrix and post-processing ----------------------------------------------


def run_interaction_matrix() -> Outputs:
    data = make_data()
    return {
        "values": to_array(data.values),
        "binary_values": to_array(data.binary_values),
        "last_timestamps_matrix": to_array(data.last_timestamps_matrix),
    }


def run_postprocessing() -> Outputs:
    scores = make_scores()
    return {
        "ExcludeItems": to_array(ExcludeItems([1, 3, 5]).apply(scores)),
        "SelectItems": to_array(SelectItems([0, 2, 4, 6]).apply(scores)),
    }


# --- Datasets ---------------------------------------------------------------------------

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "test_datasets", "datasets")

# Default filters are used where the small test samples survive them.
LOCAL_DATASETS: Dict[str, Callable[[str], datasets.Dataset]] = {
    "CiteULike": lambda path: datasets.CiteULike(path=path, filename="citeulike_sample.dat"),
    "MovieLens100K": lambda path: datasets.MovieLens100K(path=path, filename="ml-100k_sample.data"),
    "MovieLens1M": lambda path: datasets.MovieLens1M(path=path, filename="ml-1m_sample.dat"),
    "MovieLens10M": lambda path: datasets.MovieLens10M(path=path, filename="ml-10m_sample.dat"),
    "MovieLens25M": lambda path: datasets.MovieLens25M(path=path, filename="ml-25m_sample.csv"),
    "MillionSongDataset": lambda path: datasets.MillionSongDataset(
        path=path, filename="msd_train_triplets_sample.csv", use_default_filters=False
    ),
    "TasteProfile": lambda path: datasets.TasteProfile(
        path=path, filename="msd_train_triplets_sample.csv", use_default_filters=False
    ),
    "RecsysChallenge2015": lambda path: datasets.RecsysChallenge2015(path=path, filename="yoochoose-clicks.dat"),
    "RetailRocket": lambda path: datasets.RetailRocket(
        path=path, filename="retailrocket-sample.csv", use_default_filters=False
    ),
    "CosmeticsShop": lambda path: datasets.CosmeticsShop(
        path=path, filename="cosmeticsshop-sample.zip", use_default_filters=False
    ),
    "Globo": lambda path: datasets.Globo(path=path, filename="globo-sample.zip"),
}

# Datasets that unpack a downloaded archive: (module, archive in test_datasets/datasets/compressed).
DOWNLOADED_DATASETS = {
    "Netflix": ("netflix", "nf_prize_dataset.tar.gz", lambda path: datasets.Netflix(path=path)),
    "AdressaOneWeek": (
        "adressa",
        "one_week.tar.gz",
        lambda path: datasets.AdressaOneWeek(path, "adressa_views.csv", use_default_filters=False),
    ),
}


def interaction_matrix_outputs(im: InteractionMatrix) -> Outputs:
    columns = [c for c in (im.USER_IX, im.ITEM_IX, im.TIMESTAMP_IX) if c in im._df.columns]
    df = im._df[columns].sort_values(columns)
    return {"interactions": df.to_numpy(dtype=float), "shape": np.array(im.shape, dtype=float)}


def run_dataset(name: str) -> Outputs:
    # Work in a temporary copy, loading may write files next to the data.
    with tempfile.TemporaryDirectory() as path:
        if name in LOCAL_DATASETS:
            dataset = LOCAL_DATASETS[name](path)
            shutil.copyfile(os.path.join(DATASET_DIR, dataset.filename), dataset.file_path)
            return interaction_matrix_outputs(dataset.load())

        module, archive, make = DOWNLOADED_DATASETS[name]

        def mock_download(url, file_path):
            shutil.copyfile(os.path.join(DATASET_DIR, "compressed", archive), file_path)

        with patch(f"recpack.datasets.{module}._fetch_remote", partial(mock_download)):
            dataset = make(path)
            dataset.fetch_dataset(force=True)
            return interaction_matrix_outputs(dataset.load())


def run_dummy_dataset() -> Outputs:
    return interaction_matrix_outputs(datasets.DummyDataset(seed=SEED).load())


# --- All cases --------------------------------------------------------------------------

CASES: Dict[str, Callable[[], Outputs]] = {
    **{f"algorithm/{n}": (lambda n=n: run_algorithm(n)) for n in ALGORITHMS},
    **{f"metric/{n}": (lambda n=n: run_metric(n)) for n in METRICS},
    **{f"scenario/{n}": (lambda n=n: run_scenario(n)) for n in SCENARIOS},
    "matrix/InteractionMatrix": run_interaction_matrix,
    "postprocessing/filters": run_postprocessing,
    **{f"dataset/{n}": (lambda n=n: run_dataset(n)) for n in [*LOCAL_DATASETS, *DOWNLOADED_DATASETS]},
    "dataset/DummyDataset": run_dummy_dataset,
}
