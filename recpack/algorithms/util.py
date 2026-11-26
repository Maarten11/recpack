# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

from inspect import isgenerator
from itertools import islice
from typing import Iterator, List, Iterable, Union

import numpy as np
from scipy.sparse import csr_array

from recpack.matrix import InteractionMatrix, Matrix, to_binary

_TORCH_HELPERS = ("swish", "log_norm_pdf", "naive_sparse2tensor", "naive_tensor2sparse")


def __getattr__(name):
    # The torch helpers moved to recpack.algorithms.torch_util so this module does not need torch.
    # Keep ``from recpack.algorithms.util import naive_sparse2tensor`` (etc.) working.
    if name in _TORCH_HELPERS:
        from recpack.algorithms import torch_util

        return getattr(torch_util, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_users(data: Matrix) -> List[int]:
    return list(set(data.nonzero()[0]))


def get_batches(iterable: Iterable, batch_size=1000) -> Iterator[List]:
    """Get batches from an iterable.

    The final batch might contain less than batch_size entries, as it will be the remainder.

    :param iterable: List of values that will be split into batches of size `batch_size`
    :type iterable: Iterable
    :param batch_size: Size of each batch, defaults to 1000
    :type batch_size: int, optional
    :yield: Iterator of lists of values
    :rtype: Iterator[List]
    """
    if not isgenerator(iterable):
        iterable = iter(iterable)

    while True:

        batch = list(islice(iterable, 0, batch_size))
        if batch:
            yield batch
        else:
            break


def sample_rows(*args: Matrix, sample_size: int = 1000) -> List[Matrix]:
    """Samples rows from the matrices

    Rows are sampled from the nonzero rows in the first csr_array argument.
    The return value will contain a matrix for each of the matrix arguments, with only the sampled rows nonzero.

    :param sample_size: Number of rows to sample, defaults to 1000
    :type sample_size: int, optional
    :return: List of all matrices passed as args
    :rtype: List[Matrix]
    """
    nonzero_users = list(set(args[0].nonzero()[0]))
    users = np.random.choice(nonzero_users, size=min(sample_size, len(nonzero_users)), replace=False)
    sampled_matrices = []

    for mat in args:
        if type(mat) == InteractionMatrix:
            sampled_mat = mat.users_in(users)
        else:
            sampled_mat = csr_array(mat.shape)
            sampled_mat[users, :] = mat[users, :]

        sampled_matrices.append(sampled_mat)

    return sampled_matrices


def union_csr_matrices(a: csr_array, b: csr_array) -> csr_array:
    """Combine entries of 2 binary csr_matrices.


    :param a: Binary csr_array
    :type a: csr_array
    :param b: Binary csr_array
    :type b: csr_array
    :return: The union of a and b
    :rtype csr_array:
    """
    return to_binary(a + b)


def invert(x: Union[np.ndarray, csr_array]) -> Union[np.ndarray, csr_array]:
    """Invert an array.

    :param x: [description]
    :type x: [type]
    :return: [description]
    :rtype: [type]
    """
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
    elif isinstance(x, csr_array):
        ret = csr_array(x.shape)
    else:
        raise TypeError("Unsupported type for argument x.")
    ret[x.nonzero()] = 1 / x[x.nonzero()]
    return ret
