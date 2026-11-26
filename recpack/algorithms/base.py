# RecPack, An Experimentation Toolkit for Top-N Recommendation
# Copyright (C) 2020  Froomle N.V.
# License: GNU AGPLv3 - https://gitlab.com/recpack-maintainers/recpack/-/blob/master/LICENSE
# Author:
#   Lien Michiels
#   Robin Verachtert

import logging
import time
import warnings

from scipy.sparse import csr_array, lil_array
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from recpack.matrix import InteractionMatrix, to_csr_matrix, Matrix

logger = logging.getLogger("recpack")


def __getattr__(name):
    # TorchMLAlgorithm moved to recpack.algorithms.torch_base so this module does not need torch.
    # Keep ``from recpack.algorithms.base import TorchMLAlgorithm`` working.
    if name == "TorchMLAlgorithm":
        from recpack.algorithms.torch_base import TorchMLAlgorithm

        return TorchMLAlgorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class Algorithm(BaseEstimator):
    """Base class for all recpack algorithm implementations.

    This class provides fit and predict methods that handle generic
    pre and post-conditions.
    Usually a new algorithm will have to
    implement just the :meth:`_fit` and :meth:`_predict` methods.
    """

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        """Name of the object's class."""
        return self.__class__.__name__

    @property
    def identifier(self):
        """Name of the object.

        Name is made by combining the class name with the parameters
        passed at construction time.

        Constructed by recreating the initialisation call.
        Example: ``Algorithm(param_1=value)``
        """
        paramstring = ",".join((f"{k}={v}" for k, v in self.get_params().items()))
        return self.name + "(" + paramstring + ")"

    def __str__(self):
        return self.name

    def set_params(self, **params):
        """Set the parameters of the estimator.

        :param params: Estimator parameters
        :type params: dict
        """
        super().set_params(**params)

    def _fit(self, X: csr_array):
        """Stub implementation for fitting an algorithm.

        Will be called by the :meth:`fit` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix to fit the model to
        :type X: csr_array
        :raises NotImplementedError: Implement this method in the child class
        """
        raise NotImplementedError("Please implement _fit")

    def _predict(self, X: csr_array) -> csr_array:
        """Stub for predicting scores to users

        Will be called by the :meth:`predict` wrapper.
        Child classes should implement this function.

        :param X: User-item interaction matrix used as input to predict
        :type X: csr_array
        :raises NotImplementedError: Implement this method in the child class
        :return: Predictions made for all nonzero users in X
        :rtype: csr_array
        """
        raise NotImplementedError("Please implement _predict")

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Uses the sklear check_is_fitted function,
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        """
        check_is_fitted(self)

    def _check_prediction(self, X_pred: csr_array, X: csr_array) -> None:
        """Checks that the predictions matches expectations

        Checks implemented:

        - Check that all users with history got at least 1 recommendation

        For failing checks a warning is printed.

        :param X_pred: Predictions made for all nonzero users in X
        :type X_pred: csr_array
        :param X: User-item interaction matrix used as input to predict
        :type X: csr_array
        """

        users = set(X.nonzero()[0])
        predicted_users = set(X_pred.nonzero()[0])
        missing = users.difference(predicted_users)
        if len(missing) > 0:
            warnings.warn(f"{self.name} failed to recommend any items " f"for {len(missing)} users")

    def _transform_fit_input(self, X: Matrix) -> csr_array:
        """Transform the training data to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix to fit the model to
        :type X: Matrix
        :return: Transformed user-item interaction matrix to fit the model
        :rtype: csr_array
        """
        return to_csr_matrix(X, binary=True)

    def _transform_predict_input(self, X: Matrix) -> csr_array:
        """Transform the input of predict to expected type

        Data will be turned into a binary csr matrix.

        :param X: User-item interaction matrix used as input to predict
        :type X: Matrix
        :return: Transformed user-item interaction matrix used as input to predict
        :rtype: csr_array
        """
        return to_csr_matrix(X, binary=True)

    def _assert_is_interaction_matrix(self, *matrices: Matrix) -> None:
        """Make sure that the passed matrices are all an InteractionMatrix."""
        for X in matrices:
            if type(X) != InteractionMatrix:
                raise TypeError(f"{self.name} requires Interaction Matrix as input. Got {type(X)}.")

    def _assert_has_timestamps(self, *matrices: InteractionMatrix):
        """Make sure that the matrices all have timestamp information."""
        for X in matrices:
            if not X.has_timestamps:
                raise ValueError(f"{self.name} requires timestamp information in the InteractionMatrix.")

    def fit(self, X: Matrix) -> "Algorithm":
        """Fit the model to the input interaction matrix.

        After fitting the model will be ready to use for prediction.

        This function will handle some generic bookkeeping
        for each of the child classes,

        - The fit function gets timed, and this will get printed
        - Input data is converted to expected type using call to
          :meth:`_transform_predict_input`
        - The model is trained using the :meth:`_fit` method
        - :meth:`_check_fit_complete` is called to check fitting was succesful

        :param X: The interactions to fit the model on.
        :type X: Matrix
        :return: **self**, fitted algorithm
        :rtype: Algorithm
        """
        start = time.time()
        X = self._transform_fit_input(X)
        self._fit(X)

        self._check_fit_complete()
        end = time.time()
        logger.info(f"Fitting {self.name} complete - Took {end - start :.3}s")
        return self

    def predict(self, X: Matrix) -> csr_array:
        """Predicts scores, given the interactions in X

        Recommends items for each nonzero user in the X matrix.

        This function is a wrapper around the :meth:`_predict` method,
        and performs checks on in- and output data to guarantee proper computation.

        - Checks that model is fitted correctly
        - checks the output using :meth:`_check_prediction` function

        :param X: interactions to predict from.
        :type X: Matrix
        :return: The recommendation scores in a sparse matrix format.
        :rtype: csr_array
        """
        self._check_fit_complete()

        X = self._transform_predict_input(X)

        X_pred = self._predict(X)

        self._check_prediction(X_pred, X)

        return X_pred


class ItemSimilarityMatrixAlgorithm(Algorithm):
    """Base algorithm for algorithms that fit an item to item similarity model

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.
    """

    def _predict(self, X: csr_array) -> csr_array:
        """Predict scores for nonzero users in X

        Scores are computed by matrix multiplication of X
        with the stored similarity matrix.

        :param X: csr_array with interactions
        :type X: csr_array
        :return: csr_array with scores
        :rtype: csr_array
        """
        scores = X @ self.similarity_matrix_

        # If self.similarity_matrix_ is not a csr matrix,
        # scores will also not be a csr matrix
        if not isinstance(scores, csr_array):
            scores = csr_array(scores)

        return scores

    def _check_fit_complete(self):
        """Helper function to check if model was correctly fitted

        Checks implemented:

        - Checks if the algorithm has been fitted, using sklearn's `check_is_fitted`
        - Checks if the fitted similarity matrix contains similar items for each item

        For failing checks a warning is printed.
        """
        # Use super to check is fitted
        super()._check_fit_complete()

        # Additional checks on the fitted matrix.
        # Check if actually exists!
        assert hasattr(self, "similarity_matrix_")

        # Check row wise, since that will determine the recommendation options.
        items_with_score = set(self.similarity_matrix_.nonzero()[0])

        missing = self.similarity_matrix_.shape[0] - len(items_with_score)
        if missing > 0:
            warnings.warn(f"{self.name} missing similar items for {missing} items.")


class TopKItemSimilarityMatrixAlgorithm(ItemSimilarityMatrixAlgorithm):
    """Base algorithm for algorithms that fit an item to item similarity model with K similar items for every item

    Model that encodes the similarity between items is expected
    under the ``similarity_matrix_`` attribute.

    This matrix should have shape ``(|items| x |items|)``.
    This can be dense or sparse matrix depending on the algorithm used.

    Predictions are made by computing the dot product of the history vector of a user
    and the similarity matrix.

    Usually a new algorithm will have to
    implement just the :meth:`_fit` method,
    to construct the `self.similarity_matrix_` attribute.

    :param K: How many similar items will be kept per item.
    :type K: int
    """

    def __init__(self, K):
        super().__init__()
        self.K = K


class FactorizationAlgorithm(Algorithm):
    """Base class for factorization algorithms

    During fitting two matrices are constructed.

    - ``user_embedding_`` contains the users embedded in a lower dimensional space,
      shape = ``|users| x num_components``

    - ``item_embedding_`` contains the items embedded in the same dimensions
      shape = ``num_components x |items|``

    Prediction happens by multiplying a user's features with the item embedding.

    Usually a child class will have to
    implement just the :meth:`_fit` method,
    to construct the `self.user_embedding_` and `self.item_embedding_` attributes.

    :param num_components: the dimension of the feature matrices. defaults to 100
    :type num_components: int, optional
    """

    def __init__(self, num_components=100):
        super().__init__()
        self.num_components = num_components

    def _check_fit_complete(self):
        """Helper function to check that fit stored information.

        Checks implemented:

        - Checks if model is fitted, using the sklearn check_is_fitted function,
          https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        - Checks if `num_components` is correct
        """
        check_is_fitted(self)

        # Post conditions
        assert self.user_embedding_.shape[1] == self.num_components
        assert self.item_embedding_.shape[0] == self.num_components

    def _predict(self, X: csr_array) -> csr_array:
        """Predict scores for nonzero users in the interaction matrix

        For each nonzero users their embedding is multiplied with the item embeddings,
        and scores are the result of this matrix multiplication.

        Before recommending checks the shape of X to the number of
        items and users in the embeddings.

        :param X: binary interaction matrix.
        :type X: csr_array
        :return: matrix with scores for each nonzero user.
        :rtype: csr_array
        """
        assert X.shape == (self.user_embedding_.shape[0], self.item_embedding_.shape[1])
        # Get the nonzero users, for these we will recommend.
        users = list(set(X.nonzero()[0]))
        # result is a lil matrix, makes editing rows easy
        result = lil_array(X.shape)
        # Set rows of the nonzero users to the predicted scores
        result[users] = self.user_embedding_[users] @ self.item_embedding_

        return result.tocsr()
