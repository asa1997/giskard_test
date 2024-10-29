"""Load models and datasets from Github."""

import logging
from abc import ABC, abstractmethod

from giskard import Dataset

logger = logging.getLogger(__name__)


class LoaderError(RuntimeError):
    """Could not load the model and/or dataset."""


class DatasetError(LoaderError):
    """Problems related to the dataset."""


class ModelError(LoaderError):
    """Problems related to the model."""


