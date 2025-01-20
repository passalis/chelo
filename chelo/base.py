from abc import ABC, abstractmethod
import numpy as np


class CheLoDataset(ABC):
    """
    Abstract Base Class for datasets.
    """

    def __init__(self, selected_features=None, selected_targets=None):
        """
        Initialize the dataset with optional selected features and targets.

        :param selected_features: List of features to select (default: all).
        :param selected_targets: List of targets to select (default: all).
        """
        self.raw_features = None  # Immutable raw feature data
        self.raw_targets = None  # Immutable raw target data
        self.features = None  # Subset of features to use
        self.targets = None  # Subset of targets to use
        self.dataset_name = None  # Name of the dataset

        self._selected_features = selected_features
        self._selected_targets = selected_targets

    @abstractmethod
    def load_data(self):
        """
        Load the dataset and populate self.raw_features and self.raw_targets.
        """
        pass

    @abstractmethod
    def list_features(self):
        """
        List available features in the dataset.
        """
        pass

    @abstractmethod
    def list_targets(self):
        """
        List available targets in the dataset.
        """
        pass

    @abstractmethod
    def get_dataset_info(self):
        """
        Provide metadata about the dataset (e.g., source, size, description).
        """
        pass

    def select_features(self, feature_names):
        """
        Dynamically select features from the dataset.

        :param feature_names: List of feature names to select.
        """
        if not self.raw_features:
            raise ValueError(f"Dataset {self.dataset_name} not loaded yet!")
        self.features = {name: self.raw_features[name] for name in feature_names}

    def select_targets(self, target_names):
        """
        Dynamically select targets from the dataset.

        :param target_names: List of target names to select.
        """
        if not self.raw_targets:
            raise ValueError(f"Dataset {self.dataset_name} not loaded yet!")
        self.targets = {name: self.raw_targets[name] for name in target_names}

    def _apply_initial_selections(self):
        """
        Apply initial selections if specified during initialization.
        """
        if self._selected_features:
            self.select_features(self._selected_features)
        else:
            self.features = self.raw_features

        if self._selected_targets:
            self.select_targets(self._selected_targets)
        else:
            self.targets = self.raw_targets

    def size(self):
        """
        Get the size of the dataset (number of samples).
        """
        return len(next(iter(self.features.values())))

    def statistics(self):
        """
        Compute basic statistics for the features and targets.

        :return: A dictionary of statistics (mean, std, min, max) for each feature and target.
        """
        stats = {}
        for key, values in {**self.features, **self.targets}.items():
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        return stats

    def to_numpy(self):
        """
        Convert the dataset to numpy arrays.

        :return: Tuple of (features, targets) in numpy format.
        """
        return np.array(list(self.features.values())).T, np.array(list(self.targets.values())).T

    def to_pytorch(self):
        """
        Provide a PyTorch Dataset object.

        :return: A PyTorch Dataset containing features and targets.
        """
        from torch.utils.data import Dataset
        class PyTorchDataset(Dataset):
            def __init__(self, features, targets):
                self.features = features
                self.targets = targets

            def __len__(self):
                return len(self.features)

            def __getitem__(self, idx):
                return self.features[idx], self.targets[idx]

        return PyTorchDataset(np.array(list(self.features.values())).T,
                              np.array(list(self.targets.values())).T)

    def to_keras(self, batch_size=32):
        """
        Provide a Keras Sequence object for training.

        :param batch_size: Number of samples per batch.
        :return: A Keras Sequence containing features and targets.
        """
        from tensorflow.keras.utils import Sequence
        class KerasSequence(Sequence):
            def __init__(self, features, targets, batch_size):
                self.features = np.array(list(features.values())).T
                self.targets = np.array(list(targets.values())).T
                self.batch_size = batch_size

            def __len__(self):
                return int(np.ceil(len(self.features) / self.batch_size))

            def __getitem__(self, idx):
                start_idx = idx * self.batch_size
                end_idx = start_idx + self.batch_size
                return self.features[start_idx:end_idx], self.targets[start_idx:end_idx]

        return KerasSequence(self.features, self.targets, batch_size)

    def preview(self, n=5):
        """
        Preview the first n rows of the dataset.

        :return: A dictionary with the first few rows.
        """
        preview_data = {
            "features": {key: values[:n] for key, values in self.features.items()},
            "targets": {key: values[:n] for key, values in self.targets.items()},
        }
        print(preview_data)
        return preview_data
