from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class CheLoDataset(ABC):
    """
    Abstract Base Class for datasets.
    """

    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        selected_targets: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the dataset with optional selected features and targets.

        :param selected_features: List of features to select (default: all).
        :param selected_targets: List of targets to select (default: all).
        """
        self.raw_features: Optional[Dict[str, List[Any]]] = None  # Immutable raw feature data
        self.raw_targets: Optional[Dict[str, List[Any]]] = None  # Immutable raw target data
        self.features: Optional[Dict[str, List[Any]]] = None  # Subset of features to use
        self.targets: Optional[Dict[str, List[Any]]] = None  # Subset of targets to use
        self.dataset_name: Optional[str] = None  # Name of the dataset

        self._selected_features: Optional[List[str]] = selected_features
        self._selected_targets: Optional[List[str]] = selected_targets

    @abstractmethod
    def load_data(self) -> None:
        """
        Load the dataset and populate self.raw_features and self.raw_targets.
        """
        pass

    @abstractmethod
    def list_features(self) -> List[str]:
        """
        List available features in the dataset.
        """
        pass

    @abstractmethod
    def list_targets(self) -> List[str]:
        """
        List available targets in the dataset.
        """
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Provide metadata about the dataset (e.g., source, size, description).
        """
        pass

    def select_features(self, feature_names: List[str]) -> None:
        """
        Dynamically select features from the dataset.

        :param feature_names: List of feature names to select.
        """
        if not self.raw_features:
            raise ValueError(f"Dataset {self.dataset_name} not loaded yet!")
        self.features = {name: self.raw_features[name] for name in feature_names}

    def select_targets(self, target_names: List[str]) -> None:
        """
        Dynamically select targets from the dataset.

        :param target_names: List of target names to select.
        """
        if not self.raw_targets:
            raise ValueError(f"Dataset {self.dataset_name} not loaded yet!")
        self.targets = {name: self.raw_targets[name] for name in target_names}

    def _apply_initial_selections(self) -> None:
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

    def size(self) -> int:
        """
        Get the size of the dataset (number of samples).
        """
        if not self.features:
            raise ValueError("Features are not loaded.")
        return len(next(iter(self.features.values())))

    def statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute basic statistics for the features and targets.

        :return: A dictionary of statistics (mean, std, min, max) for each feature and target.
        """
        stats = {}
        if not self.features or not self.targets:
            raise ValueError("Dataset is not loaded or selections are not applied.")
        for key, values in {**self.features, **self.targets}.items():
            stats[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }
        return stats

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the dataset to numpy arrays.

        :return: Tuple of (features, targets) in numpy format.
        """
        if not self.features or not self.targets:
            raise ValueError("Dataset is not loaded or selections are not applied.")
        return (
            np.array(list(self.features.values())).T,
            np.array(list(self.targets.values())).T
        )

    def to_pytorch(self):
        """
        Provide a PyTorch Dataset object.

        :return: A PyTorch Dataset containing features and targets.
        """
        from torch.utils.data import Dataset

        class PyTorchDataset(Dataset):
            def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
                self.features = features
                self.targets = targets

            def __len__(self) -> int:
                return len(self.features)

            def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
                return self.features[idx], self.targets[idx]

        features, targets = self.to_numpy()
        return PyTorchDataset(features, targets)

    def preview(self, n: int = 5) -> Dict[str, Dict[str, List[Any]]]:
        """
        Preview the first n rows of the dataset.

        :return: A dictionary with the first few rows.
        """
        if not self.features or not self.targets:
            raise ValueError("Dataset is not loaded or selections are not applied.")
        preview_data = {
            "features": {key: values[:n] for key, values in self.features.items()},
            "targets": {key: values[:n] for key, values in self.targets.items()},
        }
        print(preview_data)
        return preview_data
