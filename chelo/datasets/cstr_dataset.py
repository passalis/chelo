from typing import List, Dict, Optional, Union


from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd
import numpy as np

@register_dataset
class CSTRDataset(CheLoDataset):
    _URL: str = "https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/CSTR_ODE_data.txt"
    _FILE_NAME: str = "CSTR_ODE_data.txt"
    _CHECKSUM: str = "757f3928146122c37efe3fa1bd67a5db"

    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        selected_targets: Optional[List[str]] = None,
        window: Optional[int] = None,
    ) -> None:
        """
        Initialize the CSTRDataset dataset.
        The dataset contains the concentrations of three species (A, B and X) through time.
        The inlet concentrations are fixed.

        :param selected_features: Features to select (default: all).
        :param selected_targets: Targets to select (default: all).
        :param window: Defines how many previous time-steps to include (default: all).
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "CSTR Dataset"
        self.dataset_url: str = "https://edgarsmdn.github.io/MLCE_book/05_Hybrid_CSTR.html"
        if window is None:
            window = 1
        self.window_size: int = window
        self._data_type = 'timeseries'

    def load_data(self) -> None:
        """
        Load the CSTRDataset dataset.
        """
        downloader: DatasetDownloader = DatasetDownloader()
        file_path: str = downloader.download(
            self._URL,
            dataset_name="cstr",
            filename=self._FILE_NAME,
            checksum=self._CHECKSUM,
        )


        data: pd.DataFrame = pd.read_csv(file_path, sep=";")
        data = data.dropna()
        self.raw_targets: Dict[str, List[Union[int, float]]] = data.iloc[self.window_size:, :].to_dict(orient="list")
        self.raw_features: Dict[str, List[Union[int, float]]] = data.to_dict(orient="list")
        for feature_name in self.raw_features:
            X = data[feature_name]
            X = np.asarray(X)
            X = np.array([X[i:i + self.window_size] for i in range(len(X) - self.window_size)])
            self.raw_features[feature_name] = X
        self._apply_initial_selections()

    def list_features(self) -> List[str]:
        """
        List the available features in the dataset.

        :return: List of feature names.
        """
        return list(self.raw_features.keys())

    def list_targets(self) -> List[str]:
        """
        List the available targets in the dataset.

        :return: List of target names.
        """
        return list(self.raw_targets.keys())

    def get_dataset_info(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get metadata about the dataset.

        :return: A dictionary containing dataset metadata.
        """
        return {
            "name": self.dataset_name,
            "description": "Dataset containing the phase envelope of various compounds.",
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url
        }
