from typing import Sequence, Dict, Optional, Union, List
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
            selected_features: Optional[Sequence[str]] = None,
            selected_targets: Optional[Sequence[str]] = None,
            window: Optional[int] = None,
    ) -> None:
        """
        Initialize the CSTR Dataset.

        The dataset contains the concentrations of three species (A, B, and X) over time.
        The inlet concentrations are fixed.

        :param selected_features: Features to select (default: all features).
        :param selected_targets: Targets to select (default: all targets).
        :param window: Number of previous time-steps to include in each feature (default: 1).
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "CSTR Dataset"
        self.dataset_url: str = "https://edgarsmdn.github.io/MLCE_book/05_Hybrid_CSTR.html"
        self.window_size: int = window if window is not None else 1
        self._data_type: str = "timeseries"

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

    def get_dataset_info(self) -> Dict[str, Union[str, Sequence[str]]]:
        """
        Retrieve metadata about the dataset.

        :return: A dictionary containing dataset metadata.
        """
        return {
            "name": self.dataset_name,
            "description": (
                "Dataset containing concentrations of three species (A, B, and X) "
                "in a continuous stirred-tank reactor (CSTR) over time."
            ),
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url,
        }
