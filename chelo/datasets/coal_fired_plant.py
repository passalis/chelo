from typing import List, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.kaggle_downloader import KaggleDatasetDownloader
import pandas as pd
from .. import _chelo_configuration


@register_dataset
class CoalFiredPlantDataset(CheLoDataset):
    _DATASET_SLUG: str = "ainalirham/coal-fired-power-plant-thermal-performance-dataset"
    _FILES: List[str] = ['dataset_combined_final.xlsm']
    _CHECKSUMS: List[str] = ["a275decc678749e39c08a9a44a48fc52"]

    def __init__(
            self,
            selected_features: Optional[List[str]] = None,
            selected_targets: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the Coal Fired Power Plant Thermal Performance Dataset.

        :param wine_type: Type of wine ('red' or 'white').
        :param selected_features: Features to select (default: all).
        :param selected_targets: Targets to select (default: all).
        """
        super().__init__(selected_features, selected_targets)
        self.dataset_name: str = f"Coal Fired Power Plant Thermal Performance Dataset"

    def load_data(self) -> None:
        """
        Load the dataset from the UCI repository or cache.
        """
        downloader = KaggleDatasetDownloader()
        for file_idx in range(len(self._FILES)):
            downloader.download_dataset(self._DATASET_SLUG, self._FILES[file_idx], self._CHECKSUMS[file_idx])
        file_path = downloader._get_file_path(self._DATASET_SLUG, self._FILES[0], )

        data: pd.DataFrame = pd.read_excel(file_path)
        data.dropna(inplace=True)
        self.raw_features = data.drop(
            columns=["Tanggal", "Unnamed: 0"]
        ).to_dict(orient="list")

        self.raw_targets = data.drop(
            columns=["Tanggal", "Unnamed: 0"]
        ).to_dict(orient="list")

        # Set default features and targets
        if self._selected_targets is None:
            self._selected_targets = ["Boiler Eff (%)"]
        if self._selected_features is None:
            self._selected_features = list(self.raw_features.keys())
            for feature in self._selected_targets:
                self._selected_features.remove(feature)
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
            "description": "Dataset containing physicochemical attributes and quality ratings of wines.",
            "wine_type": self.wine_type,
            "features": self.list_features(),
            "targets": self.list_targets(),
        }
