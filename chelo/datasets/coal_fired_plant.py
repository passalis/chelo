from typing import List, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.kaggle_downloader import KaggleDatasetDownloader
import pandas as pd


@register_dataset
class CoalFiredPlantDataset(CheLoDataset):
    """
    Dataset class for Coal Fired Power Plant Thermal Performance.

    Provides utilities to load, process, and interact with the dataset.
    """
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

        :param selected_features: List of features to select (default: all features).
        :param selected_targets: List of targets to select (default: all targets).
        """
        super().__init__(selected_features, selected_targets)
        self.dataset_name: str = "Coal Fired Power Plant Thermal Performance Dataset"
        self.dataset_url: str = ("https://www.kaggle.com/datasets/ainalirham/"
                                 "coal-fired-power-plant-thermal-performance-dataset")

    def load_data(self) -> None:
        """
        Load the dataset from Kaggle or cache, and preprocess it.

        Downloads the dataset if not already cached, removes missing values,
        and initializes the feature and target sets.
        """
        downloader: KaggleDatasetDownloader = KaggleDatasetDownloader()

        # Download and validate dataset files
        for file_name, checksum in zip(self._FILES, self._CHECKSUMS):
            downloader.download_dataset(self._DATASET_SLUG, file_name, checksum)

        # Load dataset from the downloaded file
        file_path: str = downloader._get_file_path(self._DATASET_SLUG, self._FILES[0])
        data: pd.DataFrame = pd.read_excel(file_path)
        data.dropna(inplace=True)

        # Extract raw features and targets
        self.raw_features: Dict[str, List[Union[int, float, str]]] = data.drop(
            columns=["Tanggal", "Unnamed: 0"]
        ).to_dict(orient="list")

        self.raw_targets: Dict[str, List[Union[int, float, str]]] = data.drop(
            columns=["Tanggal", "Unnamed: 0"]
        ).to_dict(orient="list")

        # Set default features and targets if none are provided
        if self._selected_targets is None:
            self._selected_targets = ["Boiler Eff (%)"]

        if self._selected_features is None:
            self._selected_features = list(self.raw_features.keys())
            for target in self._selected_targets:
                if target in self._selected_features:
                    self._selected_features.remove(target)

        self._apply_initial_selections()

    def get_dataset_info(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get metadata about the dataset.

        :return: A dictionary containing dataset metadata including
                 name, description, features, and targets.
        """
        return {
            "name": self.dataset_name,
            "description": (
                "Dataset containing thermal performance attributes of coal-fired "
                "power plants, including features like boiler efficiency."
            ),
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url
        }
