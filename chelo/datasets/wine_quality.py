from typing import List, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd


@register_dataset
class WineQualityDataset(CheLoDataset):
    _BASE_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    _FILES: Dict[str, str] = {
        "red": "winequality-red.csv",
        "white": "winequality-white.csv",
    }
    _CHECKSUMS: Dict[str, str] = {
        "red": "2daeecee174368f8a33b82c8cccae3a5",
        "white": "5d9ff0f7f716dace19e3ab4578775fd7",
    }

    def __init__(
        self,
        wine_type: str = "red",
        selected_features: Optional[List[str]] = None,
        selected_targets: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the Wine Quality Dataset.

        :param wine_type: Type of wine ('red' or 'white').
        :param selected_features: Features to select (default: all).
        :param selected_targets: Targets to select (default: all).
        """
        super().__init__(selected_features, selected_targets)
        if wine_type not in self._FILES:
            raise ValueError(f"Invalid wine_type '{wine_type}'. Must be 'red' or 'white'.")
        self.wine_type: str = wine_type
        self.dataset_name: str = f"Wine Quality ({wine_type.capitalize()})"
        self.dataset_url: str = "https://archive.ics.uci.edu/dataset/186/wine+quality"

    def load_data(self) -> None:
        """
        Load the dataset from the UCI repository or cache.
        """
        downloader: DatasetDownloader = DatasetDownloader()
        file_url: str = self._BASE_URL + self._FILES[self.wine_type]
        file_path: str = downloader.download(
            file_url,
            dataset_name="wine_quality",
            filename=self._FILES[self.wine_type],
            checksum=self._CHECKSUMS[self.wine_type],
        )

        data: pd.DataFrame = pd.read_csv(file_path, sep=";")
        self.raw_features: Dict[str, List[Union[int, float]]] = data.drop(
            columns=["quality"]
        ).to_dict(orient="list")
        self.raw_targets: Dict[str, List[int]] = {"quality": data["quality"].tolist()}
        self._apply_initial_selections()

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
            "url": self.dataset_url
        }
