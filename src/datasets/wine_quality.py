from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd

@register_dataset
class WineQualityDataset(CheLoDataset):

    BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    FILES = {
        "red": "winequality-red.csv",
        "white": "winequality-white.csv",
    }
    CHECKSUMS ={
        "red": "2daeecee174368f8a33b82c8cccae3a5",
        "white": "5d9ff0f7f716dace19e3ab4578775fd7",
    }

    def __init__(self, wine_type="red", selected_features=None, selected_targets=None):
        """
        Initialize the Wine Quality Dataset.
        :param wine_type: Type of wine ('red' or 'white').
        :param selected_features: Features to select (default: all).
        :param selected_targets: Targets to select (default: all).
        """
        super().__init__(selected_features, selected_targets)
        if wine_type not in self.FILES:
            raise ValueError(f"Invalid wine_type '{wine_type}'. Must be 'red' or 'white'.")
        self.wine_type = wine_type
        self.dataset_name = f"Wine Quality ({wine_type.capitalize()})"

    def load_data(self):
        """
        Load the dataset from the UCI repository or cache.
        """
        downloader = DatasetDownloader()
        file_url = self.BASE_URL + self.FILES[self.wine_type]
        file_path = downloader.download(file_url, dataset_name="wine_quality", filename=self.FILES[self.wine_type],
                                        checksum=self.CHECKSUMS[self.wine_type])

        data = pd.read_csv(file_path, sep=";")
        self.raw_features = data.drop(columns=["quality"]).to_dict(orient="list")
        self.raw_targets = {"quality": data["quality"].tolist()}
        self._apply_initial_selections()

    def list_features(self):
        """
        List the available features in the dataset.
        :return: List of feature names.
        """
        return list(self.raw_features.keys())

    def list_targets(self):
        """
        List the available targets in the dataset.
        :return: List of target names.
        """
        return list(self.raw_targets.keys())

    def get_dataset_info(self):
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