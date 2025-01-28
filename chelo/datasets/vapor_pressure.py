from typing import List, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd


@register_dataset
class VaporPressureDataset(CheLoDataset):
    _URL: str = "https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/Vapor_pressures.csv"
    _FILE_NAME: str = "Vapor_pressures.csv"
    _CHECKSUM: str = "ed5b1229e13b8ebd82c3a047520bf328"

    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        selected_targets: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the VaporPressureDataset Dataset.

        :param selected_features: Features to select (default: all).
        :param selected_targets: Targets to select (default: all).
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "Vapor Pressure Dataset"
        self.dataset_url: str = "https://edgarsmdn.github.io/MLCE_book/04_DNN_VLE.html"

    def load_data(self) -> None:
        """
        Load the VaporPressureDataset dataset.
        """
        downloader = DatasetDownloader()
        file_path = downloader.download(
            self._URL,
            dataset_name="vapor_pressures",
            filename=self._FILE_NAME,
            checksum=self._CHECKSUM,
        )

        try:
            data = pd.read_csv(file_path, sep=",").dropna()
        except Exception as e:
            raise RuntimeError(f"Error loading data from {file_path}: {e}")

        self.raw_features = data.drop(columns=["Pvap"], axis=1).to_dict(orient="list")
        self.raw_targets = {"p_vap": data["Pvap"].tolist()}
        self._apply_initial_selections()

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
            "url": self.dataset_url,
        }
