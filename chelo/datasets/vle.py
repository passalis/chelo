from typing import List, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd


@register_dataset
class VLEDataset(CheLoDataset):
    _URL: str = "https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/CO2_data.csv"
    _FILE_NAME: str = "CO2_data.csv"
    _CHECKSUM: str = "a3392e9a777241d7ed37e862bbe56075"

    def __init__(
        self,
        selected_features: Optional[List[str]] = None,
        selected_targets: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the VLEDataset.

        :param selected_features: Features to select (default: all).
        :param selected_targets: Targets to select (default: all).
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "VLE Dataset"
        self.dataset_url: str = "https://edgarsmdn.github.io/MLCE_book/04_DNN_VLE.html"

    def load_data(self) -> None:
        """
        Load the VLEDataset dataset.
        """
        downloader = DatasetDownloader()
        file_path = downloader.download(
            self._URL,
            dataset_name="vle",
            filename=self._FILE_NAME,
            checksum=self._CHECKSUM,
        )

        try:
            data = pd.read_csv(file_path, sep=",").dropna()
        except Exception as e:
            raise RuntimeError(f"Error loading data from {file_path}: {e}")

        # Process features and targets
        try:
            self.raw_features = data.drop(
                columns=["P_vap [kPa]", "H_vap [J/g]", "rho_vap [g/cm3]", "rho_liq [g/cm3]"]
            ).to_dict(orient="list")
            self.raw_targets = {
                "p_vap": data["P_vap [kPa]"].tolist(),
                "h_vap": data["H_vap [J/g]"].tolist(),
                "rho_vap": data["rho_vap [g/cm3]"].tolist(),
                "rho_liq": data["rho_liq [g/cm3]"].tolist(),
            }
        except KeyError as e:
            raise RuntimeError(f"Missing expected column in the dataset: {e}")

        self._apply_initial_selections()

    def get_dataset_info(self) -> Dict[str, Union[str, List[str]]]:
        """
        Get metadata about the dataset.

        :return: A dictionary containing dataset metadata.
        """
        return {
            "name": self.dataset_name,
            "description": "Dataset containing vapor-liquid equilibria of CO2.",
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url,
        }
