from typing import Sequence, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd


@register_dataset
class AmesMutagenicityDataset(CheLoDataset):
    _URL: str = "https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/mutagenicity_kNN.csv"
    _FILE_NAME: str = "mutagenicity_kNN.csv"
    _CHECKSUM: str = "0f379d83a7eb246a5ef5e139ab355d64"

    def __init__(
        self,
        selected_features: Optional[Sequence[str]] = None,
        selected_targets: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the Ames Mutagenicity dataset.

        :param selected_features: Features to select (default: all features).
        :param selected_targets: Targets to select (default: all targets).
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "Ames Q(SAR) Mutagenicity Dataset"
        self.dataset_url: str = "https://edgarsmdn.github.io/MLCE_book/02_kNN_QSPR.html"

    def load_data(self) -> None:
        """
        Load the dataset into memory.
        """
        downloader: DatasetDownloader = DatasetDownloader()
        file_path: str = downloader.download(
            self._URL,
            dataset_name="ames_mutagenicity",
            filename=self._FILE_NAME,
            checksum=self._CHECKSUM,
        )

        data: pd.DataFrame = pd.read_csv(file_path, sep=",")
        columns_to_drop = [
            "Unnamed: 0", "Id", "CAS", "SMILES", "Status",
            "Experimental value", "Predicted value"
        ]

        self.raw_features: Dict[str, Sequence[Union[int, float]]] = data.drop(
            columns=columns_to_drop, axis=1
        ).to_dict(orient="list")
        self.raw_targets: Dict[str, Sequence[int]] = {
            "mutagenicity": data["Experimental value"].tolist()
        }
        self._apply_initial_selections()

    def get_dataset_info(self) -> Dict[str, Union[str, Sequence[str]]]:
        """
        Retrieve metadata about the dataset.

        :return: A dictionary containing dataset metadata.
        """
        return {
            "name": self.dataset_name,
            "description": (
                "Dataset containing mutagenicity of various chemicals "
                "on Salmonella typhimurium (Ames test)."
            ),
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url
        }
