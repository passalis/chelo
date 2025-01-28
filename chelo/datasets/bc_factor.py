from typing import Sequence, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd


@register_dataset
class BCFactorDataset(CheLoDataset):
    _URL: str = "https://raw.githubusercontent.com/edgarsmdn/MLCE_book/main/references/BCF_training.csv"
    _FILE_NAME: str = "BCF_training.csv"
    _CHECKSUM: str = "bcf4ea5fa670952cbead1dd9b3091028"

    def __init__(
        self,
        selected_features: Optional[Sequence[str]] = None,
        selected_targets: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the Bioconcentration Factor (BCF) dataset.

        :param selected_features: Features to select (default: all features).
        :param selected_targets: Targets to select (default: all targets).
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "Bioconcentration Factor (BCF) Dataset"
        self.dataset_url: str = "https://edgarsmdn.github.io/MLCE_book/02_kNN_QSPR.html"

    def load_data(self) -> None:
        """
        Load the dataset into memory.
        """
        downloader: DatasetDownloader = DatasetDownloader()
        file_path: str = downloader.download(
            self._URL,
            dataset_name="bcf",
            filename=self._FILE_NAME,
            checksum=self._CHECKSUM,
        )

        data: pd.DataFrame = pd.read_csv(file_path, sep=",")
        columns_to_drop = ["CAS", "SMILES", "Experimental value [log(L/kg)]"]

        self.raw_features: Dict[str, Sequence[Union[int, float]]] = data.drop(
            columns=columns_to_drop, axis=1
        ).to_dict(orient="list")
        self.raw_targets: Dict[str, Sequence[float]] = {
            "bcf": data["Experimental value [log(L/kg)]"].tolist()
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
                "Dataset containing chemical properties and experimental "
                "bioconcentration factor (BCF)."
            ),
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url,
        }
