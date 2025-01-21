from typing import Optional
from .. import _chelo_configuration
import hashlib
# Export kaggle environment variables
import os
from ..utils.checksum import verify_checksum
os.environ["KAGGLE_USERNAME"] = _chelo_configuration.get("kaggle_username")
os.environ["KAGGLE_KEY"] = _chelo_configuration.get("kaggle_key")
from kaggle.api.kaggle_api_extended import KaggleApi
import os



class KaggleDatasetDownloader:
    """
    Utility class for downloading and caching datasets using the Kaggle Python API.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """
        Initialize the downloader with an optional cache directory.
        :param cache_dir: Directory to store downloaded datasets
        """
        user_dir: str = os.path.expanduser(os.path.join("~", ".chelo"))
        self.cache_dir: str = cache_dir or os.getenv("CHELO_DATASET_PATH", user_dir)
        self.cache_dir = os.path.join(self.cache_dir, "kaggle")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Initialize Kaggle API
        self.api = KaggleApi()
        self.api.authenticate()

    def _get_dataset_dir(self, dataset_slug: str) -> str:
        """
        Get the directory path for a specific dataset.
        :param dataset_slug: Kaggle dataset slug (e.g., username/dataset-name).
        :return: Path to the dataset's directory.
        """
        dataset_dir = os.path.join(self.cache_dir, dataset_slug.replace("/", "_"))
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def download_dataset(
            self, dataset_slug: str, filename: Optional[str] = None,
            checksum: Optional[str] = None
    ) -> str:
        """
        Download a Kaggle dataset and save it in the cache directory.
        :param dataset_slug: Kaggle dataset slug (e.g., username/dataset-name).
        :param filename: Specific file to return after download (optional).
        :param checksum:  Expected checksum (MD5 or SHA256) to validate the file (optional).
        :return: Path to the downloaded dataset or specific file.
        """
        dataset_dir = self._get_dataset_dir(dataset_slug)

        # Check if dataset is already downloaded
        if os.listdir(dataset_dir):
            if filename:
                file_path = os.path.join(dataset_dir, filename)
                if os.path.exists(file_path):
                    if checksum is None or verify_checksum(file_path, checksum):
                        return file_path
                    else:
                        print("Downloaded file is corrupted. Attemping to download again...")
            else:
                return dataset_dir

        # Download the dataset
        print(f"Downloading dataset '{dataset_slug}' into '{dataset_dir}'...")
        self.api.dataset_download_files(dataset_slug, path=dataset_dir, unzip=unzip)

        # Return the specific file if requested
        if filename:
            file_path = os.path.join(dataset_dir, filename)
            if os.path.exists(file_path):
                return file_path
            else:
                raise FileNotFoundError(
                    f"File '{filename}' not found after downloading the dataset."
                )
        # Verify checksum
        if checksum and not verify_checksum(file_path, checksum):
            raise ValueError(
                f"Checksum verification failed for '{filename}'. Try re-downloading the file. "
                f"If the issue persists, please open an issue at https://github.com/passalis/chelo"
            )
        return dataset_dir


    def _get_file_path(self, dataset_name: str, filename: str, ) -> str:
        """
        Get the full path for a file in the dataset's directory.
        :param dataset_name: Name of the dataset.
        :param filename: Name of the file.
        :return: Full path to the file.
        """
        dataset_dir: str = self._get_dataset_dir(dataset_name)
        return os.path.join(dataset_dir, filename)

