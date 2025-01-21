import os
import requests
from tqdm import tqdm
from typing import Optional
from ..utils.checksum import verify_checksum


class DatasetDownloader:
    """
    Utility class for downloading and caching datasets.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        """
        Initialize the downloader with an optional cache directory.
        :param cache_dir: Directory to store downloaded datasets
        """
        user_dir: str = os.path.expanduser(os.path.join("~", ".chelo"))
        self.cache_dir: str = cache_dir or os.getenv("CHELO_DATASET_PATH", user_dir)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_dataset_dir(self, dataset_name: str) -> str:
        """
        Get the directory path for a specific dataset.
        :param dataset_name: Name of the dataset.
        :return: Path to the dataset's directory.
        """
        dataset_dir: str = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def _get_file_path(self, dataset_name: str, filename: str) -> str:
        """
        Get the full path for a file in the dataset's directory.
        :param dataset_name: Name of the dataset.
        :param filename: Name of the file.
        :return: Full path to the file.
        """
        dataset_dir: str = self._get_dataset_dir(dataset_name)
        return os.path.join(dataset_dir, filename)

    def download(
        self,
        url: str,
        dataset_name: str,
        filename: Optional[str] = None,
        checksum: Optional[str] = None
    ) -> str:
        """
        Download a file for a specific dataset and save it in the dataset's folder.
        :param url: URL of the file to download.
        :param dataset_name: Name of the dataset.
        :param filename: Local filename (default: inferred from the URL).
        :param checksum: Expected checksum (MD5 or SHA256) to validate the file (optional).
        :return: Path to the downloaded file.
        """
        filename = filename or os.path.basename(url)
        file_path: str = self._get_file_path(dataset_name, filename)

        # Check if the file already exists
        if os.path.exists(file_path):
            if checksum and not verify_checksum(file_path, checksum):
                print("Checksum mismatch! Redownloading the file.")
            else:
                return file_path

        # Download the file
        print(f"Downloading '{filename}' for dataset '{dataset_name}' from {url}...")
        response: requests.Response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file with progress bar
        with open(file_path, "wb") as file, tqdm(
            total=int(response.headers.get("content-length", 0)),
            unit="B",
            unit_scale=True,
            desc=f"Downloading {filename}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                progress.update(len(chunk))

        # Verify checksum
        if checksum and not verify_checksum(file_path, checksum):
            raise ValueError(
                f"Checksum verification failed for '{filename}'. Try re-downloading the file. "
                f"If the issue persists, please open an issue at https://github.com/passalis/chelo"
            )

        print(f"File downloaded and saved at '{file_path}'.")
        return file_path
