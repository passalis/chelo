import os
import hashlib
import requests
from tqdm import tqdm
import tarfile
import zipfile

class DatasetDownloader:
    """
    Utility class for downloading and caching datasets.
    """

    def __init__(self, cache_dir=None):
        """
        Initialize the downloader with an optional cache directory.
        :param cache_dir: Directory to store downloaded datasets
        """
        user_dir = os.path.expanduser(os.path.join("~", ".chelo_datasets"))
        self.cache_dir = cache_dir or os.getenv("CHELO_DATASET_CACHE", user_dir)
        os.makedirs(self.cache_dir, exist_ok=True)


    def _get_dataset_dir(self, dataset_name):
        """
        Get the directory path for a specific dataset.
        :param dataset_name: Name of the dataset.
        :return: Path to the dataset's directory.
        """
        dataset_dir = os.path.join(self.cache_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def _get_file_path(self, dataset_name, filename):
        """
        Get the full path for a file in the dataset's directory.
        :param dataset_name: Name of the dataset.
        :param filename: Name of the file.
        :return: Full path to the file.
        """
        dataset_dir = self._get_dataset_dir(dataset_name)
        return os.path.join(dataset_dir, filename)

    def download(self, url, dataset_name, filename=None, checksum=None):
        """
        Download a file for a specific dataset and save it in the dataset's folder.
        :param url: URL of the file to download.
        :param dataset_name: Name of the dataset.
        :param filename: Local filename (default: inferred from the URL).
        :param checksum: Expected checksum (MD5 or SHA256) to validate the file (optional).
        :return: Path to the downloaded file.
        """
        filename = filename or os.path.basename(url)
        file_path = self._get_file_path(dataset_name, filename)

        # Check if the file already exists
        if os.path.exists(file_path):
            if checksum and not self._verify_checksum(file_path, checksum):
                print("Checksum mismatch! Redownloading the file.")
            else:
                return file_path

        # Download the file
        print(f"Downloading '{filename}' for dataset '{dataset_name}' from {url}...")
        response = requests.get(url, stream=True)
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
        if checksum and not self._verify_checksum(file_path, checksum):
            raise ValueError(f"Checksum verification failed for '{filename}'. Try re-downloading the file. If the issue please open an issue at https://github.com/passalis/chelo")

        print(f"File downloaded and saved at '{file_path}'.")
        return file_path

    def _verify_checksum(self, file_path, checksum):
        """
        Verify the checksum of a file.
        :param file_path: Path to the file.
        :param checksum: Expected checksum (MD5 or SHA256).
        :return: True if the checksum matches, False otherwise.
        """
        hash_func = hashlib.sha256 if len(checksum) == 64 else hashlib.md5
        with open(file_path, "rb") as file:
            file_hash = hash_func(file.read()).hexdigest()
        return file_hash == checksum



def extract_file(file_path, extract_to=None):
    """
    Extract a compressed file (.zip or .tar.gz).
    :param file_path: Path to the compressed file.
    :param extract_to: Directory to extract to (default: same as file location).
    :return: Path to the extracted directory.
    """
    extract_to = extract_to or os.path.splitext(file_path)[0]
    os.makedirs(extract_to, exist_ok=True)

    if file_path.endswith(".zip"):
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_path.endswith(".tar.gz"):
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    print(f"File extracted to '{extract_to}'.")
    return extract_to