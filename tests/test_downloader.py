import os
import pytest
from chelo.utils.downloader import DatasetDownloader


@pytest.fixture
def downloader():
    """Fixture for creating a DatasetDownloader instance."""
    return DatasetDownloader(cache_dir="./test_cache")


def test_initialization(downloader):
    """Test initialization of the DatasetDownloader."""
    assert downloader.cache_dir == "./test_cache"
    os.makedirs(downloader.cache_dir, exist_ok=True)
    assert os.path.exists(downloader.cache_dir)


def test_get_dataset_dir(downloader):
    """Test dataset directory creation and retrieval."""
    dataset_name = "test_dataset"
    dataset_dir = downloader._get_dataset_dir(dataset_name)
    assert os.path.exists(dataset_dir)
    assert dataset_dir == os.path.join(downloader.cache_dir, dataset_name)


def test_get_file_path(downloader):
    """Test file path retrieval."""
    dataset_name = "test_dataset"
    filename = "LICENSE"
    file_path = downloader._get_file_path(dataset_name, filename)
    expected_path = os.path.join(downloader.cache_dir, dataset_name, filename)
    assert file_path == expected_path


if __name__ == "__main__":
    pytest.main()
