import pytest
from chelo.datasets.cstr_dataset import CSTRDataset
from chelo.registry import DatasetRegistry


def test_cstr_dataset_basic_features_and_targets():

    dataset = CSTRDataset(window=4)
    dataset.load_data()

    # Verify that selected features and targets are present
    assert "c_A" in dataset.list_features()
    assert "c_B" in dataset.list_features()
    assert "c_X" in dataset.list_features()

    assert "c_A" in dataset.list_targets()
    assert "c_B" in dataset.list_targets()
    assert "c_X" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 4
    assert features.shape[2] == 3
    assert targets.shape[1] == 3  # One selected target


def test_cstr_dataset_default_selection():
    # Load the dataset without specifying features or targets
    dataset = CSTRDataset()
    dataset.load_data()

    # Verify that features and targets are automatically selected
    assert len(dataset.list_features()) > 0  # Ensure features are selected
    assert len(dataset.list_targets()) > 0  # Ensure targets are selected

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure features data is loaded
    assert targets.shape[0] > 0  # Ensure targets data is loaded


def test_cstr_dataset_window_functionality():
    # Load the dataset with a specific window size
    window_size = 5
    dataset = CSTRDataset(window=window_size)
    dataset.load_data()

    # Verify that the windowing works correctly
    features, targets = dataset.to_numpy()
    assert features.shape[0] == targets.shape[0]  # Ensure features and targets align
    assert features.shape[1] == window_size  # Verify that window size is applied


def test_cstr_dataset_registry():
    # Verify that CSTRDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "CSTRDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("CSTRDataset")
    dataset.load_data()
    assert len(dataset.list_targets()) > 0  # Ensure targets are available


if __name__ == "__main__":
    pytest.main()
