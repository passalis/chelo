import pytest
from chelo.datasets.vle import VLEDataset
from chelo.registry import DatasetRegistry


def test_vle_dataset_basic_features_and_targets():
    # Load the dataset with specific features and targets
    dataset = VLEDataset(
        selected_features=["T [C]"],
        selected_targets=["p_vap", "h_vap"]
    )
    dataset.load_data()
    # Verify that selected features and targets are present
    assert "T [C]" in dataset.list_features()

    assert "p_vap" in dataset.list_targets()
    assert "h_vap" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 1
    assert targets.shape[1] == 2


def test_vle_dataset_default_selection():
    # Load the dataset without specifying features or targets
    dataset = VLEDataset()
    dataset.load_data()

    # Verify that features and targets are automatically selected
    assert len(dataset.list_features()) > 0  # Ensure features are selected
    assert len(dataset.list_targets()) > 0  # Ensure targets are selected

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure features data is loaded
    assert targets.shape[0] > 0  # Ensure targets data is loaded


def test_vle_dataset_registry():
    # Verify that VLEDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "VLEDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("VLEDataset")
    dataset.load_data()
    assert len(dataset.list_targets()) > 0  # Ensure targets are available


if __name__ == "__main__":
    pytest.main()
