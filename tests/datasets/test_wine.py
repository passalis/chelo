import pytest
from chelo.datasets.wine_quality import WineQualityDataset
from chelo.registry import DatasetRegistry


def test_wine_quality_dataset_red():
    # Load the red wine dataset
    dataset = WineQualityDataset(wine_type="red", selected_features=["alcohol", "pH"], selected_targets=["quality"])
    dataset.load_data()

    # Verify that basic features indeed exist
    assert "alcohol" in dataset.list_features()
    assert "pH" in dataset.list_features()
    assert "quality" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 2  # Two selected features
    assert targets.shape[1] == 1  # One selected target


def test_wine_quality_dataset_white():
    # Load the white wine dataset
    dataset = WineQualityDataset(wine_type="white")
    dataset.load_data()

    # Verify that basic features indeed exist
    assert "alcohol" in dataset.list_features()
    assert "quality" in dataset.list_targets()

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure data is loaded
    assert targets.shape[0] > 0


# Test Cases for Dataset Registry
def test_dataset_registry():
    # Verify that WineQualityDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "WineQualityDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("WineQualityDataset", wine_type="red")
    dataset.load_data()
    assert "alcohol" in dataset.list_features()


if __name__ == "__main__":
    pytest.main()
