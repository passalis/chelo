import pytest
from chelo.datasets.vapor_pressure import VaporPressureDataset
from chelo.registry import DatasetRegistry


def test_vapor_pressure_dataset_basic_features_and_targets():
    # Load the dataset with specific features and targets
    dataset = VaporPressureDataset(
        selected_features=['SMILES', 'T'],
        selected_targets=["p_vap"]
    )
    dataset.load_data()

    # Verify that selected features and targets are present
    assert "SMILES" in dataset.list_features()  # Replace with actual feature name
    assert "T" in dataset.list_features()  # Replace with actual feature name
    assert "p_vap" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 2
    assert targets.shape[1] == 1


def test_vapor_pressure_dataset_default_selection():
    # Load the dataset without specifying features or targets
    dataset = VaporPressureDataset()
    dataset.load_data()

    # Verify that features and targets are automatically selected
    assert len(dataset.list_features()) > 0  # Ensure features are selected
    assert "p_vap" in dataset.list_targets()

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure features data is loaded
    assert targets.shape[0] > 0  # Ensure targets data is loaded


def test_vapor_pressure_dataset_registry():
    # Verify that VaporPressureDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "VaporPressureDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("VaporPressureDataset")
    dataset.load_data()
    assert "p_vap" in dataset.list_targets()


if __name__ == "__main__":
    pytest.main()
