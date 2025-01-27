import pytest
from chelo.datasets.bc_factor import BCFactorDataset
from chelo.registry import DatasetRegistry


def test_bcfactor_dataset_basic_features_and_targets():
    # Load the dataset with specific features and targets
    dataset = BCFactorDataset(
        selected_features=['NumHAcceptors', 'NumHeteroatoms', 'qed', 'TPSA', ],
        selected_targets=["bcf"]
    )
    dataset.load_data()

    # Verify that selected features and targets are present
    assert "NumHAcceptors" in dataset.list_features()  # Replace with actual feature name
    assert "qed" in dataset.list_features()  # Replace with actual feature name
    assert "bcf" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 4
    assert targets.shape[1] == 1


def test_bcfactor_dataset_default_selection():
    # Load the dataset without specifying features or targets
    dataset = BCFactorDataset()
    dataset.load_data()

    # Verify that features and targets are automatically selected
    assert len(dataset.list_features()) > 0  # Ensure features are selected
    assert "bcf" in dataset.list_targets()

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure features data is loaded
    assert targets.shape[0] > 0  # Ensure targets data is loaded


def test_bcfactor_dataset_registry():
    # Verify that BCFactorDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "BCFactorDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("BCFactorDataset")
    dataset.load_data()
    assert "bcf" in dataset.list_targets()


if __name__ == "__main__":
    pytest.main()
