import pytest
from chelo.datasets.ames_mutagenicity import AmesMutagenicityDataset
from chelo.registry import DatasetRegistry


def test_ames_mutagenicity_dataset_basic_features_and_targets():
    # Load the dataset with specific features and targets
    dataset = AmesMutagenicityDataset(
        selected_features=['NumValenceElectrons', 'qed', 'TPSA', 'MolMR', 'BalabanJ', 'BertzCT', 'MolWt', 'MolLogP'],
        selected_targets=["mutagenicity"]
    )
    dataset.load_data()

    # Verify that selected features and targets are present
    assert "NumValenceElectrons" in dataset.list_features()  # Replace with actual feature name
    assert "BalabanJ" in dataset.list_features()  # Replace with actual feature name
    assert "mutagenicity" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 8
    assert targets.shape[1] == 1


def test_ames_mutagenicity_dataset_default_selection():
    # Load the dataset without specifying features or targets
    dataset = AmesMutagenicityDataset()
    dataset.load_data()

    # Verify that features and targets are automatically selected
    assert len(dataset.list_features()) > 0  # Ensure features are selected
    assert "mutagenicity" in dataset.list_targets()

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure features data is loaded
    assert targets.shape[0] > 0  # Ensure targets data is loaded


def test_ames_mutagenicity_dataset_registry():
    # Verify that AmesMutagenicityDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "AmesMutagenicityDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("AmesMutagenicityDataset")
    dataset.load_data()
    assert "mutagenicity" in dataset.list_targets()


if __name__ == "__main__":
    pytest.main()
