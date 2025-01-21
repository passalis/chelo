import pytest
from chelo.datasets.coal_fired_plant import CoalFiredPlantDataset
from chelo.registry import DatasetRegistry


def test_coal_fired_plant_dataset_basic_features_and_targets():
    # Load the dataset with specific features and targets
    dataset = CoalFiredPlantDataset(
        selected_features=['Main steam flow (t/h)', 'Main steam temperature (boiler side) (℃)'],
        selected_targets=["Boiler Eff (%)"]
    )
    dataset.load_data()

    # Verify that selected features and targets are present
    assert "Main steam flow (t/h)" in dataset.list_features()
    assert "Main steam temperature (boiler side) (℃)" in dataset.list_features()
    assert "Boiler Eff (%)" in dataset.list_targets()

    # Verify data loading
    features, targets = dataset.to_numpy()
    assert features.shape[1] == 2  # Two selected features
    assert targets.shape[1] == 1  # One selected target


def test_coal_fired_plant_dataset_default_selection():
    # Load the dataset without specifying features or targets
    dataset = CoalFiredPlantDataset()
    dataset.load_data()

    # Verify that features and targets are automatically selected
    assert len(dataset.list_features()) > 0  # Ensure features are selected
    assert "Boiler Eff (%)" in dataset.list_targets()

    # Verify data shape
    features, targets = dataset.to_numpy()
    assert features.shape[0] > 0  # Ensure features data is loaded
    assert targets.shape[0] > 0  # Ensure targets data is loaded


# Test Cases for Dataset Registry
def test_coal_fired_plant_dataset_registry():
    # Verify that CoalFiredPlantDataset is registered
    available_datasets = DatasetRegistry.list_datasets()
    assert "CoalFiredPlantDataset" in available_datasets

    # Retrieve and load dataset
    dataset = DatasetRegistry.get_dataset("CoalFiredPlantDataset")
    dataset.load_data()
    assert "Boiler Eff (%)" in dataset.list_targets()


if __name__ == "__main__":
    pytest.main()
