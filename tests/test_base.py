import pytest
from chelo.base import CheLoDataset


# Mock Base Class Implementation for Testing
class MockDataset(CheLoDataset):
    def __init__(self, selected_features=None, selected_targets=None):
        super().__init__(selected_features, selected_targets)
        self.dataset_name = "Mock Dataset"

    def load_data(self):
        self.raw_features = {
            "Temperature": [300, 310, 320],
            "Pressure": [101.3, 98.6, 102.5],
        }
        self.raw_targets = {"Reaction Rate": [1.2, 1.5, 1.3]}
        self._apply_initial_selections()

    def list_features(self):
        return list(self.raw_features.keys())

    def list_targets(self):
        return list(self.raw_targets.keys())

    def get_dataset_info(self):
        return {
            "name": self.dataset_name,
            "description": "A mock dataset for testing.",
            "size": self.size(),
        }


# Test Cases for Base Class
def test_mock_dataset():
    dataset = MockDataset(selected_features=["Temperature"], selected_targets=["Reaction Rate"])
    dataset.load_data()

    # Verify dataset information
    assert dataset.get_dataset_info()["name"] == "Mock Dataset"
    assert dataset.list_features() == ["Temperature", "Pressure"]
    assert dataset.list_targets() == ["Reaction Rate"]

    # Verify feature and target selection
    features, targets = dataset.to_numpy()
    assert features.shape == (3, 1)
    assert targets.shape == (3, 1)


if __name__ == "__main__":
    pytest.main()
