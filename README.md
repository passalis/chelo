# CheLo (Chemical Engineering Dataset Loader) Library



<div align="center">

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Test Status (master)](https://github.com/passalis/chelo/actions/workflows/ci_master.yml/badge.svg)](https://github.com/passalis/chelo/actions/workflows/ci_master.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/passalis/chelo/badge)](https://www.codefactor.io/repository/github/passalis/chelo)
[![codecov](https://codecov.io/github/passalis/chelo/graph/badge.svg?token=BX57HE0KNF)](https://codecov.io/github/passalis/chelo)
![PyPI](https://img.shields.io/pypi/v/chelo)

<div>

## Overview
Loading a dataset is often one of the most challenging parts of building machine learning pipelines, especially for beginners. 
The **CheLo** Library is a Python library specifically designed to make machine learning more accessible to chemical engineering students, aiding in their learning journey and supporting researchers working on related projects. 
By providing an easy to use framework, this library simplifies the exploration of data-driven modeling, empowering users to access, manage, and utilize chemical engineering datasets for machine learning and statistical analysis with ease.


## Key Features
- **Dataset Standardization**: Unified API for accessing and exploring datasets.
- **Multiple Data Formats**: Provides ready to use loaders for numpy and PyTorch.
- **Preprocessing Tools**: Methods for feature/target selection, statistics, and previewing datasets.
- **Dataset Management**: Automated downloading, caching, and registry of datasets.
- **Extensibility**: Abstract base class for easy addition of new datasets.

## Disclaimer
I am not associated with any of the datasets provided in this library, nor do I host them. 
The CheLo Library solely provides tools to facilitate the downloading and loading of publicly available datasets to enhance accessibility for educational and research purposes. 
If you have any concerns, including removal requests or any other inquiries, please feel free to contact [me](https://people.auth.gr/passalis/) directly.

## Installation

To install the library, run the following command:

```bash
pip install chelo
```

To install the library in editable mode for development purposes:

```bash
git clone https://github.com/your-repo/chelo.git
cd chelo
pip install -e .
```

## Package Structure

```plaintext
chelo/                   # Root package
├── __init__.py          # Exposes core components
├── base.py              # Abstract base class and shared utilities
├── datasets/            # Dataset-specific implementations
│   └── ...              # Dataset implementations
├── utils/               # Utility functions and helpers
│   ├── __init__.py      # Utility imports
│   └──  download.py      # Dataset downloader and caching
├── registry.py          # Dataset registry
└── tests/               # Unit and integration tests
    ├── __init__.py      # Makes this directory a package
    ├── test_base.py     # Tests for the base class
    └── test_X.py        # Tests for X dataset
```

## Usage Guide

### Loading a Dataset

```python
from chelo.datasets.wine_quality import WineQualityDataset

# Instantiate the dataset
dataset = WineQualityDataset(wine_type="red", selected_features=["alcohol", "pH"], selected_targets=["quality"])

# Load data (downloads if not cached)
dataset.load_data()

# Access dataset information
info = dataset.get_dataset_info()
print("Dataset Info:", info)
```

### Accessing Data

```python
# Convert to numpy arrays
features, targets = dataset.to_numpy()
print("Features shape:", features.shape)
print("Targets shape:", targets.shape)

# Convert to PyTorch Dataset
pytorch_dataset = dataset.to_pytorch()
print("Number of samples in PyTorch Dataset:", len(pytorch_dataset))

```

### Dataset Statistics and Preview

```python
# Get basic statistics
stats = dataset.statistics()
print("Statistics:", stats)

# Preview the dataset
preview = dataset.preview(n=5)
print("Preview:", preview)
```

### Registering and Accessing Datasets

```python
from chelo.registry import DatasetRegistry

# List available datasets
print("Available Datasets:", DatasetRegistry.list_datasets())

# Retrieve and load a dataset by name
dataset = DatasetRegistry.get_dataset("WineQualityDataset", wine_type="red")
dataset.load_data()
```

## Extending the Library

To add a new dataset, create a new class that inherits from `ChemicalEngineeringDataset` and implement the required methods:

1. **Create a new dataset module:**

```plaintext
chelo/datasets/my_new_dataset.py
```

2. **Implement the dataset class:**

```python
from ..base import ChemicalEngineeringDataset

class MyNewDataset(ChemicalEngineeringDataset):
    def __init__(self, selected_features=None, selected_targets=None):
        super().__init__(selected_features, selected_targets)
        self.dataset_name = "My New Dataset"

    def load_data(self):
        # Load dataset into self.raw_features and self.raw_targets
        pass

    def list_features(self):
        return list(self.raw_features.keys())

    def list_targets(self):
        return list(self.raw_targets.keys())

    def get_dataset_info(self):
        return {"name": self.dataset_name, "description": "Description of the dataset."}
```

3. **Register the dataset:**

Add the following line to `chelo/datasets/__init__.py`:

```python
from .my_new_dataset import MyNewDataset
DatasetRegistry.register(MyNewDataset)
```

## Advanced Features

### Downloader Utility

The library includes a downloader utility for downloading and caching datasets. Files are stored in a structured cache directory (default: `~/.chelo`).

#### Example Usage

```python
from chelo.utils.download import DatasetDownloader

downloader = DatasetDownloader()

# Download a dataset file
url = "https://example.com/dataset.csv"
file_path = downloader.download(url, dataset_name="example_dataset", filename="example.csv")
print("Downloaded file path:", file_path)
```

### Dataset Registry

The registry dynamically manages available datasets, allowing users to list and retrieve datasets by name.

#### Example Usage

```python
from chelo.registry import DatasetRegistry

# List all registered datasets
print("Available Datasets:", DatasetRegistry.list_datasets())

# Retrieve a dataset by name
dataset = DatasetRegistry.get_dataset("WineQualityDataset", wine_type="white")
```

## Testing

The library includes comprehensive unit tests to ensure correctness and reliability. Run tests using `pytest`:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes and add tests.
4. Submit a pull request with a detailed description of your changes.

## License

This library is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or feedback, please contact [me](https://people.auth.gr/passalis/).

