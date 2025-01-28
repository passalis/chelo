
<div align="center">

<img src="https://raw.githubusercontent.com/passalis/chelo/main/logo.svg" width="400px">

<h2>Chemical Engineering Dataset Loader (CheLo) Library</h2>


[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/chelo/badge/?version=latest)](https://chelo.readthedocs.io/en/latest/)
[![Test Status (master)](https://github.com/passalis/chelo/actions/workflows/ci_master.yml/badge.svg)](https://github.com/passalis/chelo/actions/workflows/ci_master.yml)
[![CodeFactor](https://www.codefactor.io/repository/github/passalis/chelo/badge)](https://www.codefactor.io/repository/github/passalis/chelo)
[![codecov](https://codecov.io/github/passalis/chelo/graph/badge.svg?token=BX57HE0KNF)](https://codecov.io/github/passalis/chelo)
[![PyPI](https://img.shields.io/pypi/v/chelo)](https://pypi.org/project/chelo/)

</div>

## Overview
Loading a dataset is often one of the most challenging parts of building machine learning pipelines, especially for beginners. 
The **CheLo** Library is a Python library specifically designed to make machine learning more accessible to chemical engineering students, aiding in their learning journey and supporting researchers working on related projects. 
By providing an easy to use framework, this library simplifies the exploration of data-driven modeling, empowering users to access, manage, and utilize chemical engineering datasets for machine learning and statistical analysis with ease.
Check the [CheLo's documentation](https://chelo.readthedocs.io/en/latest/) for detailed usage instructions.


## Key Features
- **Dataset Standardization**: Unified API for accessing and exploring datasets.
- **Multiple Data Formats**: Provides ready to use loaders for numpy and PyTorch.
- **Preprocessing Tools**: Methods for feature/target selection, statistics, and previewing datasets.
- **Dataset Management**: Automated downloading, caching, and registry of datasets.
- **Extensibility**: Abstract base class for easy addition of new datasets.


## Datasets 
**CheLo** currently supports 8 datasets. You can find a list of the supported datasets [here](DATASETS.md).


## Installation

To install the library, run the following command:

```bash
pip install chelo
```

Note that for some datasets further configuration might be needed after installation (see `Configuration and Dataset Path Setup`).

## Usage Guide

### Loading a Dataset

Loading a dataset with CheLo is simple and straightforward.
Just import the desired dataset (or use `DatasetRegistry`) and call `load_data()`. 
Note that for certain datasets, such as those hosted on Kaggle, you may need to configure the library with your access credentials beforehand.


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

## Configuration 

By default, the CheLo library stores datasets in the directory `~/.chelo` (in the user's home directory). 
This default path can be customized by setting the `CHELO_DATASETS_PATH` environment variable. 
This allows you to choose a different location to store datasets and configuration files if needed.

The default dataset storage path is:

```bash
~/.chelo
```
Where ~ represents the user's home directory.
For Windows users, the path would be:
```bash
C:\Users\<USERNAME>/.chelo
```
This path is used by CheLo to download, store, and manage datasets by default.

### Custom API configuration
For some datasets credentials mights be needed to download datasets.
The CheLo library uses a ``chelo.json`` configuration file to store such settings (this file exists under the path set in  `CHELO_DATASETS_PATH`).
If the configuration file does not exist, it will be automatically created with a default structure.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes and add tests. 
4. Submit a pull request with a detailed description of your changes.

## Disclaimer
I am not associated with any of the datasets provided in this library, nor do I host them. 
The CheLo Library solely provides tools to facilitate the downloading and loading of publicly available datasets to enhance accessibility for educational and research purposes. 
Users are responsible for ensuring their use complies with the datasets' licenses and terms of use. 
Please refer to the original dataset provider for license details.
If you have any concerns, including removal requests or any other inquiries, please feel free to contact [me](https://people.auth.gr/passalis/) directly.


## License

This library is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or feedback, please contact [me](https://people.auth.gr/passalis/).

