from typing import Type, Dict, List, Any


class DatasetRegistry:
    """
    A registry to manage available datasets in CheLo.
    """
    _datasets: Dict[str, Type] = {}

    @classmethod
    def register(cls, dataset_cls: Type) -> None:
        """
        Register a dataset class with the registry.

        :param dataset_cls: The dataset class to register.
        :raises ValueError: If the dataset is already registered.
        """
        dataset_name: str = dataset_cls.__name__
        if dataset_name in cls._datasets:
            raise ValueError(f"Dataset {dataset_name} is already registered.")
        cls._datasets[dataset_name] = dataset_cls

    @classmethod
    def list_datasets(cls) -> List[str]:
        """
        List all registered datasets.

        :return: A list of names of registered datasets.
        """
        return list(cls._datasets.keys())

    @classmethod
    def get_dataset(cls, name: str, **kwargs: Any) -> Any:
        """
        Retrieve an instance of the specified dataset by name.

        :param name: Name of the dataset to retrieve.
        :param kwargs: Additional arguments to pass to the dataset constructor.
        :return: An instance of the dataset.
        :raises ValueError: If the dataset is not found.
        """
        if name not in cls._datasets:
            raise ValueError(f"Dataset {name} not found! Available datasets: {cls.list_datasets()}")
        return cls._datasets[name](**kwargs)


def register_dataset(cls: Type) -> Type:
    """
    A decorator to register a dataset class with the registry.

    :param cls: The dataset class to register.
    :return: The dataset class itself.
    """
    DatasetRegistry.register(cls)
    return cls
