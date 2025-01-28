import pickle
from typing import Any


class CacheManager:
    """
    A utility class to handle caching of processed data
    """

    @staticmethod
    def save_to_cache(data: Any, cache_path: str) -> None:
        """
        Save data to a cache file.

        :param data: Data to save.
        :param cache_path: Path to the cache file.
        """
        with open(cache_path, "wb") as cache_file:
            pickle.dump(data, cache_file)

    @staticmethod
    def load_from_cache(cache_path: str) -> Any:
        """
        Load data from a cache file.

        :param cache_path: Path to the cache file.
        :return: Loaded data.
        """
        with open(cache_path, "rb") as cache_file:
            return pickle.load(cache_file)
