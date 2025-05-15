from .base import CheLoDataset
from .registry import DatasetRegistry
import pkgutil
import importlib


__all__ = ["CheLoDataset", "DatasetRegistry",]
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if 'chelo' in loader.path:
        importlib.import_module(f".{module_name}", package=__name__)
