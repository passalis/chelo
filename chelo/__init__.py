from .base import CheLoDataset
from .registry import DatasetRegistry
from .utils.config import CheloConfig
import pkgutil
import importlib


__all__ = ["CheLoDataset", "DatasetRegistry",]
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    importlib.import_module(f".{module_name}", package=__name__)

_chelo_configuration = CheloConfig()