# import pkgutil
# import importlib
# from ..registry import DatasetRegistry
# from ..base import CheLoDataset

# def auto_register_datasets():
#     """
#     Automatically discover and register datasets in the `datasets` package.
#     """
#     for _, module_name, _ in pkgutil.iter_modules(__path__):
#         module = importlib.import_module(f"{__name__}.{module_name}")
#         for attr_name in dir(module):
#             attr = getattr(module, attr_name)
#             if isinstance(attr, type) and issubclass(attr, CheLoDataset) and attr is not CheLoDataset:
#                 DatasetRegistry.register(attr)
#
# auto_register_datasets()