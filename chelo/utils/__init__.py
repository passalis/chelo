from ..utils.config import CheloConfig
import os

_chelo_configuration = CheloConfig()
# Export Kaggle environment variables (this should be done before importing kaggle)
os.environ["KAGGLE_USERNAME"] = _chelo_configuration.get("kaggle_username")
os.environ["KAGGLE_KEY"] = _chelo_configuration.get("kaggle_key")
