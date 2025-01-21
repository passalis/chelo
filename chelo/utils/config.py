import json
import os
from typing import Any, Dict, Optional


class CheloConfig:
    """
    A class for managing the Chelo configuration file.

    Handles loading, creating, updating, and saving configuration settings for Chelo.
    """

    def __init__(self, config_file: str = "chelo.json") -> None:
        """
        Initialize the CheloConfig instance.

        :param config_file: Name of the configuration file (default: 'chelo.json').
        """
        user_dir: str = os.path.expanduser(os.path.join("~", ".chelo"))
        self.config_dir: str = os.getenv("CHELO_DATASET_PATH", user_dir)
        os.makedirs(self.config_dir, exist_ok=True)
        self.config_file: str = os.path.join(self.config_dir, config_file)

        self.default_config: Dict[str, str] = {
            "kaggle_username": "",
            "kaggle_key": ""
        }

        # Load the configuration when the instance is created
        self.config: Dict[str, Any] = {}
        self.load_config()

    def load_config(self) -> None:
        """
        Load the configuration from the file. Create the file with default values if it doesn't exist.
        """
        if not os.path.exists(self.config_file):
            print(f"Configuration file '{self.config_file}' does not exist. Creating a new one with default settings.")
            self.create_default_config()
        else:
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
            except json.JSONDecodeError:
                print("Error decoding the configuration file. Reverting to default configuration.")
                self.config = self.default_config
                self.save_config()

    def create_default_config(self) -> None:
        """
        Create a configuration file with default values.
        """
        self.config = self.default_config
        self.save_config()
        print(f"Default configuration file '{self.config_file}' created.")

    def get(self, key: str) -> Optional[Any]:
        """
        Get the value of a configuration key.

        :param key: The configuration key to retrieve.
        :return: The value associated with the key, or None if the key does not exist.
        """
        return self.config.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set the value of a configuration key and save the configuration.

        :param key: The configuration key to update.
        :param value: The new value for the key.
        """
        self.config[key] = value
        self.save_config()

    def save_config(self) -> None:
        """
        Save the current configuration to the file.
        """
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to '{self.config_file}'.")
        except IOError as e:
            print(f"Error saving configuration: {e}")
