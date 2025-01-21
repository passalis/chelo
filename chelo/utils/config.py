import json
import os

class CheloConfig:
    def __init__(self, config_file='chelo.json'):
        user_dir: str = os.path.expanduser(os.path.join("~", ".chelo"))
        self.config_file: str = os.getenv("CHELO_DATASET_PATH", user_dir)
        os.makedirs(self.config_file, exist_ok=True)
        self.config_file = os.path.join(self.config_file, config_file)

        self.default_config = {
            "kaggle_username": "",
            "kaggle_key": ""
        }
        # Load the configuration when the class is instantiated
        self.load_config()

    def load_config(self):
        """Load the configuration from the file. Create the file if it does not exist."""
        if not os.path.exists(self.config_file):
            print(f"Configuration file '{self.config_file}' does not exist. Creating a new one.")
            self.create_default_config()
        else:
            with open(self.config_file, 'r') as f:
                try:
                    self.config = json.load(f)
                except json.JSONDecodeError:
                    print("Error decoding the configuration file. Using default configuration.")
                    self.config = self.default_config

    def create_default_config(self):
        """Create the default configuration file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.default_config, f, indent=4)
        print(f"Default configuration file '{self.config_file}' created.")

    def get(self, key):
        """Get the value of a configuration key."""
        return self.config.get(key)

    def set(self, key, value):
        """Set the value of a configuration key."""
        self.config[key] = value
        self.save_config()

    def save_config(self):
        """Save the current configuration to the file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"Configuration saved to '{self.config_file}'.")

