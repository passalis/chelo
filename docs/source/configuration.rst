Configuration and Dataset Path Setup
====================================
.. _configuration-guide:

By default, the CheLo library stores datasets in the directory ``~/.chelo`` (in the user's home directory).
This default path can be customized by setting the ``CHELO_DATASETS_PATH`` environment variable.
This allows you to choose a different location to store datasets and configuration files if needed.

The default dataset storage path is:

.. code-block:: bash

   ~/.chelo

Where ``~`` represents the user's home directory.
For Windows users, the path would be:

.. code-block:: bash

   C:\Users\<USERNAME>/.chelo

This path is used by CheLo to download, store, and manage datasets by default.

Custom API Configuration
-------------------------

For some datasets, credentials might be needed to download datasets.
The CheLo library uses a ``chelo.json`` configuration file to store such settings (this file exists under the path set in ``CHELO_DATASETS_PATH``).
If the configuration file does not exist, it will be automatically created with a default structure.

Including your Kaggle credentials (username and key) and the dataset storage path.

Kaggle API Configuration
-------------------------

For datasets hosted on Kaggle, you’ll need to configure your Kaggle API credentials.
Follow these steps:

- Go to https://www.kaggle.com and log in to your account.
- Generate an API Token: Navigate to your account settings and click the 'Create New API Token' button.
  This will download a ``kaggle.json`` file to your computer, containing your ``KAGGLE_USERNAME`` and ``KAGGLE_KEY``.
- Use the credentials in the ``kaggle.json`` file to update your ``chelo.json`` configuration file. For example:

.. code-block:: json

   {
       "kaggle_username": "your_kaggle_username",
       "kaggle_key": "your_kaggle_key"
   }

Customizing the Dataset Path
----------------------------

If you want to store your datasets in a different directory, you can set the ``CHELO_DATASETS_PATH`` environment variable to your preferred directory.
The CheLo library will use this path for dataset storage instead of the default location.

To set the ``CHELO_DATASETS_PATH`` environment variable:

On Linux/macOS (using Bash):

.. code-block:: bash

   export CHELO_DATASETS_PATH="/path/to/your/datasets"

On Windows (Command Prompt):

.. code-block:: bash

   set CHELO_DATASETS_PATH=C:\path\to\your\datasets

Once the environment variable is set, CheLo will use the specified directory for storing datasets.


Note that you need to run your program from a shell or terminal where the CHELO_DATASETS_PATH environment variable has already been set. This means:

- After setting the variable (e.g., using export on Linux/macOS or set on Windows), you must remain in the same terminal session when executing your program. If you close the terminal or open a new one, the variable will no longer be set unless you configure it to load automatically.

For example:

- On Linux/macOS: If you type ``export CHELO_DATASETS_PATH="/path/to/your/datasets"`` and then run your program in the same terminal, the variable will work. However, if you open a new terminal or restart your computer, you'll need to set the variable again, unless you add it to your shell's configuration file (e.g., .bashrc or .zshrc).

- On Windows: If you use set ``CHELO_DATASETS_PATH=C:\path\to\your\datasets`` in the Command Prompt, it will only apply to that Command Prompt session. If you close it or open a new one, you’ll need to set it again unless you add it to your system's environment variables via the Control Panel or PowerShell.

This is important because the program relies on the ``CHELO_DATASETS_PATH`` variable to determine where datasets are stored. If the variable isn't set, CheLo will revert to using the default path (~/.chelo).


