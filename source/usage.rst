Usage
=====

Load a Dataset
--------------

Example of loading the Wine Quality dataset:

.. code-block:: python

   from chelo.datasets.wine_quality import WineQualityDataset

   # Load the red wine dataset
   dataset = WineQualityDataset(wine_type="red")
   dataset.load_data()

   # Get dataset information
   info = dataset.get_dataset_info()
   print(info)

Convert to Numpy Format
------------------------

.. code-block:: python

   features, targets = dataset.to_numpy()
   print("Features:", features.shape)
   print("Targets:", targets.shape)