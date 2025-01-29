from datetime import datetime
from typing import List, Dict, Optional, Union
from ..base import CheLoDataset
from ..registry import register_dataset
from ..utils.downloader import DatasetDownloader
import pandas as pd
import numpy as np
from ..utils.cache_manager import CacheManager
import os


@register_dataset
class OPSDPVDataset(CheLoDataset):
    """
    A dataset class for Open Power System Data PV dataset. Provides functionalities to download, process, and prepare
    the dataset for forecasting tasks.
    """

    # Dataset metadata
    _URLS: List[str] = [
        "https://data.open-power-system-data.org/weather_data/2020-09-16/weather_data.csv",
        "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"
    ]
    _FILES: List[str] = ["weather_data.csv", "time_series_60min_singleindex.csv"]
    _CHECKSUMS: List[str] = ["dea87ece8eded83802c8e6c740ba2e53", "3e2598ed455f85e1df970998a8552d59"]

    def __init__(
            self,
            country: str = 'GR',
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            historical_window: int = 48,
            prediction_horizon: int = 12,
            prediction_window: int = 24,
            prediction_step: int = 6,
            use_future_weather: bool = False,
            selected_features: Optional[List[str]] = None,
            selected_targets: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the OPSD PV Dataset.

        :param country: The country to use. Must be one of the available countries.
        :param start_date: The start date of the dataset. Defaults to earliest available data if not provided.
            Format: YYYY-MM-DD hour:minute:second
        :param end_date: The end date of the dataset. Defaults to the latest available data if not provided.
            Format: YYYY-MM-DD hour:minute:second
        :param historical_window: Number of time steps in the historical window for feature processing.
        :param prediction_horizon: Time steps into the future for prediction targets.
        :param prediction_window: The length of the prediction window.
        :param prediction_step: The step size for prediction data.
        :param use_future_weather: Whether to use future weather as feature (e.g., as forecast).
        :param selected_features: List of selected features to include.
        :param selected_targets: List of selected targets to include.
        """
        super().__init__(selected_features, selected_targets)

        self.dataset_name: str = "OPSD PV Dataset"
        self.dataset_url: str = "https://open-power-system-data.org/"
        self.country: str = country
        self.available_countries: List[str] = [
            'AT', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'ES', 'FR', 'GR', 'IT', 'NL', 'PT', 'RO'
        ]

        if country not in self.available_countries:
            raise ValueError(f"Invalid country: {country}. Must be one of {self.available_countries}.")

        self.start_date: datetime = start_date or datetime(2000, 1, 1, 12, 0, 0)
        self.end_date: datetime = end_date or datetime(2999, 1, 1, 10, 0, 0)

        self._data_type = 'timeseries'
        self.historical_window: int = historical_window
        self.prediction_horizon: int = prediction_horizon
        self.prediction_window: int = prediction_window
        self.prediction_step: int = prediction_step
        self.use_future_weather: bool = use_future_weather

    def load_data(self) -> None:
        """
        Download, process, and cache the dataset for the specified country and date range.
        """
        downloader: DatasetDownloader = DatasetDownloader()
        file_paths = []

        # Download required files
        for i, file_url in enumerate(self._URLS):
            file_path = downloader.download(
                file_url,
                dataset_name="opsd_pv",
                filename=self._FILES[i],
                checksum=self._CHECKSUMS[i],
            )
            file_paths.append(file_path)

        # Define cache file path
        cache_dir = os.path.join(downloader.cache_dir, "opsd_pv", ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = os.path.join(cache_dir, "processed_dataset.joblib")

        # Check if valid cache exists
        load_success: bool = False
        if os.path.exists(cache_file_path):
            try:
                df = CacheManager.load_from_cache(cache_file_path)
                load_success = True
            except Exception:
                load_success = False

        if not load_success:
            # Define relevant columns for weather and PV data
            weather_columns = ['year', 'month', 'day', 'hour']
            pv_columns = ['year', 'month', 'day', 'hour']
            for country in self.available_countries:
                weather_columns += [
                    f"{country}_temperature",
                    f"{country}_radiation_direct_horizontal",
                    f"{country}_radiation_diffuse_horizontal"
                ]
                pv_columns.append(f"{country}_solar_generation_actual")

            print("Preprocessing and caching data (this might take a while)...")
            weather_df = pd.read_csv(file_paths[0])
            pv_df = pd.read_csv(file_paths[1])

            # Add datetime-related columns
            for df in [weather_df, pv_df]:
                df['year'] = pd.to_datetime(df['utc_timestamp']).dt.year
                df['month'] = pd.to_datetime(df['utc_timestamp']).dt.month
                df['day'] = pd.to_datetime(df['utc_timestamp']).dt.day
                df['hour'] = pd.to_datetime(df['utc_timestamp']).dt.hour
                df.drop(columns=['utc_timestamp'], inplace=True)

            # Filter and interpolate the data
            weather_df = weather_df[weather_columns].interpolate().dropna()
            pv_df = pv_df[pv_columns].interpolate().dropna()

            # Merge weather and PV data
            df = pd.merge(weather_df, pv_df, on=['year', 'month', 'day', 'hour'], how='inner')

            # Save processed data to cache
            CacheManager.save_to_cache(df, cache_file_path)

        # Process and filter the data for the specified country
        country_columns = [
            'year', 'month', 'day', 'hour',
            f"{self.country}_temperature",
            f"{self.country}_radiation_direct_horizontal",
            f"{self.country}_radiation_diffuse_horizontal",
            f"{self.country}_solar_generation_actual"
        ]
        raw_data = df[country_columns].copy()
        raw_data.rename(columns={
            f"{self.country}_temperature": "temperature",
            f"{self.country}_radiation_direct_horizontal": "radiation_direct_horizontal",
            f"{self.country}_radiation_diffuse_horizontal": "radiation_diffuse_horizontal",
            f"{self.country}_solar_generation_actual": "solar_generation_actual"
        }, inplace=True)

        # Filter data based on date range
        raw_data['datetime'] = pd.to_datetime(raw_data[['year', 'month', 'day', 'hour']])
        raw_data = raw_data[(raw_data['datetime'] >= self.start_date) & (raw_data['datetime'] <= self.end_date)]
        raw_data = raw_data.drop(columns=['datetime'])
        assert len(raw_data) > 0, "Filtering dates probably lead to empty dataset."

        # Process features and targets
        self._process_features_and_targets(raw_data)
        self._apply_initial_selections()

    def _process_features_and_targets(self, raw_data: pd.DataFrame) -> None:
        """
        Process features and targets for model readiness.

        :param raw_data: Filtered raw data.
        """

        self.raw_features = raw_data.to_dict(orient="list")
        self.raw_targets = {"solar_generation_actual_target": raw_data["solar_generation_actual"].tolist()}

        target_length = len(self.raw_targets["solar_generation_actual_target"])

        # Process targets
        target_values = np.asarray(self.raw_targets["solar_generation_actual_target"])
        processed_targets = np.array([
            target_values[i + self.historical_window + self.prediction_horizon:
                          i + self.historical_window + self.prediction_horizon + self.prediction_window]
            for i in range(0, target_length - self.historical_window - self.prediction_horizon - self.prediction_window,
                           self.prediction_step)
        ])
        self.raw_targets["solar_generation_actual_target"] = processed_targets.reshape((-1, self.prediction_window))

        prediction_offset = self.prediction_horizon + self.prediction_window

        # Process features
        for feature_name, feature_values in self.raw_features.items():
            feature_array = np.asarray(feature_values)
            if (feature_name in ('temperature', 'radiation_direct_horizontal', 'radiation_diffuse_horizontal')
                    and self.use_future_weather):
                processed_features = np.array([
                    feature_array[i + prediction_offset:i + self.historical_window + prediction_offset]
                    for i in range(0, target_length - self.historical_window - prediction_offset,
                                   self.prediction_step)
                ])
            else:
                processed_features = np.array([
                    feature_array[i:i + self.historical_window]
                    for i in range(0, target_length - self.historical_window - prediction_offset,
                                   self.prediction_step)
                ])

            self.raw_features[feature_name] = processed_features

    def get_dataset_info(self) -> Dict[str, Union[str, List[str]]]:
        """Return metadata about the dataset."""
        return {
            "name": self.dataset_name,
            "description": "Open Power System Data - PV Dataset",
            "features": self.list_features(),
            "targets": self.list_targets(),
            "url": self.dataset_url
        }
