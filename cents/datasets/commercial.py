import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from cents.utils.config_loader import load_yaml, apply_overrides

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class CommercialDataset(TimeSeriesDataset):
    def __init__(self, cfg: DictConfig = None, 
                 overrides: Optional[List[str]] = None,
                 skip_heavy_processing: bool = False):
        
        """
        Initializes the commercial energy dataset.
        Original Dataset: https://github.com/buds-lab/building-data-genome-project-2
        Note: This uses the clean version of their data, which already has some preprocessing done.
        Args:
            cfg (Optional[DictConfig]): Override Hydra config; if None,
                load from `config/dataset/commercial.yaml`.
            overrides (Optional[List[str]]): Override Hydra config; if None,
                load from `config/dataset/commercial.yaml` and apply overrides.
        """

        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "commercial.yaml"))
            if overrides:
                cfg = apply_overrides(cfg, overrides)

        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        self.cfg.time_series_columns = ["energy_meter"]
        self.geography = cfg.geography

        self._load_data()
        self.time_series_column_names = cfg.time_series_columns
        self.time_series_dims = len(self.time_series_column_names)

        super().__init__(
            data=self.data,
            time_series_column_names=self.cfg.time_series_columns,
            seq_len=self.cfg.seq_len,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=skip_heavy_processing,
        )
        context_vars = self._get_context_var_dict(self.data)
        self.cfg.context_vars = context_vars

    def _load_data(self):
        """
        Loads in metadata and data for the commercial energy dataset.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        metapath = os.path.join(base_path, "metadata.csv")
        if not os.path.exists(metapath):
            raise FileNotFoundError(f"Metadata file not found at {metapath}")
        metadata = pd.read_csv(metapath, usecols=self.cfg.metadata_columns)

        data_path = os.path.join(base_path, "electricity_cleaned.csv")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        data = pd.read_csv(data_path)

        data = data.melt(
            id_vars="timestamp",          # keep timestamp as is
            var_name="dataid",            # old column names (id1, id2, etc.) become values in this column
            value_name="energy_meter"     # their values go here
        )

        data['site_id'] = data['dataid'].str.split('_').str[0]

        if self.geography:
            if self.geography not in metadata["site_id"].unique():
                raise ValueError(f"Geography {self.geography} not found in metadata")
            data = data[data["site_id"] == self.geography]
            metadata = metadata[metadata["site_id"] == self.geography]

        self.data = data
        self.metadata = metadata


    def _preprocess_data(self, data):
        """
        Creates sequences of seq_len and merges metadata. Removes any sequences with missing data.
        
        Args:
            data (pd.DataFrame): Raw DataFrame including 'energy_meter' values.
        
        Returns:
            pd.DataFrame: Metadata columns, datetime, year, month, weekday, date_day, and array-valued 'energy_meter'.
        """
        data = data.copy()

        data['datetime'] = pd.to_datetime(data['timestamp'])
        data = data.dropna(subset=['energy_meter'])  # any NaN makes the day shorter -> filtered by size later
        data = data.sort_values(by=["dataid", "datetime"])

        # grouped = data.groupby(["dataid", "datetime", "year", "month", "date_day", "weekday"])["energy_meter"].apply(np.array).reset_index()
        grouped = data.groupby(['dataid', pd.Grouper(key='datetime', freq='D')])['energy_meter']
        grouped = grouped.apply(np.asarray).reset_index(name='energy_meter')

        ## Just gonna remove any sequence with any missing values
        grouped = grouped[grouped["energy_meter"].apply(len) == self.cfg.seq_len].reset_index(drop=True)
        # grouped = grouped[grouped["energy_meter"].apply(lambda x: not np.isnan(x).any())]

        grouped['year'] = grouped['datetime'].dt.year
        grouped["month"] = grouped["datetime"].dt.month_name()
        grouped["weekday"] = grouped["datetime"].dt.day_name()
        grouped["date_day"] = grouped["datetime"].dt.day

        if grouped["energy_meter"].apply(lambda x: np.isnan(x).any()).any():
            raise ValueError("NaN values remain in grouped energy_meter sequences after filtering.")

        merged = pd.merge(grouped, self.metadata, how="left", left_on="dataid", right_on="building_id").drop(columns=["building_id"])
        merged.sort_values(by=["dataid", "datetime"], inplace=True)
        # merged = self._handle_missing_data(merged)

        # Drop rows with NaN values in context variables
        context_cols = [col for col in self.cfg.context_vars.keys() if col in merged.columns]
        nan_before = merged.shape[0]
        merged = merged.dropna(subset=context_cols)
        nan_after = merged.shape[0]
        if nan_before != nan_after:
            print(f"Dropped {nan_before - nan_after} rows with NaN values in context variables")

        return merged
    
    def _handle_missing_data(self, merged):
        raise NotImplementedError

        