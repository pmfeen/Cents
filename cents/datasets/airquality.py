from ast import Str
import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from cents.utils.config_loader import load_yaml, apply_overrides

from cents.datasets.timeseries_dataset import TimeSeriesDataset

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
# These are warnings for an error that is accounted for in the code
warnings.filterwarnings("ignore", category=RuntimeWarning) 
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AirQualityDataset(TimeSeriesDataset):
    def __init__(self, cfg: DictConfig = None,
                 overrides: Optional[List[str]] = None,
                 force_retrain_normalizer: bool = False,
                 run_dir: Optional[str] = None):
        """
        Initializes the AirQuality Dataset. Available at:
        https://doi.org/10.24432/C5RK5G.
        Hourly Air Quality at Multiple Sites in China, in many measures.
        Accompanying location and weather information.
        """

        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "airquality.yaml"))
        
        if overrides:
            cfg = apply_overrides(cfg, overrides)
        
        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        if isinstance(cfg.time_series_columns, str):
            cfg.time_series_columns = [cfg.time_series_columns]
        self.target_time_series_columns = cfg.time_series_columns
        self.geography = cfg.geography
        self.time_series_dims = cfg.time_series_dims

        self.city_names = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
                            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]
        self.context_time_series_columns = {k:v[1] for k,v in self.cfg.context_vars.items() if v[0] == "time_series"}
        self.context_series_names = list(self.context_time_series_columns.keys())
        
        self.categorical_time_series = {
            k: v[1] for k, v in self.cfg.context_vars.items() 
            if v[0] == "time_series" and v[1] is not None
        }
        self._load_data()

        super().__init__(
            data=self.data,
            time_series_column_names=self.target_time_series_columns,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=cfg.get('skip_heavy_processing', False),
            size=cfg.get('max_samples', None),
            categorical_time_series=self.categorical_time_series,
            force_retrain_normalizer=force_retrain_normalizer,
            run_dir=run_dir,
        )

    def _load_data(self):
        """
        Loads in metadata and data for commercial energy dataset.
        Categorial CVs: Year, Month, Day, Location
        Context Time Series: Temperature, Pressure, Dewpoint, Precipitation, Wind
        """

        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        if not self.geography: self.geography = self.city_names

        self.geography = [self.geography] if isinstance(self.geography, str) else self.geography
        dfs = []
        for name in self.geography:
            fname = f"PRSA_Data_{name}_20130301-20170228.csv"
            data_path = os.path.join(path, fname)

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            dfs.append(pd.read_csv(data_path)[self.cfg.data_columns])
        
        self.data = pd.concat(dfs, axis=0)

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data[["year", "month", "day", "hour"]])
        data['weekday'] = data['timestamp'].dt.day_name()

        ts_cols = self.context_series_names + self.target_time_series_columns

        data = data.sort_values(['station', 'year', 'month', 'day', 'hour'])

        # Map month integer to month name string as quickly as possible
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        data['month'] = data['month'].map(lambda x: months[x-1])

        group_keys = ["station", "year", "month", "day", "weekday"]

        grouped = (
            data.groupby(group_keys, as_index=False, sort=False)
                .agg({c: list for c in ts_cols})
        )

        # Convert lists -> numpy arrays (fast + deterministic)
        for c in ts_cols:
            grouped[c] = grouped[c].map(np.asarray)

        grouped = grouped[grouped["PM2.5"].apply(len) == self.cfg.seq_len].reset_index(
            drop=True
        )

        grouped = self._handle_missing_data(grouped)
    
        # Convert all lists in time series columns into tuples to make them hashable
        for c in ts_cols:
            grouped[c] = grouped[c].map(tuple)

        return grouped


    def _handle_missing_data(self, data):
        # Only handle missing data for numeric time series
        numeric_series = [c for c in self.context_series_names if c not in self.categorical_time_series]
        
        mask = data[numeric_series].applymap(is_all_nan).any(axis=1) if numeric_series else pd.Series([False] * len(data))
        data = data[~mask]

        for col in numeric_series:
            data[col] = data[col].apply(fill_with_row_mean)

        data[list(self.categorical_time_series.keys())]

        mask = data[list(self.categorical_time_series.keys())].applymap(is_any_nan).any(axis=1)
        data = data[~mask]

        data = data.loc[data["PM2.5"].apply(lambda x: not np.isnan(x).any())]

        return data



def is_all_nan(arr):
    return pd.isna(arr).all()

def is_any_nan(arr):
    return pd.isna(arr).any()

def fill_with_row_mean(lst):
    s = pd.Series(lst, dtype=float)
    m = s.mean(skipna=True)
    return s.fillna(m).tolist()
