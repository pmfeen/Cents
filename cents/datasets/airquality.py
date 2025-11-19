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
                overrides: Optional[List[str]] = None):
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
        self.target_time_series_columns = cfg.target_time_series_columns
        self.context_time_series_columns = cfg.context_time_series_columns
        self.geography = cfg.geography
        self.time_series_dims = len(self.target_time_series_columns)

        self.city_names = ["Aotizhongxin", "Changping", "Dingling", "Dongsi", "Guanyuan",
                            "Gucheng", "Huairou", "Nongzhanguan", "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"]

        self._load_data()

        super().__init__(
            data=self.data,
            time_series_column_names=self.time_series_columns,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=cfg.get('skip_heavy_processing', False),
            size=cfg.get('max_samples', None)
        )

    def _load_data(self):
        """
        Loads in metadata and data for commercial energy dataset.
        Categorial CVs: Year, Month, Day, Location
        Context Time Series: Temperature, Pressure, Dewpoint, Precipitation, Wind
        """

        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))

        meta_path = os.path.join(path, "metadata.csv")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")


        if not self.geography: self.geography = self.city_names

        self.geography = [self.geography] if isinstance(self.geography, str) else self.geography
        dfs = []
        for name in self.geography:
            fname = f"PRSA_Data_{name}_20130301-20170228.csv"
            data_path = os.path.join(path, fname)

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            
            dfs.append(pd.read_csv(data_path))
        
        self.data = pd.concat(dfs, axis=0)[self.cfg.data_columns]


    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data[["year", "month", "day", "hour"]])
        data['weekday'] = data['timestamp'].dt.day_name()

        ts_cols = self.context_time_series_columns + self.target_time_series_columns

        data = data.sort_values(['location', 'year', 'month', 'day', 'hour'])


        grouped = (
            data.groupby(["station", "year", "month", "day"], as_index=False)
            .agg({**{c: list for c in ts_cols},
            "weekday": 'first'})
        )

        grouped = grouped[grouped["PM2.5"].apply(len) == self.cfg.seq_len].reset_index(
            drop=True
        )

        grouped = self._handle_missing_data(grouped)

        return grouped


    def _hande_missing_data(self, data):
        mask = data[self.context_time_series_columns].applymap(is_all_nan).any(axis=1)
        data = data[~mask]

        for col in self.context_time_series_columns:
            data[col] = data[col].apply(fill_with_row_mean)

        return data



def is_all_nan(lst):
    arr = np.array(lst, dtype=float)
    return np.isnan(arr).all()


def fill_with_row_mean(lst):
    s = pd.Series(lst, dtype=float)
    m = s.mean(skipna=True)
    return s.fillna(m).tolist()
