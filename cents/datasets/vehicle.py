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


class VehicleDataset(TimeSeriesDataset):
    """
    Dataset class for Vehicle time series data.

    Handles loading, preprocessin, including normalization and context variables.

    Attributes:
        cfg (DictConfig): Hydra config for the dataset.
        name (str): Dataset name.
        normalize (bool): Whether to apply normalization.
    """

    def __init__(
        self,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
        force_retrain_normalizer: bool = False,
    ):
        """
        Initialize and preprocess the Vehicle dataset.

        Loads metadata and timeseries CSVs, then applies filtering,
        grouping, user-subsetting, and calls the base class for
        further preprocessing (normalization, merging, rarity flags).

        Args:
            cfg (Optional[DictConfig]): Override Hydra config; if None,
                load from `config/dataset/vehicle.yaml`.
            overrides (Optional[List[str]]): Override Hydra config; if None,
                load from `config/dataset/vehicle.yaml` and apply overrides.

        Raises:
            FileNotFoundError: If required CSV files are missing.
        """
        if cfg is None:
            cfg = load_yaml(os.path.join(ROOT_DIR, "config", "dataset", "vehicle.yaml"))
            if overrides:
                cfg = apply_overrides(cfg, overrides)

        self.cfg = cfg
        self.name = cfg.name
        self.normalize = cfg.normalize
        self.time_series_dims = cfg.time_series_dims
        self.num_ts_steps = cfg.num_ts_steps
        self.seq_len = self.cfg.seq_len

        self._load_data()
        
        ts_cols: List[str] = self.cfg.time_series_columns[: self.time_series_dims]


        super().__init__(
            data=self.data,
            time_series_column_names=ts_cols,
            context_var_column_names=list(self.cfg.context_vars.keys()),
            seq_len=self.cfg.seq_len,
            normalize=self.cfg.normalize,
            scale=self.cfg.scale,
            skip_heavy_processing=cfg.get('skip_heavy_processing', False),
            size=cfg.get('max_samples', None),
            force_retrain_normalizer=force_retrain_normalizer,
        )

    def _load_data(self) -> None:
        """
        Load .

        Populates self.data DataFrame.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
        """
        module_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(module_dir, "..", self.cfg.path))
        self.data = pd.read_csv(os.path.join(path, "vehicle_signal_data.csv"))


    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Convert timestamps, assemble sequences of length seq_len, and merge metadata.

        Args:
            data (pd.DataFrame): Raw concatenated grid (and solar) rows.

        Returns:
            pd.DataFrame: One row per sequence, with array-valued 'grid' and
        '''

        # Assemble sequences of length seq_len, each with a prefix context of length self.num_ts_steps
        # Time remains raw seconds, do not convert to timestamps.
        time_series_cols = self.cfg.time_series_columns[: self.time_series_dims]
        context_var_names = list(self.cfg.context_vars.keys())
        data = data.sort_values("Time").reset_index(drop=True)  # ensure increasing raw seconds

        # Only build full (context+target) window sequences that fit fully within data
        total_window = self.num_ts_steps + self.seq_len
        rolling_idxs = (
            pd.Series(np.arange(len(data)))
            .rolling(window=total_window)
            .apply(lambda x: x[0], raw=True)
            .dropna()
            .index
        )

        # Preallocate arrays for sequences, for efficiency
        out = {col: [] for col in time_series_cols}
        for cvar in context_var_names:
            out[f"{cvar}"] = []
        out["context_time"] = []

        for idx in rolling_idxs:
            window_slice = data.iloc[idx - total_window + 1 : idx + 1]
            context_slice = window_slice.iloc[:self.num_ts_steps]
            target_slice = window_slice.iloc[self.num_ts_steps:]

            # Store target sequences
            for col in time_series_cols:
                out[col].append(target_slice[col].to_numpy())

            # Store context as array(s)
            for cvar in time_series_cols:
                # Context for each variable over the context window
                if cvar in context_slice.columns:
                    out[f"context_{cvar}"].append(context_slice[cvar].to_numpy())
                else:
                    out[f"context_{cvar}"].append([None] *self.num_ts_steps)

            # Optionally, keep the raw "Time" for context window (useful to recover absolute position/relative time)
            out["context_time"].append(context_slice["Time"].to_numpy())

        out_df = pd.DataFrame(out)
        return out_df
        