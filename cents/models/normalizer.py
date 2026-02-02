from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from omegaconf import ListConfig


from cents.datasets.utils import split_timeseries
from cents.models.base import NormalizerModel
from cents.models.context import MLPContextModule, SepMLPContextModule, DynamicContextModule_CNN, DynamicContextModule_Transformer # Import to trigger registration
from cents.models.context_registry import get_context_module_cls
from cents.models.stats_head_registry import register_stats_head, get_stats_head_cls
from cents.models.registry import register_model
from cents.utils.utils import get_context_config


@register_stats_head("default", "mlp")
class MLPStatsHead(nn.Module):
    """
    Head module predicting summary statistics (mean, std, and optionally min/max z-scores) from context embedding.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        time_series_dims: int,
        do_scale: bool,
        n_layers: int = 3,
    ):
        """
        Initializes the statistics head network.

        Args:
            embedding_dim: Dimensionality of the input context embedding.
            hidden_dim: Number of units in each hidden layer.
            time_series_dims: Number of dimensions in the original time series.
            do_scale: Whether to predict scaling min/max parameters.
            n_layers: Number of hidden linear layers before the output.
        """
        super().__init__()
        self.time_series_dims = time_series_dims
        self.do_scale = do_scale
        out_dim = 4 * time_series_dims if do_scale else 2 * time_series_dims
        layers = []
        in_dim = embedding_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize the output layer properly
        # For log_sigma, initialize to small negative values so exp(log_sigma) starts around 1
        # This helps with training stability
        self._initialize_output_layer()
    
    def _initialize_output_layer(self):
        """Initialize the output layer to reasonable starting values."""
        # Get the last linear layer
        output_layer = self.net[-1]
        with torch.no_grad():
            # Initialize all weights with small values
            # nn.init.xavier_uniform_(output_layer.weight, gain=1.0)
            
            # Initialize all biases to zero first
            # nn.init.zeros_(output_layer.bias)
            
            # For log_sigma outputs (indices 1, 3, 5, ...), initialize bias to small negative
            # This makes exp(log_sigma) start around 0.1-1.0
            if self.do_scale:
                # Pattern: mu, log_sigma, z_min, z_max for each dimension
                for dim_idx in range(self.time_series_dims):
                    # log_sigma is at index 1 + 4*dim_idx
                    log_sigma_idx = 1 + 4 * dim_idx
                    # Initialize to 3.0: exp(3.0) ≈ 20, closer to typical sigma ~27
                    output_layer.bias[log_sigma_idx].fill_(3.0)
            else:
                # Pattern: mu, log_sigma for each dimension
                for dim_idx in range(self.time_series_dims):
                    # log_sigma is at index 1 + 2*dim_idx
                    log_sigma_idx = 1 + 2 * dim_idx
                    # Initialize to 3.0: exp(3.0) ≈ 20, closer to typical sigma ~27
                    output_layer.bias[log_sigma_idx].fill_(3.0)

    def forward(self, z: torch.Tensor):
        """
        Forward pass to compute predicted statistics.

        Args:
            z: Context embedding tensor of shape (batch_size, embedding_dim).

        Returns:
            pred_mu: Predicted means, shape (batch_size, time_series_dims).
            pred_sigma: Predicted standard deviations, shape (batch_size, time_series_dims).
            pred_z_min: Predicted min z-scores, or None if do_scale=False.
            pred_z_max: Predicted max z-scores, or None if do_scale=False.
            pred_log_sigma_unclamped: Unclamped log_sigma for loss computation.
        """
        out = self.net(z)
        batch_size = out.size(0)
        if self.do_scale:
            out = out.view(batch_size, 4, self.time_series_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            pred_z_min = out[:, 2, :]
            pred_z_max = out[:, 3, :]
        else:
            out = out.view(batch_size, 2, self.time_series_dims)
            pred_mu = out[:, 0, :]
            pred_log_sigma = out[:, 1, :]
            pred_z_min = None
            pred_z_max = None
        
        # Store unclamped version for loss computation BEFORE clamping
        # This must be done before any operations that might break the computation graph
        pred_log_sigma_unclamped = pred_log_sigma
        
        # Clamp log_sigma to prevent exp() from producing infinity
        # exp(88) ≈ 1.6e38 (near float32 max), so clamp to reasonable range
        pred_log_sigma_clamped = torch.clamp(pred_log_sigma, min=-10.0, max=10.0)
        pred_sigma = torch.exp(pred_log_sigma_clamped)
        return pred_mu, pred_sigma, pred_z_min, pred_z_max, pred_log_sigma_unclamped


class _NormalizerModule(nn.Module):
    """
    Wrapper module combining a context embedding and stats head for normalization.
    """

    def __init__(
        self,
        static_cond_module: nn.Module = None,
        dynamic_cond_module: nn.Module = None,
        hidden_dim: int = 512,
        time_series_dims: int = 2,
        do_scale: bool = True,
        stats_head_type: str = "mlp",
    ):
        """
        Args:
            static_cond_module: ContextModule instance for static context variables (categorical + continuous).
            dynamic_cond_module: ContextModule instance for dynamic context variables (time_series).
            hidden_dim: Hidden dimension size for the stats head.
            time_series_dims: Number of time series dimensions.
            do_scale: Whether to include scaling predictions.
            stats_head_type: Type of stats head to use (from registry).
        """
        super().__init__()
        self.static_cond_module = static_cond_module
        self.dynamic_cond_module = dynamic_cond_module
        
        # Determine embedding dimension from available modules
        if static_cond_module is not None:
            self.embedding_dim = static_cond_module.embedding_dim
        elif dynamic_cond_module is not None:
            self.embedding_dim = dynamic_cond_module.embedding_dim
        else:
            raise ValueError("At least one of static_cond_module or dynamic_cond_module must be provided")
        
        # If both modules exist, combine their embeddings
        if static_cond_module is not None and dynamic_cond_module is not None:
            # Combine embeddings from both modules
            combined_dim = static_cond_module.embedding_dim + dynamic_cond_module.embedding_dim
            self.combine_mlp = nn.Sequential(
                nn.Linear(combined_dim, self.embedding_dim),
                nn.ReLU(),
            )
        else:
            self.combine_mlp = None
        
        # Use registry to get the stats head class
        StatsHeadCls = get_stats_head_cls(stats_head_type)
        self.stats_head = StatsHeadCls(
            embedding_dim=self.embedding_dim,
            hidden_dim=hidden_dim,
            time_series_dims=time_series_dims,
            do_scale=do_scale,
        )

    def forward(self, context_vars_dict: dict):
        """
        Compute normalization parameters from categorical context.

        Args:
            context_vars_dict: Mapping of context variable names to label tensors.
                             Static vars: single values (categorical: long, continuous: float)
                             Dynamic vars: time series sequences (batch, seq_len)

        Returns:
            Tuple of (pred_mu, pred_sigma, pred_z_min, pred_z_max, pred_log_sigma_unclamped).
        """
        embeddings = []
        
        # Process static context variables
        if self.static_cond_module is not None:
            # Filter static context variables
            static_vars = {
                k: v for k, v in context_vars_dict.items()
                if k not in getattr(self, '_dynamic_var_names', [])
            }
            if static_vars:
                device = next(self.static_cond_module.parameters()).device
                static_vars = {
                    k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in static_vars.items()
                }
                static_embedding, _ = self.static_cond_module(static_vars)
                embeddings.append(static_embedding)
        
        # Process dynamic context variables
        if self.dynamic_cond_module is not None:
            # Filter dynamic context variables
            dynamic_var_names = getattr(self, '_dynamic_var_names', [])
            dynamic_vars = {
                k: v for k, v in context_vars_dict.items()
                if k in dynamic_var_names
            }
            if dynamic_vars:
                device = next(self.dynamic_cond_module.parameters()).device
                dynamic_vars = {
                    k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in dynamic_vars.items()
                }
                dynamic_embedding, _ = self.dynamic_cond_module(dynamic_vars)
                # Check for NaN in dynamic embedding
                if torch.isnan(dynamic_embedding).any() or torch.isinf(dynamic_embedding).any():
                    raise ValueError(
                        f"NaN/Inf detected in dynamic embedding. "
                        f"Dynamic vars: {list(dynamic_vars.keys())}"
                    )
                embeddings.append(dynamic_embedding)
        
        # Combine embeddings if both exist
        if len(embeddings) == 2:
            combined = torch.cat(embeddings, dim=1)
            embedding = self.combine_mlp(combined)
        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            raise ValueError("No context variables provided")
        
        return self.stats_head(embedding)


@register_model("normalizer")
class Normalizer(NormalizerModel):
    """
    Learns group-wise normalization parameters (mean, std, optional min/max) for time series by context.
    """

    def __init__(
        self,
        dataset_cfg,
        normalizer_training_cfg,
        dataset,
    ):
        """
        Initializes the Normalizer training module.

        Args:
            dataset_cfg: OmegaConf dataset config (provides context_vars, columns).
            normalizer_training_cfg: Config for normalizer training (lr, batch_size).
            dataset: Instance of TimeSeriesDataset containing data DataFrame.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])

        self.dataset_cfg = dataset_cfg
        self.normalizer_training_cfg = normalizer_training_cfg
        self.dataset = dataset

        # Get continuous variables from config if specified
        self.continuous_vars = [k for k, v in self.dataset_cfg.context_vars.items() if v[0] == "continuous"]
        self.categorical_vars = [k for k, v in self.dataset_cfg.context_vars.items() if v[0] == "categorical"]
        dynamic_vars = [k for k, v in self.dataset_cfg.context_vars.items() if v[0] == "time_series"]
        
        self.static_context_vars = self.categorical_vars + self.continuous_vars
        self.dynamic_context_vars = dynamic_vars
        self.context_vars = self.static_context_vars + self.dynamic_context_vars
        
        self.time_series_cols = dataset_cfg.time_series_columns[
            : dataset_cfg.time_series_dims
        ]
        self.time_series_dims = dataset_cfg.time_series_dims
        self.do_scale = dataset_cfg.scale
        self.seq_len = dataset_cfg.seq_len
        self.num_ts_steps = getattr(dataset_cfg, "num_ts_steps", None)  # For dynamic context length

        # Get context config
        # context_cfg = get_context_config()

        self.static_module_type = self.dataset.static_module_type
        self.dynamic_module_type = self.dataset.dynamic_module_type
        self.stats_head_type = self.dataset.stats_head_type
        
        # Get loss type from config (default to "mse")
        self.loss_type = getattr(self.normalizer_training_cfg, "loss_type", "mse")
        
        # Create static context module (for categorical + continuous)
        static_context_module = None
        if self.static_context_vars:
            StaticContextModuleCls = get_context_module_cls(self.static_module_type)
            # Filter context_vars to only static ones
            self.static_context_vars_dict = {
                k: v for k, v in self.dataset_cfg.context_vars.items() 
                if k in self.static_context_vars
            }
            static_context_module = StaticContextModuleCls(
                self.static_context_vars_dict,
                256,
            )

        # Create dynamic context module (for time_series)
        dynamic_context_module = None
        if self.dynamic_context_vars and self.dynamic_module_type is not None:
            DynamicContextModuleCls = get_context_module_cls("dynamic", self.dynamic_module_type)
            # Filter context_vars to only dynamic ones
            dynamic_context_vars_dict = {
                k: v for k, v in self.dataset_cfg.context_vars.items() 
                if k in self.dynamic_context_vars
            }
            # Use num_ts_steps for dynamic context length if available, otherwise seq_len
            dynamic_seq_len = self.num_ts_steps if self.num_ts_steps is not None else self.seq_len
            dynamic_context_module = DynamicContextModuleCls(
                dynamic_context_vars_dict,
                256,
                seq_len=dynamic_seq_len,
            )
        
        self.normalizer_model = _NormalizerModule(
            static_cond_module=static_context_module,
            dynamic_cond_module=dynamic_context_module,
            hidden_dim=512,
            time_series_dims=self.time_series_dims,
            do_scale=self.do_scale,
            stats_head_type=self.stats_head_type,
        )
        # Store dynamic var names for filtering in forward
        self.normalizer_model._dynamic_var_names = self.dynamic_context_vars
        # For backward compatibility, expose the static context module
        self.context_module = self.normalizer_model.static_cond_module
        # Expose the dynamic context module at top level so it shows in model summary
        self.dynamic_cond_module = self.normalizer_model.dynamic_cond_module

        # Will be populated in setup()
        self.sample_stats = []
        self._verify_parameters()

    def _verify_parameters(self):
        """
        Verify that all parameters including context module are registered.
        This helps debug parameter counting issues.
        """
        all_param_names = [name for name, _ in self.named_parameters()]
        context_param_names = [name for name in all_param_names if 'cond_module' in name or 'context_module' in name]
        stats_head_param_names = [name for name in all_param_names if 'stats_head' in name]
        
        if not context_param_names:
            raise RuntimeError(
                "Context module parameters not found! "
                "Expected parameters with 'cond_module' in name. "
                f"Found parameter names: {all_param_names[:10]}..."
            )
        
        print(f"[Normalizer] Found {len(context_param_names)} context module parameters")
        print(f"[Normalizer] Found {len(stats_head_param_names)} stats head parameters")
        print(f"[Normalizer] Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def setup(self, stage: Optional[str] = None):
        """
        Lightning hook: prepare training data before training.
        """
        # Compute per-sample statistics - no grouping needed
        self.sample_stats = self._compute_per_sample_stats()
        
        # Log initial predictions to check if model is in the right ballpark
        if stage == "fit" or stage is None:
            self._log_initial_predictions()
    
    def _log_initial_predictions(self):
        """Log initial model predictions to diagnose initialization issues."""
        self.eval()
        with torch.no_grad():
            # Get a sample batch
            dataloader = self.train_dataloader()
            batch = next(iter(dataloader))
            cat_vars_dict, mu_t, sigma_t, zmin_t, zmax_t = batch
            
            # Move to device
            device = next(self.parameters()).device
            cat_vars_dict = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in cat_vars_dict.items()
            }
            mu_t = mu_t.to(device)
            sigma_t = sigma_t.to(device)
            
            pred_mu, pred_sigma, pred_z_min, pred_z_max, _ = self(cat_vars_dict)
            
            print(f"\n[Initial Predictions]")
            print(f"  Target mu: mean={mu_t.mean().item():.4f}, std={mu_t.std().item():.4f}, range=[{mu_t.min().item():.4f}, {mu_t.max().item():.4f}]")
            print(f"  Predicted mu: mean={pred_mu.mean().item():.4f}, std={pred_mu.std().item():.4f}, range=[{pred_mu.min().item():.4f}, {pred_mu.max().item():.4f}]")
            print(f"  Target sigma: mean={sigma_t.mean().item():.4f}, std={sigma_t.std().item():.4f}, range=[{sigma_t.min().item():.4f}, {sigma_t.max().item():.4f}]")
            print(f"  Predicted sigma: mean={pred_sigma.mean().item():.4f}, std={pred_sigma.std().item():.4f}, range=[{pred_sigma.min().item():.4f}, {pred_sigma.max().item():.4f}]")
            print(f"  Initial loss_mu: {F.mse_loss(pred_mu, mu_t).item():.6f}")
            print(f"  Initial loss_sigma: {F.mse_loss(pred_sigma, sigma_t).item():.6f}")
            print()
        
        self.train()

    def forward(self, cat_vars_dict: dict):
        """
        Predict normalization parameters for a batch of categorical contexts.

        Args:
            cat_vars_dict: Mapping of context variable names to label tensors.

        Returns:
            Tuple of (pred_mu, pred_sigma, pred_z_min, pred_z_max, pred_log_sigma_unclamped).
        """
        return self.normalizer_model(cat_vars_dict)

    def _compute_loss_mse(self, pred_mu, pred_sigma, pred_log_sigma_unclamped, mu_t, sigma_t):
        """
        Compute MSE loss for mu and sigma.
        
        Args:
            pred_mu: Predicted means
            pred_sigma: Predicted standard deviations
            pred_log_sigma_unclamped: Unclamped log sigma predictions
            mu_t: Target means
            sigma_t: Target standard deviations
            
        Returns:
            loss_mu, loss_sigma
        """
        # Use standard MSE loss for mu
        loss_mu = F.mse_loss(pred_mu, mu_t)
        
        # Use log-space loss for sigma - this is more numerically stable
        # and handles scale differences better
        target_log_sigma = torch.log(sigma_t + 1e-8)  # Add small epsilon to avoid log(0)
        loss_sigma = F.mse_loss(pred_log_sigma_unclamped, target_log_sigma)
        
        return loss_mu, loss_sigma
    
    def _compute_loss_gaussian_nll(self, pred_mu, pred_sigma, mu_t, sigma_t):
        """
        Compute Gaussian Negative Log Likelihood loss.
        
        Treats target mu_t as observations from N(pred_mu, pred_sigma^2).
        For sigma, still uses log-space MSE since it's a scale parameter.
        
        Args:
            pred_mu: Predicted means
            pred_sigma: Predicted standard deviations
            mu_t: Target means (treated as observations)
            sigma_t: Target standard deviations
            
        Returns:
            loss_mu, loss_sigma
        """
        # Use Gaussian NLL for mu: treat mu_t as observations from N(pred_mu, pred_sigma^2)
        # GaussianNLLLoss expects: input (mean), target (observations), var (variance)
        # We need to ensure variance is positive and not too small
        pred_var = torch.clamp(pred_sigma ** 2, min=1e-6)
        gaussian_nll = nn.GaussianNLLLoss(reduction='mean')
        loss_mu = gaussian_nll(pred_mu, mu_t, pred_var)
        
        # For sigma, still use log-space MSE (sigma is a scale parameter, not a location)
        pred_log_sigma = torch.log(pred_sigma + 1e-8)
        target_log_sigma = torch.log(sigma_t + 1e-8)
        loss_sigma = F.mse_loss(pred_log_sigma, target_log_sigma)
        
        return loss_mu, loss_sigma

    def training_step(self, batch, batch_idx: int):
        """
        Training step: regress predicted stats against true group stats.

        Args:
            batch: Tuple of (cat_vars_dict, mu, sigma, zmin, zmax).
            batch_idx: Batch index.

        Returns:
            loss tensor.
        """
        context_vars_dict, mu_t, sigma_t, zmin_t, zmax_t = batch
        pred_mu, pred_sigma, pred_z_min, pred_z_max, pred_log_sigma_unclamped = self(context_vars_dict)

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss_mu, loss_sigma = self._compute_loss_mse(
                pred_mu, pred_sigma, pred_log_sigma_unclamped, mu_t, sigma_t
            )
        elif self.loss_type == "gaussian_nll":
            loss_mu, loss_sigma = self._compute_loss_gaussian_nll(
                pred_mu, pred_sigma, mu_t, sigma_t
            )
        else:
            raise ValueError(
                f"Unknown loss_type: {self.loss_type}. "
                f"Supported types: 'mse', 'gaussian_nll'"
            )
        
        total_loss = loss_mu + loss_sigma

        if self.do_scale:
            if torch.isnan(pred_z_min).any() or torch.isnan(pred_z_max).any():
                raise ValueError(
                    f"NaN detected in scale predictions at batch {batch_idx}"
                )
            loss_zmin = F.mse_loss(pred_z_min, zmin_t)
            loss_zmax = F.mse_loss(pred_z_max, zmax_t)
            total_loss += loss_zmin + loss_zmax
        else:
            loss_zmin = torch.tensor(0.0, device=total_loss.device)
            loss_zmax = torch.tensor(0.0, device=total_loss.device)
        
        # Check for NaN in loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(
                f"NaN/Inf loss detected at batch {batch_idx}. "
                f"loss_mu: {loss_mu.item():.6f}, loss_sigma: {loss_sigma.item():.6f}"
            )
        
        # Log individual components to understand what's happening
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("loss_mu", loss_mu, on_step=True, on_epoch=True, prog_bar=False)
        self.log("loss_sigma", loss_sigma, on_step=True, on_epoch=True, prog_bar=False)
        if self.do_scale:
            self.log("loss_zmin", loss_zmin, on_step=True, on_epoch=True, prog_bar=False)
            self.log("loss_zmax", loss_zmax, on_step=True, on_epoch=True, prog_bar=False)
        
        # Log prediction statistics to monitor if model is learning
        if batch_idx % 100 == 0:  # Log every 100 batches to avoid spam
            with torch.no_grad():
                self.log("pred_mu_mean", pred_mu.mean(), on_step=True, on_epoch=False)
                self.log("pred_mu_std", pred_mu.std(), on_step=True, on_epoch=False)
                self.log("pred_sigma_mean", pred_sigma.mean(), on_step=True, on_epoch=False)
                self.log("pred_sigma_std", pred_sigma.std(), on_step=True, on_epoch=False)
                self.log("target_mu_mean", mu_t.mean(), on_step=True, on_epoch=False)
                self.log("target_sigma_mean", sigma_t.mean(), on_step=True, on_epoch=False)
        
        return total_loss

    def configure_optimizers(self):
        """
        Configure optimizer for normalizer training.

        Returns:
            Adam optimizer instance.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.normalizer_training_cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
        
        return optimizer
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Monitor gradients after each training step to diagnose training issues.
        """
        if batch_idx % 100 == 0:  # Check every 100 batches
            total_norm = 0.0
            param_count = 0
            zero_grad_count = 0
            
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    if param_norm.item() < 1e-8:
                        zero_grad_count += 1
                        print(name, "HAS ZERO GRAD")
                else:
                    # Parameter has no gradient - this might indicate a problem
                    if 'cond_module' in name or 'stats_head' in name:
                        # Only warn about important parameters
                        pass
            
            total_norm = total_norm ** (1. / 2)
            
            self.log("grad_norm", total_norm, on_step=True, on_epoch=False)
            self.log("params_with_grad", param_count, on_step=True, on_epoch=False)
            
            if total_norm < 1e-6:
                print(f"[Warning] Very small gradient norm at batch {batch_idx}: {total_norm:.2e}")
            if zero_grad_count > 0:
                print(f"[Warning] {zero_grad_count} parameters have near-zero gradients at batch {batch_idx}")

    def train_dataloader(self):
        """
        Returns a DataLoader over per-group statistics samples.
        """
        ds = self._create_training_dataset()
        return DataLoader(
            ds,
            batch_size=self.normalizer_training_cfg.batch_size,
            shuffle=True,
            num_workers=4,  # Use fewer workers to reduce overhead
            persistent_workers=False,  # Disable to avoid multiprocessing cleanup issues
            pin_memory=torch.cuda.is_available(),  # Helps with GPU transfer
            prefetch_factor=2,  # Reduce prefetch to avoid memory issues
        )

    def _compute_per_sample_stats(self) -> list:
        """
        Compute statistics for each individual sample.
        This allows the model to learn context → normalization_params for all context types
        (categorical, continuous, and dynamic) without requiring grouping.

        Returns:
            List of tuples: (context_vars_dict, mu_array, std_array, zmin_array, zmax_array)
        """
        df = self.dataset.data.copy()
        sample_stats = []
        continuous_vars = getattr(self.dataset_cfg, "continuous_context_vars", None) or []
        
        for idx, row in df.iterrows():
            context_vars_dict = {}
            
            # Process static context variables (categorical + continuous)
            for var_name in self.static_context_vars:
                if var_name in row:
                    if var_name in continuous_vars:
                        context_vars_dict[var_name] = torch.tensor(row[var_name], dtype=torch.float32)
                    else:
                        context_vars_dict[var_name] = torch.tensor(row[var_name], dtype=torch.long)
            
            # Process dynamic context variables (time series)
            dynamic_ctx_dict = {}
            for var_name in self.dynamic_context_vars:
                # Check for both the original name and context_ prefix (vehicle dataset uses context_ prefix)
                ts_data = None
                if var_name in row:
                    ts_data = row[var_name]
                
                if ts_data is not None:
                    if isinstance(ts_data, np.ndarray):
                        dynamic_ctx_dict[var_name] = ts_data
                    elif isinstance(ts_data, list):
                        dynamic_ctx_dict[var_name] = np.array(ts_data)
                    else:
                        # If it's a scalar, repeat it to match the appropriate length
                        # Use num_ts_steps if available (for dynamic context), otherwise seq_len
                        context_len = self.num_ts_steps if self.num_ts_steps is not None else self.seq_len
                        dynamic_ctx_dict[var_name] = np.full(context_len, ts_data)
            
            # Compute statistics from this sample's target time series
            dimension_points = []
            for d, col_name in enumerate(self.time_series_cols):
                arr = np.array(row[col_name], dtype=np.float32).flatten()
                dimension_points.append(arr)
            
            mu_array = np.array(
                [pts.mean() for pts in dimension_points], dtype=np.float32
            )
            std_array = np.array(
                [pts.std() + 1e-8 for pts in dimension_points], dtype=np.float32
            )

            if self.do_scale:
                z_min_array = np.array(
                    [
                        (pts - mu).min() / std
                        for pts, mu, std in zip(dimension_points, mu_array, std_array)
                    ],
                    dtype=np.float32,
                )
                z_max_array = np.array(
                    [
                        (pts - mu).max() / std
                        for pts, mu, std in zip(dimension_points, mu_array, std_array)
                    ],
                    dtype=np.float32,
                )
            else:
                z_min_array = z_max_array = None
            
            sample_stats.append((
                context_vars_dict,
                dynamic_ctx_dict,
                mu_array,
                std_array,
                z_min_array,
                z_max_array,
            ))
        
        return sample_stats

    def _create_training_dataset(self) -> Dataset:
        """
        Build an internal Dataset yielding per-sample statistics.

        Returns:
            PyTorch Dataset of samples (context_vars_dict, mu, sigma, zmin, zmax).
        """
        class _TrainSet(Dataset):
            """
            Adapter Dataset to wrap per-sample statistics for DataLoader.
            """

            def __init__(self, samples, dynamic_context_vars, do_scale, dataset_cfg):
                self.samples = samples
                self.dynamic_context_vars = dynamic_context_vars
                self.do_scale = do_scale
                self.dataset_cfg = dataset_cfg

            def __len__(self) -> int:
                return len(self.samples)

            def __getitem__(self, idx: int):
                """
                Returns one training sample.

                Args:
                    idx: Index of the sample.

                Returns:
                    context_vars_dict: Tensor dict of context labels (static + dynamic).
                    mu_t: True mean tensor.
                    sigma_t: True std tensor.
                    zmin_t: True min z-score tensor or None.
                    zmax_t: True max z-score tensor or None.
                """
                context_vars_dict, dynamic_ctx_dict, mu_arr, sigma_arr, zmin_arr, zmax_arr = self.samples[idx]
                
                # Process dynamic context variables (time series) - convert to tensors
                for var_name in self.dynamic_context_vars:
                    if var_name in dynamic_ctx_dict:
                        ts_data = dynamic_ctx_dict[var_name]
                        # Convert to tensor
                        if isinstance(ts_data, np.ndarray):
                            # Check if it's categorical (integer) or numeric (float)
                            var_info = self.dataset_cfg.context_vars.get(var_name, None)
                            if var_info and var_info[1] is not None:
                                # Categorical time series
                                context_vars_dict[var_name] = torch.from_numpy(ts_data).long()
                            else:
                                # Numeric time series
                                context_vars_dict[var_name] = torch.from_numpy(ts_data).float()
                        else:
                            # Fallback: convert to array first
                            ts_array = np.array(ts_data)
                            var_info = self.dataset_cfg.context_vars.get(var_name, None)
                            if var_info and var_info[1] is not None:
                                context_vars_dict[var_name] = torch.from_numpy(ts_array).long()
                            else:
                                context_vars_dict[var_name] = torch.from_numpy(ts_array).float()
                
                mu_t = torch.from_numpy(mu_arr).float()
                sigma_t = torch.from_numpy(sigma_arr).float()
                zmin_t = torch.from_numpy(zmin_arr).float() if self.do_scale else None
                zmax_t = torch.from_numpy(zmax_arr).float() if self.do_scale else None
                return context_vars_dict, mu_t, sigma_t, zmin_t, zmax_t

        return _TrainSet(self.sample_stats, self.dynamic_context_vars, self.do_scale, self.dataset_cfg)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a DataFrame of time series using learned parameters.

        Pads or splits if needed, then applies z-score and min-max scaling.

        Args:
            df: Input DataFrame with raw time series columns.

        Returns:
            DataFrame with normalized series in same columns.
        """
        missing = [c for c in self.time_series_cols if c not in df.columns]

        if missing:
            df = split_timeseries(df, self.time_series_cols)
            missing = [c for c in self.time_series_cols if c not in df.columns]

        assert not missing, (
            "Normalizer.transform expects data in split format with columns "
            f"{self.time_series_cols}."
        )
        df_out = df.copy()
        self.eval()
        continuous_vars = getattr(self.dataset_cfg, "continuous_context_vars", None) or []
        
        # Get categorical time series from dataset if available
        categorical_ts = getattr(self.dataset, 'categorical_time_series', {})
        
        with torch.no_grad():
            for i, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Normalizing"):
                ctx = {}
                for v in self.context_vars:
                    if v in continuous_vars:
                        ctx[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                    elif v in self.dynamic_context_vars:
                        # Dynamic (time series) variable
                        if v in categorical_ts:
                            # Categorical time series - keep as long
                            ctx[v] = torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                        else:
                            # Numeric time series - convert to float32
                            ctx[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                    else:
                        # Static categorical variable
                        ctx[v] = torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                mu, sigma, zmin, zmax, _ = self(ctx)
                mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()

                for d, col in enumerate(self.time_series_cols):
                    arr = np.asarray(row[col], dtype=np.float32)
                    
                    # Skip normalization for categorical time series
                    if col in categorical_ts:
                        # Keep as integers, just ensure proper dtype
                        df_out.at[i, col] = arr.astype(np.int32)
                        continue
                    
                    # Normalize numeric time series
                    z = (arr - mu[d]) / (sigma[d] + 1e-8)
                    if self.do_scale:
                        zmin_, zmax_ = zmin[0, d].item(), zmax[0, d].item()
                        rng = (zmax_ - zmin_) + 1e-8
                        z = (z - zmin_) / rng
                    df_out.at[i, col] = z
        return df_out

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Denormalize a DataFrame of z-scored series back to original scale.

        Args:
            df: DataFrame with normalized series columns.

        Returns:
            DataFrame with denormalized series.
        """
        missing = [c for c in self.time_series_cols if c not in df.columns]

        if missing:
            df = split_timeseries(df, self.time_series_cols)
            missing = [c for c in self.time_series_cols if c not in df.columns]

        assert not missing, (
            "Normalizer.inverse_transform expects split format with columns "
            f"{self.time_series_cols}."
        )

        df_out = df.copy()
        self.eval()
        continuous_vars = getattr(self.dataset_cfg, "continuous_context_vars", None) or []
        # Get categorical time series from dataset if available
        categorical_ts = getattr(self.dataset, 'categorical_time_series', {})
        with torch.no_grad():
            for i, row in tqdm(df_out.iterrows(), total=len(df_out), desc="Inverse normalizing"):
                ctx = {}
                for v in self.context_vars:
                    if v in continuous_vars:
                        # Static continuous variable
                        ctx[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                    elif v in self.dynamic_context_vars:
                        # Dynamic (time series) variable
                        if v in categorical_ts:
                            # Categorical time series - keep as long
                            ctx[v] = torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                        else:
                            # Numeric time series - convert to float32
                            ctx[v] = torch.tensor(row[v], dtype=torch.float32).unsqueeze(0)
                    else:
                        ctx[v] = torch.tensor(row[v], dtype=torch.long).unsqueeze(0)
                mu, sigma, zmin, zmax, _ = self(ctx)
                mu, sigma = mu[0].cpu().numpy(), sigma[0].cpu().numpy()

                for d, col in enumerate(self.time_series_cols):
                    z = np.asarray(row[col], dtype=np.float32)
                    if self.do_scale:
                        zmin_, zmax_ = zmin[0, d].item(), zmax[0, d].item()
                        rng = (zmax_ - zmin_) + 1e-8
                        z = z * rng + zmin_
                    arr = z * (sigma[d] + 1e-8) + mu[d]
                    df_out.at[i, col] = arr
        return df_out
