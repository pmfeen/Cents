import copy
import math
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from contextlib import contextmanager


from cents.models.base import GenerativeModel
from cents.models.model_utils import (
    Transformer,
    cosine_beta_schedule,
    default,
    linear_beta_schedule,
    total_correlation,
)
from cents.models.registry import register_model


@register_model("diffusion_ts", "Watts_2_1D", "Watts_2_2D")
class Diffusion_TS(GenerativeModel):
    """
    PyTorch-Lightning module for a conditional time-series diffusion model.

    Uses a Transformer backbone to predict and denoise time series over
    discrete diffusion timesteps. Supports EMA smoothing and configurable
    beta schedules.

    Training objective (config model.training_objective): x0, epsilon, or v.
      - x0: predict clean sample; loss = L1/L2(model_out, x_clean).
      - epsilon: predict noise; loss = L1/L2(pred_epsilon, noise); pred_epsilon derived from model x0.
      - v: v-parameterization; loss = L1/L2(pred_v, true_v); pred_v derived from model x0.
    The network always outputs x0 (fc layer); sampling uses x0 and q_posterior unchanged.
    Variance-debugging reference:
      (a) Sampling: model predicts x0; reverse step uses q_posterior(x0,x_t,t);
          noise term = sqrt(posterior_variance). Same beta schedule in train and sample.
      (b) Normalization: per-group (context) z-score + optional [zmin,zmax] scale;
          denorm must use the same normalizer.inverse_transform. Run
          scripts/check_diffusion_consistency.py to verify norm/denorm identity.
    """

    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: OmegaConf config with fields:
                dataset.seq_len: sequence length
                dataset.time_series_dims: number of series dims
                model.*: diffusion-specific hyperparameters
                trainer: optimizer and scheduler configs
        """
        super().__init__(cfg)
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.seq_len = cfg.dataset.seq_len
        self.time_series_dims = cfg.dataset.time_series_dims
        self.eta = cfg.model.eta
        self.use_ff = cfg.model.use_ff
        self.ff_weight = default(cfg.model.reg_weight, math.sqrt(self.seq_len) / 5)
        self.embedding_dim = cfg.model.cond_emb_dim
        self.context_reconstruction_loss_weight = (
            cfg.model.context_reconstruction_loss_weight
        )
        # Verify context modules are initialized (static, dynamic, or both)
        if not hasattr(self, 'static_context_module') and not hasattr(self, 'dynamic_context_module'):
            raise ValueError("At least one context module (static or dynamic) must be initialized")

        # linear layer for denoised output
        self.fc = nn.Linear(
            self.time_series_dims + self.embedding_dim, self.time_series_dims
        )
        # Transformer backbone
        self.model = Transformer(
            n_feat=self.time_series_dims + self.embedding_dim,
            n_channel=self.seq_len,
            n_layer_enc=cfg.model.n_layer_enc,
            n_layer_dec=cfg.model.n_layer_dec,
            n_heads=cfg.model.n_heads,
            attn_pdrop=cfg.model.attn_pd,
            resid_pdrop=cfg.model.resid_pd,
            mlp_hidden_times=cfg.model.mlp_hidden_times,
            max_len=self.seq_len,
            n_embd=cfg.model.d_model,
            conv_params=[cfg.model.kernel_size, cfg.model.padding_size],
        )

        # EMA helper will be initialized on train start
        self._ema: Optional[EMA] = None

        # set up beta schedule
        if cfg.model.beta_schedule == "linear":
            betas = linear_beta_schedule(cfg.model.n_steps)
        elif cfg.model.beta_schedule == "cosine":
            betas = cosine_beta_schedule(cfg.model.n_steps)
        else:
            raise ValueError("Unknown beta schedule")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.num_timesteps = betas.shape[0]
        self.sampling_timesteps = default(
            cfg.model.sampling_timesteps, self.num_timesteps
        )
        self.fast_sampling = self.sampling_timesteps < self.num_timesteps
        self.loss_type = cfg.model.loss_type
        self.training_objective = getattr(
            cfg.model, "training_objective", "x0"
        ).lower()
        if self.training_objective not in ("x0", "eps", "v"):
            raise ValueError(
                f"training_objective must be one of x0, eps, v; got {self.training_objective}"
            )

        # register buffers for diffusion coefficients
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        # posterior coefficients
        pmc1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        pmc2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", pmc1)
        self.register_buffer("posterior_mean_coef2", pmc2)

        # Loss weighting: default (legacy), uniform, snr, or min_snr (via compute_snr_weights)
        loss_weighting = getattr(cfg.model, "loss_weighting", "default")
        min_snr_gamma = getattr(cfg.model, "min_snr_gamma", 5.0)
        if loss_weighting == "default":
            lw = torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas.clamp(min=1e-8) / 100
        else:
            lw = self.compute_snr_weights(
                alphas_cumprod,
                loss_weighting=loss_weighting,
                objective=self.training_objective,
                gamma=min_snr_gamma,
            )
        self.register_buffer("loss_weight", lw)

        # choose reconstruction loss
        if self.loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        elif self.loss_type == "l2":
            self.recon_loss_fn = F.mse_loss  # MSE for continuous RVs
        else:
            raise ValueError("Invalid loss type")

        self.auxiliary_loss = nn.CrossEntropyLoss()
        
        # Get continuous variables from config to distinguish them in loss computation
        self.continuous_context_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "continuous"]
        self.categorical_context_vars = [k for k, v in cfg.dataset.context_vars.items() if v[0] == "categorical"]

    def _get_context_embedding(self, context_vars: dict) -> Tuple[torch.Tensor, dict]:
        """
        Get combined context embedding from static and/or dynamic context modules.
        
        Args:
            context_vars: Dict of context tensors (static: single values, dynamic: time series)
            
        Returns:
            embedding: Combined embedding tensor of shape (batch_size, embedding_dim)
            all_logits: Dict of classification/regression logits from both modules
        """
        embeddings = []
        all_logits = {}
        
        # Process static context variables
        if self.static_context_module is not None:
            # Filter static context variables
            static_vars = {
                k: v for k, v in context_vars.items()
                if k not in getattr(self, 'dynamic_context_vars', [])
            }
            if static_vars:
                device = next(self.static_context_module.parameters()).device
                static_vars = {
                    k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in static_vars.items()
                }
                static_embedding, static_logits = self.static_context_module(static_vars)
                embeddings.append(static_embedding)
                all_logits.update(static_logits)
        
        # Process dynamic context variables
        if self.dynamic_context_module is not None:
            # Filter dynamic context variables
            dynamic_var_names = getattr(self, 'dynamic_context_vars', [])
            dynamic_vars = {
                k: v for k, v in context_vars.items()
                if k in dynamic_var_names
            }
            if dynamic_vars:
                device = next(self.dynamic_context_module.parameters()).device
                dynamic_vars = {
                    k: v.to(device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in dynamic_vars.items()
                }
                dynamic_embedding, dynamic_logits = self.dynamic_context_module(dynamic_vars)
                # Check for NaN in dynamic embedding
                if torch.isnan(dynamic_embedding).any() or torch.isinf(dynamic_embedding).any():
                    raise ValueError(
                        f"NaN/Inf detected in dynamic embedding. "
                        f"Dynamic vars: {list(dynamic_vars.keys())}"
                    )
                embeddings.append(dynamic_embedding)
                all_logits.update(dynamic_logits)
        
        # Combine embeddings if both exist
        if len(embeddings) == 2:
            combined = torch.cat(embeddings, dim=1)
            embedding = self.combine_mlp(combined)
        elif len(embeddings) == 1:
            embedding = embeddings[0]
        else:
            raise ValueError("No context variables provided")        
        return embedding, all_logits

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate noise given a noised sample x_t and predicted x0.

        Args:
            x_t: Noised input at timestep t, shape (B,L,C).
            t: Tensor of timesteps, shape (B,).
            x0: Predicted clean sample at t=0, shape (B,L,C).

        Returns:
            Noise prediction tensor same shape as x_t.
        """
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t - x0
        ) / self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1)

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x0 from noisy x_t and noise.

        Args:
            x_t: Noised input at timestep t, shape (B,L,C).
            t: Tensor of timesteps, shape (B,).
            noise: Actual noise added, shape (B,L,C).

        Returns:
            Reconstructed x0 tensor same shape as x_t.
        """
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1) * noise
        )

    def predict_start_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct x0 from x_t and v-parameterization.
        v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x0  =>  x0 = sqrt(alpha_bar_t) * x_t - sqrt(1 - alpha_bar_t) * v
        """
        return (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x_t
            - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * v
        )

    def predict_noise_from_v(
        self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstruct epsilon from x_t and v-parameterization.
        v = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * x0  =>  epsilon = sqrt(1 - alpha_bar_t) * x_t + sqrt(alpha_bar_t) * v
        """
        return (
            self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * x_t
            + self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * v
        )


    def compute_snr_weights(
        self,
        alphas_cumprod: torch.Tensor,
        *,
        loss_weighting: str,
        objective: str,
        gamma: float = 5.0,
    ) -> torch.Tensor:
        """
        SNR-based loss weighting per timestep.

        Args:
            alphas_cumprod: Cumulative product of alphas, shape (n_steps,).
            loss_weighting: "uniform" | "snr" | "min_snr".
            objective: "eps" | "x0" | "v" — must match training objective.
            gamma: Cap for SNR when loss_weighting == "min_snr".

        Returns:
            Weight tensor same shape as alphas_cumprod.
        """
        snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

        if loss_weighting == "uniform":
            return torch.ones_like(snr)

        if loss_weighting == "snr":
            if objective == "eps":
                return 1.0 / (snr + 1.0)
            elif objective == "x0":
                return snr / (snr + 1.0)
            elif objective == "v":
                return 1.0 / torch.sqrt(snr + 1.0)
            else:
                raise ValueError(objective)

        if loss_weighting == "min_snr":
            snr_c = torch.minimum(snr, torch.full_like(snr, gamma))
            if objective == "eps":
                return snr_c / snr.clamp(min=1e-8)
            elif objective == "x0":
                return snr_c
            elif objective == "v":
                return snr_c / (snr + 1.0)
            else:
                raise ValueError(objective)

        raise ValueError(loss_weighting)

    def q_posterior(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for q(x_{t-1} | x_t, x_0).

        Args:
            x_start: Predicted x0, shape (B,L,C).
            x_t: Noised sample at t, shape (B,L,C).
            t: Tensor of timesteps, shape (B,).

        Returns:
            Tuple of (mean, variance, log_variance) tensors for posterior.
        """
        pm = (
            self.posterior_mean_coef1[t].view(-1, 1, 1) * x_start
            + self.posterior_mean_coef2[t].view(-1, 1, 1) * x_t
        )
        pv = self.posterior_variance[t].view(-1, 1, 1)
        plv = self.posterior_log_variance_clipped[t].view(-1, 1, 1)
        return pm, pv, plv

    def forward(self, x: torch.Tensor, context_vars: dict) -> Tuple[torch.Tensor, dict]:
        """
        Single forward pass: add noise, predict denoised output and compute reconstruction loss.

        Args:
            x: Clean input batch, shape (B,L,C).
            context_vars: Dict of context tensors used for conditioning.

        Returns:
            rec_loss: Reconstruction loss tensor.
            cond_logits: Classification logits dict from context module.
        """
        # Check input x for extreme values
        # if x.abs().max() > 100.0:
        #     print(f"[Warning] Input x has extreme values: min={x.min():.4f}, max={x.max():.4f}, "
        #           f"mean={x.mean():.4f}, std={x.std():.4f}, shape={x.shape}")
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"NaN/Inf detected in input x. Shape: {x.shape}, "
                           f"NaN count: {torch.isnan(x).sum()}, Inf count: {torch.isinf(x).sum()}")
        
        b = x.shape[0]
        # t = torch.randint(0, self.num_timesteps, (b,), device=self.device)
        t = self.stratified_timesteps(b, self.num_timesteps, self.cfg.model.k_bins, device=self.device)
        embedding, cond_classification_logits = self._get_context_embedding(context_vars)
        # Check embedding for NaN/Inf
        if embedding.isnan().any() or embedding.isinf().any():
            raise ValueError(
                f"NaN/Inf detected in embedding from context module. "
                f"NaN count: {embedding.isnan().sum()}, Inf count: {embedding.isinf().sum()}, "
                f"shape: {embedding.shape}, min: {embedding.min()}, max: {embedding.max()}"
            )
        
        # Embedding should now be normalized by the context module (mean=0, std=1 per sample)
        # Check that values are in reasonable range
        if embedding.abs().max() > 100.0:
            print(f"[Warning] Embedding has large values despite normalization: "
                  f"min={embedding.min():.4f}, max={embedding.max():.4f}, "
                  f"mean={embedding.mean():.4f}, std={embedding.std():.4f}")
        
        # Check diffusion schedule parameters
        noise = torch.randn_like(x)
        x_noisy = (
            self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * x
            + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
        )
        if x_noisy.isnan().any() or x_noisy.isinf().any(): 
            raise ValueError(f"NaN/Inf detected in x_noisy. Shape: {x_noisy.shape}, "
                           f"NaN count: {torch.isnan(x_noisy).sum()}, Inf count: {torch.isinf(x_noisy).sum()}")
        # Use normalized embedding for concatenation
        embedding_expanded = embedding.unsqueeze(1).repeat(1, self.seq_len, 1)
        c = torch.cat([x_noisy, embedding_expanded], dim=-1)
        if c.isnan().any() or c.isinf().any():
            raise ValueError(f"NaN/Inf detected in concatenated input 'c'. "
                           f"Shape: {c.shape}, x_noisy stats: min={x_noisy.min():.4f}, max={x_noisy.max():.4f}, "
                           f"embedding stats: min={embedding.min():.4f}, max={embedding.max():.4f}")
        # if c.isnan().any() or c.isinf().any():
        #     raise ValueError(
        #         f"NaN/Inf detected in concatenated input 'c'. "
        #         f"x_noisy stats: mean={x_noisy.mean():.4f}, std={x_noisy.std():.4f}, "
        #         f"min={x_noisy.min():.4f}, max={x_noisy.max():.4f}. "
        #         f"embedding stats: mean={embedding.mean():.4f}, "
        #         f"std={embedding.std():.4f}, min={embedding.min():.4f}, "
        #         f"max={embedding.max():.4f}"
        #     )
        trend, season = self.model(c, t, padding_masks=None)
        x_start_pred = self.fc(trend + season)
        # Compute loss based on training objective (network always predicts x0; we derive epsilon/v as needed)
        if self.training_objective == "x0":
            loss_per_elem = self.recon_loss_fn(x_start_pred, x, reduction="none")
        elif self.training_objective == "epsilon":
            pred_noise = self.predict_noise_from_start(x_noisy, t, x_start_pred)
            loss_per_elem = self.recon_loss_fn(pred_noise, noise, reduction="none")
        else:  # v
            pred_noise = self.predict_noise_from_start(x_noisy, t, x_start_pred)
            pred_v = (
                self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * pred_noise
                - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * x_start_pred
            )
            true_v = (
                self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * noise
                - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * x
            )
            loss_per_elem = self.recon_loss_fn(pred_v, true_v, reduction="none")
        rec_loss = (
            self.loss_weight[t].view(-1, 1, 1) * loss_per_elem
        ).mean()
        return rec_loss, cond_classification_logits

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step: compute total loss including context reconstruction.

        Args:
            batch: Tuple(ts_batch, cond_batch).
            batch_idx: Index of batch (unused).

        Returns:
            total_loss: Scalar training loss.
        """
        ts_batch, cond_batch = batch
        rec_loss, cond_class_logits = self(ts_batch, cond_batch)
        
        cond_loss = 0.0


        for var_name, outputs in cond_class_logits.items():
            labels = cond_batch[var_name]
            if var_name in self.continuous_context_vars:
                loss = F.mse_loss(outputs, labels.float())
            elif var_name in self.categorical_context_vars:
                loss = self.auxiliary_loss(outputs, labels)
            
            cond_loss += loss.mean()

        #     # if var_name in self.continuous_context_vars:
        #     #     print(var_name)
        #     #     print(loss)
        #     #     print(outputs.mean(), labels.mean())

        
        # cond_loss /= len(cond_class_logits)

        h, _ = self._get_context_embedding(cond_batch)
        tc_term = (
            self.cfg.model.tc_loss_weight * total_correlation(h)
            if self.cfg.model.tc_loss_weight > 0.0
            else torch.tensor(0.0, device=self.device)
        )

        total_loss = (
            rec_loss + self.context_reconstruction_loss_weight * cond_loss + tc_term
        )
        
        # Check for NaN in total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise ValueError(
                f"NaN/Inf detected in total_loss at batch {batch_idx}. "
                f"rec_loss: {rec_loss.item():.6f}, cond_loss: {cond_loss:.6f}, tc_term: {tc_term.item():.6f}"
            )
        
        self.log_dict(
            {
                "train_loss": total_loss.item(),
                "rec_loss": rec_loss.item(),
                "cond_loss": cond_loss.item(),
                "tc_loss": tc_term,
            },
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
        )
        return total_loss

    def configure_optimizers(self) -> dict:
        """
        Set up optimizer and LR scheduler.

        Returns:
            Dict with "optimizer" and "lr_scheduler" entries.
        """
        optimizer = Adam(
            self.parameters(), lr=self.cfg.trainer.base_lr, betas=(0.9, 0.96)
        )
        scheduler = ReduceLROnPlateau(optimizer, **self.cfg.trainer.lr_scheduler_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }

    def on_train_start(self) -> None:
        """
        Initialize EMA helper at start of training.
        """
        self._ema = EMA(
            self.model,
            beta=self.cfg.model.ema_decay,
            update_every=self.cfg.model.ema_update_interval,
        )

    def on_after_backward(self) -> None:
        """
        Check gradients after backward pass but before optimizer step.
        This is the right place to inspect gradients before they're zeroed.
        """
        # Get current batch index from trainer
        if not hasattr(self.trainer, 'global_step'):
            return
        
        batch_idx = self.trainer.global_step
        
        # Debug: Check if context module parameters are getting gradients
        # Check AFTER backward pass but BEFORE optimizer step (only log occasionally)
        if batch_idx % 50 == 0:
            context_params_with_grad = []
            context_params_no_grad = []
            if self.static_context_module is not None:
                for name, param in self.static_context_module.named_parameters():
                    if param.requires_grad:
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            # Check for NaN/Inf gradients
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"[Warning] NaN/Inf gradients detected in {name}")
                            else:
                                context_params_with_grad.append((name, grad_norm))
                        else:
                            context_params_no_grad.append(name)
            
            if context_params_no_grad:
                # Group by variable name to identify which context variables are missing
                missing_vars = set()
                for param_name in context_params_no_grad:
                    # Extract variable name from parameter name (e.g., "context_embeddings.year.weight" -> "year")
                    parts = param_name.split('.')
                    if len(parts) >= 2 and parts[0] in ['context_embeddings', 'init_mlps']:
                        missing_vars.add(parts[1])
                print(f"[Warning] {len(context_params_no_grad)} context module parameters have no gradients!")
                if missing_vars:
                    pass
                    # print(f"  Missing context variables: {sorted(missing_vars)}")
                # print(f"  No grad params (sample): {context_params_no_grad[:5]}...")
            # if context_params_with_grad:
            #     avg_grad_norm = sum(g[1] for g in context_params_with_grad) / len(context_params_with_grad)
            #     max_grad_norm = max(g[1] for g in context_params_with_grad)
            #     print(f"[Debug] Context module gradients: avg_norm={avg_grad_norm:.6f}, max_norm={max_grad_norm:.6f}")

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """
        Apply EMA update after each batch end.
        """
        if hasattr(self, '_ema') and self._ema:
            self._ema.update()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Restore EMA weights from checkpoint after loading.
        """
        super().on_load_checkpoint(checkpoint)
        
        # Check if EMA weights exist in checkpoint
        state_dict = checkpoint.get('state_dict', {})
        ema_keys = [key for key in state_dict.keys() if key.startswith('_ema.')]
        
        if ema_keys:
            if not hasattr(self, '_ema') or self._ema is None:
                self._ema = EMA(
                    self.model,
                    beta=self.cfg.model.ema_decay,
                    update_every=self.cfg.model.ema_update_interval,
                )
            
            # Load EMA weights into the EMA helper
            ema_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_ema.ema_model.'):
                    # Map '_ema.ema_model.*' -> 'ema_model.*' (remove the _ema prefix)
                    ema_key = key.replace('_ema.ema_model.', 'ema_model.')
                    ema_state_dict[ema_key] = value
            
            if ema_state_dict:
                print(f"Loading {len(ema_state_dict)} EMA weights from checkpoint")
                self._ema.ema_model.load_state_dict(ema_state_dict, strict=False)
            else:
                raise ValueError("No EMA model weights found in checkpoint")
        else:
            raise ValueError("No EMA keys found in checkpoint")

    @torch.no_grad()
    def model_predictions(
        self, x: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict both noise and clean sample from current x.

        Returns:
            pred_noise: predicted noise tensor.
            x_start: predicted clean sample tensor.
        """
        c = torch.cat([x, embedding.unsqueeze(1).repeat(1, self.seq_len, 1)], dim=-1)
        trend, season = self.model(c, t, padding_masks=None)
        x_start = self.fc(trend + season)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    @torch.no_grad()
    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).

        Returns:
            pm, pv, plv, x_start: posterior parameters and predicted x0.
        """
        pred_noise, x_start = self.model_predictions(x, t, embedding)
        pm, pv, plv = self.q_posterior(x_start, x, t)
        return pm, pv, plv, x_start

    @torch.no_grad()
    def p_sample(
        self, x: torch.Tensor, t: int, embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t using posterior distribution.
        """
        bt = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
        pm, pv, plv, _ = self.p_mean_variance(x, bt, embedding)
        noise = torch.randn_like(x) if t > 0 else 0
        return pm + (0.5 * plv).exp() * noise

    @torch.no_grad()
    def sample(self, shape: Tuple[int, int, int], context_vars: dict) -> torch.Tensor:
        """
        Full reverse-pass sampling over all timesteps.

        Args:
            shape: (batch_size, seq_len, dims)
            context_vars: context conditioning dict

        Returns:
            Generated samples tensor.
        """
        x = torch.randn(shape, device=self.device)
        embedding, _ = self._get_context_embedding(context_vars)
        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t, embedding)
        return x

    @torch.no_grad()
    def fast_sample(
        self, shape: Tuple[int, int, int], context_vars: dict
    ) -> torch.Tensor:
        """
        Faster sampling using a reduced number of timesteps.
        """
        x = torch.randn(shape, device=self.device)
        embedding, _ = self._get_context_embedding(context_vars)
        times = torch.linspace(
            -1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1
        )
        times = list(reversed(times.int().tolist()))
        pairs = list(zip(times[:-1], times[1:]))
        for time, time_next in pairs:
            bt = torch.full((x.shape[0],), time, device=self.device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(x, bt, embedding)
            if time_next < 0:
                x = x_start
                continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = (
                self.eta
                * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c = (1 - alpha_next - sigma**2).sqrt()
            noise = torch.randn_like(x)
            x = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return x

    @contextmanager
    def ema_scope(self):
        if hasattr(self, "_ema") and self._ema and getattr(self.cfg.model, "use_ema_sampling", False):
            self._ema.store(self.model.parameters())
            self._ema.copy_to(self.model.parameters())
            try:
                yield
            finally:
                self._ema.restore(self.model.parameters())
        else:
            yield

    def generate(self, context_vars: dict) -> torch.Tensor:
        """
        Public entry to generate conditioned samples in batches.

        Args:
            context_vars: dict of context tensors for each sample.

        Returns:
            Complete generated tensor of shape (N, seq_len, dims).
        """
        bs = self.cfg.model.sampling_batch_size
        total = len(next(iter(context_vars.values())))
        generated_samples = []

        with self.ema_scope():
            for start_idx in tqdm(
                range(0, total, bs),
                unit="seq",
                desc="[CENTS] Generating samples",
                leave=True,
            ):
                end_idx = min(start_idx + bs, total)
                batch_context_vars = {
                    var_name: var_tensor[start_idx:end_idx]
                    for var_name, var_tensor in context_vars.items()
                }
                current_bs = end_idx - start_idx
                shape = (current_bs, self.seq_len, self.time_series_dims)

                with torch.no_grad():
                    if self.fast_sampling:
                        samples = self.fast_sample(shape, batch_context_vars)
                    else:
                        samples = self.sample(shape, batch_context_vars)


                generated_samples.append(samples.cpu())

        return torch.cat(generated_samples, dim=0)
        
    def _ensure_ema_helper(self) -> None:
        """
        Ensure EMA helper is initialized if needed for inference.
        """
        if not hasattr(self, '_ema') or self._ema is None:
            print("Initializing EMA helper for inference...")
            self._ema = EMA(
                self.model,
                beta=self.cfg.model.ema_decay,
                update_every=self.cfg.model.ema_update_interval,
            )
    def stratified_timesteps(self, batch_size: int, num_timesteps: int, k_bins: int, device=None) -> torch.Tensor:
        device = device or "cpu"
        k_bins = min(k_bins, batch_size)
        edges = torch.linspace(0, num_timesteps, steps=k_bins + 1, device=device)

        # sample one t per bin
        u = torch.rand(k_bins, device=device)
        t_bins = (edges[:-1] + u * (edges[1:] - edges[:-1])).floor().clamp_(0, num_timesteps - 1).long()

        # repeat to fill batch, then shuffle
        reps = (batch_size + k_bins - 1) // k_bins
        t = t_bins.repeat(reps)[:batch_size]
        t = t[torch.randperm(batch_size, device=device)]
        return t


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) helper for model parameters.
    """
    def __init__(self, model: nn.Module, beta: float = 0.9999, update_every: int = 10):
        super().__init__()
        self.beta = beta
        self.update_every = update_every
        self.step = 0

        # CRITICAL FIX 1: self.ema_model is the ONLY deepcopy. 
        # It holds the shadow weights.
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # CRITICAL FIX 2: We keep a reference to the LIVE model (not a copy)
        # so we can grab the latest trained weights during update().
        self.source_model = model 
        
        # Buffer to store temporary weights for the context manager
        self.collected_params = []

    def update(self) -> None:
        """
        Update the shadow parameters using the source model's current weights.
        """
        self.step += 1
        if self.step % self.update_every != 0:
            return
        
        with torch.no_grad():
            # Zip the shadow model (ema) against the live model (source)
            for ema_p, src_p in zip(self.ema_model.parameters(), self.source_model.parameters()):
                # ema_new = beta * ema_old + (1 - beta) * current_weight
                ema_p.data.mul_(self.beta).add_(src_p.data, alpha=1.0 - self.beta)

    def store(self, parameters):
        """
        Save the current parameters (of the live model) to a temporary list.
        Used by the context manager to back up weights before swapping.
        """
        self.collected_params = [param.clone().cpu() for param in parameters]

    def restore(self, parameters):
        """
        Restore the saved parameters back to the live model.
        """
        if not self.collected_params:
            raise RuntimeError("No parameters stored to restore.")
            
        for param, saved_param in zip(parameters, self.collected_params):
            param.data.copy_(saved_param.data.to(param.device))
            
        self.collected_params = [] # Clear memory

    def copy_to(self, parameters):
        """
        Copy the EMA shadow weights INTO the live model parameters.
        """
        for param, ema_param in zip(parameters, self.ema_model.parameters()):
            param.data.copy_(ema_param.data.to(param.device))