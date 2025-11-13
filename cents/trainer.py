from pathlib import Path
from typing import Dict, List, Optional

import pytorch_lightning as pl
import wandb
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cents.data_generator import DataGenerator
from cents.datasets.timeseries_dataset import TimeSeriesDataset
from cents.eval.eval import Evaluator
from cents.models.registry import get_model_cls
from cents.utils.utils import get_normalizer_training_config
from cents.utils.config_loader import load_yaml, apply_overrides

PKG_ROOT = Path(__file__).resolve().parent
CONF_DIR = PKG_ROOT / "config"


class Trainer:
    """
    Facade for training and evaluating generative time-series models.

    Supports ACGAN, Diffusion_TS and Normalizer via PyTorch Lightning and Hydra.

    Attributes:
        model_type: Identifier of the model to train/evaluate.
        dataset: TimeSeriesDataset used for training and evaluation.
        cfg: Hydra configuration object.
        model: Instantiated model object.
        pl_trainer: PyTorch Lightning Trainer.
    """

    def __init__(
        self,
        model_type: str,
        dataset: Optional[TimeSeriesDataset] = None,
        cfg: Optional[DictConfig] = None,
        overrides: Optional[List[str]] = None,
    ):
        """
        Initialize the Trainer.

        Args:
            model_type: Key of the model ("acgan", "diffusion_ts", or "normalizer").
            dataset: Dataset object required for generative models; optional for normalizer.
            cfg: Full OmegaConf DictConfig; if None, composed via Hydra.
            overrides: List of Hydra override strings.

        Raises:
            ValueError: If model_type is unknown or dataset requirements are not met.
        """
        try:
            get_model_cls(model_type)
        except ValueError:
            raise ValueError(f"Unknown model '{model_type}'")

        if model_type != "normalizer" and dataset is None:
            raise ValueError(f"Model '{model_type}' requires a TimeSeriesDataset.")

        if model_type == "normalizer" and dataset is None:
            raise ValueError("Normalizer training needs the raw dataset object.")

        self.model_type = model_type
        self.dataset = dataset
        self.cfg = cfg or self._compose_cfg(overrides or [])

        self.model = self._instantiate_model()
        self.pl_trainer = self._instantiate_trainer()

    def fit(self) -> "Trainer":
        """
        Start training.

        Returns:
            Self, to allow method chaining.
        """
        if self.model_type == "normalizer":
            self.pl_trainer.fit(self.model)
        else:
            train_loader = self.dataset.get_train_dataloader(
                batch_size=self.cfg.trainer.batch_size,
                shuffle=True,
                num_workers=6,  # Maximum for 7.5GB/10GB GPU usage
                persistent_workers=True,
            )
            self.pl_trainer.fit(self.model, train_loader, None)
        return self

    def get_data_generator(self) -> DataGenerator:
        """
        Create a DataGenerator for sampling from the trained generative model.

        Returns:
            DataGenerator bound to the trained model and dataset.

        Raises:
            RuntimeError: If called for the normalizer model (non-generative).
        """
        if self.model_type == "normalizer":
            raise RuntimeError("Normalizer is not a generative model.")

        device = (
            self.model.device
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

        gen = DataGenerator(
            model_name=self.model_type,
            device=device,
            cfg=self.cfg,
            model=self.model.eval(),
            normalizer=getattr(self.dataset, "_normalizer", None),
        )

        gen.set_dataset_spec(
            dataset_cfg=self.dataset.cfg,
            ctx_codes=self.dataset.get_context_var_codes(),
        )
        return gen

    def evaluate(self, **kwargs) -> Dict:
        """
        Run evaluation of the trained model using Evaluator.

        Args:
            **kwargs: Passed to Evaluator.evaluate_model (e.g. user_id).

        Returns:
            Dictionary of evaluation results.
        """
        evaluator = Evaluator(self.cfg, self.dataset)
        return evaluator.evaluate_model(model=self.model, **kwargs)

    def _compose_cfg(self, ov: List[str]) -> DictConfig:
        """
        Compose configuration by loading YAMLs and applying overrides.

        Structure:
            cfg.model   <- config/model/{model_type}.yaml
            cfg.trainer <- config/trainer/{model_type}.yaml
            cfg.dataset <- provided dataset.cfg (if any)
        """
        model_cfg = load_yaml(CONF_DIR / "model" / f"{self.model_type}.yaml")
        trainer_cfg = load_yaml(CONF_DIR / "trainer" / f"{self.model_type}.yaml")

        cfg = OmegaConf.create({})
        cfg.model = model_cfg
        cfg.trainer = trainer_cfg

        if self.dataset is not None:
            cfg.dataset = OmegaConf.create(
                OmegaConf.to_container(self.dataset.cfg, resolve=True)
            )

        cfg = apply_overrides(cfg, ov)

        # Ensure required top-level fields exist without Hydra
        if not hasattr(cfg, "device"):
            cfg.device = "auto"
        if not hasattr(cfg, "job_name"):
            ds_name = getattr(cfg, "dataset", {}).get("name", "dataset") if isinstance(getattr(cfg, "dataset", {}), dict) else getattr(cfg.dataset, "name", "dataset")
            ds_group = getattr(cfg, "dataset", {}).get("user_group", "all") if isinstance(getattr(cfg, "dataset", {}), dict) else getattr(cfg.dataset, "user_group", "all")
            model_name = getattr(cfg.model, "name", self.model_type)
            cfg.job_name = f"{model_name}_{ds_name}_{ds_group}"
        if not hasattr(cfg, "run_dir") or not cfg.run_dir:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cfg.run_dir = str(PKG_ROOT / "outputs" / cfg.job_name / timestamp)
        return cfg

    def _instantiate_model(self):
        """
        Instantiate the model class from the registry based on model_type.
        """
        ModelCls = get_model_cls(self.model_type)
        if self.model_type == "normalizer":
            nm_cfg = get_normalizer_training_config()
            return ModelCls(
                dataset_cfg=self.cfg.dataset,
                normalizer_training_cfg=nm_cfg,
                dataset=self.dataset,
            )
        return ModelCls(self.cfg)

    def _instantiate_trainer(self) -> pl.Trainer:
        """
        Build a PyTorch Lightning Trainer with ModelCheckpoint and loggers.

        Returns:
            Configured pl.Trainer instance.
        """
        tc = self.cfg.trainer
        callbacks = []
        # Build filename with optional context_module_type
        filename_parts = [
            self.cfg.dataset.name,
            self.model_type,
            f"dim{self.cfg.dataset.time_series_dims}"
        ]
        
        # Add context_module_type if available (from model or dataset config)
        context_module_type = getattr(
            self.cfg.model, "context_module_type", 
            getattr(self.cfg.dataset, "context_module_type", None)
        )
        if context_module_type:
            filename_parts.append(f"ctx{context_module_type}")
        
        # Add stats_head_type if available (typically in dataset config for normalizer)
        stats_head_type = getattr(self.cfg.dataset, "stats_head_type", None)
        if stats_head_type:
            filename_parts.append(f"stats{stats_head_type}")
        
        callbacks.append(
            ModelCheckpoint(
                dirpath=self.cfg.run_dir,
                filename="_".join(filename_parts),
                save_last=tc.checkpoint.save_last,
                save_on_train_epoch_end=True, ### Perhaps excessive
            )
        )
        callbacks.append(EvalAfterTraining(self.cfg, self.dataset))
        logger = False
        if getattr(self.cfg, "wandb", None) and self.cfg.wandb.enabled:
            logger = WandbLogger(
                project=self.cfg.wandb.project or "cents",
                entity=self.cfg.wandb.entity,
                name=self.cfg.wandb.name,
                save_dir=self.cfg.run_dir,
            )

        return pl.Trainer(
            max_epochs=tc.max_epochs,
            accelerator=tc.accelerator,
            strategy=tc.strategy,
            devices=tc.devices,
            precision=tc.precision,
            log_every_n_steps=tc.get("log_every_n_steps", 1),
            accumulate_grad_batches=tc.get("gradient_accumulate_every", 1),
            callbacks=callbacks,
            logger=logger,
            default_root_dir=self.cfg.run_dir,
        )


class EvalAfterTraining(Callback):
    """Run full evaluator at the *end* of training and log metrics to W&B."""

    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def on_train_end(self, trainer, pl_module):
        if not self.cfg.trainer.get("eval_after_training", False):
            return

        evaluator = Evaluator(self.cfg, self.dataset)
        results = evaluator.evaluate_model(model=pl_module)

        run = getattr(trainer.logger, "experiment", None)
        if run is not None:
            run.log(results["metrics"])
