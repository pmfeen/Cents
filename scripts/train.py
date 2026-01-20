from datetime import datetime
import pandas as pd

from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.trainer import Trainer
from cents.utils.utils import set_context_config_path, set_context_overrides
from pytorch_lightning.callbacks import EarlyStopping
import warnings
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(args) -> None:
    MODEL_NAME = args.model_name
    CR_LOSS_WEIGHT = args.cr_loss_weight
    TC_LOSS_WEIGHT = args.tc_loss_weight
    
    # Set custom context config path if provided
    if args.context_config_path:
        set_context_config_path(args.context_config_path)
    
    # Set context config overrides if provided
    if args.context_overrides:
        set_context_overrides(args.context_overrides)
    
    # Skip heavy processing for DDP compatibility

    if args.dataset == "pecanstreet":
        dataset = PecanStreetDataset(
            overrides=[f"skip_heavy_processing={args.skip_heavy_processing}, time_series_dims=1, user_group=all"],
            force_retrain_normalizer=args.force_retrain_normalizer
        )
    elif args.dataset == "commercial":
        dataset = CommercialDataset(
            overrides=[f"skip_heavy_processing={args.skip_heavy_processing}"],
            force_retrain_normalizer=args.force_retrain_normalizer
        )
    elif args.dataset == "airquality":
        dataset = AirQualityDataset(
            overrides=[f"skip_heavy_processing={args.skip_heavy_processing}"],
            force_retrain_normalizer=args.force_retrain_normalizer
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    print("Initialized Dataset")

    trainer_overrides = [
        f"trainer.max_epochs={args.epochs}",
        f"trainer.strategy={args.ddp_strategy}",
        f"trainer.devices={args.devices}",
        f"trainer.eval_after_training={args.eval_after_training}",
        f"train.accelerator={args.accelerator}",
        "trainer.early_stopping.patience=100",  # Stop if no improvement for 100 epochs
        "trainer.early_stopping.monitor=train_loss",  # Monitor training loss
        "trainer.early_stopping.mode=min",  # Stop when loss stops decreasing
        f"trainer.enable_checkpointing={args.enable_checkpointing}",  # Explicitly enable checkpointing
        "trainer.logger=False",  # Disable logger to see checkpoint messages
        f"wandb.enabled={args.wandb_enabled}",
        f"wandb.project={args.wandb_project}",
        f"wandb.entity={args.wandb_entity}",
        f"model.context_reconstruction_loss_weight={CR_LOSS_WEIGHT}",
        f"model.tc_loss_weight={TC_LOSS_WEIGHT}",
        f"wandb.name=training_dai_{MODEL_NAME}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_L{CR_LOSS_WEIGHT}_TC_{TC_LOSS_WEIGHT}_dim2",
    ]

    trainer = Trainer(
        model_type=MODEL_NAME,
        dataset=dataset,
        overrides=trainer_overrides,
    )

    trainer.fit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--model_name", type=str, default="diffusion_ts")
    parser.add_argument("--cr_loss_weight", type=float, default=0.1)
    parser.add_argument("--tc_loss_weight", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="pecanstreet")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--wandb-enabled", type=bool, default=False)
    parser.add_argument("--wandb-project", type=str, default="cents")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--eval_after_training", type=bool, default=True)
    parser.add_argument("--skip_heavy_processing", type=bool, default=True)
    parser.add_argument("--ddp-strategy", type=str, default="ddp_find_unused_parameters_false")
    parser.add_argument("--enable_checkpointing", type=bool, default=True)
    parser.add_argument("--context-config-path", type=str, default=None, 
                        help="Path to custom context config YAML file (optional)")
    parser.add_argument("--context-overrides", type=str, nargs="*", default=[],
                        help="Override context config values (e.g., 'static_context.type=mlp' 'dynamic_context.type=cnn')")
    parser.add_argument("--force-retrain-normalizer", type=bool, default=False,
                        help="Force retraining of normalizer even if cached version exists")

    args = parser.parse_args()
    main(args)
