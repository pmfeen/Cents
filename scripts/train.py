from datetime import datetime
import pandas as pd

from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping


def main() -> None:
    MODEL_NAME = "diffusion_ts"
    CR_LOSS_WEIGHT = 0.1
    TC_LOSS_WEIGHT = 0.1
    # Skip heavy processing for DDP compatibility
    dataset = CommercialDataset(
        skip_heavy_processing=True
    )

    trainer_overrides = [
        "trainer.max_epochs=5000",
        # "trainer.strategy=ddp_spawn",
        "trainer.devices=1,2,3",  # Exclude GPU 0, use GPUs 1,2,3
        # "trainer.devices=1",
        "trainer.eval_after_training=True",
        "train.accelerator=gpu",
        # "train.accelerator=cpu",
        "trainer.early_stopping.patience=100",  # Stop if no improvement for 100 epochs
        "trainer.early_stopping.monitor=train_loss",  # Monitor training loss
        "trainer.early_stopping.mode=min",  # Stop when loss stops decreasing
        "trainer.enable_checkpointing=True",  # Explicitly enable checkpointing
        "trainer.logger=False",  # Disable logger to see checkpoint messages
        "wandb.enabled=False",
        "wandb.project=cents",
        "wandb.entity=pmfeen-massachusetts-institute-of-technology",
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
    import os
    # Enable CUDA debugging for better error messages
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    main()
