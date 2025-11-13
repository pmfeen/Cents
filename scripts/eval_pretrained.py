import logging
from datetime import datetime
from typing import override

# import wandb
from omegaconf import OmegaConf

from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.eval.eval import Evaluator
from cents.utils.config_loader import load_yaml
from pathlib import Path
import torch

MODEL_KEY = "diffusion_ts"
DATASET_OVERRIDES = [
    "max_samples=10000",
    "skip_heavy_processing=True"
]

PECAN_OVERRIDES = [
    "time_series_dims=1",
    "user_group=all"
]

HOME = Path.home()

def main() -> None:
    
    model_ckpt = "cents/outputs/diffusion_ts_commercial_all/2025-11-07_15-09-33/commercial_diffusion_ts_dim1_ctxsep_mlp_statsmlp.ckpt"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    print("Loading dataset...")
    # dataset = PecanStreetDataset(overrides=DATASET_OVERRIDES + PECAN_OVERRIDES)
    dataset = CommercialDataset(overrides = DATASET_OVERRIDES)

    normalizer_ckpt = HOME / ".cache/cents/checkpoints/commercial/normalizer/commercial_normalizer_dim1_ctxsep_mlp_statsmlp.ckpt"
    # Build a minimal cfg for evaluator and generator
    eval_cfg = load_yaml("cents/config/evaluator/default.yaml")
    top_cfg = load_yaml("cents/config/config.yaml")
    cfg = OmegaConf.create({})
    cfg.evaluator = eval_cfg
    cfg.wandb = top_cfg.get("wandb", {})
    cfg.device = top_cfg.get("device", "auto")
    cfg.model = OmegaConf.create(OmegaConf.to_container(OmegaConf.load(f"cents/config/model/{MODEL_KEY}.yaml"), resolve=True))
    cfg.dataset = OmegaConf.create(
        OmegaConf.to_container(dataset.cfg, resolve=True)
    )
    # Enable EMA sampling to use the EMA weights from checkpoint
    cfg.model.use_ema_sampling = True
    cfg.eval_pv_shift = eval_cfg.get("eval_pv_shift", False)
    cfg.eval_metrics = eval_cfg.get("eval_metrics", True)
    cfg.eval_context_sparse = eval_cfg.get("eval_context_sparse", True)
    cfg.save_results = eval_cfg.get("save_results", False)
    cfg.eval_disentanglement = eval_cfg.get("eval_disentanglement", True)
    cfg.job_name = eval_cfg.get("job_name", "default_job")
    cfg.save_results = True
    cfg.save_dir = HOME / f"cents/outputs/diffusion_ts_commercial_all/2025-11-07_15-09-33/eval"
    print("Dataset spec set. Setting up DataGenerator...")

    # Use the fixed checkpoint with DataGenerator
    gen = DataGenerator(model_type = MODEL_KEY, dataset=dataset)
    print("Loading checkpoint... EMA sampling enabled - will use EMA weights for generation")
    gen.load_from_checkpoint(model_ckpt, normalizer_ckpt)

    gen.set_dataset_spec(gen.model.cfg.dataset, dataset.get_context_var_codes())
    cfg.dataset = gen.model.cfg.dataset
    
    print("Checkpoint loaded")

    print("Evaluating model...")
    results = Evaluator(cfg, dataset).evaluate_model(data_generator=gen)
    print(results)


if __name__ == "__main__":
    main()
