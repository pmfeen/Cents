import logging
from datetime import datetime

import wandb
from omegaconf import OmegaConf

from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.eval.eval import Evaluator
from cents.utils.config_loader import load_yaml


MODEL_KEY = "Watts_1_2D"
DATASET_OVERRIDES = [
    "user_group=pv_users",
    "time_series_dims=2",
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    if wandb.run is None:
        wandb.init(
            project="cents",
            name=f"{MODEL_KEY}-eval-only-run_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            entity="pmfeen-massachusetts-institute-of-technology",
        )

    # Dataset with simple overrides (no Hydra)
    dataset = PecanStreetDataset(overrides=DATASET_OVERRIDES)

    # Build a minimal cfg for evaluator and generator
    eval_cfg = load_yaml("cents/config/evaluator/default.yaml")
    top_cfg = load_yaml("cents/config/config.yaml")
    cfg = OmegaConf.create({})
    cfg.evaluator = eval_cfg
    cfg.wandb = top_cfg.get("wandb", {})
    cfg.device = top_cfg.get("device", "auto")
    cfg.model = OmegaConf.create({"name": MODEL_KEY})
    cfg.dataset = OmegaConf.create(
        OmegaConf.to_container(dataset.cfg, resolve=True)
    )

    # Use the fixed checkpoint with DataGenerator
    gen = DataGenerator(MODEL_KEY)
    gen.set_dataset_spec(cfg.dataset, dataset.get_context_var_codes())

    results = Evaluator(cfg, dataset).evaluate_model(data_generator=gen)
    print(results)


if __name__ == "__main__":
    main()
