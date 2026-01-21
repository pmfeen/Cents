import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
import argparse

from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.eval.eval import Evaluator
from cents.utils.config_loader import load_yaml
from cents.utils.utils import set_context_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
DATASET_OVERRIDES = ["max_samples=10000", "skip_heavy_processing=True"]
PECAN_OVERRIDES = ["time_series_dims=1", "user_group=all"]


def _load_dataset(name: str, overrides: list):
    """Load a dataset by name with optional overrides."""
    if name == "pecanstreet":
        return PecanStreetDataset(overrides=DATASET_OVERRIDES + PECAN_OVERRIDES + (overrides or []))
    if name == "commercial":
        return CommercialDataset(overrides=DATASET_OVERRIDES + (overrides or []))
    if name == "airquality":
        return AirQualityDataset(overrides=DATASET_OVERRIDES + (overrides or []))
    raise ValueError(f"Dataset {name} not supported. Use: pecanstreet, commercial, airquality.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model using comprehensive metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt or .pt).",
    )
    parser.add_argument(
        "--normalizer-ckpt",
        type=str,
        default=None,
        help="Path to normalizer checkpoint. If omitted, evaluation uses normalized space.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="diffusion_ts",
        help="Model type (e.g. diffusion_ts) used to load the checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pecanstreet",
        choices=("pecanstreet", "commercial", "airquality"),
        help="Dataset name (must match the one used to train the model).",
    )
    parser.add_argument(
        "--dataset-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Extra dataset overrides, e.g. time_series_dims=1.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results. If None, uses default location based on model checkpoint path.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=None,
        help="Job name for evaluation run. If None, uses default from evaluator config.",
    )
    parser.add_argument(
        "--evaluator-config",
        type=str,
        default="cents/config/evaluator/default.yaml",
        help="Path to evaluator config YAML file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cents/config/config.yaml",
        help="Path to main config YAML file.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA sampling (EMA is used by default when present in checkpoint).",
    )
    parser.add_argument(
        "--eval-pv-shift",
        action="store_true",
        help="Enable PV shift evaluation.",
    )
    parser.add_argument(
        "--no-eval-metrics",
        action="store_true",
        help="Disable evaluation metrics computation.",
    )
    parser.add_argument(
        "--no-eval-context-sparse",
        action="store_true",
        help="Disable sparse context evaluation.",
    )
    parser.add_argument(
        "--no-eval-disentanglement",
        action="store_true",
        help="Disable disentanglement evaluation.",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Disable saving evaluation results.",
    )
    parser.add_argument(
        "--context-config-path",
        type=str,
        default=None,
        help="Path to custom context config YAML file (optional).",
    )
    args = parser.parse_args()

    # Set custom context config path if provided
    if args.context_config_path:
        set_context_config_path(args.context_config_path)

    logging.info("Loading dataset %s...", args.dataset)
    overrides = list(args.dataset_overrides) if args.dataset_overrides else []
    if args.dataset == "pecanstreet" and "time_series_dims" not in str(overrides):
        overrides = overrides + ["time_series_dims=1", "user_group=all"]
    
    dataset = _load_dataset(args.dataset, overrides)

    # Load configs
    eval_cfg = load_yaml(args.evaluator_config)
    top_cfg = load_yaml(args.config)
    
    cfg = OmegaConf.create({})
    cfg.evaluator = eval_cfg
    cfg.wandb = top_cfg.get("wandb", {})
    cfg.device = top_cfg.get("device", "auto")
    cfg.model = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.load(f"cents/config/model/{args.model_type}.yaml"), resolve=True)
    )
    cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))
    
    # Set EMA sampling
    cfg.model.use_ema_sampling = not args.no_ema
    
    # Set evaluation flags (use config defaults if not overridden)
    cfg.eval_pv_shift = args.eval_pv_shift if args.eval_pv_shift else eval_cfg.get("eval_pv_shift", False)
    cfg.eval_metrics = False if args.no_eval_metrics else eval_cfg.get("eval_metrics", True)
    cfg.eval_context_sparse = False if args.no_eval_context_sparse else eval_cfg.get("eval_context_sparse", True)
    cfg.eval_disentanglement = False if args.no_eval_disentanglement else eval_cfg.get("eval_disentanglement", True)
    cfg.save_results = False if args.no_save_results else True
    
    # Set job name
    cfg.job_name = args.job_name if args.job_name else eval_cfg.get("job_name", "default_job")
    
    # Set save directory
    if args.save_dir:
        cfg.save_dir = Path(args.save_dir)
    else:
        # Default: use model checkpoint directory + /eval
        model_ckpt_path = Path(args.model_ckpt)
        cfg.save_dir = model_ckpt_path.parent / "eval"
    
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir, exist_ok=True)
        logging.info("Created evaluation directory: %s", cfg.save_dir)

    logging.info("Setting up DataGenerator (model_type=%s)...", args.model_type)
    gen = DataGenerator(model_type=args.model_type, dataset=dataset)
    
    logging.info("Loading checkpoint... EMA sampling %s", "enabled" if cfg.model.use_ema_sampling else "disabled")
    gen.load_from_checkpoint(args.model_ckpt, args.normalizer_ckpt)
    
    # Ensure EMA setting is applied to the config used by the model at generate time
    target = getattr(gen.model, "cfg", None) or gen.cfg
    if target is not None and hasattr(target, "model"):
        target.model.use_ema_sampling = cfg.model.use_ema_sampling
    
    gen.set_dataset_spec(gen.model.cfg.dataset, dataset.get_context_var_codes())
    cfg.dataset = gen.model.cfg.dataset
    
    logging.info("Checkpoint loaded. Starting evaluation...")
    results = Evaluator(cfg, dataset).evaluate_model(data_generator=gen)
    logging.info("Evaluation complete!")
    print(results)


if __name__ == "__main__":
    main()
