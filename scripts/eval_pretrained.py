import logging
import os
from pathlib import Path
from typing import Tuple
import json

import torch
import torch.nn.functional as F

from omegaconf import OmegaConf
import argparse

from cents.data_generator import DataGenerator
from cents.models.registry import get_model_type_from_hf_name
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.eval.eval import Evaluator
from cents.utils.config_loader import load_yaml, apply_overrides
from cents.utils.utils import set_context_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
DATASET_OVERRIDES = ["normalize=False"]
PECAN_OVERRIDES = ["time_series_dims=1", "user_group=all"]

CONFIG_DATASET_DIR = Path(__file__).resolve().parent.parent / "cents" / "config" / "dataset"


def _load_dataset_config(dataset_name: str, overrides: list) -> OmegaConf:
    """Load dataset-specific config from config/dataset/{dataset_name}.yaml and apply overrides."""
    config_path = CONFIG_DATASET_DIR / f"{dataset_name}.yaml"
    if not config_path.exists():
        raise ValueError(
            f"Dataset config not found for '{dataset_name}' at {config_path}. "
            f"Available: {[p.name for p in CONFIG_DATASET_DIR.glob('*.yaml')]}"
        )
    cfg = load_yaml(str(config_path))
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def _infer_dataset_shape_from_ckpt(
    ckpt_path: str, cond_emb_dim: int
) -> Tuple[int, int]:
    """
    Infer seq_len and time_series_dims from a Diffusion_TS checkpoint state_dict
    so the model can be built with the same architecture as when the checkpoint was saved.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Keys may be "model.pos_enc.pe" (Lightning) or "pos_enc.pe" (raw)
    for pe_key in ("model.pos_enc.pe", "pos_enc.pe"):
        if pe_key in state_dict:
            # shape (1, seq_len, d_model)
            seq_len = int(state_dict[pe_key].shape[1])
            break
    else:
        raise ValueError(
            "Could not infer seq_len from checkpoint (no pos_enc.pe key in state_dict)"
        )
    # combine_s: Conv1d(n_embd, n_feat, ...) -> weight shape (n_feat, n_embd, k)
    # n_feat = time_series_dims + cond_emb_dim
    for cs_key in ("model.combine_s.weight", "combine_s.weight"):
        if cs_key in state_dict:
            n_feat = int(state_dict[cs_key].shape[0])
            time_series_dims = n_feat - cond_emb_dim
            if time_series_dims < 1:
                time_series_dims = 1
            break
    else:
        raise ValueError(
            "Could not infer time_series_dims from checkpoint (no combine_s.weight in state_dict)"
        )
    return seq_len, time_series_dims


def _load_dataset(name: str, dataset_cfg: OmegaConf):
    """Load a dataset by name using dataset-specific config (from config/dataset/{name}.yaml)."""
    if name == "pecanstreet":
        return PecanStreetDataset(cfg=dataset_cfg)
    if name == "commercial":
        return CommercialDataset(cfg=dataset_cfg)
    if name == "airquality":
        return AirQualityDataset(cfg=dataset_cfg)
    raise ValueError(f"Dataset {name} not supported. Use: pecanstreet, commercial, airquality.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model using comprehensive metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint (.ckpt or .pt). Required unless --model-key is set.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="HuggingFace model key (e.g. Watts_2_1D). If set, model and normalizer are loaded from HF instead of --model-ckpt.",
    )
    parser.add_argument(
        "--normalizer-ckpt",
        type=str,
        default=None,
        help="Path to normalizer checkpoint. If omitted, evaluation uses normalized space (or HF normalizer when using --model-key).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type (e.g. diffusion_ts) used to load the checkpoint. Inferred from --model-key when loading from HF.",
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
        help="Directory to save evaluation results. If None, uses checkpoint parent + /eval or outputs/eval/<model-key> when using --model-key.",
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
        "--ema",
        action="store_true",
        help="Enable EMA sampling.",
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

    if not args.model_ckpt and not args.model_key:
        parser.error("One of --model-ckpt or --model-key is required.")
    if args.model_ckpt and args.model_key:
        parser.error("Use only one of --model-ckpt or --model-key.")

    # Set custom context config path if provided
    if args.context_config_path:
        set_context_config_path(args.context_config_path)

    logging.info("Loading dataset %s...", args.dataset)
    overrides = list(DATASET_OVERRIDES)
    if args.dataset == "pecanstreet":
        overrides = overrides + PECAN_OVERRIDES
    # Use pretrained normalizer from checkpoint/HF: skip dataset normalizer init so it doesn't train
    if args.model_key:
        overrides = overrides + ["normalize=False"]
    # Watts (and most pretrained) normalizers use scale=True (do_scale); match so stats_head shape loads
    if args.model_key:
        overrides = overrides + ["scale=True"]
    if args.dataset_overrides:
        overrides = overrides + list(args.dataset_overrides)
    dataset_cfg = _load_dataset_config(args.dataset, overrides)
    dataset = _load_dataset(args.dataset, dataset_cfg)

    # Resolve model type (from key when loading from HF, else from args)
    if args.model_key:
        model_type = get_model_type_from_hf_name(args.model_key)
    else:
        model_type = args.model_type or "diffusion_ts"

    # Load configs
    eval_cfg = load_yaml(args.evaluator_config)
    top_cfg = load_yaml(args.config)

    cfg = OmegaConf.create({})
    cfg.evaluator = eval_cfg
    cfg.wandb = top_cfg.get("wandb", {})
    cfg.device = "cuda:2"
    cfg.model = OmegaConf.create(
        OmegaConf.to_container(OmegaConf.load(f"cents/config/model/{model_type}.yaml"), resolve=True)
    )
    cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))

    # print("EVAL CONFIG:")
    # print(cfg)

    # When loading from a local checkpoint, infer seq_len and time_series_dims from the
    # checkpoint so the model is built with the same architecture (avoids shape mismatch).
    if args.model_ckpt and Path(args.model_ckpt).suffix == ".ckpt":
        try:
            ckpt_seq_len, ckpt_time_series_dims = _infer_dataset_shape_from_ckpt(
                args.model_ckpt, cond_emb_dim=int(cfg.model.cond_emb_dim)
            )
            if ckpt_seq_len != cfg.dataset.seq_len or ckpt_time_series_dims != cfg.dataset.time_series_dims:
                logging.info(
                    "Checkpoint has seq_len=%s, time_series_dims=%s; overriding dataset config to match.",
                    ckpt_seq_len, ckpt_time_series_dims,
                )
                cfg.dataset.seq_len = ckpt_seq_len
                cfg.dataset.time_series_dims = ckpt_time_series_dims
                dataset.cfg.seq_len = ckpt_seq_len
                dataset.cfg.time_series_dims = ckpt_time_series_dims
                dataset.seq_len = ckpt_seq_len
                dataset.time_series_dims = ckpt_time_series_dims
        except (KeyError, ValueError) as e:
            logging.warning(
                "Could not infer dataset shape from checkpoint (%s). Using eval dataset config; shape mismatch may occur.",
                e,
            )

    # Set EMA sampling
    cfg.model.use_ema_sampling = args.ema

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
    elif args.model_key:
        cfg.save_dir = Path("outputs/eval") / args.model_key
    else:
        model_ckpt_path = Path(args.model_ckpt)
        cfg.save_dir = model_ckpt_path.parent / "eval"

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir, exist_ok=True)
        logging.info("Created evaluation directory: %s", cfg.save_dir)

    use_hf = args.model_key is not None
    if use_hf:
        logging.info("Setting up DataGenerator from HuggingFace (model_key=%s)...", args.model_key)
        gen = DataGenerator(model_name=args.model_key, dataset=dataset, cfg=cfg)
    else:
        logging.info("Setting up DataGenerator (model_type=%s)...", model_type)
        gen = DataGenerator(model_type=model_type, dataset=dataset, cfg=cfg)
        logging.info("Loading checkpoint... EMA sampling %s", "enabled" if cfg.model.use_ema_sampling else "disabled")
        gen.load_from_checkpoint(args.model_ckpt, args.normalizer_ckpt)

    # Ensure EMA setting is applied to the config used by the model at generate time
    target = getattr(gen.model, "cfg", None) or gen.cfg
    if target is not None and hasattr(target, "model"):
        target.model.use_ema_sampling = cfg.model.use_ema_sampling

    # gen.set_dataset_spec(gen.model.cfg.dataset, dataset.get_context_var_codes())
    cfg.dataset = gen.model.cfg.dataset
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60 + "\n")
    results = Evaluator(cfg, dataset).evaluate_model(data_generator=gen)

    print("\n📊 METRICS:")
    print("-" * 60)
    metrics = results.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subval in value.items():
                print(f"  {subkey}: {subval:.6f}" if isinstance(subval, (int, float)) else f"  {subkey}: {subval}")
        else:
            print(f"{key}: {value:.6f}" if isinstance(value, (int, float)) else f"{key}: {value}")
    
    # Results are automatically saved if save_results=True
    if args.save_dir:
        with open(Path(args.save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n✅ Results saved to: {Path(args.save_dir) / "metrics.json"}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()