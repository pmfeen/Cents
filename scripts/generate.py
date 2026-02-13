#!/usr/bin/env python3
"""
Generate synthetic time series samples from a trained model.

Supports:
  - Random context: sample context from the dataset's support (including continuous).
  - Explicit context: provide context as JSON (categorical: int codes; continuous: z-scored floats).
  - Output to Parquet (default) or CSV.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from cents.data_generator import DataGenerator
from cents.datasets.pecanstreet import PecanStreetDataset
from cents.datasets.commercial import CommercialDataset
from cents.datasets.airquality import AirQualityDataset
from cents.datasets.utils import convert_generated_data_to_df
from cents.utils.config_loader import load_yaml
from cents.utils.utils import set_context_config_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
DATASET_OVERRIDES = ["max_samples=10000", "skip_heavy_processing=True"]
PECAN_OVERRIDES = ["time_series_dims=1", "user_group=all"]


def _load_dataset(name: str, overrides: list):
    if name == "pecanstreet":
        return PecanStreetDataset(overrides=DATASET_OVERRIDES + PECAN_OVERRIDES + (overrides or []))
    if name == "commercial":
        return CommercialDataset(overrides=DATASET_OVERRIDES + (overrides or []))
    if name == "airquality":
        return AirQualityDataset(overrides=DATASET_OVERRIDES + (overrides or []))
    raise ValueError(f"Dataset {name} not supported. Use: pecanstreet, commercial, airquality.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic time series from a trained model.",
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
        help="Path to normalizer checkpoint. If omitted, output stays in normalized space.",
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
        "-n",
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default="samples.parquet",
        help="Output path for generated samples.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format. Parquet preserves array columns better.",
    )
    parser.add_argument(
        "--random-context",
        action="store_true",
        help="Sample context randomly from the dataset support (categorical and continuous).",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help='Explicit context as JSON, e.g. \'{"weekday":0,"month":3}\'. '
        "Categorical: int codes. Continuous: z-scored (normalized) floats.",
    )
    parser.add_argument(
        "--context-config-path",
        type=str,
        default=None,
        help="Path to custom context config YAML (optional).",
    )
    parser.add_argument(
        "--dataset-overrides",
        type=str,
        nargs="*",
        default=[],
        help="Extra dataset overrides, e.g. time_series_dims=1.",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Disable EMA sampling (EMA is used by default when present in the checkpoint).",
    )
    args = parser.parse_args()

    use_random = args.random_context
    use_explicit = args.context is not None and args.context.strip() != ""
    if not use_random and not use_explicit:
        parser.error("Provide either --random-context or --context (JSON).")
    if use_random and use_explicit:
        parser.error("Provide only one of --random-context or --context.")

    if args.context_config_path:
        set_context_config_path(args.context_config_path)

    overrides = list(args.dataset_overrides) if args.dataset_overrides else []

    logging.info("Loading dataset %s...", args.dataset)
    dataset = _load_dataset(args.dataset, overrides)
    cfg = OmegaConf.create({})
    cfg.model = load_yaml(Path("cents/config/model") / f"{args.model_type}.yaml")
    cfg.dataset = OmegaConf.create(OmegaConf.to_container(dataset.cfg, resolve=True))
    cfg.model.use_ema_sampling = not args.no_ema

    logging.info("Setting up DataGenerator (model_type=%s)...", args.model_type)
    gen = DataGenerator(model_type=args.model_type, dataset=dataset, cfg=cfg)
    gen.load_from_checkpoint(args.model_ckpt, args.normalizer_ckpt)
    # Ensure EMA setting is applied to the config used by the model at generate time
    target = getattr(gen.model, "cfg", None) or gen.cfg
    if target is not None and hasattr(target, "model"):
        target.model.use_ema_sampling = cfg.model.use_ema_sampling
    gen.set_dataset_spec(gen.model.cfg.dataset, dataset.get_context_var_codes())

    if use_random:
        # Sample a new random context for each of the n samples
        contexts = [dataset.sample_random_context_vars() for _ in range(args.num_samples)]
        ctx_batch = {
            k: torch.stack([c[k] for c in contexts]).to(gen.device)
            for k in contexts[0].keys()
        }
        logging.info("Generating %d samples with %d random contexts...", args.num_samples, args.num_samples)
        with torch.no_grad():
            ts = gen.model.generate(ctx_batch)
        df = convert_generated_data_to_df(ts, ctx_batch, decode=False)
    else:
        context_dict = json.loads(args.context)
        for k, v in context_dict.items():
            if isinstance(v, float):
                context_dict[k] = v
            else:
                context_dict[k] = int(v)
        gen.set_context(**context_dict)
        logging.info("Generating %d samples with context %s...", args.num_samples, context_dict)
        df = gen.generate(args.num_samples)

    if gen.normalizer is not None:
        df = gen.normalizer.inverse_transform(df)
        logging.info("Inverse-normalized outputs to original scale.")
    else:
        logging.warning("No normalizer loaded; outputs are in normalized space.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)
    logging.info("Wrote %d samples to %s", len(df), out.resolve())


if __name__ == "__main__":
    main()
