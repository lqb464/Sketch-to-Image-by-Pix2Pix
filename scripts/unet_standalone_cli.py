"""CLI for running the standalone U-Net sketch-to-image model from YAML config."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "notebook_unet_standalone.yaml"

COMMON_KEYS = {
    "dataroot",
    "name",
    "checkpoints_dir",
    "model",
    "input_nc",
    "output_nc",
    "ngf",
    "ndf",
    "netD",
    "netG",
    "n_layers_D",
    "norm",
    "freeze_encoder",
    "init_type",
    "init_gain",
    "no_dropout",
    "dataset_mode",
    "direction",
    "serial_batches",
    "num_threads",
    "batch_size",
    "load_size",
    "crop_size",
    "max_dataset_size",
    "preprocess",
    "no_flip",
    "display_winsize",
    "epoch",
    "load_iter",
    "verbose",
    "suffix",
    "use_wandb",
    "wandb_project_name",
    "wandb_project",
    "wandb_entity",
    "loss",
    "lambda_L1",
    "seed",
}

TRAIN_KEYS = COMMON_KEYS | {
    "display_freq",
    "update_html_freq",
    "print_freq",
    "no_html",
    "save_latest_freq",
    "save_epoch_freq",
    "save_by_iter",
    "continue_train",
    "epoch_count",
    "phase",
    "n_epochs",
    "n_epochs_decay",
    "beta1",
    "lr",
    "gan_mode",
    "pool_size",
    "lr_policy",
    "lr_decay_iters",
    "eval_test_freq",
    "eval_phase",
    "num_test",
    "eval_batch_size",
    "eval_num_threads",
    "eval_flip",
    "eval_num_fid_images",
    "eval_kid_subsets",
    "eval_kid_subset_size",
    "no_eval_test_images",
    "eval_max_dataset_size",
}

TEST_KEYS = COMMON_KEYS | {
    "results_dir",
    "aspect_ratio",
    "phase",
    "eval",
    "num_test",
}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _parse_scalar(value: str) -> Any:
    parsed = yaml.safe_load(value)
    return parsed


def _apply_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must use KEY=VALUE format, got: {item}")
        key, value = item.split("=", 1)
        config[key.strip()] = _parse_scalar(value)


def _append_arg(argv: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    flag = f"--{key}"
    if isinstance(value, bool):
        if value:
            argv.append(flag)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            argv.extend([flag, str(item)])
        return
    argv.extend([flag, str(value)])


def _build_command(mode: str, config: dict[str, Any], extra_args: list[str]) -> list[str]:
    config = dict(config)
    config["model"] = "unet_standalone"

    script = ROOT / ("train.py" if mode == "train" else "test.py")
    allowed = TRAIN_KEYS if mode == "train" else TEST_KEYS
    argv = [sys.executable, str(script)]

    ignored = sorted(key for key in config if key not in allowed)
    if ignored:
        print(f"[unet-cli] Ignoring unsupported {mode} config keys: {', '.join(ignored)}")

    for key in sorted(allowed):
        if key in config:
            _append_arg(argv, key, config[key])
    argv.extend(extra_args)
    return argv


def main() -> int:
    parser = argparse.ArgumentParser(description="Run standalone U-Net train/test from YAML config.")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "test"):
        subparser = subparsers.add_parser(mode, help=f"run {mode}.py")
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
        subparser.add_argument("--set", dest="overrides", action="append", default=[], help="override config value, e.g. --set dataroot=./datasets/cufs")
        subparser.add_argument("--dry-run", action="store_true", help="print the generated command without running it")
        subparser.add_argument("extra_args", nargs=argparse.REMAINDER, help="extra raw args forwarded after a -- separator")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = _load_config(config_path)
    _apply_overrides(config, args.overrides)

    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    command = _build_command(args.mode, config, extra_args)
    print("[unet-cli] " + " ".join(command))
    if args.dry_run:
        return 0
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
