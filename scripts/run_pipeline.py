#!/usr/bin/env python3
"""
One-shot Kaggle pipeline: prepare -> verify -> train -> evaluate vs TG-CUP.

  !python scripts/run_pipeline.py --config configs/kaggle_gpu.yaml --fresh
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _root() -> str:
  return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd: list[str]) -> None:
  print("\n" + "=" * 70)
  print(">>>", " ".join(cmd))
  print("=" * 70)
  subprocess.check_call(cmd, cwd=_root())


def main() -> None:
  parser = argparse.ArgumentParser(description="Prepare + verify + train + evaluate")
  parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
  parser.add_argument("--raw-dir", default="cup2_dataset")
  parser.add_argument("--processed-dir", default="data/processed")
  parser.add_argument("--max-samples", type=int, default=None)
  parser.add_argument("--fresh", action="store_true",
                      help="Delete processed data + checkpoints first")
  parser.add_argument("--skip-prepare", action="store_true")
  parser.add_argument("--skip-verify", action="store_true")
  parser.add_argument("--skip-train", action="store_true")
  parser.add_argument("--eval-beam-size", type=int, default=5)
  parser.add_argument(
    "--start-phase",
    choices=("both", "detection", "update"),
    default="both",
    help="Pass update to skip stage-1 if checkpoints/best.pt exists",
  )
  args = parser.parse_args()

  os.chdir(_root())

  import yaml
  with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
  max_samples = args.max_samples or cfg["data"]["max_samples"]

  if args.fresh:
    for path in (args.processed_dir, cfg["training"]["checkpoint_dir"]):
      if os.path.isdir(path):
        print(f"Removing {path} ...")
        shutil.rmtree(path)

  if not args.skip_prepare:
    vocab_max = str(cfg["data"].get("vocab_max_size", 30000))
    run([
      sys.executable, "scripts/prepare_data.py",
      "--raw-dir", args.raw_dir,
      "--output-dir", args.processed_dir,
      "--max-samples", str(max_samples),
      "--vocab-max-size", vocab_max,
    ])

  if not args.skip_verify:
    run([
      sys.executable, "scripts/verify_setup.py",
      "--config", args.config,
      "--processed-dir", args.processed_dir,
    ])

  if not args.skip_train:
    train_cmd = [
      sys.executable, "scripts/train.py",
      "--config", args.config,
      "--processed-dir", args.processed_dir,
      "--start-phase", args.start_phase,
    ]
    run(train_cmd)

  eval_beam = args.eval_beam_size or cfg["model"].get("beam_size", 5)
  run([
    sys.executable, "scripts/evaluate.py",
    "--config", args.config,
    "--processed-dir", args.processed_dir,
    "--checkpoint", os.path.join(cfg["training"]["checkpoint_dir"], "best.pt"),
    "--beam-size", str(eval_beam),
  ])


if __name__ == "__main__":
  main()
