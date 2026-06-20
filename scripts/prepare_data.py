#!/usr/bin/env python3
"""Prepare cleaned dataset for GC-TGCUP."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import prepare_datasets


def main():
  parser = argparse.ArgumentParser(description="Clean and split CUP2 dataset")
  parser.add_argument("--raw-dir", default="cup2_dataset")
  parser.add_argument("--output-dir", default="data/processed")
  parser.add_argument("--max-samples", type=int, default=5000)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  print(f"Preparing {args.max_samples} samples from {args.raw_dir} ...")
  train_ds, valid_ds, test_ds, vocab = prepare_datasets(
    raw_dir=args.raw_dir,
    processed_dir=args.output_dir,
    max_samples=args.max_samples,
    seed=args.seed,
  )

  stats_path = os.path.join(args.output_dir, "stats.json")
  with open(stats_path) as f:
    stats = json.load(f)

  print("\n=== Data Preparation Complete ===")
  print(f"  Train : {len(train_ds)}")
  print(f"  Valid : {len(valid_ds)}")
  print(f"  Test  : {len(test_ds)}")
  print(f"  Vocab : {len(vocab)} tokens")
  print(f"  Positive (outdated) : {stats.get('positive', '?')}")
  print(f"  Negative (no update): {stats.get('negative', '?')}")
  print(f"  NCIU samples        : {stats.get('nciu', '?')}")
  print(f"  Output dir          : {args.output_dir}")


if __name__ == "__main__":
  main()
