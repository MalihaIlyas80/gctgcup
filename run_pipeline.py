#!/usr/bin/env python3
"""
End-to-end pipeline: prepare data → train → evaluate.
Usage: python run_pipeline.py
"""
import subprocess
import sys


def run(cmd):
  print(f"\n>>> {' '.join(cmd)}\n")
  result = subprocess.run(cmd, check=False)
  if result.returncode != 0:
    sys.exit(result.returncode)


if __name__ == "__main__":
  py = sys.executable
  run([py, "scripts/prepare_data.py", "--max-samples", "1000"])
  run([py, "scripts/train.py", "--epochs", "5"])
  run([py, "scripts/evaluate.py"])
