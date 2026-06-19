#!/usr/bin/env python3
"""Smoke test: 50 samples, 2 epochs – verifies full pipeline."""
import os
import subprocess
import sys

py = sys.executable
root = os.path.dirname(os.path.dirname(__file__))
os.chdir(root)

subprocess.check_call([py, "scripts/prepare_data.py", "--max-samples", "50"])
subprocess.check_call([py, "scripts/train.py", "--epochs", "2"])
subprocess.check_call([py, "scripts/evaluate.py"])
print("\nQuick test PASSED.")
