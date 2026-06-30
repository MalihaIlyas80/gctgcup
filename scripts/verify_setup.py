#!/usr/bin/env python3
"""2-minute preflight check before the expensive Kaggle GPU run."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import collate_fn, CUPDataset
from src.models.gctgcup import GCTGCUP


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
  parser.add_argument("--processed-dir", default="data/processed")
  args = parser.parse_args()

  with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  train_path = os.path.join(args.processed_dir, "train.jsonl")
  if not os.path.exists(train_path):
    raise SystemExit(f"Missing {train_path} — run prepare_data first.")

  samples = []
  with open(train_path, encoding="utf-8") as f:
    for i, line in enumerate(f):
      if i >= 4:
        break
      samples.append(json.loads(line))

  ds = CUPDataset(samples, long_threshold=cfg["data"]["long_comment_threshold"])
  batch = collate_fn([ds[i] for i in range(min(2, len(ds)))])

  model = GCTGCUP(
    hidden_dim=cfg["model"]["hidden_dim"],
    dropout=cfg["model"]["dropout"],
    graphcodebert_name=cfg["model"]["graphcodebert"],
    freeze_bert=cfg["model"]["freeze_graphcodebert"],
    update_model_name=cfg["model"]["update_model"],
    max_src_len=cfg["model"]["max_src_len"],
    max_tgt_len=cfg["model"]["max_tgt_len"],
    max_edit_chars=cfg["model"].get("max_edit_chars", 350),
    max_ast_chars=cfg["model"].get("max_ast_chars", 180),
    det_loss_weight=cfg["model"].get("det_loss_weight", 0.10),
    upd_loss_weight=cfg["model"].get("upd_loss_weight", 0.90),
  ).to(device)

  for k, v in batch.items():
    if isinstance(v, torch.Tensor):
      batch[k] = v.to(device)

  model.train()
  out = model(batch)
  loss = out["loss"].item()
  if not (0 < loss < 100):
    raise SystemExit(f"Suspicious loss={loss} — abort before full train.")
  print(f"Forward OK | loss={loss:.4f}")

  model.eval()
  with torch.no_grad():
    preds, beams = model.generate(
      batch["src_methods"][:1],
      batch["dst_methods"][:1],
      batch["src_descs"][:1],
      batch["edit_texts"][:1],
      batch["ast_texts"][:1],
      beam_size=2,
      force_update=True,
      max_len=cfg["model"]["max_decode_len"],
    )
  print(f"Generate OK | pred={preds[0]!r}")
  print(f"Reference   | ref ={batch['dst_descs'][0]!r}")
  if not preds[0].strip():
    raise SystemExit("Empty prediction — abort before full train.")

  print("\n=== ALL PREFLIGHT CHECKS PASSED ===")


if __name__ == "__main__":
  main()
