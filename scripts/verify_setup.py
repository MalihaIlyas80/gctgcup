#!/usr/bin/env python3
"""Quick preflight before the expensive Kaggle run."""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import CUPDataset, Vocabulary, collate_fn
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

  vocab_path = os.path.join(args.processed_dir, "vocab.json")
  if not os.path.exists(vocab_path):
    raise SystemExit(f"Missing {vocab_path}")

  vocab = Vocabulary.load(vocab_path)
  samples = []
  with open(os.path.join(args.processed_dir, "train.jsonl"), encoding="utf-8") as f:
    for i, line in enumerate(f):
      if i >= 4:
        break
      samples.append(json.loads(line))

  ds = CUPDataset(samples, vocab, long_threshold=cfg["data"]["long_comment_threshold"])
  batch = collate_fn([ds[i] for i in range(min(2, len(ds)))])

  model = GCTGCUP(
    vocab_size=len(vocab),
    hidden_dim=cfg["model"]["hidden_dim"],
    num_heads=cfg["model"]["num_heads"],
    num_encoder_layers=cfg["model"]["num_encoder_layers"],
    num_decoder_layers=cfg["model"]["num_decoder_layers"],
    ggnn_steps=cfg["model"]["ggnn_steps"],
    dropout=cfg["model"]["dropout"],
    graphcodebert_name=cfg["model"]["graphcodebert"],
    freeze_bert=cfg["model"]["freeze_graphcodebert"],
    long_threshold=cfg["data"]["long_comment_threshold"],
    edit_weight=cfg["model"].get("edit_weight", 8.0),
  ).to(device)

  for k, v in batch.items():
    if isinstance(v, torch.Tensor):
      batch[k] = v.to(device)

  out = model(batch)
  print(f"Forward OK | loss={out['loss'].item():.4f}")

  src_tok = [" ".join(t) for t in batch["src_tokens_list"]]
  gen_ids, no_upd, _, surf, _ = model.generate(
    batch["src_ids"], batch["edit_ids"],
    batch["src_methods"], batch["dst_methods"],
    batch["graphs"],
    beam_size=1, force_update=True,
    comments=batch["src_descs"], src_descs=src_tok,
    src_tokens_list=batch["src_tokens_list"],
    code_change_seqs=batch["code_change_seqs"],
    id2token=vocab.id2token,
    return_beam_candidates=True,
  )
  pred = surf[0] if surf[0] else (" ".join(vocab.decode(gen_ids[0])) if gen_ids[0] else no_upd[0])
  ref = " ".join(batch["dst_tokens_list"][0])
  print(f"Generate OK | pred={pred[:80]!r}")
  print(f"Reference   | ref ={ref[:80]!r}")
  if not pred.strip():
    raise SystemExit("Empty prediction — abort.")
  print("\n=== ALL PREFLIGHT CHECKS PASSED ===")


if __name__ == "__main__":
  main()
