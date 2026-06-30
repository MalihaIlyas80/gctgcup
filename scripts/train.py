#!/usr/bin/env python3
"""Train GC-TGCUP two-stage comment updating model."""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import CUPDataset, Vocabulary, collate_fn
from src.models.gctgcup import GCTGCUP
from src.training.trainer import Trainer


def load_jsonl(path):
  samples = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      samples.append(json.loads(line))
  return samples


def build_optimizer(model, cfg):
  bert_params, other_params = [], []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if "bert" in name or "code_encoder" in name:
      bert_params.append(p)
    else:
      other_params.append(p)
  return torch.optim.AdamW([
    {"params": other_params, "lr": cfg["training"]["learning_rate"]},
    {"params": bert_params,  "lr": cfg["training"].get("graphcodebert_lr", 1e-5)},
  ], weight_decay=cfg["training"]["weight_decay"])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
  parser.add_argument("--processed-dir", default="data/processed")
  parser.add_argument("--epochs", type=int, default=None)
  args = parser.parse_args()

  with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")

  vocab = Vocabulary.load(os.path.join(args.processed_dir, "vocab.json"))
  train_ds = CUPDataset(
    load_jsonl(os.path.join(args.processed_dir, "train.jsonl")), vocab,
    long_threshold=cfg["data"]["long_comment_threshold"],
  )
  valid_ds = CUPDataset(
    load_jsonl(os.path.join(args.processed_dir, "valid.jsonl")), vocab,
    long_threshold=cfg["data"]["long_comment_threshold"],
  )

  num_workers = min(4, (os.cpu_count() or 1) // 2)
  train_loader = DataLoader(
    train_ds, batch_size=cfg["training"]["batch_size"],
    shuffle=True, collate_fn=collate_fn, num_workers=num_workers,
    persistent_workers=num_workers > 0,
  )
  valid_loader = DataLoader(
    valid_ds, batch_size=cfg["training"]["batch_size"],
    shuffle=False, collate_fn=collate_fn, num_workers=num_workers,
    persistent_workers=num_workers > 0,
  )

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
    det_loss_weight=cfg["model"].get("det_loss_weight", 0.15),
    upd_loss_weight=cfg["model"].get("upd_loss_weight", 0.85),
  )

  optimizer = build_optimizer(model, cfg)
  pos_weight = None
  if "pos_weight" in cfg["training"]:
    pos_weight = torch.tensor(cfg["training"]["pos_weight"], dtype=torch.float)
    print(f"Detection pos_weight: {pos_weight.item():.2f}")

  trainer = Trainer(
    model, train_loader, valid_loader, optimizer, device,
    checkpoint_dir=cfg["training"]["checkpoint_dir"],
    patience=cfg["training"]["patience"],
    vocab=vocab,
    max_valid_batches=cfg["training"].get("max_valid_batches", 12),
    pos_weight=pos_weight,
    grad_accumulation_steps=cfg["training"].get("grad_accumulation_steps", 4),
    det_threshold=cfg["model"].get("det_threshold", 0.5),
  )

  epochs = args.epochs or cfg["training"]["update_epochs"]
  print(f"\nTraining GC-TGCUP for {epochs} epochs ...")
  result = trainer.fit(epochs)

  report_path = os.path.join(cfg["training"]["checkpoint_dir"], "training_history.json")
  with open(report_path, "w") as f:
    json.dump(result, f, indent=2)
  print(f"\nTraining complete. Best score: {result.get('best_val_score', 0):.2f}")
  print(f"History saved to {report_path}")


if __name__ == "__main__":
  main()
