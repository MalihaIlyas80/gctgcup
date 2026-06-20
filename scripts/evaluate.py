#!/usr/bin/env python3
"""Evaluate GC-TGCUP vs TG-CUP baseline metrics."""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import CUPDataset, Vocabulary, collate_fn
from src.evaluation.metrics import compute_all_metrics
from src.models.gctgcup import GCTGCUP


def load_jsonl(path):
  samples = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      samples.append(json.loads(line))
  return samples


@torch.no_grad()
def evaluate_model(model, loader, vocab, device):
  model.eval()
  predictions, references, sources = [], [], []
  det_preds, det_labels = [], []
  is_nciu, is_long = [], []
  beam_cands = []

  for batch in loader:
    for k, v in batch.items():
      if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

    det_logits = model.detect(batch["src_methods"], batch["dst_methods"], batch["src_descs"])
    preds = (torch.sigmoid(det_logits) >= 0.45).long().cpu().tolist()
    det_preds.extend(preds)
    det_labels.extend(batch["labels"].long().cpu().tolist())

    gen_ids, no_upd_texts, beam_ids = model.generate(
      batch["src_ids"], batch["edit_ids"],
      batch["src_methods"], batch["dst_methods"],
      batch["graphs"],
      max_len=50, beam_size=5,
      comments=batch["src_descs"],
      src_descs=batch["src_descs"],
      return_beam_candidates=True,
    )
    for ids, no_upd, cands, ref, src in zip(
      gen_ids, no_upd_texts, beam_ids, batch["dst_descs"], batch["src_descs"]
    ):
      if no_upd is not None:
        # No update predicted: use original comment directly (exact match preserved)
        pred_text = no_upd
        beam_cands.append([no_upd] * 5)
      else:
        pred_text = " ".join(vocab.decode(ids))
        beam_cands.append([" ".join(vocab.decode(c)) for c in cands])
      predictions.append(pred_text)
      references.append(ref)
      sources.append(src)

    is_nciu.extend(batch["is_nciu"].cpu().tolist())
    is_long.extend(batch["is_long"].cpu().tolist())

  return compute_all_metrics(
    predictions, references, sources,
    beam_candidates=beam_cands,
    det_preds=det_preds, det_labels=det_labels,
    is_nciu=is_nciu, is_long=is_long,
  )


def print_comparison(metrics, baseline):
  print("\n" + "=" * 70)
  print(f"{'Metric':<25} {'GC-TGCUP':>12} {'TG-CUP':>12} {'Delta':>12}")
  print("-" * 70)
  mapping = {
    "accuracy": "accuracy",
    "recall_at_5": "recall_at_5",
    "gleu": "gleu",
    "meteor": "meteor",
    "sari": "sari",
    "bleu": "bleu",
  }
  for key, bkey in mapping.items():
    ours = getattr(metrics, key)
    base = baseline.get(bkey, 0)
    delta = ours - base
    sign = "+" if delta >= 0 else ""
    print(f"{key:<25} {ours:>11.2f}% {base:>11.2f}% {sign}{delta:>10.2f}%")

  print("-" * 70)
  print(f"{'det_accuracy':<25} {metrics.det_accuracy:>11.2f}%")
  print(f"{'det_precision':<25} {metrics.det_precision:>11.2f}%")
  print(f"{'det_recall':<25} {metrics.det_recall:>11.2f}%")
  print(f"{'det_f1':<25} {metrics.det_f1:>11.2f}%")
  print(f"{'nciu_accuracy':<25} {metrics.nciu_accuracy:>11.2f}%")
  print(f"{'long_comment_accuracy':<25} {metrics.long_comment_accuracy:>11.2f}%")
  print("=" * 70)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/default.yaml")
  parser.add_argument("--processed-dir", default="data/processed")
  parser.add_argument("--checkpoint", default="checkpoints/best.pt")
  args = parser.parse_args()

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  vocab = Vocabulary.load(os.path.join(args.processed_dir, "vocab.json"))
  test_ds = CUPDataset(load_jsonl(os.path.join(args.processed_dir, "test.jsonl")), vocab)
  test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                           shuffle=False, collate_fn=collate_fn)

  model = GCTGCUP(
    vocab_size=len(vocab),
    hidden_dim=cfg["model"]["hidden_dim"],
    num_heads=cfg["model"]["num_heads"],
    num_encoder_layers=cfg["model"]["num_encoder_layers"],
    num_decoder_layers=cfg["model"]["num_decoder_layers"],
    ggnn_steps=cfg["model"]["ggnn_steps"],
    dropout=cfg["model"]["dropout"],
    graphcodebert_name=cfg["model"]["graphcodebert"],
    long_threshold=cfg["data"]["long_comment_threshold"],
  ).to(device)

  if os.path.exists(args.checkpoint):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint}")
  else:
    print("WARNING: No checkpoint found – evaluating untrained model.")

  metrics = evaluate_model(model, test_loader, vocab, device)
  print_comparison(metrics, cfg["evaluation"]["tgcup_baseline"])

  report = metrics.to_dict()
  report["vs_tgcup"] = {
    k: round(getattr(metrics, k) - cfg["evaluation"]["tgcup_baseline"].get(k, 0), 2)
    for k in ["accuracy", "recall_at_5", "gleu", "meteor", "sari", "bleu"]
  }
  out_path = os.path.join(cfg["training"]["checkpoint_dir"], "evaluation_report.json")
  with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
  main()
