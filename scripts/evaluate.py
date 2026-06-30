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
def evaluate_model(model, loader, vocab, device, det_threshold=0.5, beam_size=5,
                   max_len=50, max_batches=None):
  model.eval()
  predictions, references, sources = [], [], []
  det_preds, det_labels = [], []
  is_nciu, is_long, outdated = [], [], []
  beam_cands = []

  for bi, batch in enumerate(loader):
    if max_batches is not None and bi >= max_batches:
      break
    for k, v in batch.items():
      if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

    det_logits = model.detect(batch["src_methods"], batch["dst_methods"], batch["src_descs"])
    preds = (torch.sigmoid(det_logits) >= det_threshold).long().cpu().tolist()
    det_preds.extend(preds)
    det_labels.extend(batch["labels"].long().cpu().tolist())

    src_tok_texts = [" ".join(t) for t in batch["src_tokens_list"]]
    ref_tok_texts = [" ".join(t) for t in batch["dst_tokens_list"]]

    gen_ids, no_upd_texts, beam_ids, surface_texts, beam_surfaces = model.generate(
      batch["src_ids"], batch["edit_ids"],
      batch["src_methods"], batch["dst_methods"],
      batch["graphs"],
      max_len=max_len, beam_size=beam_size,
      det_threshold=det_threshold,
      comments=batch["src_descs"],
      src_descs=src_tok_texts,
      src_tokens_list=batch["src_tokens_list"],
      id2token=vocab.id2token,
      return_beam_candidates=True,
      force_update=True,
    )
    for ids, no_upd, cands, surf, surf_cands, ref, src in zip(
      gen_ids, no_upd_texts, beam_ids, surface_texts, beam_surfaces,
      ref_tok_texts, src_tok_texts,
    ):
      if no_upd is not None:
        pred_text = no_upd
        beam_cands.append([no_upd] * beam_size)
      else:
        pred_text = surf or " ".join(vocab.decode(ids))
        beam_cands.append(surf_cands if surf_cands else [" ".join(vocab.decode(c)) for c in cands])
      predictions.append(pred_text)
      references.append(ref)
      sources.append(src)

    is_nciu.extend(batch["is_nciu"].cpu().tolist())
    is_long.extend(batch["is_long"].cpu().tolist())
    outdated.extend(batch["labels"].long().cpu().tolist())

  return compute_all_metrics(
    predictions, references, sources,
    beam_candidates=beam_cands,
    det_preds=det_preds, det_labels=det_labels,
    is_nciu=is_nciu, is_long=is_long,
    outdated=outdated,
  )


def print_comparison(metrics, baseline):
  print("\n" + "=" * 70)
  print(f" UPDATE STAGE  |  outdated-only subset (N={metrics.num_outdated})  |  vs TG-CUP")
  print("=" * 70)
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
  wins = 0
  for key, bkey in mapping.items():
    ours = getattr(metrics, key)
    base = baseline.get(bkey, 0)
    delta = ours - base
    sign = "+" if delta >= 0 else ""
    wins += int(delta >= 0)
    print(f"{key:<25} {ours:>11.2f}% {base:>11.2f}% {sign}{delta:>10.2f}%")
  print("-" * 70)
  print(f"{'nciu_accuracy':<25} {metrics.nciu_accuracy:>11.2f}%   (N={metrics.num_nciu})")
  print(f"{'long_comment_accuracy':<25} {metrics.long_comment_accuracy:>11.2f}%   (N={metrics.num_long})")
  print("-" * 70)
  print(f"  -> GC-TGCUP beats TG-CUP on {wins}/6 update metrics")

  print("\n" + "=" * 70)
  print(f" DETECTION STAGE  |  full set (N={metrics.num_total})  |  TG-CUP has none")
  print("=" * 70)
  print(f"{'det_accuracy':<25} {metrics.det_accuracy:>11.2f}%")
  print(f"{'det_precision':<25} {metrics.det_precision:>11.2f}%")
  print(f"{'det_recall':<25} {metrics.det_recall:>11.2f}%")
  print(f"{'det_f1':<25} {metrics.det_f1:>11.2f}%")
  print("=" * 70)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/kaggle_gpu.yaml")
  parser.add_argument("--processed-dir", default="data/processed")
  parser.add_argument("--checkpoint", default="checkpoints/best.pt")
  parser.add_argument("--beam-size", type=int, default=None)
  parser.add_argument("--max-batches", type=int, default=None)
  args = parser.parse_args()

  with open(args.config, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  vocab = Vocabulary.load(os.path.join(args.processed_dir, "vocab.json"))
  test_ds = CUPDataset(
    load_jsonl(os.path.join(args.processed_dir, "test.jsonl")), vocab,
    long_threshold=cfg["data"]["long_comment_threshold"],
  )
  test_loader = DataLoader(
    test_ds, batch_size=cfg["training"]["batch_size"],
    shuffle=False, collate_fn=collate_fn,
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
    freeze_bert=cfg["model"].get("freeze_graphcodebert", False),
    long_threshold=cfg["data"]["long_comment_threshold"],
    edit_weight=cfg["model"].get("edit_weight", 8.0),
    det_loss_weight=cfg["model"].get("det_loss_weight", 0.15),
    upd_loss_weight=cfg["model"].get("upd_loss_weight", 0.85),
  ).to(device)

  if os.path.exists(args.checkpoint):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint}")
  else:
    print("WARNING: No checkpoint found.")

  beam_size = args.beam_size or cfg["model"].get("beam_size", 5)
  metrics = evaluate_model(
    model, test_loader, vocab, device,
    det_threshold=cfg["model"].get("det_threshold", 0.5),
    beam_size=beam_size,
    max_len=cfg["model"].get("max_decode_len", 50),
    max_batches=args.max_batches,
  )
  print_comparison(metrics, cfg["evaluation"]["tgcup_baseline"])

  report = metrics.to_dict()
  out_path = os.path.join(cfg["training"]["checkpoint_dir"], "evaluation_report.json")
  with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
  print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
  main()
