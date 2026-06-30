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

from src.data.dataset import CUPDataset, collate_fn
from src.evaluation.metrics import compute_all_metrics
from src.models.gctgcup import GCTGCUP


def load_jsonl(path):
  samples = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      samples.append(json.loads(line))
  return samples


@torch.no_grad()
def evaluate_model(model, loader, device, det_threshold=0.5, beam_size=5, max_len=64,
                   qualitative=None, max_batches=None):
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
    labels = batch["labels"].long().cpu().tolist()
    det_labels.extend(labels)

    # Source = old comment, reference = new comment (raw text).
    src_texts = batch["src_descs"]
    ref_texts = batch["dst_descs"]

    # force_update=True -> always generate (TG-CUP-style pure update quality).
    pred_texts, beam_b = model.generate(
      batch["src_methods"], batch["dst_methods"],
      batch["src_descs"], batch["edit_texts"], batch["ast_texts"],
      max_len=max_len, beam_size=beam_size,
      det_threshold=det_threshold,
      force_update=True,
    )
    for pred_text, cands, ref, src, lab in zip(pred_texts, beam_b, ref_texts, src_texts, labels):
      predictions.append(pred_text)
      references.append(ref)
      sources.append(src)
      beam_cands.append(cands)
      # Collect qualitative examples on OUTDATED samples (the update task).
      if qualitative is not None and lab == 1 and len(qualitative) < 20:
        qualitative.append({
          "OLD (source)": src,
          "PRED (model)": pred_text,
          "NEW (reference)": ref,
          "exact_match": pred_text.strip() == ref.strip(),
        })

    is_nciu.extend(batch["is_nciu"].cpu().tolist())
    is_long.extend(batch["is_long"].cpu().tolist())
    outdated.extend(labels)

  return compute_all_metrics(
    predictions, references, sources,
    beam_candidates=beam_cands,
    det_preds=det_preds, det_labels=det_labels,
    is_nciu=is_nciu, is_long=is_long,
    outdated=outdated,
  )


def print_comparison(metrics, baseline):
  # ── Option A: UPDATE stage on outdated-only subset (fair TG-CUP setting) ──
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
  print(f"{'nciu_accuracy':<25} {metrics.nciu_accuracy:>11.2f}%   (NCIU robustness, N={metrics.num_nciu})")
  print(f"{'long_comment_accuracy':<25} {metrics.long_comment_accuracy:>11.2f}%   (long comments, N={metrics.num_long})")
  print("-" * 70)
  print(f"  -> GC-TGCUP beats TG-CUP on {wins}/{len(mapping)} update metrics")

  # ── Option B: DETECTION stage on full imbalanced set (TG-CUP has none) ──
  print("\n" + "=" * 70)
  print(f" DETECTION STAGE  |  full imbalanced set (N={metrics.num_total}, "
        f"pos={metrics.det_pos_ratio:.1f}%)  |  TG-CUP has none")
  print("=" * 70)
  print(f"{'det_accuracy':<25} {metrics.det_accuracy:>11.2f}%")
  print(f"{'det_precision':<25} {metrics.det_precision:>11.2f}%")
  print(f"{'det_recall':<25} {metrics.det_recall:>11.2f}%")
  print(f"{'det_f1':<25} {metrics.det_f1:>11.2f}%")
  print("=" * 70)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/default.yaml")
  parser.add_argument("--processed-dir", default="data/processed")
  parser.add_argument("--checkpoint", default="checkpoints/best.pt")
  parser.add_argument("--beam-size", type=int, default=None,
                      help="Override beam size (use 1 for a fast diagnostic)")
  parser.add_argument("--max-batches", type=int, default=None,
                      help="Only evaluate the first N batches (fast diagnostic)")
  args = parser.parse_args()

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  test_ds = CUPDataset(load_jsonl(os.path.join(args.processed_dir, "test.jsonl")),
                       long_threshold=cfg["data"]["long_comment_threshold"])
  test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"],
                           shuffle=False, collate_fn=collate_fn)

  model = GCTGCUP(
    hidden_dim=cfg["model"]["hidden_dim"],
    dropout=cfg["model"]["dropout"],
    graphcodebert_name=cfg["model"]["graphcodebert"],
    freeze_bert=cfg["model"].get("freeze_graphcodebert", True),
    long_threshold=cfg["data"]["long_comment_threshold"],
    update_model_name=cfg["model"].get("update_model", "google/flan-t5-base"),
    max_src_len=cfg["model"].get("max_src_len", 512),
    max_tgt_len=cfg["model"].get("max_tgt_len", 128),
    max_edit_chars=cfg["model"].get("max_edit_chars", 400),
    max_ast_chars=cfg["model"].get("max_ast_chars", 200),
    det_loss_weight=cfg["model"].get("det_loss_weight", 0.15),
    upd_loss_weight=cfg["model"].get("upd_loss_weight", 0.85),
  ).to(device)

  if os.path.exists(args.checkpoint):
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {args.checkpoint}")
  else:
    print("WARNING: No checkpoint found – evaluating untrained model.")

  qualitative = []
  beam_size = args.beam_size or cfg["model"].get("beam_size", 5)
  metrics = evaluate_model(
    model, test_loader, device,
    det_threshold=cfg["model"].get("det_threshold", 0.5),
    beam_size=beam_size,
    max_len=cfg["model"].get("max_decode_len", 128),
    qualitative=qualitative,
    max_batches=args.max_batches,
  )

  # ── Qualitative diagnostic: are predictions in the SAME text space as refs? ──
  print("\n" + "=" * 70)
  print(" QUALITATIVE SAMPLES (outdated subset)  —  OLD -> PRED vs NEW")
  print("=" * 70)
  for i, ex in enumerate(qualitative):
    print(f"\n[{i}] exact_match={ex['exact_match']}")
    print(f"  OLD : {ex['OLD (source)']!r}")
    print(f"  PRED: {ex['PRED (model)']!r}")
    print(f"  NEW : {ex['NEW (reference)']!r}")
  qpath = os.path.join(cfg["training"]["checkpoint_dir"], "qualitative_samples.json")
  with open(qpath, "w", encoding="utf-8") as f:
    json.dump(qualitative, f, ensure_ascii=False, indent=2)
  print(f"\n(saved {len(qualitative)} examples to {qpath})")

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
