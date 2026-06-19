#!/usr/bin/env python3
"""
Compare GC-TGCUP vs simplified TG-CUP baseline on the same test split.
Run after training GC-TGCUP, or with untrained weights for smoke test.
"""
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
from src.models.tgcup_baseline import TGCUPBaseline


def load_jsonl(path):
  samples = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      samples.append(json.loads(line))
  return samples


@torch.no_grad()
def run_model(model, loader, vocab, device, has_detection=False):
  model.eval()
  predictions, references, sources = [], [], []
  det_preds, det_labels = [], []
  is_nciu, is_long = [], []

  for batch in loader:
    for k, v in batch.items():
      if isinstance(v, torch.Tensor):
        batch[k] = v.to(device)

    if has_detection:
      det_logits = model.detect(batch["src_methods"], batch["dst_methods"], batch["src_descs"])
      det_preds.extend((torch.sigmoid(det_logits) >= 0.5).long().cpu().tolist())
      det_labels.extend(batch["labels"].long().cpu().tolist())
      gen_ids = model.generate(
        batch["src_ids"], batch["edit_ids"],
        batch["src_methods"], batch["dst_methods"],
        batch["graphs"], comments=batch["src_descs"],
      )
    else:
      # baseline: always attempt update
      gen_ids = []
      for i in range(batch["src_ids"].size(0)):
        single = {k: (v[i:i+1] if isinstance(v, torch.Tensor) else [v[i]]) for k, v in batch.items()}
        # simple greedy decode via forward (teacher forcing not used at inference)
        sep = torch.full((1, 1), 4, dtype=torch.long, device=device)
        combined = torch.cat([single["src_ids"], sep, single["edit_ids"]], dim=1)[:, :512]
        mask = combined.ne(0)
        seq_enc = model.seq_encoder(model._embed(combined), src_key_padding_mask=~mask)
        graph_enc, graph_mask = model._encode_graph_batch(single["graphs"], device, model.vocab_size)
        memory = torch.cat([seq_enc, graph_enc], dim=1)
        mem_mask = torch.cat([mask, graph_mask], dim=1)
        seq = torch.tensor([[1]], device=device)
        for _ in range(50):
          tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.size(1), device=device)
          dec = model.decoder(
            tgt=model._embed(seq), memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=~mem_mask,
          )
          nxt = model.output_proj(dec[:, -1]).argmax(-1, keepdim=True)
          seq = torch.cat([seq, nxt], dim=1)
          if nxt.item() == 2:
            break
        gen_ids.append(seq[0, 1:].tolist())

    for ids, ref, src in zip(gen_ids, batch["dst_descs"], batch["src_descs"]):
      predictions.append(" ".join(vocab.decode(ids)))
      references.append(ref)
      sources.append(src)
    is_nciu.extend(batch["is_nciu"].cpu().tolist())
    is_long.extend(batch["is_long"].cpu().tolist())

  return compute_all_metrics(
    predictions, references, sources,
    det_preds=det_preds if has_detection else None,
    det_labels=det_labels if has_detection else None,
    is_nciu=is_nciu, is_long=is_long,
  )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default="configs/default.yaml")
  parser.add_argument("--processed-dir", default="data/processed")
  parser.add_argument("--gctgcup-checkpoint", default="checkpoints/best.pt")
  args = parser.parse_args()

  with open(args.config) as f:
    cfg = yaml.safe_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  vocab = Vocabulary.load(os.path.join(args.processed_dir, "vocab.json"))
  test_ds = CUPDataset(load_jsonl(os.path.join(args.processed_dir, "test.jsonl")), vocab)
  loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], collate_fn=collate_fn)

  common = dict(
    vocab_size=len(vocab),
    hidden_dim=cfg["model"]["hidden_dim"],
    num_heads=cfg["model"]["num_heads"],
    num_encoder_layers=cfg["model"]["num_encoder_layers"],
    num_decoder_layers=cfg["model"]["num_decoder_layers"],
    ggnn_steps=cfg["model"]["ggnn_steps"],
    dropout=cfg["model"]["dropout"],
  )

  baseline = TGCUPBaseline(**common).to(device)
  gctgcup = GCTGCUP(
    **common,
    graphcodebert_name=cfg["model"]["graphcodebert"],
    freeze_bert=cfg["model"]["freeze_graphcodebert"],
    long_threshold=cfg["data"]["long_comment_threshold"],
  ).to(device)

  if os.path.exists(args.gctgcup_checkpoint):
    ckpt = torch.load(args.gctgcup_checkpoint, map_location=device)
    gctgcup.load_state_dict(ckpt["model_state"])

  print("Evaluating TG-CUP baseline ...")
  m_base = run_model(baseline, loader, vocab, device, has_detection=False)
  print("Evaluating GC-TGCUP ...")
  m_ours = run_model(gctgcup, loader, vocab, device, has_detection=True)

  paper = cfg["evaluation"]["tgcup_baseline"]
  print("\n" + "=" * 80)
  print(f"{'Metric':<22} {'TG-CUP(paper)':>14} {'Baseline':>12} {'GC-TGCUP':>12} {'Ours vs Paper':>14}")
  print("-" * 80)
  for key in ["accuracy", "recall_at_5", "gleu", "meteor", "sari", "bleu"]:
    b = getattr(m_base, key)
    o = getattr(m_ours, key)
    p = paper.get(key, 0)
    print(f"{key:<22} {p:>13.2f}% {b:>11.2f}% {o:>11.2f}% {o - p:>+13.2f}%")
  print("-" * 80)
  print(f"{'det_f1':<22} {'N/A':>14} {'N/A':>12} {m_ours.det_f1:>11.2f}%")
  print(f"{'nciu_accuracy':<22} {'N/A':>14} {m_base.nciu_accuracy:>11.2f}% {m_ours.nciu_accuracy:>11.2f}%")
  print(f"{'long_comment_accuracy':<22} {'N/A':>14} {m_base.long_comment_accuracy:>11.2f}% {m_ours.long_comment_accuracy:>11.2f}%")
  print("=" * 80)

  out = {"baseline": m_base.to_dict(), "gctgcup": m_ours.to_dict(), "paper_tgcup": paper}
  path = os.path.join(cfg["training"]["checkpoint_dir"], "comparison_report.json")
  with open(path, "w") as f:
    json.dump(out, f, indent=2)
  print(f"\nSaved: {path}")


if __name__ == "__main__":
  main()
