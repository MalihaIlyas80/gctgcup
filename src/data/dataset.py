"""
Dataset loading and collation for GC-TGCUP.

The model is fully text-conditioned:
  * Stage 1 detection uses GraphCodeBERT on raw code/comment text.
  * Stage 2 update is a pretrained CodeT5 fed with
        old comment + code-edit sequence + AST-diff summary.
So there is no closed vocabulary / id tensors here — we precompute the
edit-sequence text and AST-diff summary once at prepare time and store them in
the processed jsonl, keeping per-epoch data loading cheap (no javalang at train).
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .ast_diff import build_ast_diff_graph
from .cleaning import (
    CommentCleaner,
    build_edit_sequence,
    clean_sample,
    flatten_edit_sequence,
    is_valid_sample,
)

# AST-diff edge type for "node value updated old->new" (see ast_diff.EDGE_UPDATE).
_EDGE_UPDATE = 2


def build_edit_text(s: Dict[str, Any]) -> str:
  """Flatten a sample's code-change sequence into a single edit string."""
  old_t, new_t, acts = build_edit_sequence(s.get("code_change_seq", []))
  return " ".join(flatten_edit_sequence(old_t, new_t, acts))


def build_ast_text(old_code: str, new_code: str, max_items: int = 30) -> str:
  """Serialise the AST-difference graph into a compact text summary.

  Captures TG-CUP's structural signal as text so it can condition the CodeT5
  updater: updated nodes (old -> new value) and changed/inserted value nodes.
  """
  try:
    graph = build_ast_diff_graph(old_code, new_code)
  except Exception:
    graph = None
  if graph is None:
    return ""

  id2node = {n.node_id: n for n in graph.nodes}
  parts: List[str] = []
  seen = set()

  for src, dst, et in graph.edges:
    if et == _EDGE_UPDATE:
      o, n = id2node.get(src), id2node.get(dst)
      if o is not None and n is not None:
        frag = f"{o.node_type} {o.value} -> {n.value}".strip()
        if frag not in seen:
          seen.add(frag)
          parts.append(frag)

  for n in graph.nodes:
    if len(parts) >= max_items:
      break
    if n.is_value_node and n.value:
      frag = f"{n.node_type} {n.value}".strip()
      if frag not in seen:
        seen.add(frag)
        parts.append(frag)

  return " ; ".join(parts[:max_items])


class CUPDataset(Dataset):
  """Single split of comment-update data (text-only items)."""

  def __init__(
    self,
    samples: List[Dict[str, Any]],
    max_comment_len: int = 128,
    long_threshold: int = 25,
    **_ignored,
  ):
    self.samples = samples
    self.max_comment_len = max_comment_len
    self.long_threshold = long_threshold

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    s = self.samples[idx]
    src_text = s.get("src_desc", "") or ""
    dst_text = s.get("dst_desc", "") or ""
    # Prefer precomputed fields (written by prepare_datasets); fall back live.
    edit_text = s.get("edit_text")
    if edit_text is None:
      edit_text = build_edit_text(s)
    ast_text = s.get("ast_text")
    if ast_text is None:
      ast_text = build_ast_text(s.get("src_method", ""), s.get("dst_method", ""))

    label = int(bool(s.get("label", True)))
    is_long = len(src_text.split()) > self.long_threshold
    is_nciu = bool(s.get("is_nciu", False))

    return {
      "idx": s.get("idx", idx),
      "src_desc": src_text,
      "dst_desc": dst_text,
      "src_method": s.get("src_method", ""),
      "dst_method": s.get("dst_method", ""),
      "edit_text": edit_text,
      "ast_text": ast_text,
      "label": torch.tensor(label, dtype=torch.float),
      "is_long": is_long,
      "is_nciu": is_nciu,
    }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
  return {
    "labels": torch.stack([b["label"] for b in batch]),
    "src_descs": [b["src_desc"] for b in batch],
    "dst_descs": [b["dst_desc"] for b in batch],
    "src_methods": [b["src_method"] for b in batch],
    "dst_methods": [b["dst_method"] for b in batch],
    "edit_texts": [b["edit_text"] for b in batch],
    "ast_texts": [b["ast_text"] for b in batch],
    "is_long": torch.tensor([b["is_long"] for b in batch], dtype=torch.bool),
    "is_nciu": torch.tensor([b["is_nciu"] for b in batch], dtype=torch.bool),
  }


def _stream_jsonl(path: str, max_lines: Optional[int] = None):
  with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
      if max_lines and i >= max_lines:
        break
      yield json.loads(line)


def load_and_clean_split(
  paths: List[str],
  max_samples: int,
  cleaner: CommentCleaner,
  seed: int = 42,
  target_pos_ratio: float = 0.35,
) -> List[Dict[str, Any]]:
  """
  Stream, clean, validate, deduplicate.
  Actively collects positive (outdated) samples to avoid class imbalance.
  """
  seen = set()
  positive: List[Dict[str, Any]] = []
  negative: List[Dict[str, Any]] = []
  rng = random.Random(seed)

  target_pos = max(1, int(max_samples * target_pos_ratio))
  max_scan = max_samples * 50

  scanned = 0
  for path in paths:
    if not os.path.exists(path):
      continue
    for raw in _stream_jsonl(path):
      scanned += 1
      if scanned > max_scan and len(positive) >= target_pos and len(negative) >= max_samples:
        break
      cleaned = clean_sample(raw, cleaner)
      if not is_valid_sample(cleaned):
        continue
      key = (cleaned.get("src_method"), cleaned.get("src_desc"), cleaned.get("dst_desc"))
      if key in seen:
        continue
      seen.add(key)
      if cleaned.get("label"):
        if len(positive) < target_pos * 3:
          positive.append(cleaned)
      else:
        if len(negative) < max_samples * 2:
          negative.append(cleaned)
    if scanned > max_scan and len(positive) >= target_pos:
      break

  rng.shuffle(positive)
  rng.shuffle(negative)
  n_pos = min(len(positive), target_pos)
  n_neg = min(len(negative), max_samples - n_pos)
  samples = positive[:n_pos] + negative[:n_neg]
  rng.shuffle(samples)
  return samples[:max_samples]


def _precompute_fields(samples: List[Dict[str, Any]]) -> None:
  """Add edit_text + ast_text to each sample in-place (one-time, at prepare)."""
  for s in samples:
    s["edit_text"] = build_edit_text(s)
    s["ast_text"] = build_ast_text(s.get("src_method", ""), s.get("dst_method", ""))


def prepare_datasets(
  raw_dir: str,
  processed_dir: str,
  max_samples: int = 1000,
  train_ratio: float = 0.8,
  valid_ratio: float = 0.1,
  seed: int = 42,
  long_threshold: int = 25,
  **_ignored,
) -> Tuple[CUPDataset, CUPDataset, CUPDataset, Dict[str, Any]]:
  os.makedirs(processed_dir, exist_ok=True)
  cleaner = CommentCleaner()

  paths = [
    os.path.join(raw_dir, "train.jsonl"),
    os.path.join(raw_dir, "valid.jsonl"),
    os.path.join(raw_dir, "test.jsonl"),
  ]
  all_samples = load_and_clean_split(paths, max_samples, cleaner, seed)

  # stratified split by label (keeps outdated/no-update ratio in each split)
  rng = random.Random(seed)
  pos = [s for s in all_samples if s.get("label")]
  neg = [s for s in all_samples if not s.get("label")]
  rng.shuffle(pos)
  rng.shuffle(neg)

  def _split(lst: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    n = len(lst)
    n_tr = int(n * train_ratio)
    n_va = int(n * valid_ratio)
    return lst[:n_tr], lst[n_tr:n_tr + n_va], lst[n_tr + n_va:]

  train_pos, valid_pos, test_pos = _split(pos)
  train_neg, valid_neg, test_neg = _split(neg)
  train_s = train_pos + train_neg
  valid_s = valid_pos + valid_neg
  test_s = test_pos + test_neg
  rng.shuffle(train_s)
  rng.shuffle(valid_s)
  rng.shuffle(test_s)

  # Precompute edit-sequence text + AST-diff summary so training never pays the
  # javalang parse cost again (huge per-epoch speedup vs building graphs live).
  for split in (train_s, valid_s, test_s):
    _precompute_fields(split)

  stats = {
    "total": len(all_samples),
    "train": len(train_s),
    "valid": len(valid_s),
    "test": len(test_s),
    "positive": sum(1 for s in all_samples if s.get("label")),
    "negative": sum(1 for s in all_samples if not s.get("label")),
    "nciu": sum(1 for s in all_samples if s.get("is_nciu")),
  }
  with open(os.path.join(processed_dir, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

  for name, data in [("train", train_s), ("valid", valid_s), ("test", test_s)]:
    with open(os.path.join(processed_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
      for s in data:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

  return (
    CUPDataset(train_s, long_threshold=long_threshold),
    CUPDataset(valid_s, long_threshold=long_threshold),
    CUPDataset(test_s, long_threshold=long_threshold),
    stats,
  )
