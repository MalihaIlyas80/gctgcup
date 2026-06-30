"""
Dataset loading, vocabulary, and collation for GC-TGCUP.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .ast_diff import ASTDiffGraph, build_ast_diff_graph
from .cleaning import (
    CommentCleaner,
    build_edit_sequence,
    clean_sample,
    flatten_edit_sequence,
    is_valid_sample,
)

SPECIAL_TOKENS = [
    "<pad>", "<s>", "</s>", "<unk>", "<sep>",
    "<before>", "<after>", "<equal>", "<replace>", "<insert>", "<delete>",
    "<keep>", "<copy>", "<local_edit>",
]


class Vocabulary:
  def __init__(self):
    self.token2id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    self.id2token = {i: t for t, i in self.token2id.items()}

  def __len__(self) -> int:
    return len(self.token2id)

  def add_token(self, token: str) -> int:
    if token not in self.token2id:
      idx = len(self.token2id)
      self.token2id[token] = idx
      self.id2token[idx] = token
    return self.token2id[token]

  def encode(self, tokens: List[str], add_special: bool = True) -> List[int]:
    ids = []
    if add_special:
      ids.append(self.token2id["<s>"])
    for t in tokens:
      ids.append(self.token2id.get(t, self.token2id["<unk>"]))
    if add_special:
      ids.append(self.token2id["</s>"])
    return ids

  def decode(self, ids: List[int], skip_special: bool = True) -> List[str]:
    special = {self.token2id["<pad>"], self.token2id["<s>"], self.token2id["</s>"]}
    tokens = []
    for i in ids:
      if skip_special and i in special:
        continue
      tokens.append(self.id2token.get(i, "<unk>"))
    return tokens

  def save(self, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
      json.dump(self.token2id, f, ensure_ascii=False, indent=2)

  @classmethod
  def load(cls, path: str) -> "Vocabulary":
    vocab = cls()
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    for tok, idx in data.items():
      vocab.token2id[tok] = idx
      vocab.id2token[idx] = tok
    return vocab

  @classmethod
  def from_mix_vocab(cls, path: str) -> "Vocabulary":
    """Load from cup2_dataset/mix_vocab.json token_word2id."""
    vocab = cls()
    with open(path, encoding="utf-8") as f:
      data = json.load(f)
    for tok, idx in data.get("token_word2id", {}).items():
      vocab.token2id[tok] = idx
      vocab.id2token[idx] = tok
    return vocab


def tokenize_comment(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    import re
    return [t for t in re.findall(r"\w+|<con>|[^\w\s]", text) if t.strip()]


def _sample_comment_tokens(s: Dict[str, Any]) -> Tuple[List[str], List[str]]:
  """Tokens actually fed to the model (mirror CUPDataset.__getitem__)."""
  src_tokens = s.get("src_desc_tokens") or tokenize_comment(s.get("src_desc", ""))
  dst_tokens = s.get("dst_desc_tokens") or tokenize_comment(s.get("dst_desc", ""))
  return src_tokens, dst_tokens


def build_vocabulary(train_samples: List[Dict[str, Any]], max_size: int = 30000) -> "Vocabulary":
  """
  Build a CLOSED vocabulary from the training comments only.

  This is the single most important fix vs the old 100k mix_vocab: with a few
  thousand training samples, a 100k-token output softmax never trains, which
  collapses BLEU/GLEU/METEOR. A data-driven vocab (~8-15k) keeps the output
  layer fully trainable, and the pointer-generator copy head handles any rare
  out-of-vocabulary identifiers at inference time.
  """
  from collections import Counter

  counter: Counter = Counter()
  for s in train_samples:
    src_tokens, dst_tokens = _sample_comment_tokens(s)
    counter.update(src_tokens)
    counter.update(dst_tokens)

  vocab = Vocabulary()  # special tokens occupy ids 0..len(SPECIAL_TOKENS)-1
  for tok, _freq in counter.most_common():
    if len(vocab) >= max_size:
      break
    vocab.add_token(tok)
  return vocab


class CUPDataset(Dataset):
  """Single split of comment-update data."""

  def __init__(
    self,
    samples: List[Dict[str, Any]],
    vocab: Vocabulary,
    max_comment_len: int = 128,
    max_edit_len: int = 512,
    long_threshold: int = 25,
    build_graphs: bool = True,
  ):
    self.samples = samples
    self.vocab = vocab
    self.max_comment_len = max_comment_len
    self.max_edit_len = max_edit_len
    self.long_threshold = long_threshold
    self.build_graphs = build_graphs

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    s = self.samples[idx]
    src_tokens = s.get("src_desc_tokens") or tokenize_comment(s["src_desc"])
    dst_tokens = s.get("dst_desc_tokens") or tokenize_comment(s["dst_desc"])

    old_t, new_t, acts = build_edit_sequence(s.get("code_change_seq", []))
    edit_flat = flatten_edit_sequence(old_t, new_t, acts)

    src_ids = self.vocab.encode(src_tokens[: self.max_comment_len])
    dst_ids = self.vocab.encode(dst_tokens[: self.max_comment_len])
    edit_ids = self.vocab.encode(edit_flat[: self.max_edit_len], add_special=False)
    edit_ids = [self.vocab.token2id["<s>"]] + edit_ids + [self.vocab.token2id["</s>"]]

    label = int(bool(s.get("label", True)))
    is_long = len(src_tokens) > self.long_threshold
    is_nciu = bool(s.get("is_nciu", False))

    item: Dict[str, Any] = {
      "idx": s.get("idx", idx),
      "src_desc": s["src_desc"],
      "dst_desc": s["dst_desc"],
      "src_method": s["src_method"],
      "dst_method": s["dst_method"],
      "src_ids": torch.tensor(src_ids, dtype=torch.long),
      "dst_ids": torch.tensor(dst_ids, dtype=torch.long),
      "edit_ids": torch.tensor(edit_ids, dtype=torch.long),
      "src_tokens": src_tokens,
      "dst_tokens": dst_tokens,
      "label": torch.tensor(label, dtype=torch.float),
      "is_long": is_long,
      "is_nciu": is_nciu,
    }

    if self.build_graphs:
      graph = build_ast_diff_graph(s["src_method"], s["dst_method"])
      if graph is None:
        graph = ASTDiffGraph(
          nodes=[__import__("src.data.ast_diff", fromlist=["ASTNode"]).ASTNode(0, "Method", "method", True)],
          edges=[],
          value_node_indices=[0],
        )
      item["graph"] = graph

    return item


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
  pad_id = 0

  def pad_1d(tensors, pad_value=0):
    return pad_sequence(tensors, batch_first=True, padding_value=pad_value)

  out = {
    "src_ids": pad_1d([b["src_ids"] for b in batch], pad_id),
    "dst_ids": pad_1d([b["dst_ids"] for b in batch], pad_id),
    "edit_ids": pad_1d([b["edit_ids"] for b in batch], pad_id),
    "labels": torch.stack([b["label"] for b in batch]),
    "src_descs": [b["src_desc"] for b in batch],
    "dst_descs": [b["dst_desc"] for b in batch],
    "src_methods": [b["src_method"] for b in batch],
    "dst_methods": [b["dst_method"] for b in batch],
    "src_tokens_list": [b["src_tokens"] for b in batch],
    "dst_tokens_list": [b["dst_tokens"] for b in batch],
    "is_long": torch.tensor([b["is_long"] for b in batch], dtype=torch.bool),
    "is_nciu": torch.tensor([b["is_nciu"] for b in batch], dtype=torch.bool),
    "graphs": [b.get("graph") for b in batch],
  }
  return out


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


def prepare_datasets(
  raw_dir: str,
  processed_dir: str,
  max_samples: int = 1000,
  train_ratio: float = 0.8,
  valid_ratio: float = 0.1,
  seed: int = 42,
  vocab_max_size: int = 30000,
) -> Tuple[CUPDataset, CUPDataset, CUPDataset, Vocabulary]:
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

  # Build a CLOSED vocabulary from training comments (NOT the 100k mix_vocab).
  # This keeps the output softmax trainable; the copy head covers rare OOV tokens.
  vocab = build_vocabulary(train_s, max_size=vocab_max_size)
  vocab.save(os.path.join(processed_dir, "vocab.json"))

  stats = {
    "total": len(all_samples),
    "train": len(train_s),
    "valid": len(valid_s),
    "test": len(test_s),
    "positive": sum(1 for s in all_samples if s.get("label")),
    "negative": sum(1 for s in all_samples if not s.get("label")),
    "nciu": sum(1 for s in all_samples if s.get("is_nciu")),
    "vocab_size": len(vocab),
  }
  with open(os.path.join(processed_dir, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

  for name, data in [("train", train_s), ("valid", valid_s), ("test", test_s)]:
    with open(os.path.join(processed_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
      for s in data:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

  return (
    CUPDataset(train_s, vocab),
    CUPDataset(valid_s, vocab),
    CUPDataset(test_s, vocab),
    vocab,
  )
