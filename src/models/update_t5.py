"""
Stage 2: Update head built on a PRETRAINED code-aware seq2seq model (CodeT5).

Why pretrained?  A from-scratch Transformer decoder needs hundreds of thousands
of examples to learn faithful copy+edit generation, so on a ~20k dataset it tops
out at ~2-3% exact match.  CodeT5 is pretrained on code+text and is extremely
data-efficient at copy-and-edit, which is exactly the comment-update task.  This
is what lets the model beat TG-CUP on a SMALL dataset (the data-efficiency thesis).

The update is conditioned on a multimodal text input:
    old comment  +  code-edit sequence  +  AST-diff summary
so all of TG-CUP's modalities (semantics + structure + edit) reach the decoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn

# Bump this string when the loader changes — helps verify Kaggle pulled latest code.
TOKENIZER_LOADER_VERSION = "vocab_merges_direct_v3"


def _resolve_hub_file(model_name: str, filename: str) -> str:
  local = Path(model_name) / filename
  if local.is_file():
    return str(local)
  from huggingface_hub import hf_hub_download
  return hf_hub_download(model_name, filename)


def _read_merges(merges_path: str) -> List[Tuple[str, str]]:
  """Parse merges.txt, skipping header / malformed lines."""
  merges: List[Tuple[str, str]] = []
  with open(merges_path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split()
      if len(parts) >= 2:
        merges.append((parts[0], parts[1]))
  return merges


class _CodeT5TokenizerWrapper:
  """Minimal HuggingFace-compatible wrapper around a `tokenizers` BPE object."""

  def __init__(self, tokenizer, pad_token_id: int, special_ids: set):
    self._tok = tokenizer
    self.pad_token_id = pad_token_id
    self._special_ids = special_ids

  def __call__(
    self,
    texts: Union[str, List[str]],
    max_length: int,
    truncation: bool = True,
    padding: bool = True,
    return_tensors: str = "pt",
  ):
    if isinstance(texts, str):
      texts = [texts]
    encodings = self._tok.encode_batch(texts)
    ids_list: List[List[int]] = []
    for enc in encodings:
      ids = list(enc.ids)
      if truncation and len(ids) > max_length:
        ids = ids[:max_length]
      ids_list.append(ids)

    if padding:
      max_len = max(len(x) for x in ids_list) if ids_list else 1
      padded, attn = [], []
      for ids in ids_list:
        pad_len = max_len - len(ids)
        padded.append(ids + [self.pad_token_id] * pad_len)
        attn.append([1] * len(ids) + [0] * pad_len)
      input_ids = torch.tensor(padded, dtype=torch.long)
      attention_mask = torch.tensor(attn, dtype=torch.long)
    else:
      input_ids = torch.tensor(ids_list, dtype=torch.long)
      attention_mask = torch.ones_like(input_ids)

    return SimpleNamespace(input_ids=input_ids, attention_mask=attention_mask)

  def batch_decode(
    self,
    token_ids: Union[torch.Tensor, Sequence[Sequence[int]]],
    skip_special_tokens: bool = True,
  ) -> List[str]:
    if isinstance(token_ids, torch.Tensor):
      rows = token_ids.tolist()
    else:
      rows = [list(r) for r in token_ids]
    out: List[str] = []
    for row in rows:
      if skip_special_tokens:
        row = [t for t in row if t not in self._special_ids]
      out.append(self._tok.decode(row).strip())
    return out


def _load_codet5_tokenizer(model_name: str) -> _CodeT5TokenizerWrapper:
  """Build CodeT5 tokenizer from vocab.json + merges.txt only.

  Does NOT call transformers AutoTokenizer / RobertaTokenizerFast — those break
  on recent Kaggle images (`extra_special_tokens`, bad merges parsing).
  """
  from tokenizers import Tokenizer
  from tokenizers.decoders import Metaspace as MetaspaceDecoder
  from tokenizers.models import BPE
  from tokenizers.pre_tokenizers import Metaspace
  from tokenizers.processors import RobertaProcessing

  vocab_path = _resolve_hub_file(model_name, "vocab.json")
  merges_path = _resolve_hub_file(model_name, "merges.txt")

  with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)
  merges = _read_merges(merges_path)

  pad_id = int(vocab.get("<pad>", 1))
  special_ids = {
    int(vocab.get("<s>", 0)),
    pad_id,
    int(vocab.get("</s>", 2)),
    int(vocab.get("<unk>", 3)),
    int(vocab.get("<mask>", 4)),
  }

  bpe = BPE(vocab, merges)
  tok = Tokenizer(bpe)
  tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
  tok.decoder = MetaspaceDecoder(replacement="▁", prepend_scheme="always")
  tok.post_processor = RobertaProcessing(
    ("</s>", int(vocab.get("</s>", 2))),
    ("<s>", int(vocab.get("<s>", 0))),
    add_prefix_space=False,
    trim_offsets=True,
  )

  print(f"CodeT5 tokenizer loaded ({TOKENIZER_LOADER_VERSION}) from vocab+merges")
  return _CodeT5TokenizerWrapper(tok, pad_id, special_ids)


class UpdateT5(nn.Module):
  def __init__(
    self,
    model_name: str = "Salesforce/codet5-small",
    max_src_len: int = 320,
    max_tgt_len: int = 64,
  ):
    super().__init__()
    from transformers import T5ForConditionalGeneration

    self.tokenizer = _load_codet5_tokenizer(model_name)
    self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    self.max_src_len = max_src_len
    self.max_tgt_len = max_tgt_len

  # ── input construction ──────────────────────────────────────────────
  def _build_inputs(
    self,
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
  ) -> List[str]:
    texts = []
    for c, e, a in zip(old_comments, edit_texts, ast_texts):
      c = (c or "").strip()
      e = (e or "").strip()
      a = (a or "").strip()
      texts.append(
        f"update comment: {c} code change: {e} ast diff: {a}"
      )
    return texts

  def _encode(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = self.tokenizer(
      texts,
      max_length=self.max_src_len,
      truncation=True,
      padding=True,
      return_tensors="pt",
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)

  # ── training loss ───────────────────────────────────────────────────
  def forward(
    self,
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
    new_comments: List[str],
  ) -> torch.Tensor:
    device = next(self.parameters()).device
    texts = self._build_inputs(old_comments, edit_texts, ast_texts)
    # T5 is numerically unstable in fp16; always run it in fp32 even when the
    # surrounding trainer uses AMP (GraphCodeBERT still benefits from AMP).
    with torch.autocast(device_type=device.type, enabled=False):
      input_ids, attn = self._encode(texts, device)
      labels = self.tokenizer(
        new_comments,
        max_length=self.max_tgt_len,
        truncation=True,
        padding=True,
        return_tensors="pt",
      ).input_ids.to(device)
      labels[labels == self.tokenizer.pad_token_id] = -100
      out = self.t5(input_ids=input_ids, attention_mask=attn, labels=labels)
    return out.loss

  # ── inference (beam search) ─────────────────────────────────────────
  @torch.no_grad()
  def generate(
    self,
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
    num_beams: int = 5,
    max_len: int = 64,
  ) -> Tuple[List[str], List[List[str]]]:
    device = next(self.parameters()).device
    texts = self._build_inputs(old_comments, edit_texts, ast_texts)
    num_beams = max(1, num_beams)
    with torch.autocast(device_type=device.type, enabled=False):
      input_ids, attn = self._encode(texts, device)
      gen = self.t5.generate(
        input_ids=input_ids,
        attention_mask=attn,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_length=max_len,
        early_stopping=True,
        no_repeat_ngram_size=3,
      )
    decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
    decoded = [t.strip() for t in decoded]

    b = len(old_comments)
    beams = [decoded[i * num_beams:(i + 1) * num_beams] for i in range(b)]
    best = [cands[0] if cands else "" for cands in beams]
    return best, beams
