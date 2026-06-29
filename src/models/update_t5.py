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

from typing import List, Tuple

import torch
import torch.nn as nn


def _load_codet5_tokenizer(model_name: str):
  """Load CodeT5's tokenizer robustly across transformers versions.

  CodeT5 ships a RoBERTa-style BPE tokenizer (vocab.json + merges.txt) and no
  fast `tokenizer.json`.  On some (newer) transformers versions AutoTokenizer
  fails: the slow->fast conversion asks for sentencepiece, and the slow path
  itself raises `extra_special_tokens must be ...`.  RobertaTokenizerFast builds
  the fast tokenizer directly from vocab+merges and sidesteps both issues.
  """
  errors = []
  # 1) Fast RoBERTa tokenizer straight from vocab.json + merges.txt.
  try:
    from transformers import RobertaTokenizerFast
    return RobertaTokenizerFast.from_pretrained(model_name)
  except Exception as e:  # noqa: BLE001
    errors.append(f"RobertaTokenizerFast: {e}")
  # 2) Generic Auto (fast).
  try:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)
  except Exception as e:  # noqa: BLE001
    errors.append(f"AutoTokenizer(fast): {e}")
  # 3) Generic Auto (slow).
  try:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, use_fast=False)
  except Exception as e:  # noqa: BLE001
    errors.append(f"AutoTokenizer(slow): {e}")
  raise RuntimeError(
    "Could not load CodeT5 tokenizer for '%s'. Tried:\n  - %s"
    % (model_name, "\n  - ".join(errors))
  )


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
