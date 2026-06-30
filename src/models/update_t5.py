"""
Stage 2: Pretrained seq2seq updater (Flan-T5).

Uses HuggingFace AutoTokenizer — no custom BPE loader (that caused BLEU ~1%).
Flan-T5 is instruction-tuned for text rewriting; with the old comment in the
input it copy-edits efficiently on small data — the key to beating TG-CUP.

Conditioned on: old comment + code-edit sequence + AST-diff summary.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

UPDATE_MODEL_VERSION = "flan_t5_hf_v1"


class UpdateT5(nn.Module):
  def __init__(
    self,
    model_name: str = "google/flan-t5-base",
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    max_edit_chars: int = 350,
    max_ast_chars: int = 180,
  ):
    super().__init__()
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.t5 = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    self.max_src_len = max_src_len
    self.max_tgt_len = max_tgt_len
    self.max_edit_chars = max_edit_chars
    self.max_ast_chars = max_ast_chars

    self._verify_tokenizer()
    print(f"Update model loaded ({UPDATE_MODEL_VERSION}): {model_name}")

  def _verify_tokenizer(self) -> None:
    tests = (
      "Returns the maximum value.",
      "Update Java comment test case.",
    )
    for text in tests:
      enc = self.tokenizer(text, return_tensors="pt")
      dec = self.tokenizer.decode(enc.input_ids[0], skip_special_tokens=True)
      if dec.strip() != text.strip():
        raise RuntimeError(f"Tokenizer round-trip failed: {text!r} -> {dec!r}")

  def _build_inputs(
    self,
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
  ) -> List[str]:
    texts = []
    for c, e, a in zip(old_comments, edit_texts, ast_texts):
      c = (c or "").strip()
      e = (e or "").strip()[: self.max_edit_chars]
      a = (a or "").strip()[: self.max_ast_chars]
      texts.append(
        "Update the Java method comment.\n"
        f"Old comment: {c}\n"
        f"Code edit: {e}\n"
        f"AST diff: {a}\n"
        "Updated comment:"
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

  def forward(
    self,
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
    new_comments: List[str],
  ) -> torch.Tensor:
    device = next(self.parameters()).device
    texts = self._build_inputs(old_comments, edit_texts, ast_texts)
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

  @torch.no_grad()
  def generate(
    self,
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
    num_beams: int = 5,
    max_len: int = 128,
  ) -> Tuple[List[str], List[List[str]]]:
    device = next(self.parameters()).device
    texts = self._build_inputs(old_comments, edit_texts, ast_texts)
    num_beams = max(1, num_beams)
    cfg = self.t5.config
    with torch.autocast(device_type=device.type, enabled=False):
      input_ids, attn = self._encode(texts, device)
      gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        max_new_tokens=max_len,
        decoder_start_token_id=cfg.decoder_start_token_id,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=cfg.pad_token_id,
      )
      if num_beams > 1:
        gen_kwargs["length_penalty"] = 1.0
        gen_kwargs["early_stopping"] = True
      gen = self.t5.generate(**gen_kwargs)
    decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
    decoded = [t.strip() for t in decoded]

    b = len(old_comments)
    beams = [decoded[i * num_beams:(i + 1) * num_beams] for i in range(b)]
    best = [cands[0] if cands else "" for cands in beams]
    return best, beams
