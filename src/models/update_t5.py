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

TOKENIZER_LOADER_VERSION = "vocab_merges_direct_v5"


def _resolve_hub_file(model_name: str, filename: str) -> str:
  local = Path(model_name) / filename
  if local.is_file():
    return str(local)
  from huggingface_hub import hf_hub_download
  return hf_hub_download(model_name, filename)


class _CodeT5TokenizerWrapper:
  """Minimal HuggingFace-compatible wrapper around a `tokenizers` BPE object."""

  def __init__(
    self,
    tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    special_ids: set,
  ):
    self._tok = tokenizer
    self.pad_token_id = pad_token_id
    self.eos_token_id = eos_token_id
    self._special_ids = special_ids

  def _encode_batch(
    self,
    texts: List[str],
    max_length: int,
    truncation: bool,
    add_special_tokens: bool,
  ) -> List[List[int]]:
    if add_special_tokens:
      encodings = self._tok.encode_batch(texts)
    else:
      # Targets: no leading <s> — T5 supplies decoder_start_token_id itself.
      encodings = self._tok.encode_batch(texts, add_special_tokens=False)
    ids_list: List[List[int]] = []
    for enc in encodings:
      ids = list(enc.ids)
      if not add_special_tokens:
        ids = ids + [self.eos_token_id]
      if truncation and len(ids) > max_length:
        ids = ids[:max_length]
        if ids[-1] != self.eos_token_id and self.eos_token_id in ids[:-1]:
          pass
        elif ids[-1] != self.eos_token_id:
          ids[-1] = self.eos_token_id
      ids_list.append(ids)
    return ids_list

  def __call__(
    self,
    texts: Union[str, List[str]],
    max_length: int,
    truncation: bool = True,
    padding: bool = True,
    return_tensors: str = "pt",
    add_special_tokens: bool = True,
  ):
    if isinstance(texts, str):
      texts = [texts]
    ids_list = self._encode_batch(texts, max_length, truncation, add_special_tokens)

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
  """Build CodeT5 tokenizer from vocab.json + merges.txt only."""
  from tokenizers import Tokenizer
  from tokenizers.decoders import Metaspace as MetaspaceDecoder
  from tokenizers.models import BPE
  from tokenizers.pre_tokenizers import Metaspace
  from tokenizers.processors import RobertaProcessing

  vocab_path = _resolve_hub_file(model_name, "vocab.json")
  merges_path = _resolve_hub_file(model_name, "merges.txt")

  with open(vocab_path, encoding="utf-8") as f:
    vocab = json.load(f)

  pad_id = int(vocab.get("<pad>", 1))
  eos_id = int(vocab.get("</s>", 2))
  special_ids = {
    int(vocab.get("<s>", 0)),
    pad_id,
    eos_id,
    int(vocab.get("<unk>", 3)),
    int(vocab.get("<mask>", 4)),
  }

  # Load BPE directly from hub files (preserves merge rank order).
  bpe = BPE(vocab=vocab_path, merges=merges_path)
  tok = Tokenizer(bpe)
  # CodeT5 / RoBERTa BPE uses GPT-2-style "Ġ" as the word-boundary marker, NOT "▁".
  # Using "▁" (SentencePiece) was the root cause of BLEU ~1% despite falling loss.
  tok.pre_tokenizer = Metaspace(replacement="Ġ", prepend_scheme="always")
  tok.decoder = MetaspaceDecoder(replacement="Ġ", prepend_scheme="always")
  # MUST match HuggingFace RobertaTokenizer: add_prefix_space=True.
  tok.post_processor = RobertaProcessing(
    ("</s>", eos_id),
    ("<s>", int(vocab.get("<s>", 0))),
    add_prefix_space=True,
    trim_offsets=False,
  )

  wrapper = _CodeT5TokenizerWrapper(tok, pad_id, eos_id, special_ids)
  # Sanity check: round-trip must preserve text (catches wrong pretokenizer).
  for test in ("Returns the maximum value.", "update comment test"):
    rt = wrapper.batch_decode(
      wrapper(test, max_length=32, add_special_tokens=True).input_ids,
      skip_special_tokens=True,
    )[0]
    if rt != test:
      raise RuntimeError(f"Tokenizer round-trip failed: {test!r} -> {rt!r}")

  print(f"CodeT5 tokenizer loaded ({TOKENIZER_LOADER_VERSION}) from vocab+merges")
  return wrapper


class UpdateT5(nn.Module):
  def __init__(
    self,
    model_name: str = "Salesforce/codet5-small",
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    max_edit_chars: int = 400,
    max_ast_chars: int = 200,
  ):
    super().__init__()
    from transformers import T5ForConditionalGeneration

    self.tokenizer = _load_codet5_tokenizer(model_name)
    self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
    self.max_src_len = max_src_len
    self.max_tgt_len = max_tgt_len
    self.max_edit_chars = max_edit_chars
    self.max_ast_chars = max_ast_chars

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
      # Old comment first (most important for copy+edit); truncate long tail via max_src_len.
      texts.append(f"update comment: {c} code change: {e} ast diff: {a}")
    return texts

  def _encode(self, texts: List[str], device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = self.tokenizer(
      texts,
      max_length=self.max_src_len,
      truncation=True,
      padding=True,
      add_special_tokens=True,
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
        add_special_tokens=False,
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
        gen_kwargs["length_penalty"] = 1.1
        gen_kwargs["early_stopping"] = True
      gen = self.t5.generate(**gen_kwargs)
    decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
    decoded = [t.strip() for t in decoded]

    b = len(old_comments)
    beams = [decoded[i * num_beams:(i + 1) * num_beams] for i in range(b)]
    best = [cands[0] if cands else "" for cands in beams]
    return best, beams
