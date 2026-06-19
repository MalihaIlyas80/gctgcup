"""
GraphCodeBERT encoder for deep code semantics (thesis gap: semantic weakness).
Mandatory component – encodes old+new code jointly.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class GraphCodeBERTEncoder(nn.Module):
  def __init__(
    self,
    model_name: str = "microsoft/graphcodebert-base",
    hidden_dim: int = 256,
    freeze: bool = False,
    max_length: int = 256,
  ):
    super().__init__()
    self.max_length = max_length
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.bert = AutoModel.from_pretrained(model_name)
    bert_dim = self.bert.config.hidden_size

    if freeze:
      for p in self.bert.parameters():
        p.requires_grad = False

    self.proj = nn.Sequential(
      nn.Linear(bert_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),
      nn.Dropout(0.1),
    )

  def _encode_texts(self, texts: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    enc = self.tokenizer(
      texts,
      padding=True,
      truncation=True,
      max_length=self.max_length,
      return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    out = self.bert(**enc)
    hidden = self.proj(out.last_hidden_state)
    mask = enc["attention_mask"].bool()
    return hidden, mask

  def encode_code_pair(
    self,
    old_codes: List[str],
    new_codes: List[str],
    device: torch.device,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode concatenated old/new code with [SEP] for change semantics."""
    pairs = []
    for old, new in zip(old_codes, new_codes):
      pairs.append(f"{old} [SEP] {new}")
    return self._encode_texts(pairs, device)

  def encode_comment(
    self,
    comments: List[str],
    device: torch.device,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    return self._encode_texts(comments, device)

  def pooled_code_repr(
    self,
    old_codes: List[str],
    new_codes: List[str],
    device: torch.device,
  ) -> torch.Tensor:
    hidden, mask = self.encode_code_pair(old_codes, new_codes, device)
    mask_f = mask.unsqueeze(-1).float()
    pooled = (hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-6)
    return pooled
