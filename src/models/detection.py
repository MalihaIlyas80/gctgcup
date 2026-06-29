"""
Stage 1: Outdated Comment Detection.
Gap addressed: TG-CUP has no detection stage.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .graphcodebert_encoder import GraphCodeBERTEncoder


class OutdatedCommentDetector(nn.Module):
  """
  Binary classifier: is the old comment outdated given code change?
  Fuses GraphCodeBERT code semantics + comment encoding via cross-attention.
  """

  def __init__(
    self,
    hidden_dim: int = 256,
    graphcodebert_name: str = "microsoft/graphcodebert-base",
    freeze_bert: bool = False,
    dropout: float = 0.1,
  ):
    super().__init__()
    self.code_encoder = GraphCodeBERTEncoder(
      model_name=graphcodebert_name,
      hidden_dim=hidden_dim,
      freeze=freeze_bert,
    )
    # shared BERT backbone but separate projection for comment domain
    self.comment_encoder = self.code_encoder
    self.comment_proj = nn.Sequential(
      nn.Linear(hidden_dim, hidden_dim),
      nn.LayerNorm(hidden_dim),
      nn.GELU(),
    )
    self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout, batch_first=True)
    self.classifier = nn.Sequential(
      nn.Linear(hidden_dim * 3, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, 1),
    )

  def forward(
    self,
    old_codes: List[str],
    new_codes: List[str],
    comments: List[str],
    code_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  ) -> torch.Tensor:
    device = next(self.parameters()).device
    # Reuse a precomputed code encoding when available (avoids a duplicate
    # GraphCodeBERT forward pass per step).
    if code_cache is not None:
      code_h, code_mask = code_cache
    else:
      code_h, code_mask = self.code_encoder.encode_code_pair(old_codes, new_codes, device)
    com_h_raw, com_mask = self.comment_encoder.encode_comment(comments, device)
    com_h = self.comment_proj(com_h_raw)

    attn_out, _ = self.cross_attn(
      query=com_h,
      key=code_h,
      value=code_h,
      key_padding_mask=~code_mask,
    )

    def masked_mean(h, mask):
      m = mask.unsqueeze(-1).float()
      return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    code_pool = masked_mean(code_h, code_mask)
    com_pool = masked_mean(com_h, com_mask)
    cross_pool = masked_mean(attn_out, com_mask)

    logits = self.classifier(torch.cat([code_pool, com_pool, cross_pool], dim=-1))
    return logits.squeeze(-1)

  def predict(self, old_codes, new_codes, comments, threshold=0.5):
    self.eval()
    with torch.no_grad():
      logits = self.forward(old_codes, new_codes, comments)
      probs = torch.sigmoid(logits)
      preds = (probs >= threshold).long()
    return preds, probs
