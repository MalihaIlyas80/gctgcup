"""
Local Edit Decoder for long/complex comments (thesis gap).
Instead of full rewrite, uses copy-from-old-comment + local replacement.
Pointer-generator style mechanism.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalEditDecoder(nn.Module):
  """
  For long comments: generate with copy attention over source comment tokens.
  p(w) = p_gen * p_vocab(w) + (1-p_gen) * p_copy(w)
  """

  def __init__(self, hidden_dim: int, vocab_size: int):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.gen_gate = nn.Linear(hidden_dim * 2, 1)
    self.copy_attn = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
    self.output_proj = nn.Linear(hidden_dim, vocab_size)

  def forward(
    self,
    decoder_hidden: torch.Tensor,
    encoder_out: torch.Tensor,
    encoder_mask: torch.Tensor,
    src_ids: torch.Tensor,
  ) -> torch.Tensor:
    """
    decoder_hidden: (B, D)
    encoder_out: (B, S, D) – old comment encoder states
  src_ids: (B, S)
    Returns logits (B, vocab_size) with copy distribution merged.
    """
    if decoder_hidden.dim() == 1:
      decoder_hidden = decoder_hidden.unsqueeze(0)
    B, S, D = encoder_out.shape

    query = decoder_hidden.unsqueeze(1).expand(B, S, -1)
    energy = torch.tanh(self.copy_attn(torch.cat([query, encoder_out], dim=-1)))
    energy = energy.sum(-1)
    energy = energy.masked_fill(~encoder_mask, float("-inf"))
    copy_dist = F.softmax(energy, dim=-1)

    vocab_logits = self.output_proj(decoder_hidden)
    p_gen = torch.sigmoid(self.gen_gate(torch.cat([
      decoder_hidden,
      (encoder_out * encoder_mask.unsqueeze(-1).float()).sum(1) / encoder_mask.sum(1, keepdim=True).clamp(min=1).float(),
    ], dim=-1)))

    vocab_prob = F.softmax(vocab_logits, dim=-1)
    copy_prob = torch.zeros(B, self.vocab_size, device=decoder_hidden.device)
    copy_prob.scatter_add_(1, src_ids.clamp(min=0), copy_dist * (1 - p_gen))

    final_prob = p_gen * vocab_prob + copy_prob
    return torch.log(final_prob.clamp(min=1e-9))
