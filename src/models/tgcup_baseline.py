"""
Simplified TG-CUP baseline (paper Section 3) for ablation comparison.
Transformer encoder + GGNN + decoder WITHOUT GraphCodeBERT / detection / local edit.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ast_diff import ASTDiffGraph
from .ggnn import GGNN
from .gctgcup import PositionalEncoding


class TGCUPBaseline(nn.Module):
  def __init__(self, vocab_size: int, hidden_dim: int = 256, num_heads: int = 8,
               num_encoder_layers: int = 4, num_decoder_layers: int = 4,
               ggnn_steps: int = 6, dropout: float = 0.1):
    super().__init__()
    self.vocab_size = vocab_size
    self.pad_id = 0
    self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
    self.pos_enc = PositionalEncoding(hidden_dim)
    self.ggnn = GGNN(hidden_dim, vocab_size, num_steps=ggnn_steps)

    enc_layer = nn.TransformerEncoderLayer(
      d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
      dropout=dropout, batch_first=True,
    )
    self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
    dec_layer = nn.TransformerDecoderLayer(
      d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
      dropout=dropout, batch_first=True,
    )
    self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
    self.output_proj = nn.Linear(hidden_dim, vocab_size)

  def _embed(self, ids):
    return self.pos_enc(self.token_embed(ids))

  def _encode_graph_batch(self, graphs: List[ASTDiffGraph], device, vocab_size):
    node_ids_batch = []
    for g in graphs:
      ids = [hash(n.value if n.is_value_node else n.node_type) % (vocab_size - 10) + 10 for n in g.nodes]
      node_ids_batch.append(torch.tensor(ids, dtype=torch.long, device=device))
    return self.ggnn.forward_batch(node_ids_batch, graphs)

  def forward(self, batch: Dict, teacher_forcing: bool = True) -> Dict[str, torch.Tensor]:
    device = batch["src_ids"].device
    sep = torch.full((batch["src_ids"].size(0), 1), 4, dtype=torch.long, device=device)
    combined = torch.cat([batch["src_ids"], sep, batch["edit_ids"]], dim=1)[:, :512]
    mask = combined.ne(self.pad_id)
    seq_enc = self.seq_encoder(self._embed(combined), src_key_padding_mask=~mask)
    graph_enc, graph_mask = self._encode_graph_batch(batch["graphs"], device, self.vocab_size)
    memory = torch.cat([seq_enc, graph_enc], dim=1)
    mem_mask = torch.cat([mask, graph_mask], dim=1)

    tgt_in = batch["dst_ids"][:, :-1]
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=device)
    decoded = self.decoder(
      tgt=self._embed(tgt_in), memory=memory, tgt_mask=tgt_mask,
      tgt_key_padding_mask=tgt_in.eq(self.pad_id),
      memory_key_padding_mask=~mem_mask,
    )
    logits = self.output_proj(decoded)
    loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), batch["dst_ids"][:, 1:].reshape(-1), ignore_index=self.pad_id)
    return {"loss": loss, "upd_loss": loss, "det_loss": torch.tensor(0.0, device=device)}
