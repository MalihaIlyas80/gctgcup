"""
Gated Graph Neural Network for AST-Difference Graph (TG-CUP Section 3.1).
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from src.data.ast_diff import ASTDiffGraph, EDGE_AST, EDGE_COMPOUND, EDGE_ORDER, EDGE_UPDATE


class GGNN(nn.Module):
  NUM_EDGE_TYPES = 4

  def __init__(self, hidden_dim: int, vocab_size: int, num_steps: int = 6):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.num_steps = num_steps
    self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

    self.W_z = nn.Linear(hidden_dim * self.NUM_EDGE_TYPES, hidden_dim)
    self.U_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.W_r = nn.Linear(hidden_dim * self.NUM_EDGE_TYPES, hidden_dim)
    self.U_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.W_h = nn.Linear(hidden_dim * self.NUM_EDGE_TYPES, hidden_dim)
    self.U_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

  def _propagate(self, h: torch.Tensor, adj_lists: List[List[List[int]]]) -> torch.Tensor:
    """h: (N, D); adj_lists: per edge type list of adjacency lists."""
  # batch handled outside – single graph at a time in collate
    N, D = h.shape
    msgs = []
    for et in range(self.NUM_EDGE_TYPES):
      adj = adj_lists[et]
      m = torch.zeros(N, D, device=h.device)
      for src, nbrs in enumerate(adj):
        if nbrs:
          m[src] = h[nbrs].mean(0)
      msgs.append(m)
    return torch.cat(msgs, dim=-1)

  def forward_single(self, node_ids: torch.Tensor, graph: ASTDiffGraph) -> torch.Tensor:
    """Return value-node states (L, D)."""
    device = node_ids.device
    h = self.embed(node_ids)

    adj_lists = []
    for et in range(self.NUM_EDGE_TYPES):
      adj = [[] for _ in range(graph.num_nodes)]
      for src, dst, edge_t in graph.edges:
        if edge_t == et:
          adj[src].append(dst)
      adj_lists.append(adj)

    for _ in range(self.num_steps):
      a = self._propagate(h, adj_lists)
      z = torch.sigmoid(self.W_z(a) + self.U_z(h))
      r = torch.sigmoid(self.W_r(a) + self.U_r(h))
      h_tilde = torch.tanh(self.W_h(a) + self.U_h(r * h))
      h = (1 - z) * h + z * h_tilde

    value_idx = graph.value_node_indices or [0]
    return h[value_idx]

  def forward_batch(
    self,
    node_ids_batch: List[torch.Tensor],
    graphs: List[ASTDiffGraph],
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad value-node outputs. Returns (B, max_L, D), mask."""
    outputs = []
    for node_ids, g in zip(node_ids_batch, graphs):
      v = self.forward_single(node_ids.to(next(self.parameters()).device), g)
      outputs.append(v)

    max_len = max(o.size(0) for o in outputs)
    D = outputs[0].size(-1)
    device = outputs[0].device
    padded = torch.zeros(len(outputs), max_len, D, device=device)
    mask = torch.zeros(len(outputs), max_len, dtype=torch.bool, device=device)
    for i, o in enumerate(outputs):
      padded[i, : o.size(0)] = o
      mask[i, : o.size(0)] = True
    return padded, mask
