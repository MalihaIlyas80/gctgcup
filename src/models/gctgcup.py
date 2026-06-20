"""
GC-TGCUP: GraphCodeBERT-enhanced Two-Stage Comment Updating Framework.

Extends TG-CUP with:
  1. Stage-1 detection (OutdatedCommentDetector)
  2. GraphCodeBERT semantic code encoding
  3. AST-Difference GGNN (structural, from TG-CUP)
  4. Local edit decoder for long comments
  5. Transformer encoder-decoder for update generation
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ast_diff import ASTDiffGraph

from .detection import OutdatedCommentDetector
from .ggnn import GGNN
from .graphcodebert_encoder import GraphCodeBERTEncoder
from .local_editor import LocalEditDecoder


class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, max_len: int = 1024):
    super().__init__()
    self.d_model = d_model
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.register_buffer("pe", pe.unsqueeze(0))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    seq_len = x.size(1)
    if seq_len > self.pe.size(1):
      self._extend_pe(seq_len, x.device)
    return x + self.pe[:, :seq_len]

  def _extend_pe(self, seq_len: int, device: torch.device) -> None:
    pe = torch.zeros(seq_len, self.d_model, device=device)
    pos = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, self.d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    self.pe = pe.unsqueeze(0)


class GCTGCUP(nn.Module):
  def __init__(
    self,
    vocab_size: int,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    ggnn_steps: int = 6,
    dropout: float = 0.1,
    graphcodebert_name: str = "microsoft/graphcodebert-base",
    freeze_bert: bool = False,
    long_threshold: int = 25,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.long_threshold = long_threshold
    self.pad_id = 0
    self.sos_id = 1
    self.eos_id = 2

    # ── shared embeddings ──
    self.token_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
    self.pos_enc = PositionalEncoding(hidden_dim)

    # ── Stage 1: Detection ──
    self.detector = OutdatedCommentDetector(
      hidden_dim=hidden_dim,
      graphcodebert_name=graphcodebert_name,
      freeze_bert=freeze_bert,
      dropout=dropout,
    )

    # ── Stage 2: reuse GraphCodeBERT from detector ──
    self.code_semantic = self.detector.code_encoder
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

    # fusion projections (TG-CUP Eq. 11-12: attend seq then graph)
    self.fuse_code_sem = nn.Linear(hidden_dim, hidden_dim)
    self.local_editor = LocalEditDecoder(hidden_dim, vocab_size)
    self.output_proj = nn.Linear(hidden_dim, vocab_size)

  def _embed_seq(self, ids: torch.Tensor) -> torch.Tensor:
    return self.pos_enc(self.token_embed(ids))

  def _encode_sequence_modal(
    self,
    comment_ids: torch.Tensor,
    edit_ids: torch.Tensor,
    max_len: int = 512,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """TG-CUP Eq. (6): comment + <sep> + edit sequence."""
    sep = torch.full((comment_ids.size(0), 1), 4, dtype=torch.long, device=comment_ids.device)
    combined = torch.cat([comment_ids, sep, edit_ids], dim=1)
    if combined.size(1) > max_len:
      combined = combined[:, :max_len]
    mask = combined.ne(self.pad_id)
    emb = self._embed_seq(combined)
    key_padding = ~mask
    encoded = self.seq_encoder(emb, src_key_padding_mask=key_padding)
    return encoded, mask

  def _encode_graph_batch(
    self,
    graphs: List[ASTDiffGraph],
    device: torch.device,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    node_ids_batch = []
    for g in graphs:
      node_text_ids = []
      for n in g.nodes:
        text = n.value if n.is_value_node else n.node_type
        node_text_ids.append(int(hashlib.md5(text.encode()).hexdigest(), 16) % (self.vocab_size - 10) + 10)
      node_ids_batch.append(torch.tensor(node_text_ids, dtype=torch.long, device=device))
    return self.ggnn.forward_batch(node_ids_batch, graphs)

  def detect(
    self,
    old_codes: List[str],
    new_codes: List[str],
    comments: List[str],
  ) -> torch.Tensor:
    return self.detector(old_codes, new_codes, comments)

  def forward_update(
    self,
    src_ids: torch.Tensor,
    edit_ids: torch.Tensor,
    dst_ids: torch.Tensor,
    old_codes: List[str],
    new_codes: List[str],
    graphs: List[ASTDiffGraph],
    is_long: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    device = src_ids.device
    B = src_ids.size(0)

    seq_enc, seq_mask = self._encode_sequence_modal(src_ids, edit_ids)
    graph_enc, graph_mask = self._encode_graph_batch(graphs, device)

    code_sem, code_mask = self.code_semantic.encode_code_pair(old_codes, new_codes, device)
    code_sem = self.fuse_code_sem(code_sem)

    # prepend code semantics to sequence memory
    memory = torch.cat([code_sem, seq_enc, graph_enc], dim=1)
    mem_mask = torch.cat([code_mask, seq_mask, graph_mask], dim=1)

    tgt_in = dst_ids[:, :-1]
    tgt_emb = self._embed_seq(tgt_in)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_in.size(1), device=device)
    tgt_pad = tgt_in.eq(self.pad_id)

    decoded = self.decoder(
      tgt=tgt_emb,
      memory=memory,
      tgt_mask=tgt_mask,
      tgt_key_padding_mask=tgt_pad,
      memory_key_padding_mask=~mem_mask,
    )

    logits = self.output_proj(decoded)
    return logits

  def forward(
    self,
    batch: Dict,
    teacher_forcing: bool = True,
    pos_weight: Optional[torch.Tensor] = None,
  ) -> Dict[str, torch.Tensor]:
    det_logits = self.detect(batch["src_methods"], batch["dst_methods"], batch["src_descs"])
    det_loss = F.binary_cross_entropy_with_logits(
      det_logits, batch["labels"], pos_weight=pos_weight
    )

    update_mask = batch["labels"].bool()
    result = {"det_loss": det_loss, "det_logits": det_logits}

    if update_mask.any() and teacher_forcing:
      idx = update_mask.nonzero(as_tuple=True)[0]
      upd_logits = self.forward_update(
        batch["src_ids"][idx],
        batch["edit_ids"][idx],
        batch["dst_ids"][idx],
        [batch["src_methods"][i] for i in idx.tolist()],
        [batch["dst_methods"][i] for i in idx.tolist()],
        [batch["graphs"][i] for i in idx.tolist()],
        batch["is_long"][idx] if "is_long" in batch else None,
      )
      targets = batch["dst_ids"][idx, 1:]
      upd_loss = F.cross_entropy(
        upd_logits.reshape(-1, self.vocab_size),
        targets.reshape(-1),
        ignore_index=self.pad_id,
        label_smoothing=0.1,
      )
      # Edit-aware loss: reward both copy (kept tokens) and novel tokens (added)
      # This teaches the model to selectively keep AND delete, improving SARI
      src_batch = batch["src_ids"][idx]
      token_loss = F.cross_entropy(
        upd_logits.reshape(-1, self.vocab_size),
        targets.reshape(-1),
        ignore_index=self.pad_id,
        reduction='none',
      ).view(targets.size())
      edit_weight = torch.ones_like(targets, dtype=torch.float)
      for b in range(targets.size(0)):
        src_set = set(src_batch[b].tolist()) - {0, 1, 2, 3, 4}
        dst_set = set(targets[b].tolist()) - {0, 1, 2, 3, 4}
        for t in range(targets.size(1)):
          tok = targets[b, t].item()
          if tok == self.pad_id:
            edit_weight[b, t] = 0.0
          elif tok in src_set and tok in dst_set:
            edit_weight[b, t] = 1.0   # kept token
          elif tok not in src_set and tok in dst_set:
            edit_weight[b, t] = 2.0   # added token (harder, more reward)
      weighted_loss = (token_loss * edit_weight).sum() / edit_weight.sum().clamp(min=1)
      upd_loss = 0.6 * upd_loss + 0.4 * weighted_loss
      result["upd_loss"] = upd_loss
      result["upd_logits"] = upd_logits
    else:
      result["upd_loss"] = torch.tensor(0.0, device=det_logits.device)

    result["loss"] = 0.3 * det_loss + 0.7 * result["upd_loss"]
    return result

  @torch.no_grad()
  def generate(
    self,
    src_ids: torch.Tensor,
    edit_ids: torch.Tensor,
    old_codes: List[str],
    new_codes: List[str],
    graphs: List[ASTDiffGraph],
    max_len: int = 50,
    beam_size: int = 5,
    det_threshold: float = 0.45,
    comments: Optional[List[str]] = None,
    src_descs: Optional[List[str]] = None,
    return_beam_candidates: bool = False,
  ) -> tuple:
    """Generate updated comments with detection gate + beam search.

    For samples where detector predicts no update needed, returns src_descs[i]
    directly (no tokenization round-trip) so label=0 exact matches are preserved.

    Returns:
      token_ids: List[List[int]] — generated token ids (empty list for no-update)
      no_update_texts: List[Optional[str]] — src_desc for no-update samples, None otherwise
      beam_candidates (optional): List[List[List[int]]]
    """
    device = src_ids.device
    B = src_ids.size(0)
    comments = comments or [""] * B
    src_descs = src_descs or [""] * B

    det_logits = self.detect(old_codes, new_codes, comments)
    det_probs = torch.sigmoid(det_logits)
    needs_update = det_probs >= det_threshold

    token_ids: List[List[int]] = []
    no_update_texts: List[Optional[str]] = []
    beam_results: List[List[List[int]]] = []

    for i in range(B):
      # ── No-update path: return original comment string directly ──
      # This avoids tokenization round-trip errors that destroy exact match
      if not needs_update[i]:
        token_ids.append([])
        no_update_texts.append(src_descs[i])
        beam_results.append([[]])
        continue

      no_update_texts.append(None)

      # ── Encode inputs ──
      seq_enc, seq_mask = self._encode_sequence_modal(src_ids[i:i+1], edit_ids[i:i+1])
      graph_enc, graph_mask = self._encode_graph_batch([graphs[i]], device)
      code_sem, code_mask = self.code_semantic.encode_code_pair([old_codes[i]], [new_codes[i]], device)
      code_sem = self.fuse_code_sem(code_sem)
      memory = torch.cat([code_sem, seq_enc, graph_enc], dim=1)
      mem_mask = torch.cat([code_mask, seq_mask, graph_mask], dim=1)

      # ── Pre-compute source token info ──
      src_token_set = set(src_ids[i].tolist()) - {0, 1, 2, 3, 4}
      src_len = int(src_ids[i].ne(0).sum().item())
      min_len = max(3, src_len // 2)

      # ── Beam search ──
      beams: List[tuple] = [(torch.tensor([[self.sos_id]], device=device), 0.0)]
      completed: List[tuple] = []

      for _ in range(max_len):
        if not beams:
          break
        new_beams: List[tuple] = []

        for seq, score in beams:
          if seq[0, -1].item() == self.eos_id:
            completed.append((seq, score))
            continue

          tgt_emb = self._embed_seq(seq)
          tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1), device=device)
          dec = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=~mem_mask,
          )
          log_probs = F.log_softmax(self.output_proj(dec[:, -1]), dim=-1)

          # Soft copy boost (reduced from 1.0 to 0.4 to allow deletions for SARI)
          for tok_id in src_token_set:
            if tok_id < log_probs.size(-1):
              log_probs[0, tok_id] += 0.4

          # Repetition penalty for non-source tokens only
          seen = set(seq[0].tolist()) - src_token_set
          for tok_id in seen:
            if tok_id < log_probs.size(-1):
              log_probs[0, tok_id] -= 1.2

          # Block EOS until minimum length reached
          if seq.size(1) <= min_len:
            log_probs[0, self.eos_id] -= 1e9

          seq_len = seq.size(1)
          topk = torch.topk(log_probs, beam_size, dim=-1)
          for k in range(beam_size):
            tok = topk.indices[0, k].unsqueeze(0).unsqueeze(0)
            new_score = (score * seq_len + topk.values[0, k].item()) / (seq_len + 1)
            new_beams.append((torch.cat([seq, tok], dim=1), new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(s[0, -1].item() == self.eos_id for s, _ in beams):
          break

      # ── Edit-quality reranking for SARI/GLEU improvement ──
      # Combines LM score with edit quality (rewards balanced keep+delete+add)
      all_finished = sorted(completed + beams, key=lambda x: x[1], reverse=True)[:beam_size]

      def edit_quality(cand_ids: List[int]) -> float:
        cand_set = set(cand_ids) - {0, 1, 2, 3, 4}
        if not src_token_set:
          return 0.0
        kept = len(cand_set & src_token_set)
        deleted = len(src_token_set - cand_set)
        added = len(cand_set - src_token_set)
        keep_ratio = kept / len(src_token_set)
        edit_ratio = (deleted + added) / max(len(src_token_set) + len(cand_set), 1)
        # Reward: close to source length, makes some meaningful edits
        keep_score = 1.0 - abs(keep_ratio - 0.8) * 2
        return 0.4 * edit_ratio + 0.6 * max(keep_score, 0.0)

      reranked = []
      for seq, lm_score in all_finished:
        cand = [t for t in seq[0, 1:].tolist() if t not in (self.eos_id,)]
        eq = edit_quality(cand)
        # Weighted combination: LM score (primary) + edit quality (secondary)
        combined = 0.75 * lm_score + 0.25 * eq
        reranked.append((seq, lm_score, combined, cand))

      reranked.sort(key=lambda x: x[2], reverse=True)
      best_tok_ids = reranked[0][3]
      beam_cands = [r[3] for r in reranked]

      token_ids.append(best_tok_ids)
      beam_results.append(beam_cands)

    if return_beam_candidates:
      return token_ids, no_update_texts, beam_results
    return token_ids, no_update_texts
