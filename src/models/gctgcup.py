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
    self.unk_id = 3

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

    # ── Output head with weight tying (helps generation under small data) ──
    self.output_proj = nn.Linear(hidden_dim, vocab_size, bias=False)
    self.output_proj.weight = self.token_embed.weight  # tie input/output embeddings

    # ── Pointer-generator copy head (new comment ≈ old comment + small edits) ──
    # Copies tokens from the (old comment + code-edit) source sequence. This is
    # the main fix for low BLEU/GLEU/METEOR: rare identifiers are copied, not
    # generated from a huge softmax.
    self.copy_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
    self.p_gen = nn.Linear(hidden_dim * 2, 1)

  def _embed_seq(self, ids: torch.Tensor) -> torch.Tensor:
    return self.pos_enc(self.token_embed(ids))

  def _encode_sequence_modal(
    self,
    comment_ids: torch.Tensor,
    edit_ids: torch.Tensor,
    max_len: int = 512,
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TG-CUP Eq. (6): comment + <sep> + edit sequence.

    Returns (encoded, mask, combined_ids). combined_ids is needed by the
    pointer-generator copy head to scatter copy-probabilities onto the vocab.
    """
    sep = torch.full((comment_ids.size(0), 1), 4, dtype=torch.long, device=comment_ids.device)
    combined = torch.cat([comment_ids, sep, edit_ids], dim=1)
    if combined.size(1) > max_len:
      combined = combined[:, :max_len]
    mask = combined.ne(self.pad_id)
    emb = self._embed_seq(combined)
    key_padding = ~mask
    encoded = self.seq_encoder(emb, src_key_padding_mask=key_padding)
    return encoded, mask, combined

  def _mixture_logprobs(
    self,
    decoded: torch.Tensor,
    seq_enc: torch.Tensor,
    seq_mask: torch.Tensor,
    seq_ids: torch.Tensor,
    return_copy: bool = False,
  ):
    """Pointer-generator mixture of generation + copy distributions.

    decoded:  (B, T, D) decoder states
    seq_enc:  (B, S, D) encoder states of (old comment + edit) source
    seq_mask: (B, S) bool, True for real tokens
    seq_ids:  (B, S) vocab ids of the source tokens (for copy scatter)
    Returns log P(token) of shape (B, T, V); also copy_attn (B, T, S) if asked.
    """
    # Force fp32 here: under AMP, probabilities + log(clamp(1e-9)) would
    # underflow in fp16 and produce NaNs.
    with torch.autocast(device_type=decoded.device.type, enabled=False):
      decoded = decoded.float()
      seq_enc = seq_enc.float()
      B, T, _ = decoded.shape

      gen_prob = F.softmax(self.output_proj(decoded), dim=-1)          # (B, T, V)

      keys = self.copy_key(seq_enc)                                    # (B, S, D)
      scores = torch.bmm(decoded, keys.transpose(1, 2))                # (B, T, S)
      scores = scores / (self.hidden_dim ** 0.5)
      scores = scores.masked_fill(~seq_mask.unsqueeze(1), float("-inf"))
      copy_attn = F.softmax(scores, dim=-1)                            # (B, T, S)
      context = torch.bmm(copy_attn, seq_enc)                          # (B, T, D)

      p_gen = torch.sigmoid(self.p_gen(torch.cat([decoded, context], dim=-1)))  # (B, T, 1)

      final = p_gen * gen_prob                                         # (B, T, V)
      src_index = seq_ids.unsqueeze(1).expand(B, T, seq_ids.size(1))   # (B, T, S)
      final = final.scatter_add(2, src_index, (1.0 - p_gen) * copy_attn)
      logp = torch.log(final.clamp(min=1e-9))
      if return_copy:
        return logp, copy_attn
      return logp

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
    code_h: torch.Tensor,
    code_mask: torch.Tensor,
    graphs: List[ASTDiffGraph],
  ) -> torch.Tensor:
    """code_h / code_mask: precomputed GraphCodeBERT code-pair encoding (reused)."""
    device = src_ids.device

    seq_enc, seq_mask, seq_ids = self._encode_sequence_modal(src_ids, edit_ids)
    graph_enc, graph_mask = self._encode_graph_batch(graphs, device)

    code_sem = self.fuse_code_sem(code_h)

    # Multimodal memory: GraphCodeBERT code semantics + edit/comment seq + AST-GGNN
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

    # Pointer-generator: copy over the (old comment + edit) source sequence
    return self._mixture_logprobs(decoded, seq_enc, seq_mask, seq_ids)

  def _diff_aware_nll(
    self,
    logprobs: torch.Tensor,
    targets: torch.Tensor,
    src_ids: torch.Tensor,
    edit_weight: float = 3.0,
  ) -> torch.Tensor:
    """NLL that UP-WEIGHTS target tokens absent from the old comment.

    For outdated samples the new comment ≈ old comment + a few edits, so plain
    NLL is minimised by copying the old comment (high BLEU, ~0 exact match).
    By weighting the *edited* tokens (those not present in the old comment) we
    force the model to actually learn the edit instead of collapsing to copy.
    """
    B, T, V = logprobs.shape
    nll = F.nll_loss(
      logprobs.reshape(-1, V), targets.reshape(-1),
      ignore_index=self.pad_id, reduction="none",
    ).view(B, T)

    in_src = torch.zeros(B, V, dtype=torch.bool, device=targets.device)
    in_src.scatter_(1, src_ids.clamp(min=0), True)
    tgt_in_src = in_src.gather(1, targets.clamp(min=0))            # (B, T)

    weights = torch.where(
      tgt_in_src,
      torch.ones_like(targets, dtype=logprobs.dtype),
      torch.full_like(targets, edit_weight, dtype=logprobs.dtype),
    )
    weights = weights.masked_fill(targets.eq(self.pad_id), 0.0)

    nll = nll * weights
    return nll.sum() / weights.sum().clamp(min=1.0)

  def forward(
    self,
    batch: Dict,
    teacher_forcing: bool = True,
    pos_weight: Optional[torch.Tensor] = None,
  ) -> Dict[str, torch.Tensor]:
    device = next(self.parameters()).device
    # Encode code pair ONCE with GraphCodeBERT, then reuse for detection + update.
    code_h, code_mask = self.code_semantic.encode_code_pair(
      batch["src_methods"], batch["dst_methods"], device
    )
    det_logits = self.detector(
      batch["src_methods"], batch["dst_methods"], batch["src_descs"],
      code_cache=(code_h, code_mask),
    )
    det_loss = F.binary_cross_entropy_with_logits(
      det_logits, batch["labels"], pos_weight=pos_weight
    )

    update_mask = batch["labels"].bool()
    result = {"det_loss": det_loss, "det_logits": det_logits}

    if update_mask.any() and teacher_forcing:
      idx = update_mask.nonzero(as_tuple=True)[0]
      upd_logprobs = self.forward_update(
        batch["src_ids"][idx],
        batch["edit_ids"][idx],
        batch["dst_ids"][idx],
        code_h[idx],
        code_mask[idx],
        [batch["graphs"][i] for i in idx.tolist()],
      )
      targets = batch["dst_ids"][idx, 1:]
      # Diff-aware NLL: up-weight edited tokens so the model learns to EDIT,
      # not just copy the old comment (the cause of ~0 exact-match accuracy).
      upd_loss = self._diff_aware_nll(
        upd_logprobs, targets, batch["src_ids"][idx],
      )
      result["upd_loss"] = upd_loss
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
    decode_fn=None,
    force_update: bool = False,
  ) -> Tuple[List[str], List[List[str]]]:
    """Generate updated comments via beam search, decoding to text.

    With byte-level BPE there is no OOV: the copy mixture already places mass on
    the correct in-vocab subword id, so we beam-search over ids and detokenize
    with decode_fn (the tokenizer's decoder). decode_fn: List[int] -> str.

    force_update=True bypasses the detection gate (TG-CUP-style: always update).

    Returns:
      pred_texts:  List[str]        — best predicted comment per sample
      beam_texts:  List[List[str]]  — beam candidates per sample (for Recall@k)
    """
    device = src_ids.device
    B = src_ids.size(0)
    comments = comments or [""] * B
    src_descs = src_descs or [""] * B
    if decode_fn is None:
      decode_fn = lambda ids: " ".join(str(t) for t in ids)

    code_h, code_mask = self.code_semantic.encode_code_pair(old_codes, new_codes, device)
    det_logits = self.detector(old_codes, new_codes, comments, code_cache=(code_h, code_mask))
    det_probs = torch.sigmoid(det_logits)
    needs_update = torch.ones_like(det_probs, dtype=torch.bool) if force_update \
      else det_probs >= det_threshold

    pred_texts: List[str] = []
    beam_texts: List[List[str]] = []

    for i in range(B):
      if not needs_update[i]:
        pred_texts.append(src_descs[i])
        beam_texts.append([src_descs[i]] * 5)
        continue

      seq_enc, seq_mask, seq_ids = self._encode_sequence_modal(src_ids[i:i+1], edit_ids[i:i+1])
      graph_enc, graph_mask = self._encode_graph_batch([graphs[i]], device)
      code_sem = self.fuse_code_sem(code_h[i:i+1])
      memory = torch.cat([code_sem, seq_enc, graph_enc], dim=1)
      mem_mask = torch.cat([code_mask[i:i+1], seq_mask, graph_mask], dim=1)

      # beam = (ids_tensor, score)
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
          log_probs = self._mixture_logprobs(dec[:, -1:], seq_enc, seq_mask, seq_ids)[0, -1]

          # Block degenerate loops (same token 3x in a row).
          if seq.size(1) >= 2:
            last, last2 = seq[0, -1].item(), seq[0, -2].item()
            if last == last2 and last not in (self.sos_id, self.eos_id, self.pad_id):
              log_probs[last] -= 1e9
          # Require at least one real token before EOS.
          if seq.size(1) == 1:
            log_probs[self.eos_id] -= 1e9

          seq_len = seq.size(1)
          topk = torch.topk(log_probs, beam_size, dim=-1)
          for k in range(beam_size):
            tok = topk.indices[k].view(1, 1)
            new_score = (score * seq_len + topk.values[k].item()) / (seq_len + 1)
            new_beams.append((torch.cat([seq, tok], dim=1), new_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
        if all(s[0, -1].item() == self.eos_id for s, _ in beams):
          break

      finished = sorted(completed + beams, key=lambda x: x[1], reverse=True)[:beam_size]
      cand_id_lists = [
        [t for t in seq[0, 1:].tolist() if t != self.eos_id] for seq, _ in finished
      ] or [[]]
      cand_texts = [decode_fn(ids) for ids in cand_id_lists]
      pred_texts.append(cand_texts[0])
      beam_texts.append(cand_texts)

    return pred_texts, beam_texts
