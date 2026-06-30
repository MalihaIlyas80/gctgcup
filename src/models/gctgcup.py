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

import difflib
import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.ast_diff import ASTDiffGraph
from src.data.cleaning import apply_comment_code_renames, comment_has_code_rename

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
    edit_weight: float = 8.0,
    det_loss_weight: float = 0.15,
    upd_loss_weight: float = 0.85,
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.long_threshold = long_threshold
    self.edit_weight = edit_weight
    self.det_loss_weight = det_loss_weight
    self.upd_loss_weight = upd_loss_weight
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

  @staticmethod
  def _copy_surface_aligned(
    src_tokens: List[str],
    edit_ids_row: torch.Tensor,
    id2token: Dict[int, str],
  ) -> List[str]:
    """One surface string per encoder position (comment + sep + edit)."""
    row = ["<s>"] + list(src_tokens) + ["</s>", "<sep>"]
    for tid in edit_ids_row.tolist():
      row.append(id2token.get(int(tid), "<unk>"))
    return row

  def _mixture_logprobs(
    self,
    decoded: torch.Tensor,
    seq_enc: torch.Tensor,
    seq_mask: torch.Tensor,
    seq_ids: torch.Tensor,
    return_components: bool = False,
  ):
    """Pointer-generator mixture of generation + copy distributions.

    decoded:  (B, T, D) decoder states
    seq_enc:  (B, S, D) encoder states of (old comment + edit) source
    seq_mask: (B, S) bool, True for real tokens
    seq_ids:  (B, S) vocab ids of the source tokens (for copy scatter)
    Returns log P(token) of shape (B, T, V), or (log_probs, p_gen, copy_attn).
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
      log_probs = torch.log(final.clamp(min=1e-9))
      if return_components:
        return log_probs, p_gen, copy_attn
      return log_probs

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

  def _sequence_diff_aware_nll(
    self,
    logprobs: torch.Tensor,
    targets: torch.Tensor,
    src_tokens_batch: List[List[str]],
    dst_tokens_batch: List[List[str]],
    edit_weight: float = 5.0,
  ) -> torch.Tensor:
    """Up-weight only tokens that differ between old and new comment (aligned diff)."""
    B, T, V = logprobs.shape
    nll = F.nll_loss(
      logprobs.reshape(-1, V), targets.reshape(-1),
      ignore_index=self.pad_id, reduction="none",
    ).view(B, T)

    weights = torch.ones_like(targets, dtype=logprobs.dtype)
    for b in range(B):
      dst_toks = dst_tokens_batch[b]
      pos_w = torch.ones(T, device=targets.device, dtype=logprobs.dtype)
      sm = difflib.SequenceMatcher(
        None, src_tokens_batch[b], dst_toks, autojunk=False,
      )
      for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
          continue
        for j in range(j1, min(j2, len(dst_toks))):
          if j < T:
            pos_w[j] = edit_weight
      weights[b] = pos_w

    weights = weights.masked_fill(targets.eq(self.pad_id), 0.0)
    nll = nll * weights
    return nll.sum() / weights.sum().clamp(min=1.0)

  def _zero_trainable_loss(self) -> torch.Tensor:
    """Differentiable zero — safe backward when a batch has no update targets."""
    loss = torch.tensor(0.0, device=next(self.parameters()).device)
    for p in self.parameters():
      if p.requires_grad:
        loss = loss + p.sum() * 0.0
    return loss

  def forward(
    self,
    batch: Dict,
    teacher_forcing: bool = True,
    pos_weight: Optional[torch.Tensor] = None,
    phase: str = "joint",
  ) -> Dict[str, torch.Tensor]:
    """phase: 'joint' | 'detection' (stage-1 only) | 'update' (stage-2 only)."""
    device = next(self.parameters()).device
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
    upd_loss = torch.tensor(0.0, device=det_logits.device)

    if phase != "detection" and update_mask.any() and teacher_forcing:
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
      src_tok = [batch["src_tokens_list"][i] for i in idx.tolist()]
      dst_tok = [batch["dst_tokens_list"][i] for i in idx.tolist()]
      upd_loss = self._sequence_diff_aware_nll(
        upd_logprobs, targets, src_tok, dst_tok,
        edit_weight=self.edit_weight,
      )
    elif phase == "update":
      upd_loss = self._zero_trainable_loss()
    result["upd_loss"] = upd_loss

    if phase == "detection":
      result["loss"] = det_loss
    elif phase == "update":
      result["loss"] = upd_loss
    else:
      result["loss"] = (
        self.det_loss_weight * det_loss + self.upd_loss_weight * upd_loss
      )
    return result

  def set_training_phase(self, phase: str) -> None:
    """Freeze/unfreeze submodules for sequential two-stage training."""
    upd_modules = [
      self.token_embed, self.pos_enc, self.ggnn, self.seq_encoder,
      self.decoder, self.output_proj, self.copy_key, self.p_gen, self.fuse_code_sem,
    ]
    if phase == "detection":
      for m in upd_modules:
        for p in m.parameters():
          p.requires_grad = False
      for p in self.detector.parameters():
        p.requires_grad = True
    elif phase == "update":
      # Keep detection head frozen; fine-tune GraphCodeBERT + update decoder.
      for p in self.detector.classifier.parameters():
        p.requires_grad = False
      for p in self.detector.comment_proj.parameters():
        p.requires_grad = False
      for p in self.detector.cross_attn.parameters():
        p.requires_grad = False
      for p in self.code_semantic.parameters():
        p.requires_grad = True
      for m in upd_modules:
        for p in m.parameters():
          p.requires_grad = True
    else:
      for p in self.parameters():
        p.requires_grad = True

  @classmethod
  def merge_src_prediction(cls, src_tokens: List[str], pred_tokens: List[str]) -> str:
    """Merge copy-stable spans from src with edited spans from the model."""
    sm = difflib.SequenceMatcher(None, src_tokens, pred_tokens, autojunk=False)
    merged: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
      if tag == "equal":
        merged.extend(src_tokens[i1:i2])
      elif tag in ("replace", "insert"):
        merged.extend(pred_tokens[j1:j2])
    return cls._format_surface(merged)

  @classmethod
  def _dedupe_candidates(cls, candidates: List[str]) -> List[str]:
    seen, out = set(), []
    for c in candidates:
      c = c.strip()
      if c and c not in seen:
        seen.add(c)
        out.append(c)
    return out

  @classmethod
  def _pick_primary_prediction(
    cls,
    src_tokens: List[str],
    model_pred: str,
    rule_pred: str,
    merge_pred: str,
    code_change_seq: list,
  ) -> str:
    """Prefer rule-based rename when code edits touch the comment."""
    if rule_pred and comment_has_code_rename(src_tokens, code_change_seq):
      return rule_pred
    if merge_pred:
      return merge_pred
    return model_pred

  _SKIP_OUTPUT = frozenset({
    "<sep>", "<s>", "</s>", "<pad>", "<unk>",
    "<before>", "<after>", "<equal>", "<replace>", "<insert>", "<delete>",
  })

  @classmethod
  def _format_surface(cls, tokens: List[str]) -> str:
    return " ".join(t for t in tokens if t not in cls._SKIP_OUTPUT)

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
    src_tokens_list: Optional[List[List[str]]] = None,
    code_change_seqs: Optional[List[list]] = None,
    id2token: Optional[Dict[int, str]] = None,
    return_beam_candidates: bool = False,
    force_update: bool = False,
  ) -> tuple:
    """Two-stage inference: detect outdated → generate update (or keep original).

    force_update=True skips the detection gate (TG-CUP-style update eval only).
    Surface-token beam search copies OOV identifiers verbatim for exact-match accuracy.
    """
    device = src_ids.device
    B = src_ids.size(0)
    comments = comments or [""] * B
    src_descs = src_descs or [""] * B
    src_tokens_list = src_tokens_list or [[] for _ in range(B)]
    code_change_seqs = code_change_seqs or [[] for _ in range(B)]
    id2token = id2token or {}

    code_h, code_mask = self.code_semantic.encode_code_pair(old_codes, new_codes, device)
    det_logits = self.detector(old_codes, new_codes, comments, code_cache=(code_h, code_mask))
    det_probs = torch.sigmoid(det_logits)
    if force_update:
      needs_update = torch.ones_like(det_probs, dtype=torch.bool)
    else:
      needs_update = det_probs >= det_threshold

    token_ids: List[List[int]] = []
    no_update_texts: List[Optional[str]] = []
    surface_texts: List[str] = []
    beam_results: List[List[List[int]]] = []
    beam_surface_results: List[List[str]] = []

    for i in range(B):
      if not needs_update[i]:
        token_ids.append([])
        no_update_texts.append(src_descs[i])
        surface_texts.append(src_descs[i])
        beam_results.append([[]])
        beam_surface_results.append([src_descs[i]])
        continue

      no_update_texts.append(None)

      seq_enc, seq_mask, seq_ids = self._encode_sequence_modal(src_ids[i:i+1], edit_ids[i:i+1])
      graph_enc, graph_mask = self._encode_graph_batch([graphs[i]], device)
      code_sem = self.fuse_code_sem(code_h[i:i+1])
      code_mask_i = code_mask[i:i+1]
      memory = torch.cat([code_sem, seq_enc, graph_enc], dim=1)
      mem_mask = torch.cat([code_mask_i, seq_mask, graph_mask], dim=1)

      combined_surface = self._copy_surface_aligned(
        src_tokens_list[i], edit_ids[i], id2token,
      )

      # Surface-aware beam: copy actions emit source surface tokens (OOV-safe).
      beams: List[tuple] = [(torch.tensor([[self.sos_id]], device=device), [], 0.0)]
      completed: List[tuple] = []

      for _ in range(max_len):
        if not beams:
          break
        new_beams: List[tuple] = []
        for seq, surf, score in beams:
          if seq[0, -1].item() == self.eos_id:
            completed.append((seq, surf, score))
            continue

          tgt_emb = self._embed_seq(seq)
          tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq.size(1), device=device)
          dec = self.decoder(
            tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask,
            memory_key_padding_mask=~mem_mask,
          )
          log_probs, p_gen, copy_attn = self._mixture_logprobs(
            dec[:, -1:], seq_enc, seq_mask, seq_ids, return_components=True,
          )
          lp = log_probs[0, 0]
          pg = float(p_gen[0, 0, 0].item())
          ca = copy_attn[0, 0]
          seq_len = seq.size(1)

          if seq.size(1) == 1:
            lp = lp.clone()
            lp[self.eos_id] -= 1e9

          actions: List[tuple] = []

          copy_scores = (1.0 - pg) * ca
          copy_scores = copy_scores.masked_fill(~seq_mask[0], float("-inf"))
          n_copy = min(beam_size, int((copy_scores > -1e8).sum().item()))
          if n_copy > 0:
            top_copy = torch.topk(copy_scores, n_copy)
            for val, pos in zip(top_copy.values, top_copy.indices):
              pos_i = int(pos.item())
              tok_id = int(seq_ids[0, pos_i].item())
              tok_str = combined_surface[pos_i] if pos_i < len(combined_surface) else id2token.get(tok_id, "<unk>")
              if tok_str in self._SKIP_OUTPUT:
                continue
              actions.append(("copy", tok_id, tok_str, float(val.item())))

          gen_lp = lp.clone()
          if seq.size(1) >= 2:
            last, last2 = seq[0, -1].item(), seq[0, -2].item()
            if last == last2 and last not in (self.sos_id, self.eos_id, self.pad_id):
              gen_lp[last] -= 1e9

          top_gen = torch.topk(gen_lp, beam_size)
          for val, tok_id_t in zip(top_gen.values, top_gen.indices):
            tok_id = int(tok_id_t.item())
            if tok_id in (self.sos_id, self.pad_id):
              continue
            if tok_id == self.eos_id and len(surf) == 0:
              continue
            tok_str = id2token.get(tok_id, "<unk>")
            if tok_id == self.eos_id:
              tok_str = "</s>"
            actions.append(("gen", tok_id, tok_str, float(val.item())))

          for _, tok_id, tok_str, act_score in actions:
            new_seq = torch.cat([seq, torch.tensor([[tok_id]], device=device)], dim=1)
            new_surf = list(surf)
            if tok_str != "</s>":
              new_surf.append(tok_str)
            new_score = (score * seq_len + act_score) / (seq_len + 1)
            new_beams.append((new_seq, new_surf, new_score))

        beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size * 2]
        if all(b[0][0, -1].item() == self.eos_id for b in beams):
          break

      finished = sorted(completed + beams, key=lambda x: x[2], reverse=True)[:beam_size]
      cand_ids = [[t for t in seq[0, 1:].tolist() if t not in (self.eos_id,)] for seq, _, _ in finished]
      cand_surfs = [self._format_surface(surf) for _, surf, _ in finished]

      model_pred = cand_surfs[0] if cand_surfs else ""
      pred_toks = model_pred.split() if model_pred else []
      rule_pred = self._format_surface(
        apply_comment_code_renames(src_tokens_list[i], code_change_seqs[i]),
      )
      merge_pred = (
        self.merge_src_prediction(src_tokens_list[i], pred_toks) if pred_toks else ""
      )
      all_cands = self._dedupe_candidates([rule_pred, merge_pred] + cand_surfs)
      primary = self._pick_primary_prediction(
        src_tokens_list[i], model_pred, rule_pred, merge_pred, code_change_seqs[i],
      )

      token_ids.append(cand_ids[0] if cand_ids else [])
      surface_texts.append(primary)
      beam_results.append(cand_ids if cand_ids else [[]])
      beam_surface_results.append(all_cands[: max(beam_size, 1)] or [""])

    if return_beam_candidates:
      return token_ids, no_update_texts, beam_results, surface_texts, beam_surface_results
    return token_ids, no_update_texts, surface_texts
