"""
GC-TGCUP: GraphCodeBERT-enhanced Two-Stage Comment Updating Framework.

Two stages, addressing TG-CUP's gaps:
  Stage 1 — Detection : OutdatedCommentDetector (GraphCodeBERT) decides whether
            the old comment is outdated given the code change.  TG-CUP has no
            detection stage.
  Stage 2 — Update    : UpdateT5, a PRETRAINED code-aware seq2seq model
            (CodeT5) conditioned on the multimodal input
                old comment + code-edit sequence + AST-diff summary.
            Using a pretrained generator is what makes the model data-efficient
            enough to beat TG-CUP on a small (~20k) dataset.

GraphCodeBERT (mandatory) drives detection / code semantics; the AST-diff and
edit sequence carry TG-CUP's structural + edit modalities into the updater.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .detection import OutdatedCommentDetector
from .update_t5 import UpdateT5


class GCTGCUP(nn.Module):
  def __init__(
    self,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    graphcodebert_name: str = "microsoft/graphcodebert-base",
    freeze_bert: bool = False,
    long_threshold: int = 25,
    update_model_name: str = "Salesforce/codet5-base",
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    max_edit_chars: int = 400,
    max_ast_chars: int = 200,
    det_loss_weight: float = 0.15,
    upd_loss_weight: float = 0.85,
    **_ignored,
  ):
    super().__init__()
    self.long_threshold = long_threshold
    self.det_loss_weight = det_loss_weight
    self.upd_loss_weight = upd_loss_weight

    # ── Stage 1: Detection (GraphCodeBERT) ──
    self.detector = OutdatedCommentDetector(
      hidden_dim=hidden_dim,
      graphcodebert_name=graphcodebert_name,
      freeze_bert=freeze_bert,
      dropout=dropout,
    )
    self.code_semantic = self.detector.code_encoder

    # ── Stage 2: Update (pretrained CodeT5) ──
    self.updater = UpdateT5(
      model_name=update_model_name,
      max_src_len=max_src_len,
      max_tgt_len=max_tgt_len,
      max_edit_chars=max_edit_chars,
      max_ast_chars=max_ast_chars,
    )

  # ── Stage 1 ──────────────────────────────────────────────────────────
  def detect(
    self,
    old_codes: List[str],
    new_codes: List[str],
    comments: List[str],
  ) -> torch.Tensor:
    return self.detector(old_codes, new_codes, comments)

  # ── joint training step ──────────────────────────────────────────────
  def forward(
    self,
    batch: Dict,
    teacher_forcing: bool = True,
    pos_weight: Optional[torch.Tensor] = None,
  ) -> Dict[str, torch.Tensor]:
    device = next(self.parameters()).device

    det_logits = self.detector(
      batch["src_methods"], batch["dst_methods"], batch["src_descs"]
    )
    det_loss = F.binary_cross_entropy_with_logits(
      det_logits, batch["labels"], pos_weight=pos_weight
    )
    result = {"det_loss": det_loss, "det_logits": det_logits}

    # Update loss only on OUTDATED samples (label == 1): those are the ones
    # that actually have a target edit to learn.
    update_mask = batch["labels"].bool()
    if update_mask.any():
      idx = update_mask.nonzero(as_tuple=True)[0].tolist()
      upd_loss = self.updater(
        [batch["src_descs"][i] for i in idx],
        [batch["edit_texts"][i] for i in idx],
        [batch["ast_texts"][i] for i in idx],
        [batch["dst_descs"][i] for i in idx],
      )
      result["upd_loss"] = upd_loss
    else:
      result["upd_loss"] = torch.zeros((), device=device)

    result["loss"] = (
      self.det_loss_weight * det_loss + self.upd_loss_weight * result["upd_loss"]
    )
    return result

  # ── inference ────────────────────────────────────────────────────────
  @torch.no_grad()
  def generate(
    self,
    old_codes: List[str],
    new_codes: List[str],
    old_comments: List[str],
    edit_texts: List[str],
    ast_texts: List[str],
    beam_size: int = 5,
    det_threshold: float = 0.5,
    force_update: bool = False,
    max_len: int = 128,
  ) -> Tuple[List[str], List[List[str]]]:
    """Two-stage inference.

    force_update=True bypasses the detection gate (TG-CUP-style: always update),
    used to measure pure update quality on the outdated-only subset.

    Returns:
      pred_texts:  best predicted comment per sample
      beam_texts:  beam candidates per sample (for Recall@k)
    """
    det_logits = self.detector(old_codes, new_codes, old_comments)
    det_probs = torch.sigmoid(det_logits)
    needs_update = (
      torch.ones_like(det_probs, dtype=torch.bool)
      if force_update else det_probs >= det_threshold
    )

    best, beams = self.updater.generate(
      old_comments, edit_texts, ast_texts,
      num_beams=beam_size, max_len=max_len,
    )

    pred_texts: List[str] = []
    beam_texts: List[List[str]] = []
    for i in range(len(old_comments)):
      if bool(needs_update[i]):
        pred_texts.append(best[i])
        beam_texts.append(beams[i] or [best[i]])
      else:
        # not outdated -> keep the comment unchanged
        pred_texts.append(old_comments[i])
        beam_texts.append([old_comments[i]] * max(beam_size, 1))

    return pred_texts, beam_texts
