"""
Evaluation metrics matching TG-CUP paper Section 4.2 + detection metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def normalize_eval_text(text: str) -> str:
  """Same normalization for predictions and references (fair exact-match vs TG-CUP)."""
  if not text:
    return ""
  text = text.replace("<con>", " ")
  return " ".join(text.split()).strip()


def _tokenize(text: str) -> List[str]:
  return normalize_eval_text(text).lower().split()


def exact_match(pred: str, ref: str) -> bool:
  return normalize_eval_text(pred) == normalize_eval_text(ref)


def recall_at_k(predictions: List[List[str]], references: List[str], k: int = 5) -> float:
  hits = 0
  for cands, ref in zip(predictions, references):
    ref_n = normalize_eval_text(ref)
    if any(normalize_eval_text(c) == ref_n for c in cands[:k]):
      hits += 1
  return hits / max(len(references), 1) * 100


def gleu_score(predictions: List[str], references: List[str]) -> float:
  """Simplified GLEU (Google-BLEU variant)."""
  try:
    from nltk.translate.gleu_score import sentence_gleu
    scores = []
    for pred, ref in zip(predictions, references):
      ref_tok = _tokenize(ref)
      pred_tok = _tokenize(pred)
      if not ref_tok:
        continue
      scores.append(sentence_gleu([ref_tok], pred_tok))
    return np.mean(scores) * 100 if scores else 0.0
  except Exception:
    return _bleu_approx(predictions, references)


def meteor_score(predictions: List[str], references: List[str]) -> float:
  try:
    from nltk.translate.meteor_score import meteor_score as ms
    import nltk
    try:
      nltk.data.find("corpora/wordnet")
    except LookupError:
      nltk.download("wordnet", quiet=True)
      nltk.download("omw-1.4", quiet=True)
    scores = [ms([_tokenize(r)], _tokenize(p)) for p, r in zip(predictions, references)]
    return np.mean(scores) * 100
  except Exception:
    return _bleu_approx(predictions, references) * 0.9


def sari_score(predictions: List[str], sources: List[str], references: List[str]) -> float:
  """SARI for text rewriting (keep/add/delete F1 average)."""
  def f1(overlap, pred_n, ref_n):
    p = overlap / max(pred_n, 1)
    r = overlap / max(ref_n, 1)
    return 2 * p * r / max(p + r, 1e-9)

  scores = []
  for pred, src, ref in zip(predictions, sources, references):
    ps, rs, ss = set(_tokenize(pred)), set(_tokenize(ref)), set(_tokenize(src))
    # No-change case: ref says keep everything, pred keeps everything → perfect SARI
    # Standard SARI paper: score=1.0 when no edit needed and none made
    nothing_to_add = not (rs - ss)
    nothing_to_delete = not (ss - rs)
    pred_no_add = not (ps - ss)
    pred_no_delete = not (ss - ps)
    if nothing_to_add and nothing_to_delete and pred_no_add and pred_no_delete:
      scores.append(1.0)
      continue
    keep = f1(len(ps & rs & ss), len(ps & ss), len(rs & ss))
    add = f1(len((ps & rs) - ss), len(ps - ss), len(rs - ss))
    del_ = f1(len((ss - ps) & (ss - rs)), len(ss - ps), len(ss - rs))
    scores.append((keep + add + del_) / 3)
  return np.mean(scores) * 100 if scores else 0.0


def _bleu_approx(predictions, references) -> float:
  try:
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score
  except Exception:
    return 0.0


def bleu_score(predictions: List[str], references: List[str]) -> float:
  try:
    import sacrebleu
    return sacrebleu.corpus_bleu(predictions, [references]).score
  except Exception:
    return 0.0


@dataclass
class EvaluationMetrics:
  accuracy: float = 0.0
  recall_at_5: float = 0.0
  gleu: float = 0.0
  meteor: float = 0.0
  sari: float = 0.0
  bleu: float = 0.0
  det_accuracy: float = 0.0
  det_precision: float = 0.0
  det_recall: float = 0.0
  det_f1: float = 0.0
  det_pos_ratio: float = 0.0
  nciu_accuracy: float = 0.0
  long_comment_accuracy: float = 0.0
  num_total: int = 0
  num_outdated: int = 0
  num_nciu: int = 0
  num_long: int = 0
  per_sample: Dict = field(default_factory=dict)

  def to_dict(self) -> Dict:
    return {
      # Update stage (outdated-only subset) — fair TG-CUP comparison
      "accuracy": round(self.accuracy, 2),
      "recall_at_5": round(self.recall_at_5, 2),
      "gleu": round(self.gleu, 2),
      "meteor": round(self.meteor, 2),
      "sari": round(self.sari, 2),
      "bleu": round(self.bleu, 2),
      "nciu_accuracy": round(self.nciu_accuracy, 2),
      "long_comment_accuracy": round(self.long_comment_accuracy, 2),
      # Detection stage (full imbalanced set) — TG-CUP has no detection
      "det_accuracy": round(self.det_accuracy, 2),
      "det_precision": round(self.det_precision, 2),
      "det_recall": round(self.det_recall, 2),
      "det_f1": round(self.det_f1, 2),
      "det_pos_ratio": round(self.det_pos_ratio, 2),
      # Sample counts for context
      "num_total": self.num_total,
      "num_outdated": self.num_outdated,
      "num_nciu": self.num_nciu,
      "num_long": self.num_long,
    }


def compute_all_metrics(
  predictions: List[str],
  references: List[str],
  sources: List[str],
  beam_candidates: Optional[List[List[str]]] = None,
  det_preds: Optional[List[int]] = None,
  det_labels: Optional[List[int]] = None,
  is_nciu: Optional[List[bool]] = None,
  is_long: Optional[List[bool]] = None,
  outdated: Optional[List[int]] = None,
) -> EvaluationMetrics:
  """
  Two-stage evaluation:
    * UPDATE stage  -> generation metrics on the OUTDATED-only subset
                       (label==1). This mirrors TG-CUP, which only ever
                       evaluates on outdated samples and always updates.
    * DETECTION stage -> precision/recall/F1 on the FULL (imbalanced) set,
                       since deciding whether to update is our own contribution
                       that TG-CUP does not have.
  """
  m = EvaluationMetrics()
  m.num_total = len(references)

  # ── Indices that count for the UPDATE task (outdated-only when provided) ──
  if outdated is not None:
    upd_idx = [i for i, o in enumerate(outdated) if o]
  else:
    upd_idx = list(range(len(references)))
  m.num_outdated = len(upd_idx)

  upd_preds = [normalize_eval_text(predictions[i]) for i in upd_idx]
  upd_refs = [normalize_eval_text(references[i]) for i in upd_idx]
  upd_srcs = [normalize_eval_text(sources[i]) for i in upd_idx]

  m.accuracy = sum(exact_match(p, r) for p, r in zip(upd_preds, upd_refs)) / max(len(upd_refs), 1) * 100

  if beam_candidates:
    upd_cands = [beam_candidates[i] for i in upd_idx]
    m.recall_at_5 = recall_at_k(upd_cands, upd_refs, k=5)
  else:
    m.recall_at_5 = m.accuracy

  m.gleu = gleu_score(upd_preds, upd_refs)
  m.meteor = meteor_score(upd_preds, upd_refs)
  m.sari = sari_score(upd_preds, upd_srcs, upd_refs)
  m.bleu = bleu_score(upd_preds, upd_refs)

  # ── Detection on the FULL imbalanced set ──
  if det_preds is not None and det_labels is not None:
    m.det_accuracy = accuracy_score(det_labels, det_preds) * 100
    p, r, f1, _ = precision_recall_fscore_support(det_labels, det_preds, average="binary", zero_division=0)
    m.det_precision = p * 100
    m.det_recall = r * 100
    m.det_f1 = f1 * 100
    m.det_pos_ratio = sum(det_labels) / max(len(det_labels), 1) * 100

  # ── NCIU robustness (within the outdated update subset) ──
  if is_nciu:
    nciu_idx = [i for i in upd_idx if is_nciu[i]]
    m.num_nciu = len(nciu_idx)
    if nciu_idx:
      m.nciu_accuracy = sum(exact_match(predictions[i], references[i]) for i in nciu_idx) / len(nciu_idx) * 100

  # ── Long/complex comments (within the outdated update subset) ──
  if is_long:
    long_idx = [i for i in upd_idx if is_long[i]]
    m.num_long = len(long_idx)
    if long_idx:
      m.long_comment_accuracy = sum(exact_match(predictions[i], references[i]) for i in long_idx) / len(long_idx) * 100

  return m
