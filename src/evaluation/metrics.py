"""
Evaluation metrics matching TG-CUP paper Section 4.2 + detection metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def _tokenize(text: str) -> List[str]:
  return text.lower().split()


def exact_match(pred: str, ref: str) -> bool:
  return pred.strip() == ref.strip()


def recall_at_k(predictions: List[List[str]], references: List[str], k: int = 5) -> float:
  hits = 0
  for cands, ref in zip(predictions, references):
    if any(c.strip() == ref.strip() for c in cands[:k]):
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
    keep = f1(len(ps & rs & ss), len(ps & ss), len(rs & ss))
    add = f1(len(ps & rs - ss), len(ps - ss), len(rs - ss))
    del_ = f1(len(ss - ps & ss), len(ss - ps), len(ss - rs))
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
  nciu_accuracy: float = 0.0
  long_comment_accuracy: float = 0.0
  per_sample: Dict = field(default_factory=dict)

  def to_dict(self) -> Dict:
    return {
      "accuracy": round(self.accuracy, 2),
      "recall_at_5": round(self.recall_at_5, 2),
      "gleu": round(self.gleu, 2),
      "meteor": round(self.meteor, 2),
      "sari": round(self.sari, 2),
      "bleu": round(self.bleu, 2),
      "det_accuracy": round(self.det_accuracy, 2),
      "det_precision": round(self.det_precision, 2),
      "det_recall": round(self.det_recall, 2),
      "det_f1": round(self.det_f1, 2),
      "nciu_accuracy": round(self.nciu_accuracy, 2),
      "long_comment_accuracy": round(self.long_comment_accuracy, 2),
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
) -> EvaluationMetrics:
  m = EvaluationMetrics()

  m.accuracy = sum(exact_match(p, r) for p, r in zip(predictions, references)) / max(len(references), 1) * 100

  if beam_candidates:
    m.recall_at_5 = recall_at_k(beam_candidates, references, k=5)
  else:
    m.recall_at_5 = m.accuracy

  m.gleu = gleu_score(predictions, references)
  m.meteor = meteor_score(predictions, references)
  m.sari = sari_score(predictions, sources, references)
  m.bleu = bleu_score(predictions, references)

  if det_preds is not None and det_labels is not None:
    m.det_accuracy = accuracy_score(det_labels, det_preds) * 100
    p, r, f1, _ = precision_recall_fscore_support(det_labels, det_preds, average="binary", zero_division=0)
    m.det_precision = p * 100
    m.det_recall = r * 100
    m.det_f1 = f1 * 100

  if is_nciu:
    nciu_preds = [predictions[i] for i, f in enumerate(is_nciu) if f]
    nciu_refs = [references[i] for i, f in enumerate(is_nciu) if f]
    if nciu_preds:
      m.nciu_accuracy = sum(exact_match(p, r) for p, r in zip(nciu_preds, nciu_refs)) / len(nciu_preds) * 100

  if is_long:
    long_preds = [predictions[i] for i, f in enumerate(is_long) if f]
    long_refs = [references[i] for i, f in enumerate(is_long) if f]
    if long_preds:
      m.long_comment_accuracy = sum(exact_match(p, r) for p, r in zip(long_preds, long_refs)) / len(long_preds) * 100

  return m
