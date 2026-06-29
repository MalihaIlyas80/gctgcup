"""
Subword (byte-level BPE) tokenizer for GC-TGCUP comment/edit text.

Why subwords (matching TG-CUP): a closed WORD-level vocabulary cannot reproduce
out-of-vocabulary identifiers (renamed methods/variables), which capped exact
match near zero. A byte-level BPE has NO out-of-vocabulary tokens (it covers all
bytes) and produces shorter, far more learnable units, so the decoder can
reproduce the exact updated comment. encode->decode is loss-less, so references
and predictions live in the same detokenized text space (also fixes sacrebleu's
detokenization warning).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

# Special tokens occupy the first, fixed ids. The model hard-codes
# pad=0, sos=1, eos=2, unk=3, sep=4 — keep this order in sync.
SPECIAL_TOKENS = [
    "<pad>", "<s>", "</s>", "<unk>", "<sep>",
    "<before>", "<after>", "<equal>", "<replace>", "<insert>", "<delete>",
    "<keep>", "<copy>", "<local_edit>",
]


class SubwordTokenizer:
  """Thin wrapper around a HuggingFace byte-level BPE tokenizer."""

  def __init__(self, tok):
    self._tok = tok
    self.pad_id = tok.token_to_id("<pad>")
    self.sos_id = tok.token_to_id("<s>")
    self.eos_id = tok.token_to_id("</s>")
    self.unk_id = tok.token_to_id("<unk>")
    self.sep_id = tok.token_to_id("<sep>")
    vocab = tok.get_vocab()
    self.token2id: Dict[str, int] = vocab
    self.id2token: Dict[int, str] = {i: t for t, i in vocab.items()}

  def __len__(self) -> int:
    return self._tok.get_vocab_size()

  # ── encode / decode ──────────────────────────────────────────────────────
  def encode_text(self, text: str) -> List[int]:
    """Encode raw text to subword ids (no <s>/</s>)."""
    return self._tok.encode(text or "").ids

  def encode_with_special(self, text: str, max_len: int) -> List[int]:
    core = self.encode_text(text)[:max_len]
    return [self.sos_id] + core + [self.eos_id]

  def decode(self, ids: Iterable[int]) -> str:
    """Decode ids back to detokenized text, skipping special tokens."""
    ids = [int(i) for i in ids]
    return self._tok.decode(ids, skip_special_tokens=True).strip()

  # ── persistence ────────────────────────────────────────────────────────────
  def save(self, path: str) -> None:
    self._tok.save(path)

  @classmethod
  def load(cls, path: str) -> "SubwordTokenizer":
    from tokenizers import Tokenizer
    return cls(Tokenizer.from_file(path))

  @classmethod
  def train(cls, texts: Iterable[str], vocab_size: int, save_path: str,
            min_frequency: int = 2) -> "SubwordTokenizer":
    """Train a byte-level BPE on the given texts and persist it."""
    from tokenizers import ByteLevelBPETokenizer

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(
      texts,
      vocab_size=max(vocab_size, len(SPECIAL_TOKENS) + 256),
      min_frequency=min_frequency,
      special_tokens=SPECIAL_TOKENS,
      show_progress=False,
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    bpe.save(save_path)
    from tokenizers import Tokenizer
    return cls(Tokenizer.from_file(save_path))
