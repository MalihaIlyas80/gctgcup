"""
Professional data cleaning pipeline for comment-update dataset.
Follows TG-CUP paper Section 2.2: remove URLs, emails, irrelevant noise.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Optional

# ── regex patterns ──────────────────────────────────────────────────────────
_URL_RE = re.compile(
    r"https?://[^\s\)>\"']+|www\.[^\s\)>\"']+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_JAVADOC_TAG_RE = re.compile(r"@(param|return|throws|see|since|author|version|deprecated)\b")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")
_BLANK_LINE_RE = re.compile(r"^\s*$", re.MULTILINE)
# CUP2 stores word boundaries as <con>; normalize so CodeT5 + exact-match align.
_CON_TOKEN_RE = re.compile(r"<con>")


def normalize_comment_text(text: str) -> str:
  """Normalize comment text for training and TG-CUP-fair evaluation."""
  if not text:
    return ""
  text = _CON_TOKEN_RE.sub(" ", text)
  text = _MULTI_SPACE_RE.sub(" ", text)
  return text.strip()


@dataclass
class CommentCleaner:
    """Configurable comment cleaner matching TG-CUP preprocessing."""

    remove_urls: bool = True
    remove_emails: bool = True
    normalize_unicode: bool = True
    strip_javadoc_tags: bool = False  # keep {@link …} – structural info
    lowercase: bool = False

    def clean(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.strip()
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        text = _CONTROL_CHAR_RE.sub("", text)
        text = _HTML_ENTITY_RE.sub(" ", text)

        if self.remove_urls:
            text = _URL_RE.sub(" ", text)
        if self.remove_emails:
            text = _EMAIL_RE.sub(" ", text)

        # collapse inline whitespace but preserve newlines for structure
        lines = []
        for line in text.splitlines():
            line = _MULTI_SPACE_RE.sub(" ", line).strip()
            if line:
                lines.append(line)
        text = "\n".join(lines)

        if self.lowercase:
            text = text.lower()

        return normalize_comment_text(text)


_DEFAULT_CLEANER = CommentCleaner()


def clean_comment(text: str, cleaner: Optional[CommentCleaner] = None) -> str:
    return (cleaner or _DEFAULT_CLEANER).clean(text)


def clean_code(text: str) -> str:
    """Lightweight code normalisation – preserve structure."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    return text.strip()


def is_valid_sample(
    sample: Dict[str, Any],
    min_comment_len: int = 3,
    max_code_len: int = 5000,
) -> bool:
    """Filter broken / trivial samples (TG-CUP Section 4.1)."""
    src_desc = sample.get("src_desc", "") or ""
    dst_desc = sample.get("dst_desc", "") or ""
    src_code = sample.get("src_method", "") or ""
    dst_code = sample.get("dst_method", "") or ""

    if len(src_code) > max_code_len or len(dst_code) > max_code_len:
        return False
    if not src_code.strip() or not dst_code.strip():
        return False
    if len(src_desc.strip()) < min_comment_len:
        return False

    label = sample.get("label", True)
    if label and len(dst_desc.strip()) < min_comment_len:
        return False

    return True


def build_edit_sequence(
    code_change_seq: list,
    action_map: Optional[Dict[str, int]] = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Convert pre-computed code_change_seq to TG-CUP format:
    <before> old <after> new <action>  triplets.
    """
    old_tokens, new_tokens, actions = [], [], []
    for triple in code_change_seq:
        if len(triple) != 3:
            continue
        old_t, new_t, act = triple
        old_tokens.append(old_t)
        new_tokens.append(new_t)
        actions.append(act)
    return old_tokens, new_tokens, actions


def flatten_edit_sequence(
    old_tokens: list[str],
    new_tokens: list[str],
    actions: list[str],
) -> list[str]:
    """TG-CUP Equation (1): interleaved edit items."""
    seq: list[str] = []
    for o, n, a in zip(old_tokens, new_tokens, actions):
        seq.extend(["<before>", o, "<after>", n, f"<{a}>"])
    return seq


def apply_comment_code_renames(
    src_tokens: list[str],
    code_change_seq: list,
) -> list[str]:
    """Backward-compatible alias for rename-only edits."""
    return apply_comment_edits_from_code_change(src_tokens, code_change_seq)


def apply_comment_edits_from_code_change(
    src_tokens: list[str],
    code_change_seq: list,
) -> list[str]:
    """
    Apply code AST-diff edits to the old comment (TG-CUP local-edit prior).
    replace / delete / insert from code_change_seq are mirrored in comment tokens.
    """
    out = list(src_tokens)
    for triple in code_change_seq or []:
        if len(triple) != 3:
            continue
        old_t, new_t, act = triple[0], triple[1], triple[2]
        if old_t in ("<con>", "<pad>") or (not old_t and not new_t):
            continue

        if act == "replace" and old_t and new_t:
            for i, tok in enumerate(out):
                if tok == old_t or tok.lower() == old_t.lower():
                    out[i] = new_t
        elif act == "delete" and old_t and len(old_t) > 2:
            out = [t for t in out if t.lower() != old_t.lower()]
        elif act == "insert" and new_t and len(new_t) > 2:
            if new_t.lower() in [x.lower() for x in out]:
                continue
            placed = False
            if old_t and len(old_t) > 2:
                for i, tok in enumerate(out):
                    if tok.lower() == old_t.lower():
                        out.insert(i + 1, new_t)
                        placed = True
                        break
            if not placed:
                out.append(new_t)
    return out


def fuse_rule_and_model(
    rule_tokens: list[str],
    model_tokens: list[str],
) -> list[str]:
    """Keep rule-stable spans; use model tokens for insert/replace regions."""
    if not model_tokens:
        return list(rule_tokens)
    if not rule_tokens:
        return list(model_tokens)
    sm = difflib.SequenceMatcher(None, rule_tokens, model_tokens, autojunk=False)
    fused: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            fused.extend(rule_tokens[i1:i2])
        elif tag in ("replace", "insert"):
            fused.extend(model_tokens[j1:j2])
        elif tag == "delete":
            pass
    return fused


def comment_has_code_edit(src_tokens: list[str], code_change_seq: list) -> bool:
    """True when any code edit plausibly affects the old comment."""
    src_lower = {t.lower() for t in src_tokens}
    src_text = " ".join(src_tokens).lower()
    for triple in code_change_seq or []:
        if len(triple) != 3:
            continue
        old_t, new_t, act = triple[0], triple[1], triple[2]
        if act == "replace" and old_t and old_t.lower() in src_lower:
            return True
        if act == "delete" and old_t and len(old_t) > 2 and old_t.lower() in src_lower:
            return True
        if act == "insert" and new_t and len(new_t) > 2 and new_t.lower() not in src_lower:
            if not old_t or old_t.lower() in src_text:
                return True
    return False


def comment_has_code_rename(src_tokens: list[str], code_change_seq: list) -> bool:
    src_lower = {t.lower() for t in src_tokens}
    for triple in code_change_seq or []:
        if len(triple) == 3 and triple[2] == "replace":
            old_t = triple[0]
            if old_t and old_t.lower() in src_lower:
                return True
    return False


def is_nciu_sample(sample: Dict[str, Any]) -> bool:
    """
    NCIU (Non-Code-Indicative Update): code changes don't appear in comment.
    TG-CUP RQ4: samples HEB-CUP cannot update.
    Heuristic: no changed code token appears in old comment.
    """
    src_desc = (sample.get("src_desc") or "").lower()
    code_change_seq = sample.get("code_change_seq") or []
    changed_tokens = set()
    for triple in code_change_seq:
        if len(triple) == 3 and triple[2] in ("replace", "insert", "delete"):
            if triple[0] and triple[0] not in ("<con>", "<pad>"):
                changed_tokens.add(triple[0].lower())
            if triple[1] and triple[1] not in ("<con>", "<pad>"):
                changed_tokens.add(triple[1].lower())
    if not changed_tokens:
        return True
    return not any(tok in src_desc for tok in changed_tokens if len(tok) > 2)


def clean_sample(sample: Dict[str, Any], cleaner: Optional[CommentCleaner] = None) -> Dict[str, Any]:
    """Return a cleaned copy of one dataset record."""
    cleaner = cleaner or _DEFAULT_CLEANER
    out = dict(sample)
    out["src_desc"] = clean_comment(sample.get("src_desc", ""), cleaner)
    out["dst_desc"] = clean_comment(sample.get("dst_desc", ""), cleaner)
    out["src_method"] = clean_code(sample.get("src_method", ""))
    out["dst_method"] = clean_code(sample.get("dst_method", ""))
    out["is_nciu"] = is_nciu_sample(out)
    return out
