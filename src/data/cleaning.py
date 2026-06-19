"""
Professional data cleaning pipeline for comment-update dataset.
Follows TG-CUP paper Section 2.2: remove URLs, emails, irrelevant noise.
"""
from __future__ import annotations

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

        return text.strip()


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
