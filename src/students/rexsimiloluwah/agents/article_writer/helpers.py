import random
import re
import time
from typing import List, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import Config


def enforce_word_limit(
    text,
    min_words=950,
    max_words=1050
):
    words = text.split()
    count = len(words)

    if count > max_words:
        # trim down while trying not to cut mid-sentence
        trimmed = " ".join(words[:max_words])
        if not trimmed.endswith(('.', '!', '?')):
            trimmed += "..."
        return trimmed

    elif count < min_words:
        # pad with a simple marker (better: re-prompt expansion)
        padding = " ".join(["[additional detail]" for _ in range(min_words - count)])
        return text + "\n\n" + padding

    return text


# Helper functions
def safe_llm_invoke(llm: ChatGoogleGenerativeAI, prompt: str, max_retries: int = Config.LLM_CALL_MAX_RETRIES) -> str:
    """Safely invoke LLM with retry logic"""
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return response.content  # type: ignore
        except Exception as e:
            error_text = str(e).lower()
            print(f"⚠️  Attempt {attempt}/{max_retries} failed: {e}")

            if "429" in error_text or "quota" in error_text:
                wait_time = min(10 * attempt, 60)
                print(f"🚦 Rate limit - waiting {wait_time}s...")
                time.sleep(wait_time)
                continue

            if "timeout" in error_text or "connection" in error_text:
                time.sleep(5 * attempt)
                continue

            if attempt < max_retries:
                time.sleep(5 * attempt)
                continue

            raise

    return "(LLM invocation failed after retries)"


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _word_count(s: str) -> int:
    return len(s.split())


def sentence_reduce_random(  # noqa: C901
    text: str,
    max_words: int,
    *,
    preserve_first_last: bool = True,
    protect_intro_outro_words: int = 200,   # NEW: protect first/last N words
    seed: Optional[int] = 42,
    protected_sections: Optional[List[str]] = None,
) -> Tuple[str, int]:
    """
    Randomly remove full sentences until total word count <= max_words.
    - Preserves markdown structure lines and code blocks.
    - Never removes sentences from protected sections (e.g., Introduction/Conclusion).
    - Optionally protects the first and last N words globally.
    Returns (reduced_text, reduced_word_count).
    """
    if _word_count(text) <= max_words:
        return text, _word_count(text)

    rng = random.Random(seed)
    if protected_sections is None:
        protected_sections = ["introduction", "conclusion"]

    protected_set = {s.strip().lower() for s in protected_sections}
    lines = text.splitlines()
    in_code = False

    # Sentence splits
    line_sentences: List[Optional[List[str]]] = []
    line_protected: List[bool] = []
    removable_indices: List[Tuple[int, int]] = []

    structural_prefixes = tuple(["#", ">", "-", "*", "+"])
    numeric_bullet = re.compile(r"^\s*\d+\.\s+")
    header_pat = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*$")

    current_is_protected = False
    for li, raw in enumerate(lines):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            in_code = not in_code
            line_sentences.append(None)
            line_protected.append(False)
            continue
        m = header_pat.match(line)
        if m:
            key = re.sub(r"[^a-z\s]", "", m.group(1).strip().lower())
            current_is_protected = any(ps in key for ps in protected_set)
            line_sentences.append(None)
            line_protected.append(False)
            continue
        if in_code or line.strip().startswith(structural_prefixes) or numeric_bullet.match(line):
            line_sentences.append(None)
            line_protected.append(False)
            continue
        if not line.strip():
            line_sentences.append(None)
            line_protected.append(False)
            continue
        sents = _SENT_SPLIT.split(line.strip())
        line_sentences.append(sents)
        line_protected.append(current_is_protected)

    # Flatten sentences for index mapping
    all_sentences = []
    for li, sents in enumerate(line_sentences):
        if sents is None:
            continue
        for si, sent in enumerate(sents):
            all_sentences.append((li, si, sent))

    # Compute word offsets for global protection of first/last N words
    word_offsets = []
    total = 0
    for li, si, sent in all_sentences:
        w = len(sent.split())
        word_offsets.append((li, si, total, total + w))  # sentence spans
        total += w

    intro_guard = protect_intro_outro_words
    outro_guard = max(0, total - protect_intro_outro_words)

    # First/last sentence guards
    first_guard = (all_sentences[0][0], all_sentences[0][1]) if preserve_first_last else None
    last_guard = (all_sentences[-1][0], all_sentences[-1][1]) if preserve_first_last else None

    # Collect candidates
    for (li, si, start, end), (_, _, sent) in zip(word_offsets, all_sentences):
        if line_sentences[li] is None or sent is None:
            continue
        if line_protected[li]:
            continue
        if first_guard and (li, si) == first_guard:
            continue
        if last_guard and (li, si) == last_guard:
            continue
        # 🔒 protect first/last N words globally
        if start < intro_guard or end > outro_guard:
            continue
        removable_indices.append((li, si))

    rng.shuffle(removable_indices)

    def reconstruct_text() -> str:
        out = []
        for li, sents in enumerate(line_sentences):
            if sents is None:
                out.append(lines[li])
            else:
                joined = " ".join(s for s in sents if s is not None)
                out.append(joined)
        return "\n".join(out).strip()

    total_words = _word_count(reconstruct_text())
    if total_words <= max_words:
        return reconstruct_text(), total_words

    # Remove until under budget
    for li, si in removable_indices:
        sents = line_sentences[li]
        if sents and 0 <= si < len(sents) and sents[si] is not None:
            sents[si] = None  # type: ignore
            total_words = _word_count(reconstruct_text())
            if total_words <= max_words:
                reduced = reconstruct_text()
                return reduced, total_words

    # Fallback hard cap
    reduced = reconstruct_text()
    words = reduced.split()
    if len(words) > max_words:
        reduced = " ".join(words[:max_words])
    return reduced, _word_count(reduced)


def _sentence_safe_enforce(text: str, mn: int, mx: int) -> str:
    try:
        import re
        words = text.split()
        if len(words) <= mx:
            return text
        sentences = re.split(r'(?<=[.!?])\s+', text)
        out, total = [], 0
        for s in sentences:
            w = len(s.split())
            if total + w > mx:
                break
            out.append(s)
            total += w
        return " ".join(out)
    except Exception:
        # Fallback: hard cap (last resort)
        return " ".join(text.split()[:mx])
