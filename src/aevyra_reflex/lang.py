# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language detection for dataset inputs.

Uses Unicode character-range heuristics — no external dependencies required.
The detection is intentionally coarse: it identifies the dominant script family
and maps it to the language name used in the reasoning-model instruction.

Supported script families and their natural-language names:

  Chinese (Simplified/Traditional)  →  "Chinese"
  Japanese (Hiragana/Katakana + CJK) →  "Japanese"
  Korean (Hangul)                    →  "Korean"
  Arabic                             →  "Arabic"
  Hebrew                             →  "Hebrew"
  Cyrillic (Russian etc.)            →  "Russian"
  Devanagari (Hindi etc.)            →  "Hindi"
  Thai                               →  "Thai"
  Greek                              →  "Greek"
  Latin (default)                    →  "English"

When the dominant script is Latin, "English" is returned regardless of the
actual Western language, since the vast majority of prompts optimized with
Reflex are English-language tasks.  If a more precise Latin-language name is
needed in future, a word-frequency or n-gram approach can be layered on.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Unicode ranges (half-open intervals, inclusive start / exclusive end)
# ---------------------------------------------------------------------------

_RANGES: list[tuple[int, int, str]] = [
    # Japanese-specific scripts come first — if Hiragana/Katakana is present
    # it's almost certainly Japanese even if CJK characters are also there.
    (0x3040, 0x30FF + 1, "Japanese"),   # Hiragana + Katakana
    (0x31F0, 0x31FF + 1, "Japanese"),   # Katakana Phonetic Extensions
    # CJK block is shared by Chinese, Japanese, and Korean; we treat it as
    # Chinese unless Japanese scripts were already detected above.
    (0x4E00, 0x9FFF + 1, "Chinese"),    # CJK Unified Ideographs (main block)
    (0x3400, 0x4DBF + 1, "Chinese"),    # CJK Extension A
    (0x20000, 0x2A6DF + 1, "Chinese"),  # CJK Extension B
    (0xF900, 0xFAFF + 1, "Chinese"),    # CJK Compatibility Ideographs
    (0xAC00, 0xD7AF + 1, "Korean"),     # Hangul Syllables
    (0x1100, 0x11FF + 1, "Korean"),     # Hangul Jamo
    (0x0600, 0x06FF + 1, "Arabic"),     # Arabic
    (0x0750, 0x077F + 1, "Arabic"),     # Arabic Supplement
    (0xFB50, 0xFDFF + 1, "Arabic"),     # Arabic Presentation Forms-A
    (0x0590, 0x05FF + 1, "Hebrew"),     # Hebrew
    (0x0400, 0x04FF + 1, "Russian"),    # Cyrillic
    (0x0500, 0x052F + 1, "Russian"),    # Cyrillic Supplement
    (0x0900, 0x097F + 1, "Hindi"),      # Devanagari
    (0x0E00, 0x0E7F + 1, "Thai"),       # Thai
    (0x0370, 0x03FF + 1, "Greek"),      # Greek and Coptic
]


def _script_of(cp: int) -> str | None:
    """Return the language label for a Unicode code point, or None for Latin/other."""
    for start, end, lang in _RANGES:
        if start <= cp < end:
            return lang
    return None


def detect_language(texts: list[str], sample_chars: int = 2000) -> str:
    """Detect the dominant language of a list of text samples.

    Args:
        texts: Sample strings from the dataset (e.g. user questions).
        sample_chars: Maximum characters to examine per text to keep this fast.

    Returns:
        A language name like ``"English"``, ``"Chinese"``, ``"Japanese"``, etc.
        Falls back to ``"English"`` when no non-Latin script exceeds the
        threshold, since Latin-script languages are overwhelmingly English in
        practice for Reflex workloads.
    """
    if not texts:
        return "English"

    counts: dict[str, int] = {}
    total_alpha = 0

    for text in texts:
        for ch in text[:sample_chars]:
            cp = ord(ch)
            # Skip ASCII — it's either English or punctuation shared across langs
            if cp < 0x0100:
                if ch.isalpha():
                    total_alpha += 1
                continue
            lang = _script_of(cp)
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
                total_alpha += 1

    if not counts:
        return "English"

    dominant_lang = max(counts, key=lambda k: counts[k])
    dominant_count = counts[dominant_lang]

    # Only override English when non-Latin characters make up >15 % of
    # alphabetic content — avoids mislabelling English text that contains
    # a handful of quoted terms or proper nouns in another script.
    if total_alpha > 0 and dominant_count / total_alpha >= 0.15:
        return dominant_lang

    return "English"
