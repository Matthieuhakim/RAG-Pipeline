"""Answer shaping — switch prompt templates by detected intent.

Classifies user queries into answer shapes (list, comparison, summary,
definition, general) using lightweight regex heuristics and selects a
matching system prompt that guides the LLM toward structured output.
No extra LLM call is needed for classification.
"""

from __future__ import annotations

import re
from enum import Enum


class AnswerShape(Enum):
    LIST = "list"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    DEFINITION = "definition"
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Detection patterns (checked in order; first match wins)
# ---------------------------------------------------------------------------

_LIST_RE = re.compile(
    r"\b(?:"
    r"list|enumerate|what\s+are\s+(?:the|all|some)"
    r"|name\s+(?:the|all|some)"
    r"|give\s+(?:me\s+)?(?:a\s+)?(?:list|examples)"
    r"|how\s+many\s+(?:types|kinds|ways|steps)"
    r"|what\s+(?:types|kinds|ways|steps)"
    r"|which\s+(?:methods|techniques|approaches|strategies|factors)"
    r")\b",
    re.IGNORECASE,
)

_COMPARISON_RE = re.compile(
    r"\b(?:"
    r"compar(?:e|ison)|differenc(?:e|es)|differentiate"
    r"|(?:vs\.?|versus)\b"
    r"|similarities?\s+(?:and|between)"
    r"|(?:how\s+(?:does|do|is|are)\s+\w+\s+)?(?:differ\b|contrast)"
    r"|advantages?\s+(?:and|over)"
    r"|pros?\s+(?:and|&)\s+cons?"
    r"|trade[\s\-]?offs?"
    r")\b",
    re.IGNORECASE,
)

_SUMMARY_RE = re.compile(
    r"\b(?:"
    r"summar(?:y|ize|ise)|overview"
    r"|(?:give|provide)\s+(?:me\s+)?(?:a\s+)?(?:brief|short|quick)"
    r"|main\s+(?:points?|ideas?|findings?|takeaways?)"
    r"|key\s+(?:points?|ideas?|findings?|takeaways?)"
    r"|in\s+(?:a\s+)?nutshell"
    r"|(?:can\s+you\s+)?(?:briefly|concisely)\s+(?:explain|describe)"
    r"|tl;?\s*dr"
    r")\b",
    re.IGNORECASE,
)

_DEFINITION_RE = re.compile(
    r"\b(?:"
    r"(?:what\s+is|what\s+are|what's)\s+(?:a\s+|an\s+|the\s+)?"
    r"|defin(?:e|ition)"
    r"|(?:what\s+does|what\s+do)\s+\w+\s+mean"
    r"|meaning\s+of"
    r"|explain\s+(?:the\s+)?(?:concept|term|notion)"
    r")\b",
    re.IGNORECASE,
)

_SHAPE_RULES: list[tuple[re.Pattern[str], AnswerShape]] = [
    (_SUMMARY_RE, AnswerShape.SUMMARY),
    (_COMPARISON_RE, AnswerShape.COMPARISON),
    (_LIST_RE, AnswerShape.LIST),
    (_DEFINITION_RE, AnswerShape.DEFINITION),
]


# ---------------------------------------------------------------------------
# Prompt templates per shape
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS: dict[AnswerShape, str] = {
    AnswerShape.LIST: (
        "You are a helpful assistant. Answer the user's question based ONLY "
        "on the provided context. Format your answer as a numbered or "
        "bulleted list. Each item should be concise. If the context does not "
        "contain enough information, say so clearly."
    ),
    AnswerShape.COMPARISON: (
        "You are a helpful assistant. Answer the user's question based ONLY "
        "on the provided context. Structure your answer as a comparison: "
        "use a table or side-by-side format highlighting similarities and "
        "differences. If the context does not contain enough information, "
        "say so clearly."
    ),
    AnswerShape.SUMMARY: (
        "You are a helpful assistant. Answer the user's question based ONLY "
        "on the provided context. Provide a concise summary in 2-4 sentences "
        "capturing the most important points. If the context does not contain "
        "enough information, say so clearly."
    ),
    AnswerShape.DEFINITION: (
        "You are a helpful assistant. Answer the user's question based ONLY "
        "on the provided context. Start with a clear, one-sentence definition, "
        "then elaborate with key details from the context. If the context does "
        "not contain enough information, say so clearly."
    ),
    AnswerShape.GENERAL: (
        "You are a helpful assistant. Answer the user's question based ONLY "
        "on the provided context. If the context does not contain enough "
        "information, say so clearly."
    ),
}


def detect_answer_shape(query: str) -> AnswerShape:
    """Classify a query into an answer shape using regex heuristics."""
    for pattern, shape in _SHAPE_RULES:
        if pattern.search(query):
            return shape
    return AnswerShape.GENERAL


def get_system_prompt(shape: AnswerShape) -> str:
    """Return the system prompt for a given answer shape."""
    return _SYSTEM_PROMPTS[shape]
