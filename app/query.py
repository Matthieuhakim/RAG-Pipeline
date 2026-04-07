from __future__ import annotations

import json
import re

from app.generation import chat_completion
from app.models import IntentDecision


async def detect_intent(query: str, summaries: dict[str, str]) -> IntentDecision:
    if not summaries:
        return IntentDecision(
            needs_retrieval=False,
            reasoning="No documents are available, so retrieval cannot help.",
        )

    summary_lines = "\n".join(
        f"- {filename}: {summary}" for filename, summary in sorted(summaries.items())
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You decide whether a user query needs knowledge-base retrieval. "
                "Respond with JSON only: "
                '{"needs_retrieval": true|false, "reasoning": "brief explanation"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                "Given a knowledge base containing the following documents:\n"
                f"{summary_lines}\n\n"
                f"The user asks: {query!r}\n"
                "Does this query require searching the knowledge base?"
            ),
        },
    ]
    response = await chat_completion(messages, temperature=0.0, max_tokens=120)
    return _parse_intent_response(response)


async def transform_query(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite user questions into concise search queries for retrieval. "
                "Return only the rewritten query."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rewrite this user question as an optimal search query for retrieving "
                "relevant document chunks. Return only the rewritten query.\n\n"
                f"{query}"
            ),
        },
    ]
    rewritten = await chat_completion(messages, temperature=0.1, max_tokens=100)
    return rewritten.strip()


async def answer_directly(query: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    return await chat_completion(messages, temperature=0.3, max_tokens=400)


def _parse_intent_response(response: str) -> IntentDecision:
    cleaned = response.strip()
    cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return IntentDecision(
                needs_retrieval=True,
                reasoning="The model response was not valid JSON, defaulting to retrieval.",
            )
        payload = json.loads(match.group(0))

    try:
        return IntentDecision.model_validate(payload)
    except Exception:
        return IntentDecision(
            needs_retrieval=True,
            reasoning="The model response was malformed, defaulting to retrieval.",
        )
