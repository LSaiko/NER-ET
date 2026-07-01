"""Claude Resolver + Synthesizer.

Dedupes surface-form variants, resolves low-confidence labels via context,
and synthesizes a structured JSON summary. Never raises — returns None on
failure so main.py can fall back to raw BERT output.
"""
import json
import logging
import os

import anthropic

from app.schemas import ResolvedEntities

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"

RESOLVER_SYSTEM = """You are an entity resolution specialist.

You receive raw NER output from a BERT token classifier and must
produce a clean structured JSON object. Do not invent entities.
Work only with what BERT detected.

Your tasks:
1. DEDUPLICATE: Merge entity variants that refer to the same real-world
   referent. Use the longest, most complete surface form as canonical.
   Example: "John Smith", "Smith", "J. Smith" → canonical "John Smith"

2. RESOLVE AMBIGUITY: For entities with score < 0.75, use the
   surrounding context snippet to confirm or correct the label.

3. SYNTHESIZE: Return a JSON object with exactly this schema:

{
  "canonical_entities": [
    {
      "label": "PERSON" | "ORG" | "LOC" | "DATE",
      "canonical_text": "...",
      "variants": ["...", "..."],
      "confidence": 0.0–1.0
    }
  ],
  "entity_counts": { "PERSON": 0, "ORG": 0, "LOC": 0, "DATE": 0 },
  "resolution_notes": "One sentence plain-English summary of what was resolved."
}

Return ONLY valid JSON. No markdown. No preamble. No explanation."""

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.endswith("```"):
            s = s[: -3]
    return s.strip()


def resolve_entities(
    raw_entities: list[dict],
    source_text: str,
    max_retries: int = 2,
) -> ResolvedEntities | None:
    """
    Invokes Claude Resolver. Returns None on failure — never raises.
    Claude receives: raw entity list + first 500 chars of source text as context.
    """
    user_msg = (
        f"Entities (JSON array):\n{json.dumps(raw_entities, indent=2)}\n\n"
        f"Context snippet (first 500 chars):\n{source_text[:500]}"
    )

    for attempt in range(max_retries + 1):  # 1 try + max_retries retries
        try:
            msg = client.messages.create(
                model=_MODEL,
                max_tokens=300,
                system=RESOLVER_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            parsed = json.loads(_strip_fences(msg.content[0].text))
            return ResolvedEntities.model_validate(parsed)
        except anthropic.APIError as e:
            logger.warning("resolver API error (attempt %d): %s", attempt + 1, e)
        except (json.JSONDecodeError, ValueError) as e:
            # bad/invalid JSON or schema mismatch — retry
            logger.warning("resolver parse/validate error (attempt %d): %s", attempt + 1, e)

    logger.warning("resolver failed after %d attempts; falling back", max_retries + 1)
    return None
