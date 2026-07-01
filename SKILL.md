# SKILL.md — Claude Resolver Contract

## Skill Name
entity-resolver

## Claude's Job (what it IS)
- Deduplicate surface-form variants of the same named entity
- Resolve low-confidence entities (score < 0.75) using context
- Produce a structured JSON summary of canonical entities

## Claude's Job (what it IS NOT)
- NOT a token classifier
- NOT a retriever
- NOT a general-purpose assistant during this pipeline run
- NOT permitted to invent entities not present in BERT output

## Invocation Contract
Input  → JSON array of raw NER entities from bert-base-NER + spaCy
Output → Structured JSON matching ResolvedEntities schema (see schemas.py)
Tokens → max_tokens: 300
Model  → claude-sonnet-4-6

## Trigger Condition
Claude is invoked ONLY when at least one of:
  a) Any entity has score < 0.75
  b) Two or more surface forms share a token overlap ≥ 0.6
  c) Entity count > 10 (summary synthesis always warranted)

## Fallback
If Claude API fails after 2 retries, return raw BERT output with
resolved=False flag in NERResponse. Never block the API response.
