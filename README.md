# NER-ET

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![HuggingFace](https://img.shields.io/badge/HuggingFace-bert--base--NER-yellow)
![Claude API](https://img.shields.io/badge/Claude-Resolver-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-green)

Extract named entities (PERSON, ORG, LOC, DATE) from any document via REST API.
BERT classifies — Claude resolves.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      NER-ET Pipeline                     │
│                                                         │
│  POST /extract                                          │
│       │                                                 │
│       ▼                                                 │
│  ┌──────────────────────┐                              │
│  │  dslim/bert-base-NER │  ──── PERSON, ORG, LOC       │
│  │  (token classifier)  │                              │
│  └──────────────────────┘                              │
│       │                                                 │
│  ┌────────────────┐                                     │
│  │  spaCy         │  ──── DATE entities                 │
│  │  en_core_web_sm│                                     │
│  └────────────────┘                                     │
│       │                                                 │
│       ▼   [if score<0.75 OR variants OR count>10]       │
│  ┌──────────────────────┐                              │
│  │  Claude Resolver     │  ──── Deduplicate + Summarize │
│  │  (Synthesizer role)  │                              │
│  └──────────────────────┘                              │
│       │                                                 │
│       ▼                                                 │
│  NERResponse JSON                                       │
└─────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
git clone https://github.com/LSaiko/NER-ET
cd NER-ET
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # add your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

## Example Request / Response

Request:

```json
POST /extract
{
  "text": "Dr. Sarah Chen signed off on the Medcore IQ protocol in Boston on March 4, 2024."
}
```

Response (abbreviated):

```json
{
  "entities": [
    { "label": "PERSON", "text": "Sarah Chen", "score": 0.98, "start": 4, "end": 14 },
    { "label": "ORG",    "text": "Medcore",    "score": 0.91, "start": 38, "end": 45 },
    { "label": "LOC",    "text": "Boston",     "score": 0.99, "start": 63, "end": 69 },
    { "label": "DATE",   "text": "March 4, 2024", "score": 1.0, "start": 73, "end": 87 }
  ],
  "resolved": {
    "canonical_entities": [
      { "label": "PERSON", "canonical_text": "Dr. Sarah Chen",
        "variants": ["Sarah Chen"], "confidence": 0.98 }
    ],
    "entity_counts": { "PERSON": 1, "ORG": 1, "LOC": 1, "DATE": 1 },
    "resolution_notes": "Dr. title form promoted as canonical. No ambiguities found."
  },
  "claude_invoked": true,
  "model": "dslim/bert-base-NER",
  "char_count": 88,
  "entity_count": 4
}
```

## Skills Demonstrated

| Skill | Where |
|-------|-------|
| Token classification | `ner_pipeline.py` — bert-base-NER aggregation |
| DATE hybrid extraction | `ner_pipeline.py` — spaCy secondary pass |
| FastAPI REST design | `main.py` — lifespan, CORSMiddleware, response_model |
| Pydantic v2 schemas | `schemas.py` — ConfigDict, Literal types |
| Claude post-processing | `resolver.py` — Resolver + Synthesizer roles |
| JSON output formatting | `resolver.py` — structured output with retry |
| Entity deduplication | `resolver.py` — surface-form variant merging |
| Pytest with mocking | `tests/test_ner.py` — zero real API calls in CI |

## Interview Q&A

**Q: How does BERT tokenization affect entity span detection?**

BERT uses WordPiece tokenization, which splits uncommon words into subword
tokens (e.g. "Medcore" → "Med", "##core"). When a named entity spans multiple
subword tokens, the NER model must label each sub-token and the `aggregation_strategy`
parameter controls how these are merged. With "simple", the label of the first
sub-token governs the group and scores are averaged. This matters for entity
boundaries: a hyphenated name like "Smith-Jones" may produce split spans that
require post-processing alignment — which is exactly where the Claude Resolver
adds value by canonicalizing incomplete surface forms.

**Q: Why does Claude act as a Resolver rather than running a second classification pass?**

LLMs are unreliable zero-shot token classifiers at span-level precision.
What they excel at is reasoning over structured output to make
deduplication and ambiguity-resolution decisions — tasks that require
reading comprehension rather than sequence labeling. Keeping roles
separated (BERT classifies, Claude resolves) also makes each component
independently testable and replaceable.

## ValiOOP / docforge Integration

```
NER-ET JSON output → ValiOOP/docforge entity traceability layer
```

Entity mapping for medical device validation documents:

```
  PERSON → Responsible party (author, approver, reviewer)
  ORG    → Device name, department, supplier
  LOC    → Test site, manufacturing location
  DATE   → Protocol date, approval date, change date
```

To integrate:

```python
import httpx
response = httpx.post("http://localhost:8000/extract", json={"text": doc_text})
entities = response.json()["resolved"]["canonical_entities"]
# Pass entities to docforge traceability builder
```
