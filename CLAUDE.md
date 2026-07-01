# CLAUDE.md — NER-ET Session Rules

## Project
Named Entity Recognition Extractor Tool
github.com/LSaiko/NER-ET

## Claude's Role in This Codebase
Claude is the **Resolver** and **Synthesizer**.
- Claude does NOT classify tokens. That is bert-base-NER's job.
- Claude does NOT retrieve data. That is FastAPI + HuggingFace's job.
- Claude receives raw NER output and resolves ambiguities, deduplicates
  surface-form variants, and produces a structured JSON summary.

## Ponytail Discipline (FULL level)
Before writing any custom code, climb the ladder:
1. Can I skip it entirely?
2. Can stdlib handle it?
3. Is there a platform-native solution?
4. Is there a one-line installed-dep call?
5. Only then: minimum custom code.

## Hard Constraints
- Windows runtime: `num_workers=0` on ALL DataLoaders, no multiprocessing
- No hardcoded secrets — all credentials via `python-dotenv` from `.env`
- Pydantic v2 syntax only (model_config, field_validator — NOT @validator)
- No albumentations, no albucore
- Use `pathlib.Path` everywhere — no `os.path`
- Claude API calls: max_tokens ≤ 300 for resolver, structured JSON only
- Retry logic required on all Claude API calls (max 2 retries)
- All paths relative to repo root

## File Layout
ner-et/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI entry point
│   ├── ner_pipeline.py   # HuggingFace inference (bert-base-NER + spaCy DATE)
│   ├── resolver.py       # Claude Resolver + Synthesizer
│   └── schemas.py        # Pydantic v2 request/response models
├── tests/
│   ├── __init__.py
│   └── test_ner.py       # pytest + httpx AsyncClient; mock all external calls
├── CLAUDE.md             # This file
├── SKILL.md              # Claude role contract
├── requirements.txt      # Pinned versions
├── .env.example
├── .gitignore
├── Dockerfile
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md

## Self-Check Before Committing
- [ ] All tests pass: `pytest tests/ -v`
- [ ] API boots cleanly: `uvicorn app.main:app --reload`
- [ ] `/health` returns 200
- [ ] `/extract` returns valid NERResponse JSON
- [ ] No hardcoded API keys anywhere in tracked files
- [ ] `.env` is in `.gitignore`
- [ ] README contains ASCII architecture diagram and Interview Q&A section
