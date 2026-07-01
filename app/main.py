"""FastAPI entry point — NER-ET."""
import logging
from contextlib import asynccontextmanager
from itertools import combinations

from dotenv import load_dotenv

load_dotenv()  # before app creation — resolver reads ANTHROPIC_API_KEY

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from app.ner_pipeline import extract_entities  # noqa: E402
from app.resolver import resolve_entities  # noqa: E402
from app.schemas import NERRequest, NERResponse  # noqa: E402

logger = logging.getLogger("uvicorn")

_MODEL = "dslim/bert-base-NER"


@asynccontextmanager
async def lifespan(app: FastAPI):
    extract_entities("warmup test string")  # pre-warm BERT + spaCy
    print("NER pipeline warmed up.")
    yield


app = FastAPI(title="NER-ET", lifespan=lifespan)

# Restrict origins in production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _tokens(text: str) -> set[str]:
    return set(text.lower().split())


def _token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of whitespace tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def should_invoke_claude(entities: list[dict]) -> bool:
    """SKILL.md triggers: (a) any score < 0.75, (b) 2+ surface forms with
    token overlap >= 0.6, (c) entity count > 10."""
    if len(entities) > 10:
        return True
    if any(e["score"] < 0.75 for e in entities):
        return True
    for a, b in combinations(entities, 2):
        if _token_overlap(a["text"], b["text"]) >= 0.6:
            return True
    return False


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": _MODEL}


@app.post("/extract", response_model=NERResponse)
def extract(request: NERRequest) -> NERResponse:
    try:
        raw = extract_entities(request.text)

        claude_invoked = should_invoke_claude(raw)
        resolved = resolve_entities(raw, request.text) if claude_invoked else None

        return NERResponse(
            entities=raw,
            resolved=resolved,
            claude_invoked=claude_invoked,
            model=_MODEL,
            char_count=len(request.text),
            entity_count=len(raw),
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("extract failed")
        raise HTTPException(status_code=500, detail=str(e))
