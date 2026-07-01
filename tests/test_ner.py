"""pytest + httpx AsyncClient. All external calls (BERT, spaCy, Claude) mocked."""
import pytest
from httpx import ASGITransport, AsyncClient

import app.main as main
from app.schemas import CanonicalEntity, EntityCounts, ResolvedEntities

pytestmark = pytest.mark.asyncio


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=main.app), base_url="http://t")


def _high_conf_entities() -> list[dict]:
    return [
        {"label": "PERSON", "text": "Sarah Chen", "score": 0.98, "start": 4, "end": 14},
        {"label": "ORG", "text": "Medcore", "score": 0.91, "start": 38, "end": 45},
        {"label": "LOC", "text": "Boston", "score": 0.99, "start": 63, "end": 69},
        {"label": "DATE", "text": "March 4, 2024", "score": 1.0, "start": 73, "end": 87},
    ]


def _resolved() -> ResolvedEntities:
    return ResolvedEntities(
        canonical_entities=[
            CanonicalEntity(label="PERSON", canonical_text="Dr. Sarah Chen",
                            variants=["Sarah Chen"], confidence=0.98),
        ],
        entity_counts=EntityCounts(PERSON=1, ORG=1, LOC=1, DATE=1),
        resolution_notes="No ambiguities found.",
    )


VALID_TEXT = "Dr. Sarah Chen signed off on the Medcore IQ protocol in Boston on March 4, 2024."


async def test_health_ok():
    async with _client() as ac:
        r = await ac.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "dslim/bert-base-NER"}


async def test_extract_returns_ner_response(monkeypatch):
    monkeypatch.setattr(main, "extract_entities", lambda t: _high_conf_entities())
    monkeypatch.setattr(main, "resolve_entities", lambda raw, text: _resolved())
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": VALID_TEXT})
    assert r.status_code == 200
    body = r.json()
    for key in ("entities", "resolved", "claude_invoked", "model", "char_count", "entity_count"):
        assert key in body
    assert body["entity_count"] == 4
    assert body["char_count"] == len(VALID_TEXT)


async def test_extract_entity_labels_valid(monkeypatch):
    monkeypatch.setattr(main, "extract_entities", lambda t: _high_conf_entities())
    monkeypatch.setattr(main, "resolve_entities", lambda raw, text: _resolved())
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": VALID_TEXT})
    labels = {e["label"] for e in r.json()["entities"]}
    assert labels <= {"PERSON", "ORG", "LOC", "DATE"}


async def test_extract_claude_not_invoked_high_confidence(monkeypatch):
    monkeypatch.setattr(main, "extract_entities", lambda t: _high_conf_entities())
    # resolve_entities must NOT be called; make it explode if it is.
    monkeypatch.setattr(main, "resolve_entities",
                        lambda raw, text: (_ for _ in ()).throw(AssertionError("should not run")))
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": VALID_TEXT})
    body = r.json()
    assert body["claude_invoked"] is False
    assert body["resolved"] is None


async def test_extract_claude_invoked_low_confidence(monkeypatch):
    low = _high_conf_entities()
    low[0]["score"] = 0.5  # trips trigger (a)
    monkeypatch.setattr(main, "extract_entities", lambda t: low)
    monkeypatch.setattr(main, "resolve_entities", lambda raw, text: _resolved())
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": VALID_TEXT})
    body = r.json()
    assert body["claude_invoked"] is True
    assert body["resolved"] is not None


async def test_extract_text_too_short():
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": "short"})
    assert r.status_code == 422


async def test_extract_text_too_long():
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": "a" * 50_001})
    assert r.status_code == 422


async def test_fallback_on_claude_failure(monkeypatch):
    low = _high_conf_entities()
    low[0]["score"] = 0.5  # trigger Claude
    monkeypatch.setattr(main, "extract_entities", lambda t: low)
    monkeypatch.setattr(main, "resolve_entities", lambda raw, text: None)  # simulate failure
    async with _client() as ac:
        r = await ac.post("/extract", json={"text": VALID_TEXT})
    body = r.json()
    assert body["claude_invoked"] is True
    assert body["resolved"] is None
