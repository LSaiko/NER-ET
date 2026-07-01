"""HuggingFace NER (dslim/bert-base-NER) + spaCy DATE extraction.

Models load ONCE at import — never per-request. Windows: no multiprocessing,
so pipelines run single-threaded (num_workers=0 semantics).
"""
from functools import lru_cache

# BERT entity_group -> our label. MISC intentionally absent (dropped).
_LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}


# ponytail: lazy singletons — loaded ONCE on first use (or lifespan warmup),
# never per-request. Deferred so importing this module (tests, CI) doesn't
# pull models. Heavy imports live inside the loaders.
@lru_cache(maxsize=1)
def _get_ner():
    from transformers import pipeline
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")


@lru_cache(maxsize=1)
def _get_nlp():
    import spacy
    return spacy.load("en_core_web_sm")  # DATE only — not PER/ORG/LOC


def extract_entities(text: str) -> list[dict]:
    """
    Returns list of dicts compatible with EntityItem schema.
    Keys: label, text, score, start, end
    """
    out: list[dict] = []

    for e in _get_ner()(text):
        label = _LABEL_MAP.get(e["entity_group"])
        if label is None:  # drops MISC and anything unexpected
            continue
        out.append({
            "label": label,
            "text": e["word"],
            "score": float(e["score"]),
            "start": int(e["start"]),
            "end": int(e["end"]),
        })

    for ent in _get_nlp()(text).ents:
        if ent.label_ != "DATE":
            continue
        out.append({
            "label": "DATE",
            "text": ent.text,
            "score": 1.0,  # spaCy gives no score; dates are exact matches
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return out
