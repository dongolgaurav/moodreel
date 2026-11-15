"""Recommendation algorithms.

Three approaches are implemented, each producing a score from 0 to 1.

* Content-based filtering. TF-IDF vectors of plot, keywords and genres are
  compared with cosine similarity, against a typed request, a seed movie or
  a watchlist profile. For movie-to-movie similarity this is combined with
  overlap in genres, keywords, cast and director.
* Sentiment-based matching. The VADER tone of each plot summary is compared
  with the tone the requested mood aims for.
* Hybrid. A weighted blend of the two scores above and a vote-adjusted
  quality score. Signals that are unavailable (no mood chosen, nothing to
  compare content against) are dropped and the remaining weights rescaled,
  so the ranking always uses everything it knows and nothing it does not.

Every function also returns the evidence behind a score, so the interface can
explain each recommendation instead of presenting a bare ranking.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel

import features
import nlp

METHODS = ["Hybrid", "Content match", "Mood match", "Top rated"]
DEFAULT_WEIGHTS = {"content": 0.45, "mood": 0.25, "quality": 0.30}
# Each single-signal method is the hybrid with all weight on one signal.
_METHOD_WEIGHTS = {
    "Content match": {"content": 1.0, "mood": 0.0, "quality": 0.0},
    "Mood match": {"content": 0.0, "mood": 1.0, "quality": 0.0},
    "Top rated": {"content": 0.0, "mood": 0.0, "quality": 1.0},
}
# Movie-to-movie similarity: how much each kind of overlap counts.
SIMILARITY_WEIGHTS = {"text": 0.45, "genre": 0.25, "keyword": 0.15, "cast": 0.10, "director": 0.05}


def _normalise(values) -> np.ndarray:
    """Scale scores so the best one is 1. Raw cosine scores are tiny numbers."""
    array = np.asarray(values, dtype=float)
    top = array.max() if array.size else 0.0
    return array / top if top > 0 else np.zeros_like(array)


def _readable_terms(terms: list[str], limit: int = 3) -> list[str]:
    """Pick up to ``limit`` distinct terms, skipping words already covered."""
    chosen: list[str] = []
    for term in (nlp.display_term(t) for t in terms):
        words = set(term.split())
        if any(words <= set(c.split()) or set(c.split()) <= words for c in chosen):
            continue
        chosen.append(term)
        if len(chosen) == limit:
            break
    return chosen


# --------------------------------------------------------------------------
# Content-based filtering
# --------------------------------------------------------------------------

def text_similarity(frame: pd.DataFrame, query_text: str) -> tuple[np.ndarray, list[list[str]]]:
    """Cosine similarity between each movie's text and a query text.

    Returns the score for every row and, for each row, the terms that
    contributed most to its score (used for explanations).
    """
    count = len(frame)
    empty = (np.zeros(count), [[] for _ in range(count)])
    if count == 0 or not query_text.strip():
        return empty
    vectorizer = nlp.make_vectorizer()
    try:
        # The query is fitted together with the movies so that it shares
        # their vocabulary and IDF weights.
        matrix = vectorizer.fit_transform(frame["text"].tolist() + [query_text])
    except ValueError:  # every document was empty after stop-word removal
        return empty
    movies, query = matrix[:-1], matrix[-1]
    # TF-IDF rows are L2-normalised, so the dot product is cosine similarity.
    scores = linear_kernel(movies, query).ravel()

    names = vectorizer.get_feature_names_out()
    contributions = movies.multiply(query).tocsr()
    terms: list[list[str]] = []
    for index in range(count):
        row = contributions.getrow(index)
        strongest = row.indices[np.argsort(row.data)[::-1][:8]]
        terms.append(_readable_terms([names[j] for j in strongest]))
    return scores, terms


def _jaccard(first: list[str], second: list[str]) -> tuple[float, list[str]]:
    """Share of items two lists have in common, and the common items."""
    a, b = set(first), set(second)
    if not a or not b:
        return 0.0, []
    shared = [item for item in first if item in b]
    return len(shared) / len(a | b), shared


def pair_overlap(seed: dict, other: dict) -> dict:
    """Structured overlap between two movie records."""
    genre, shared_genres = _jaccard(seed["genres"], other["genres"])
    keyword, shared_keywords = _jaccard(seed["keywords"], other["keywords"])
    shared_cast = [name for name in seed["cast"] if name in set(other["cast"])]
    shared_directors = [name for name in seed["directors"] if name in set(other["directors"])]
    return {
        "genre": genre,
        "keyword": keyword,
        "cast": min(1.0, len(shared_cast) / 2),  # two shared leads is a full match
        "director": 1.0 if shared_directors else 0.0,
        "shared_genres": shared_genres,
        "shared_keywords": shared_keywords,
        "shared_cast": shared_cast,
        "shared_directors": shared_directors,
    }


def explain_overlap(seed: dict, other: dict, overlap: dict, terms: list[str]) -> list[str]:
    """Turn overlap evidence into sentences a user can read."""
    reasons: list[str] = []
    if overlap["shared_genres"]:
        reasons.append("Same genres: " + ", ".join(overlap["shared_genres"]))
    if overlap["shared_keywords"]:
        reasons.append("Shared themes: " + ", ".join(overlap["shared_keywords"][:4]))
    # Plot terms that merely repeat a genre or a shared keyword add nothing.
    covered = {word.lower() for genre in overlap["shared_genres"] for word in genre.split()}
    covered |= {keyword.lower() for keyword in overlap["shared_keywords"]}
    terms = [t for t in terms if t not in covered and not set(t.split()) & covered]
    if terms:
        reasons.append("Similar plot language: " + ", ".join(terms))
    if overlap["shared_directors"]:
        reasons.append("Same director: " + ", ".join(overlap["shared_directors"]))
    if overlap["shared_cast"]:
        reasons.append("Also stars " + ", ".join(overlap["shared_cast"][:3]))
    if nlp.tone_label(seed["tone"]) == nlp.tone_label(other["tone"]):
        reasons.append(f"Same tone: {nlp.tone_label(other['tone']).lower()}")
    return reasons or ["Loosely related by TMDB audience behaviour"]


def similar_movies(seed: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Rank candidate movies by similarity to a seed movie, with reasons.

    Adds two columns: ``similarity`` (0 to 1) and ``reasons`` (list of text).
    """
    pool = frame[frame["id"] != seed["id"]].reset_index(drop=True)
    if pool.empty:
        return pool.assign(similarity=[], reasons=[])
    text_scores, terms = text_similarity(pool, seed["text"])
    text_scores = _normalise(text_scores)

    similarity, reasons = [], []
    for index, row in enumerate(pool.to_dict("records")):
        overlap = pair_overlap(seed, row)
        similarity.append(
            SIMILARITY_WEIGHTS["text"] * text_scores[index]
            + sum(SIMILARITY_WEIGHTS[k] * overlap[k] for k in ("genre", "keyword", "cast", "director"))
        )
        reasons.append(explain_overlap(seed, row, overlap, terms[index]))
    pool = pool.assign(similarity=similarity, reasons=reasons)
    return pool.sort_values("similarity", ascending=False).reset_index(drop=True)


def compare_pair(first: dict, second: dict) -> tuple[float, list[str]]:
    """Similarity between exactly two movies, with reasons."""
    pair = pd.DataFrame([second])
    text_scores, terms = text_similarity(pair, first["text"])
    overlap = pair_overlap(first, second)
    score = SIMILARITY_WEIGHTS["text"] * min(1.0, float(text_scores[0]) * 3) + sum(
        SIMILARITY_WEIGHTS[k] * overlap[k] for k in ("genre", "keyword", "cast", "director")
    )
    return score, explain_overlap(first, second, overlap, terms[0])


# --------------------------------------------------------------------------
# Hybrid ranking (also covers the single-signal methods)
# --------------------------------------------------------------------------

def rank(
    frame: pd.DataFrame,
    method: str = "Hybrid",
    weights: dict | None = None,
    mood: str | None = None,
    mood_target: float | None = None,
    content: np.ndarray | None = None,
    terms: list[list[str]] | None = None,
    content_label: str = "Matches your request",
) -> tuple[pd.DataFrame, list[str]]:
    """Score and sort movies. Returns the ranked frame and the signals used.

    ``mood`` is one of ``nlp.MOODS``; ``mood_target`` can be given instead as
    a raw tone value. ``content`` holds one content-similarity score per row.
    Adds the columns content_score, mood_score, quality_score, score and why.
    """
    ranked = frame.copy()
    if ranked.empty:
        return ranked.assign(score=[], why=[]), []
    if mood_target is None and mood in nlp.MOODS:
        mood_target = nlp.MOODS[mood]

    weighted = features.weighted_rating(ranked)
    ranked["quality_score"] = weighted / 10
    ranked["mood_score"] = (
        ranked["tone"].apply(lambda tone: nlp.mood_match(tone, mood_target))
        if mood_target is not None else np.nan
    )
    has_content = content is not None and len(content) == len(ranked) and np.max(content) > 0
    ranked["content_score"] = _normalise(content) if has_content else np.nan

    chosen = _METHOD_WEIGHTS.get(method, weights or DEFAULT_WEIGHTS)
    # Keep only signals that carry weight and actually exist for this request.
    usable = {
        signal: weight for signal, weight in chosen.items()
        if weight > 0 and ranked[f"{signal}_score"].notna().any()
    }
    if not usable:
        usable = {"quality": 1.0}  # nothing else to go on: fall back to quality
    total = sum(usable.values())
    ranked["score"] = sum(ranked[f"{s}_score"] * w for s, w in usable.items()) / total

    reasons = []
    for position, row in enumerate(ranked.to_dict("records")):
        parts: list[str] = []
        if "content" in usable and terms and terms[position]:
            parts.append(f"{content_label}: {', '.join(terms[position])}")
        if "mood" in usable and row["mood_score"] >= 0.7:
            parts.append("Plot tone suits " + (f"a {mood.lower()} mood" if mood else "your usual taste"))
        parts.append(f"Vote-adjusted rating {weighted.iloc[position]:.1f}")
        reasons.append(" · ".join(parts))
    ranked["why"] = reasons

    ranked = ranked.sort_values(["score", "quality_score"], ascending=False).reset_index(drop=True)
    return ranked, list(usable)


# --------------------------------------------------------------------------
# Watchlist profile
# --------------------------------------------------------------------------

def build_profile(watchlist: pd.DataFrame) -> dict:
    """Summarise a watchlist as text, genre preferences and typical tone."""
    genre_counts = Counter(genre for genres in watchlist["genres"] for genre in genres)
    total = sum(genre_counts.values()) or 1
    return {
        "text": " ".join(watchlist["text"]),
        "genre_share": {genre: count / total for genre, count in genre_counts.items()},
        "top_genres": [genre for genre, _ in genre_counts.most_common(3)],
        "tone": float(watchlist["tone"].mean()),
        "rating": float(watchlist["rating"].mean()),
    }


def profile_recommend(watchlist: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates for a user, based on what is in their watchlist.

    Content similarity is 70% text similarity to the combined watchlist and
    30% preference for the genres the user saves most. It is then blended
    with tone closeness and quality through the same hybrid ranking.
    """
    pool = candidates[~candidates["id"].isin(watchlist["id"])].reset_index(drop=True)
    if pool.empty or watchlist.empty:
        return pool.assign(score=[], why=[]) if pool.empty else pool
    profile = build_profile(watchlist)
    text_scores, terms = text_similarity(pool, profile["text"])
    genre_affinity = pool["genres"].apply(
        lambda genres: sum(profile["genre_share"].get(g, 0) for g in genres)
    )
    content = 0.7 * _normalise(text_scores) + 0.3 * _normalise(genre_affinity)
    ranked, _ = rank(
        pool,
        method="Hybrid",
        mood_target=profile["tone"],
        content=content,
        terms=terms,
        content_label="Echoes your watchlist",
    )
    return ranked
