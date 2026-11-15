"""Candidate generation: decides which movies each recommender gets to rank.

Recommending happens in two stages. This module is stage one: it asks TMDB
for a pool of plausible candidates and turns them into a feature DataFrame.
Stage two (``recommender.py``) scores and orders that pool. Splitting the work
this way keeps API usage low, because only a few dozen movies ever need full
details, while still drawing candidates from TMDB's entire catalogue.
"""

from __future__ import annotations

import pandas as pd

import features
import tmdb_client as tmdb
from filters import Filters

MAX_POOL = 140           # most movies enriched for a single request
ACCLAIMED_MIN_VOTES = 300  # floor that keeps obscure 10/10 entries out


def frame_for(movie_ids: list[int]) -> pd.DataFrame:
    """Fetch details for the given ids and return them as a feature frame."""
    ids = list(dict.fromkeys(movie_ids))[:MAX_POOL]
    details = tmdb.movies_details(ids)
    return features.build_frame([details[i] for i in ids if i in details])


def genre_ids() -> dict[str, int]:
    """Genre name -> TMDB id."""
    return {name: genre_id for genre_id, name in tmdb.genre_map().items()}


# --------------------------------------------------------------------------
# Resolving names typed by the user into TMDB ids
# --------------------------------------------------------------------------

def resolve_person(name: str) -> dict | None:
    """Best TMDB match for a person's name, or None."""
    results = tmdb.search_person(name.strip()) if name.strip() else []
    return {"id": results[0]["id"], "name": results[0]["name"]} if results else None


def resolve_people(text: str) -> tuple[list[dict], list[str]]:
    """Resolve a comma-separated list of names. Returns (found, not found)."""
    found, missing = [], []
    for name in [part.strip() for part in text.split(",") if part.strip()]:
        person = resolve_person(name)
        if person:
            found.append(person)
        else:
            missing.append(name)
    return found, missing


def resolve_movie(title: str) -> dict | None:
    """Best TMDB match for a movie title, or None."""
    results = tmdb.search_movie(title.strip()) if title.strip() else []
    return {"id": results[0]["id"], "title": results[0]["title"]} if results else None


def keyword_ids(free_text: str) -> list[int]:
    """Map theme words such as "heist" onto ids in TMDB's keyword vocabulary."""
    phrases = [free_text] + [word for word in free_text.split() if len(word) >= 4][:3]
    ids: list[int] = []
    for phrase in dict.fromkeys(phrases):
        results = tmdb.search_keyword(phrase)
        exact = [r for r in results if r["name"].lower() == phrase.lower()]
        ids += [r["id"] for r in (exact[:1] or results[:1])]
    return list(dict.fromkeys(ids))[:4]


# --------------------------------------------------------------------------
# Candidate pools
# --------------------------------------------------------------------------

def discover_pool(flt: Filters, free_text: str = "", seed_id: int | None = None) -> pd.DataFrame:
    """Candidates for the Discover page.

    Four sources are merged so the pool is not just "whatever is popular":
    popular movies matching the filters, acclaimed movies matching the
    filters, movies tagged with the request's themes, and (when the request
    names a movie) TMDB's neighbours of that movie. The filters are then
    re-applied to the detailed records, so every result satisfies all of them.
    """
    base = flt.to_discover_params(genre_ids())
    found = tmdb.discover({**base, "sort_by": "popularity.desc"}, pages=2)
    found += tmdb.discover(
        {**base, "sort_by": "vote_average.desc",
         "vote_count.gte": max(flt.min_votes, ACCLAIMED_MIN_VOTES)},
        pages=1,
    )
    if free_text:
        themes = keyword_ids(free_text)
        if themes:
            found += tmdb.discover(
                {**base, "with_keywords": "|".join(str(t) for t in themes),
                 "sort_by": "popularity.desc"},
                pages=2,
            )
    if seed_id:
        found += tmdb.movie_recommendations(seed_id) + tmdb.movie_similar(seed_id)
    return flt.apply(frame_for([movie["id"] for movie in found]))


def similar_pool(seed: dict) -> pd.DataFrame:
    """Candidates for "more like this": TMDB neighbours plus same-genre films."""
    found = tmdb.movie_recommendations(seed["id"]) + tmdb.movie_similar(seed["id"])
    ids = genre_ids()
    main_genres = [str(ids[g]) for g in seed["genres"][:2] if g in ids]
    if main_genres:
        found += tmdb.discover(
            {"with_genres": ",".join(main_genres), "vote_count.gte": 200,
             "sort_by": "popularity.desc"},
            pages=2,
        )
    frame = frame_for([movie["id"] for movie in found])
    # Very low vote counts usually mean incomplete metadata and weak matches.
    return frame[frame["votes"] >= 300].reset_index(drop=True)


def profile_pool(watchlist: pd.DataFrame) -> pd.DataFrame:
    """Candidates for watchlist-based recommendations."""
    found: list[dict] = []
    for movie_id in watchlist["id"].tolist()[:6]:
        found += tmdb.movie_recommendations(int(movie_id))
    ids = genre_ids()
    top_genres = watchlist["genres"].explode().value_counts().index.tolist()[:2]
    genre_param = "|".join(str(ids[g]) for g in top_genres if g in ids)
    if genre_param:
        found += tmdb.discover(
            {"with_genres": genre_param, "vote_count.gte": 300, "vote_average.gte": 6.5,
             "sort_by": "popularity.desc"},
            pages=2,
        )
    return frame_for([movie["id"] for movie in found])


def trending_frame(window: str = "week", limit: int = 12) -> pd.DataFrame:
    """Movies trending on TMDB, in TMDB's order."""
    return frame_for([movie["id"] for movie in tmdb.trending(window)[:limit]])
