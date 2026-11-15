"""Data processing and feature engineering.

Turns TMDB's nested JSON into one flat record per movie and assembles the
pandas DataFrame that every recommender and chart in the app works from.
This module has no network or Streamlit dependency, so it is easy to test.
"""

from __future__ import annotations

import pandas as pd

import nlp

TOP_CAST = 6  # billed actors kept per movie
STANDARD_CERTIFICATIONS = {"G", "PG", "PG-13", "R", "NC-17"}

COLUMNS = [
    "id", "title", "year", "release_date", "runtime", "certification",
    "genres", "overview", "tagline", "keywords", "cast", "directors",
    "director", "rating", "votes", "popularity", "budget", "revenue",
    "language", "poster_path", "text", "tone",
]


def _us_certification(details: dict) -> str:
    """Return the US certification (G, PG, PG-13, R, NC-17), or "" if unknown.

    TMDB lists one entry per release (theatrical, digital, and so on). Many
    carry no certification and some carry "NR" for an unrated cut, so a
    standard MPA rating is preferred over any other non-empty value.
    """
    found: list[str] = []
    for country in (details.get("release_dates") or {}).get("results", []):
        if country.get("iso_3166_1") == "US":
            found = [r["certification"] for r in country.get("release_dates", []) if r.get("certification")]
    for certification in found:
        if certification in STANDARD_CERTIFICATIONS:
            return certification
    return found[0] if found else ""


def to_record(details: dict) -> dict:
    """Flatten one TMDB movie detail payload into a single-level record."""
    release_date = details.get("release_date") or ""
    credits = details.get("credits") or {}
    cast = sorted(credits.get("cast", []), key=lambda c: c.get("order", 999))
    directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    genres = [g["name"] for g in details.get("genres", [])]
    keywords = [k["name"] for k in (details.get("keywords") or {}).get("keywords", [])]
    overview = details.get("overview") or ""

    return {
        "id": details["id"],
        "title": details.get("title") or "Untitled",
        "year": int(release_date[:4]) if release_date[:4].isdigit() else None,
        "release_date": release_date,
        "runtime": details.get("runtime") or None,
        "certification": _us_certification(details),
        "genres": genres,
        "overview": overview,
        "tagline": details.get("tagline") or "",
        "keywords": keywords,
        "cast": [c["name"] for c in cast[:TOP_CAST]],
        "directors": directors,
        "director": ", ".join(directors),
        "rating": float(details.get("vote_average") or 0),
        "votes": int(details.get("vote_count") or 0),
        "popularity": float(details.get("popularity") or 0),
        "budget": int(details.get("budget") or 0),
        "revenue": int(details.get("revenue") or 0),
        "language": details.get("original_language") or "",
        "poster_path": details.get("poster_path"),
        # Text used for TF-IDF. Keywords and genres are repeated so that these
        # curated labels weigh more than incidental words in the plot summary.
        "text": " ".join([overview, " ".join(keywords), " ".join(keywords), " ".join(genres)]),
        # Sentiment of the plot summary, from -1 (dark) to +1 (uplifting).
        "tone": nlp.tone_score(overview),
    }


def build_frame(details_list: list[dict]) -> pd.DataFrame:
    """Build the movie DataFrame, one row per movie, without duplicates."""
    records = [to_record(d) for d in details_list if d and d.get("id")]
    frame = pd.DataFrame(records, columns=COLUMNS)
    return frame.drop_duplicates(subset="id").reset_index(drop=True)


def weighted_rating(frame: pd.DataFrame, prior_votes: int = 500) -> pd.Series:
    """Bayesian average rating that corrects for small vote counts.

    A movie rated 9.0 by 20 people is less trustworthy than one rated 8.2 by
    20,000. Each rating is pulled toward the pool's mean rating in proportion
    to how few votes it has: WR = v/(v+m) * R + m/(v+m) * C, where v is the
    vote count, R the rating, C the pool mean and m the prior weight.
    """
    if frame.empty:
        return pd.Series(dtype=float)
    pool_mean = frame["rating"].mean()
    votes = frame["votes"]
    return (votes / (votes + prior_votes)) * frame["rating"] + (
        prior_votes / (votes + prior_votes)
    ) * pool_mean


def format_money(amount: int | float | None) -> str:
    """Format a dollar amount compactly, for example $1.2B or $45M."""
    if not amount:
        return "Not reported"
    if amount >= 1e9:
        return f"${amount / 1e9:.1f}B"
    if amount >= 1e6:
        return f"${amount / 1e6:.0f}M"
    return f"${amount:,.0f}"
