"""TMDB API client: every network call the app makes lives in this module.

Design decisions
----------------
* The API key is never hard-coded. It is read from the ``TMDB_API_KEY``
  environment variable or from Streamlit secrets, so the key stays out of the
  source code and out of version control.
* Every endpoint wrapper is cached. List-style endpoints (discover, search,
  trending) use ``st.cache_data`` with a time-to-live, and per-movie details
  are kept in a shared in-memory store so overlapping result sets never fetch
  the same movie twice. This keeps the app fast and well under TMDB's rate
  limits.
* Movie details are fetched concurrently, with credits, keywords and
  certifications bundled into a single request per movie through
  ``append_to_response``.
* Failures are raised as ``TMDBError`` with a message that is safe to show to
  users. The request URL is never included, because it can contain the key.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
TIMEOUT_SECONDS = 10
MAX_ATTEMPTS = 3
DETAIL_WORKERS = 8
DETAIL_STORE_LIMIT = 4000  # movies kept in memory before the store is reset
PLACEHOLDER_KEY = "PASTE_YOUR_KEY_HERE"

_session = requests.Session()


class TMDBError(Exception):
    """A TMDB request failed. The message is safe to display in the UI."""


# --------------------------------------------------------------------------
# Low-level request handling
# --------------------------------------------------------------------------

def get_api_key() -> str | None:
    """Return the configured TMDB credential, or None when it is missing."""
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets["TMDB_API_KEY"]).strip()
        except Exception:  # no secrets file, or the entry is absent
            key = ""
    if not key or key == PLACEHOLDER_KEY:
        return None
    return key


def _request(path: str, params: dict | None = None, key: str | None = None) -> dict:
    """GET one TMDB endpoint, retrying on rate limits and transient errors."""
    key = key or get_api_key()
    if not key:
        raise TMDBError("No TMDB API key is configured.")

    query = dict(params or {})
    headers = {"accept": "application/json"}
    # TMDB issues two credentials: a short v3 key (sent as a query parameter)
    # and a long v4 read token (sent as a bearer header). Accept either.
    if key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {key}"
    else:
        query["api_key"] = key

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = _session.get(
                BASE_URL + path, params=query, headers=headers, timeout=TIMEOUT_SECONDS
            )
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS - 1:
                # "from None" drops the original exception, whose text
                # includes the full URL and therefore the API key.
                raise TMDBError(
                    "Could not reach TMDB. Check your internet connection and try again."
                ) from None
            time.sleep(0.5 * (attempt + 1))
            continue

        if response.status_code == 429:  # rate limited: wait as instructed
            time.sleep(float(response.headers.get("Retry-After", 1)))
            continue
        if response.status_code >= 500:  # TMDB-side problem: brief back-off
            time.sleep(0.5 * (attempt + 1))
            continue
        if response.status_code == 401:
            raise TMDBError("TMDB rejected the API key. Check that it was copied correctly.")
        if response.status_code == 404:
            raise TMDBError("TMDB could not find that item.")
        if not response.ok:
            raise TMDBError(f"TMDB returned an unexpected error (status {response.status_code}).")
        return response.json()

    raise TMDBError("TMDB is busy right now. Please try again in a moment.")


# --------------------------------------------------------------------------
# Cached endpoint wrappers
# --------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner=False)
def genre_map() -> dict[int, str]:
    """Map TMDB genre ids to names, for example {28: "Action"}."""
    data = _request("/genre/movie/list", {"language": "en-US"})
    return {g["id"]: g["name"] for g in data.get("genres", [])}


@st.cache_data(ttl=3600, show_spinner=False)
def discover(params: dict, pages: int = 2) -> list[dict]:
    """Run the discover endpoint and return up to ``pages`` pages of movies."""
    movies: list[dict] = []
    for page in range(1, pages + 1):
        data = _request(
            "/discover/movie",
            {"include_adult": "false", "language": "en-US", **params, "page": page},
        )
        movies.extend(data.get("results", []))
        if page >= data.get("total_pages", 1):
            break
    return movies


@st.cache_data(ttl=3600, show_spinner=False)
def search_movie(query: str) -> list[dict]:
    """Search movies by title, most relevant first."""
    data = _request("/search/movie", {"query": query, "include_adult": "false"})
    return data.get("results", [])


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def search_person(name: str) -> list[dict]:
    """Search people by name, most popular first."""
    data = _request("/search/person", {"query": name, "include_adult": "false"})
    return sorted(data.get("results", []), key=lambda p: p.get("popularity", 0), reverse=True)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def search_keyword(query: str) -> list[dict]:
    """Search TMDB's keyword vocabulary, for example "heist"."""
    return _request("/search/keyword", {"query": query}).get("results", [])


@st.cache_data(ttl=1800, show_spinner=False)
def trending(window: str = "week") -> list[dict]:
    """Movies trending today ("day") or this week ("week")."""
    return _request(f"/trending/movie/{window}").get("results", [])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def movie_recommendations(movie_id: int) -> list[dict]:
    """TMDB's own recommendations for a movie, used only as a candidate pool."""
    return _request(f"/movie/{movie_id}/recommendations").get("results", [])


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def movie_similar(movie_id: int) -> list[dict]:
    """TMDB's genre and keyword neighbours, used only as a candidate pool."""
    return _request(f"/movie/{movie_id}/similar").get("results", [])


# --------------------------------------------------------------------------
# Movie details (shared store plus concurrent fetching)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _detail_store() -> dict[int, dict]:
    """One dictionary shared by all sessions: movie id -> full detail payload."""
    return {}


def _fetch_details(movie_id: int, key: str) -> dict | None:
    """Fetch one movie with credits, keywords and certifications attached."""
    try:
        return _request(
            f"/movie/{movie_id}",
            {"append_to_response": "keywords,credits,release_dates", "language": "en-US"},
            key=key,
        )
    except TMDBError:
        return None  # one missing movie should not sink a whole result page


def movies_details(movie_ids: list[int]) -> dict[int, dict]:
    """Return full details for many movies, fetching only the ones not stored."""
    store = _detail_store()
    wanted = list(dict.fromkeys(movie_ids))  # de-duplicate, keep order
    missing = [m for m in wanted if m not in store]
    if missing:
        key = get_api_key()
        if not key:
            raise TMDBError("No TMDB API key is configured.")
        if len(store) + len(missing) > DETAIL_STORE_LIMIT:
            store.clear()
        # The key is resolved here, in the main thread, and passed to workers
        # so that no worker thread needs to touch Streamlit's secrets.
        with ThreadPoolExecutor(max_workers=DETAIL_WORKERS) as pool:
            for movie_id, data in zip(missing, pool.map(lambda m: _fetch_details(m, key), missing)):
                if data:
                    store[movie_id] = data
    return {m: store[m] for m in wanted if m in store}


def movie_details(movie_id: int) -> dict | None:
    """Return full details for one movie, or None when it cannot be loaded."""
    return movies_details([movie_id]).get(movie_id)


def poster_url(poster_path: str | None, size: str = "w342") -> str | None:
    """Build a full poster image URL from the path TMDB returns."""
    return f"{IMAGE_BASE_URL}{size}{poster_path}" if poster_path else None
