"""Reusable interface components shared by every page.

Keeping the movie card, the movie search box and the page header in one place
is what gives the app a consistent look: every page shows a movie the same
way and offers the same two actions on it.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

import features
import nlp
import state
import tmdb_client as tmdb

APP_NAME = "MoodReel"
TAGLINE = "Tell it your mood. Get tonight's movie."

# Filled in by app.py so components can link to pages.
PAGES: dict[str, object] = {}

_STYLE = """
<style>
  .block-container { padding-top: 2.2rem; max-width: 1250px; }
  [data-testid="stImage"] img { border-radius: 8px; }
  .mr-noposter {
    aspect-ratio: 2 / 3; border-radius: 8px; background: #1A1F2B;
    border: 1px dashed #3A4254; color: #8A93A6; font-size: 0.8rem;
    display: flex; align-items: center; justify-content: center;
  }
  .mr-tagline { color: #A9B1C3; margin-top: -0.6rem; }
</style>
"""


def inject_style() -> None:
    st.markdown(_STYLE, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<p class='mr-tagline'>{subtitle}</p>", unsafe_allow_html=True)


def require_api_key() -> None:
    """Stop the page with setup instructions when no API key is configured."""
    if tmdb.get_api_key():
        return
    st.warning(
        "**MoodReel needs a TMDB API key before it can load movies.**\n\n"
        "- **Running locally:** open `.streamlit/secrets.toml` and replace the "
        "placeholder with your key.\n"
        "- **On Streamlit Cloud:** open the app's Settings, choose Secrets, and add "
        '`TMDB_API_KEY = "your key"`.'
    )
    st.stop()


# --------------------------------------------------------------------------
# Small formatting helpers
# --------------------------------------------------------------------------

def as_int(value) -> int | None:
    """Convert a possibly-missing number (None or NaN) to int or None."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(value)


def movie_label(movie: dict) -> str:
    year = as_int(movie.get("year"))
    return f"{movie['title']} ({year})" if year else movie["title"]


def meta_line(movie: dict) -> str:
    """One-line summary: rating, votes, runtime, certification, tone."""
    parts = [f"★ {movie['rating']:.1f} ({movie['votes']:,} votes)" if movie["votes"] else "Not yet rated"]
    runtime = as_int(movie.get("runtime"))
    if runtime:
        parts.append(f"{runtime} min")
    if movie.get("certification"):
        parts.append(movie["certification"])
    parts.append(f"{nlp.tone_label(movie['tone'])} tone")
    return " · ".join(parts)


def shorten(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------
# Movie card and grid
# --------------------------------------------------------------------------

def _open_similar(movie: dict) -> None:
    """Send a movie to the Similar page's search box and go there."""
    set_picker("pick_similar", int(movie["id"]), movie["title"])
    st.switch_page(PAGES["similar"])


def movie_card(
    movie: dict,
    key: str,
    why: str | list[str] | None = None,
    score: float | None = None,
    score_label: str = "Match",
    position: int | None = None,
) -> None:
    """Draw one movie: poster, facts, the reason it was recommended, actions."""
    movie_id = int(movie["id"])
    with st.container(border=True):
        poster_column, body = st.columns([1, 3])
        url = tmdb.poster_url(movie.get("poster_path"))
        if url:
            poster_column.image(url, width="stretch")
        else:
            poster_column.markdown("<div class='mr-noposter'>No poster</div>", unsafe_allow_html=True)

        with body:
            heading = f"**{position}. {movie_label(movie)}**" if position else f"**{movie_label(movie)}**"
            if score is not None and not math.isnan(score):
                heading += f"&nbsp; :orange-badge[{score_label} {score:.0%}]"
            st.markdown(heading)
            st.caption(meta_line(movie))
            if movie.get("genres"):
                st.caption(" / ".join(movie["genres"]))
            credits = []
            if movie.get("director"):
                credits.append(f"Directed by {movie['director']}")
            if movie.get("cast"):
                credits.append("With " + ", ".join(movie["cast"][:3]))
            if credits:
                st.caption(" · ".join(credits))
            st.write(shorten(movie.get("overview") or "No plot summary is available."))

            if isinstance(why, list) and why:
                st.markdown("\n".join(f"- {reason}" for reason in why))
            elif isinstance(why, str) and why:
                st.caption(f":material/lightbulb: {why}")

            save, similar, details = st.columns(3)
            if state.in_watchlist(movie_id):
                save.button("Saved", key=f"btn_{key}_{movie_id}_rm", icon=":material/bookmark_added:",
                            help="Remove from watchlist", on_click=state.remove_from_watchlist,
                            args=(movie_id,), width="stretch")
            else:
                save.button("Watchlist", key=f"btn_{key}_{movie_id}_add", icon=":material/bookmark_add:",
                            help="Save to watchlist", on_click=state.add_to_watchlist,
                            args=(movie,), width="stretch")
            if similar.button("Similar", key=f"btn_{key}_{movie_id}_sim", icon=":material/hub:",
                              help="Find movies like this one", width="stretch"):
                _open_similar(movie)
            with details.popover("More", icon=":material/info:", width="stretch"):
                if movie.get("tagline"):
                    st.markdown(f"*{movie['tagline']}*")
                st.markdown(
                    f"**Released:** {movie.get('release_date') or 'Unknown'}  \n"
                    f"**Language:** {(movie.get('language') or 'unknown').upper()}  \n"
                    f"**Budget:** {features.format_money(movie.get('budget'))}  \n"
                    f"**Box office:** {features.format_money(movie.get('revenue'))}  \n"
                    f"**TMDB popularity:** {movie.get('popularity', 0):.0f}  \n"
                    f"**Plot tone score:** {movie['tone']:+.2f} (−1 dark, +1 uplifting)"
                )
                if movie.get("keywords"):
                    st.caption("Keywords: " + ", ".join(movie["keywords"][:12]))
                st.write(movie.get("overview") or "")


def movie_grid(
    frame: pd.DataFrame,
    key: str,
    why_column: str | None = None,
    score_column: str | None = None,
    score_label: str = "Match",
    numbered: bool = False,
) -> None:
    """Draw movies two to a row."""
    movies = frame.to_dict("records")
    for start in range(0, len(movies), 2):
        for offset, column in enumerate(st.columns(2)):
            if start + offset >= len(movies):
                continue
            movie = movies[start + offset]
            with column:
                movie_card(
                    movie,
                    key=key,
                    why=movie.get(why_column) if why_column else None,
                    score=movie.get(score_column) if score_column else None,
                    score_label=score_label,
                    position=start + offset + 1 if numbered else None,
                )


# --------------------------------------------------------------------------
# Movie search box
# --------------------------------------------------------------------------

def set_picker(key: str, movie_id: int, title: str) -> None:
    """Pre-fill a movie search box from elsewhere in the app."""
    st.session_state[f"{key}_query"] = title
    st.session_state[f"{key}_choice"] = movie_id
    st.session_state[f"{key}_pinned"] = {"id": movie_id, "title": title}


def movie_picker(label: str, key: str) -> int | None:
    """Search box plus a list of matches. Returns the chosen TMDB movie id."""
    query = st.text_input(label, key=f"{key}_query", placeholder="Type a movie title")
    if not query.strip():
        return None
    matches = tmdb.search_movie(query.strip())[:8]
    labels = {
        m["id"]: f"{m['title']} ({(m.get('release_date') or '')[:4] or 'year unknown'})"
        for m in matches
    }
    # A movie sent here from a card is kept as an option even if the title
    # search would not rank it in the first few results (remakes, sequels).
    pinned = st.session_state.get(f"{key}_pinned")
    if pinned and pinned["title"] == query and pinned["id"] not in labels:
        labels = {pinned["id"]: pinned["title"], **labels}
    if not labels:
        st.warning(f"No movie found for “{query}”. Check the spelling or try a shorter title.")
        return None
    if st.session_state.get(f"{key}_choice") not in labels:
        st.session_state.pop(f"{key}_choice", None)
    return st.selectbox("Choose the exact movie", list(labels), format_func=labels.get, key=f"{key}_choice")
