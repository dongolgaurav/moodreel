"""Session state: filter defaults, the parsed search and the watchlist.

Streamlit reruns the whole script on every interaction, so anything that must
survive a rerun lives in ``st.session_state``. Keys follow a naming scheme:

* ``f_...``    filter and option widgets
* ``pick_...`` movie search boxes
* ``q_...``    what the last natural language search was understood to mean
* ``watchlist`` and ``results`` hold data shared between pages
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import features

MIN_YEAR = 1920
THIS_YEAR = date.today().year
VOTE_STEPS = [0, 50, 100, 200, 500, 1000, 2500, 5000, 10000]
TIME_MODES = ["Year range", "Decade", "Exact dates"]
DECADES = [f"{start}s" for start in range(2020, MIN_YEAR - 1, -10)]
GENRE_MODES = ["All selected (AND)", "Any selected (OR)"]
PERSIST_PREFIXES = ("f_", "pick_")

FILTER_DEFAULTS = {
    "f_time_mode": "Year range",
    "f_years": (MIN_YEAR, THIS_YEAR),
    "f_decade": "1990s",
    "f_date_from": date(THIS_YEAR - 1, 1, 1),
    "f_date_to": date.today(),
    "f_min_rating": 6.0,
    "f_min_votes": 200,
    "f_runtime": (40, 240),
    "f_certs": [],
    "f_language": "Any",
    "f_actors": "",
    "f_director": "",
    "f_genres": [],
    "f_genre_mode": GENRE_MODES[0],
    "f_genres_exclude": [],
    "f_mood": "Any",
}
OPTION_DEFAULTS = {
    "f_query": "",
    "f_method": "Hybrid",
    "f_count": 12,
    "f_w_content": 0.45,
    "f_w_mood": 0.25,
    "f_w_quality": 0.30,
    "f_trend_window": "This week",
    "f_insight_source": "Latest Discover results",
}
QUERY_DEFAULTS = {"q_free": "", "q_seed": None, "q_notes": [], "q_warnings": []}


def init_state() -> None:
    """Create missing keys and keep widget values alive across pages.

    Streamlit discards a widget's value when the widget is not drawn, which
    would reset the filters every time the user visits another page. Writing
    each value back to itself marks it as ordinary state, so it is kept.
    """
    for key, value in {**FILTER_DEFAULTS, **OPTION_DEFAULTS, **QUERY_DEFAULTS}.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("watchlist", {})
    st.session_state.setdefault("results", None)
    for key in list(st.session_state.keys()):
        if key.startswith(PERSIST_PREFIXES):
            st.session_state[key] = st.session_state[key]


def reset_filters(clear_query: bool = True) -> None:
    """Return every filter, and optionally the search box, to its default."""
    for key, value in {**FILTER_DEFAULTS, **QUERY_DEFAULTS}.items():
        st.session_state[key] = value
    if clear_query:
        st.session_state["f_query"] = ""


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------

def add_to_watchlist(movie: dict) -> None:
    """Save a movie. Only the base feature columns are stored."""
    st.session_state["watchlist"][int(movie["id"])] = {c: movie.get(c) for c in features.COLUMNS}


def remove_from_watchlist(movie_id: int) -> None:
    st.session_state["watchlist"].pop(int(movie_id), None)


def in_watchlist(movie_id: int) -> bool:
    return int(movie_id) in st.session_state["watchlist"]


def watchlist_frame() -> pd.DataFrame:
    """The watchlist as a feature DataFrame, in the order movies were added."""
    return pd.DataFrame(list(st.session_state["watchlist"].values()), columns=features.COLUMNS)
