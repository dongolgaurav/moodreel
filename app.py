"""MoodReel: a mood-aware movie recommender built on the TMDB API.

Entry point. Run locally with:  streamlit run app.py

Module map
----------
tmdb_client.py  API access and caching
features.py     data processing and feature engineering
filters.py      the filtering system
nlp.py          TF-IDF text features, sentiment analysis, query parsing
recommender.py  content-based, sentiment-based and hybrid ranking
pipeline.py     candidate generation (which movies get ranked)
state.py        session state: filters, search, watchlist
ui.py           shared interface components
views.py        the six pages
"""

from pathlib import Path

import streamlit as st

import state
import ui
import views

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title=f"{ui.APP_NAME} | {ui.TAGLINE}", page_icon=str(ASSETS / "icon.svg"),
                   layout="wide")
state.init_state()
ui.inject_style()
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "icon.svg"))

# Navigation: one entry per page, grouped by what the user is trying to do.
ui.PAGES.update({
    "discover": st.Page(views.discover_page, title="Discover", icon=":material/search:",
                        url_path="discover", default=True),
    "similar": st.Page(views.similar_page, title="More like this", icon=":material/hub:",
                       url_path="similar"),
    "watchlist": st.Page(views.watchlist_page, title="Watchlist", icon=":material/bookmark:",
                         url_path="watchlist"),
    "trending": st.Page(views.trending_page, title="Trending", icon=":material/trending_up:",
                        url_path="trending"),
    "insights": st.Page(views.insights_page, title="Insights", icon=":material/insights:",
                        url_path="insights"),
    "compare": st.Page(views.compare_page, title="Compare", icon=":material/compare_arrows:",
                       url_path="compare"),
})
navigation = st.navigation({
    "Find a movie": [ui.PAGES["discover"], ui.PAGES["similar"], ui.PAGES["trending"]],
    "Your space": [ui.PAGES["watchlist"]],
    "Explore": [ui.PAGES["insights"], ui.PAGES["compare"]],
})
navigation.run()

# TMDB's terms of use require this attribution notice.
st.sidebar.caption(
    f"{ui.APP_NAME} uses the TMDB API but is not endorsed or certified by TMDB. "
    "Movie data and images are provided by themoviedb.org."
)
