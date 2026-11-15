"""The six pages of the app. Each function draws one page.

Pages only orchestrate: they read widgets, call the pipeline and the
recommenders, and hand the results to the shared components in ``ui.py``.
"""

from __future__ import annotations

import functools
import math

import pandas as pd
import plotly.express as px
import streamlit as st

import features
import nlp
import pipeline
import recommender
import state
import tmdb_client as tmdb
import ui
from filters import CERTIFICATIONS, LANGUAGES, Filters

EXAMPLE_QUERIES = [
    "feel-good 90s comedy under two hours",
    "highly rated sci-fi like Inception",
    "tense heist thriller, not too long",
    "family friendly animated adventure from the last 5 years",
]


def safe_page(page):
    """Show API problems as a friendly message instead of a stack trace."""
    @functools.wraps(page)
    def wrapper():
        try:
            page()
        except tmdb.TMDBError as error:
            st.error(f"{error}")
    return wrapper


# ==========================================================================
# Discover
# ==========================================================================

def _snap(value: float, step: int, up: bool = False) -> int:
    """Round a number onto a slider's step grid."""
    return int((math.ceil if up else math.floor)(value / step) * step)


def apply_query(text: str | None = None) -> None:
    """Callback: parse the search box and load its meaning into the filters.

    The parsed request is written into the same session keys the sidebar
    widgets use, so the user sees exactly how the sentence was understood and
    can adjust any filter afterwards.
    """
    session = st.session_state
    if text is not None:
        session["f_query"] = text
    parsed = nlp.parse_query(session.get("f_query", ""))
    state.reset_filters(clear_query=False)
    warnings: list[str] = []
    free_text = parsed.free_text

    if parsed.year_min or parsed.year_max:
        low = min(max(parsed.year_min or state.MIN_YEAR, state.MIN_YEAR), state.THIS_YEAR)
        high = max(min(parsed.year_max or state.THIS_YEAR, state.THIS_YEAR), low)
        session["f_time_mode"] = "Year range"
        session["f_years"] = (low, high)
    if parsed.runtime_min or parsed.runtime_max:
        low = min(max(_snap(parsed.runtime_min or 40, 5), 40), 240)
        high = max(min(_snap(parsed.runtime_max or 240, 5, up=True), 240), low)
        session["f_runtime"] = (low, high)
    if parsed.min_rating is not None:
        session["f_min_rating"] = min(9.0, math.floor(parsed.min_rating * 2) / 2)
    if parsed.min_votes is not None:
        session["f_min_votes"] = max(v for v in state.VOTE_STEPS if v <= parsed.min_votes)
    session["f_certs"] = [c for c in parsed.certifications if c in CERTIFICATIONS]
    language_names = {code: name for name, code in LANGUAGES.items()}
    if parsed.language in language_names:
        session["f_language"] = language_names[parsed.language]
    session["f_genres"] = parsed.genres_include
    session["f_genres_exclude"] = parsed.genres_exclude
    session["f_genre_mode"] = state.GENRE_MODES[0 if parsed.genre_mode == "all" else 1]
    if parsed.mood:
        session["f_mood"] = parsed.mood

    # Names and titles need TMDB to confirm they exist. Anything TMDB does
    # not recognise is treated as a theme word rather than silently dropped.
    try:
        actors = []
        for name in parsed.actors:
            person = pipeline.resolve_person(name)
            if person:
                actors.append(person["name"])
            else:
                free_text = f"{free_text} {name}".strip()
        session["f_actors"] = ", ".join(actors)
        if parsed.director:
            person = pipeline.resolve_person(parsed.director)
            if person:
                session["f_director"] = person["name"]
            else:
                warnings.append(f"No director found for “{parsed.director}”.")
        if parsed.similar_to:
            session["q_seed"] = pipeline.resolve_movie(parsed.similar_to)
            if not session["q_seed"]:
                free_text = f"{free_text} {parsed.similar_to}".strip()
    except tmdb.TMDBError as error:
        warnings.append(str(error))

    session["q_free"] = free_text
    session["q_notes"] = parsed.notes()
    session["q_warnings"] = warnings


def _sidebar_filters(genres: list[str]) -> Filters:
    """Draw the filter controls and return them as a Filters object."""
    session = st.session_state
    flt = Filters()
    with st.sidebar:
        st.subheader("Filters")

        # ---- Temporal -----------------------------------------------------
        with st.expander("When it was released", expanded=True):
            mode = st.radio("Filter by", state.TIME_MODES, key="f_time_mode", horizontal=True,
                            label_visibility="collapsed")
            if mode == "Year range":
                low, high = st.slider("Release years", state.MIN_YEAR, state.THIS_YEAR, key="f_years")
                # The slider's end stops mean "no limit" on that side.
                flt.date_from = f"{low}-01-01" if low > state.MIN_YEAR else None
                flt.date_to = f"{high}-12-31" if high < state.THIS_YEAR else None
            elif mode == "Decade":
                decade = int(st.selectbox("Decade", state.DECADES, key="f_decade")[:4])
                flt.date_from, flt.date_to = f"{decade}-01-01", f"{decade + 9}-12-31"
            else:
                start = st.date_input("Released on or after", key="f_date_from")
                end = st.date_input("Released on or before", key="f_date_to")
                if start > end:
                    st.error("The start date is after the end date, so the two were swapped.")
                    start, end = end, start
                flt.date_from, flt.date_to = start.isoformat(), end.isoformat()

        # ---- Quality ------------------------------------------------------
        with st.expander("Quality", expanded=True):
            flt.min_rating = st.slider("Minimum rating", 0.0, 9.0, step=0.5, key="f_min_rating")
            flt.min_votes = st.select_slider(
                "Minimum number of votes", state.VOTE_STEPS, key="f_min_votes",
                help="A higher number keeps only widely seen movies and makes ratings more reliable.",
            )

        # ---- Genres -------------------------------------------------------
        with st.expander("Genres", expanded=True):
            flt.genres_include = st.multiselect("Include", genres, key="f_genres")
            mode = st.radio("Match", state.GENRE_MODES, key="f_genre_mode",
                            disabled=len(flt.genres_include) < 2)
            flt.genre_mode = "all" if mode == state.GENRE_MODES[0] else "any"
            flt.genres_exclude = st.multiselect("Exclude (NOT)", genres, key="f_genres_exclude")
            clash = set(flt.genres_include) & set(flt.genres_exclude)
            if clash:
                st.warning(f"{', '.join(sorted(clash))} is both included and excluded, so nothing can match.")

        # ---- Content ------------------------------------------------------
        with st.expander("Runtime, certification, language"):
            low, high = st.slider("Runtime (minutes)", 40, 240, step=5, key="f_runtime")
            flt.runtime_min = low if low > 40 else None
            flt.runtime_max = high if high < 240 else None
            flt.certifications = st.multiselect("US certification", CERTIFICATIONS, key="f_certs")
            language = st.selectbox("Original language", ["Any", *LANGUAGES], key="f_language")
            flt.language = LANGUAGES.get(language)

        # ---- Personnel ----------------------------------------------------
        with st.expander("Cast and director"):
            actors_text = st.text_input("Actors", key="f_actors", placeholder="e.g. Tom Hanks, Meg Ryan",
                                        help="Separate several actors with commas. All must appear.")
            director_text = st.text_input("Director", key="f_director", placeholder="e.g. Greta Gerwig")
            actors, missing = pipeline.resolve_people(actors_text)
            flt.cast_ids = [person["id"] for person in actors]
            director = pipeline.resolve_person(director_text) if director_text.strip() else None
            if director:
                flt.director_id, flt.director_name = director["id"], director["name"]
            elif director_text.strip():
                missing.append(director_text.strip())
            matched = [person["name"] for person in actors] + ([director["name"]] if director else [])
            if matched:
                st.caption("Matched: " + ", ".join(matched))
            if missing:
                st.warning("Not found on TMDB: " + ", ".join(missing))

        st.button("Reset all filters", on_click=state.reset_filters, width="stretch",
                  icon=":material/restart_alt:", key="btn_reset")
    return flt


@safe_page
def discover_page() -> None:
    ui.page_header("Find tonight's movie", "Describe what you feel like watching, or set the filters yourself.")
    ui.require_api_key()
    session = st.session_state
    genres = sorted(tmdb.genre_map().values())

    # ---- Natural language search -----------------------------------------
    with st.form("search", border=False):
        box, button = st.columns([6, 1], vertical_alignment="bottom")
        box.text_input("What are you in the mood for?", key="f_query",
                       placeholder="e.g. feel-good 90s comedy under two hours with Tom Hanks")
        button.form_submit_button("Search", type="primary", on_click=apply_query, width="stretch")
    st.caption("Try one of these:")
    for index, example in enumerate(EXAMPLE_QUERIES):
        if index % 2 == 0:
            example_columns = st.columns(2)
        example_columns[index % 2].button(example, key=f"btn_example_{example}", on_click=apply_query, args=(example,),
                      width="stretch")

    if session["q_notes"]:
        badges = " ".join(f":blue-badge[{note}]" for note in session["q_notes"])
        st.markdown(f"**Understood as** &nbsp;{badges}")
    for warning in session["q_warnings"]:
        st.warning(warning)

    flt = _sidebar_filters(genres)

    # ---- Ranking options ---------------------------------------------------
    method_column, mood_column, count_column = st.columns([2, 2, 1])
    method = method_column.selectbox(
        "Recommendation approach", recommender.METHODS, key="f_method",
        help="Hybrid blends content match, mood match and rating. The others use one signal only.",
    )
    mood = mood_column.selectbox(
        "Mood", ["Any", *nlp.MOODS], key="f_mood",
        help="Matched against the sentiment of each plot summary.",
    )
    count = count_column.selectbox("Show", [6, 12, 20, 30], key="f_count")
    weights = recommender.DEFAULT_WEIGHTS
    if method == "Hybrid":
        with st.expander("Adjust how the hybrid score is weighted"):
            sliders = st.columns(3)
            weights = {
                "content": sliders[0].slider("Content match", 0.0, 1.0, step=0.05, key="f_w_content"),
                "mood": sliders[1].slider("Mood match", 0.0, 1.0, step=0.05, key="f_w_mood"),
                "quality": sliders[2].slider("Rating", 0.0, 1.0, step=0.05, key="f_w_quality"),
            }
            st.caption("Weights are rescaled to add up to 100%. A signal with nothing to "
                       "measure, such as mood when no mood is chosen, is left out.")

    # ---- Candidates ----------------------------------------------------------
    seed = session["q_seed"]
    free_text = session["q_free"]
    with st.spinner("Finding movies…"):
        pool = pipeline.discover_pool(flt, free_text, seed["id"] if seed else None)
        if seed:
            pool = pool[pool["id"] != seed["id"]].reset_index(drop=True)
    if pool.empty:
        session["results"] = None
        st.warning(
            "**No movies match all of these filters together.** Try widening the release "
            "years, lowering the minimum rating or vote count, removing a genre, or "
            "switching genres to “Any selected (OR)”."
        )
        return

    # ---- Content signal: what should the movies be similar to? -----------
    content, terms, label = None, None, "Matches your request"
    watchlist = state.watchlist_frame()
    if seed:
        seed_details = tmdb.movie_details(seed["id"])
        if seed_details:
            seed_text = features.to_record(seed_details)["text"] + " " + free_text
            content, terms = recommender.text_similarity(pool, seed_text)
            label = f"Close to {seed['title']}"
    elif free_text:
        content, terms = recommender.text_similarity(pool, free_text)
    elif not watchlist.empty:
        content, terms = recommender.text_similarity(pool, recommender.build_profile(watchlist)["text"])
        label = "Echoes your watchlist"

    ranked, used = recommender.rank(
        pool, method=method, weights=weights, mood=None if mood == "Any" else mood,
        content=content, terms=terms, content_label=label,
    )
    session["results"] = ranked

    # ---- Results -------------------------------------------------------------
    signal_names = {"content": "content match", "mood": "mood match", "quality": "vote-adjusted rating"}
    st.subheader(f"{min(count, len(ranked))} picks from {len(ranked)} matching movies")
    summary = "Ranked by " + ", ".join(signal_names[s] for s in used) + "."
    active = flt.describe()
    if active:
        summary += " Filters: " + "; ".join(active) + "."
    st.caption(summary)
    if method == "Content match" and "content" not in used:
        st.info("Content match needs something to compare against: describe a theme in the "
                "search box, name a movie (“like Inception”), or save movies to your watchlist. "
                "Showing the best-rated matches for now.")
    elif method == "Mood match" and "mood" not in used:
        st.info("Choose a mood to rank by mood. Showing the best-rated matches for now.")
    elif free_text and content is not None and content.max() == 0:
        st.info(f"No plot mentions “{free_text}”, so results are ranked on the other signals.")

    ui.movie_grid(ranked.head(count), key="discover", why_column="why", score_column="score")


# ==========================================================================
# Similar movies
# ==========================================================================

@safe_page
def similar_page() -> None:
    ui.page_header("More like this", "Pick a movie you enjoyed and see what resembles it, and why.")
    ui.require_api_key()
    movie_id = ui.movie_picker("A movie you liked", key="pick_similar")
    if not movie_id:
        st.info("Type a title above to start. You can also press “Similar” on any movie card.")
        return
    details = tmdb.movie_details(movie_id)
    if not details:
        st.error("That movie could not be loaded from TMDB. Please pick another.")
        return
    seed = features.to_record(details)

    st.subheader("Your pick")
    ui.movie_card(seed, key="seed")

    with st.spinner("Comparing plots, themes, cast and crew…"):
        ranked = recommender.similar_movies(seed, pipeline.similar_pool(seed))
    if ranked.empty:
        st.warning("TMDB has too little information about this movie to find close matches.")
        return
    st.subheader(f"Closest matches to {seed['title']}")
    weights = recommender.SIMILARITY_WEIGHTS
    st.caption(
        f"Similarity combines plot and keyword text ({weights['text']:.0%}), genres "
        f"({weights['genre']:.0%}), shared themes ({weights['keyword']:.0%}), cast "
        f"({weights['cast']:.0%}) and director ({weights['director']:.0%})."
    )
    ui.movie_grid(ranked.head(10), key="similar", why_column="reasons",
                  score_column="similarity", score_label="Similarity")


# ==========================================================================
# Watchlist
# ==========================================================================

@safe_page
def watchlist_page() -> None:
    ui.page_header("Your watchlist", "Saved movies shape the recommendations below.")
    ui.require_api_key()
    watchlist = state.watchlist_frame()
    if watchlist.empty:
        st.info("Your watchlist is empty. Press “Watchlist” on any movie to save it here. "
                "Two or three saved movies are enough to get personal recommendations.")
        return

    profile = recommender.build_profile(watchlist)
    first, second, third, fourth = st.columns(4)
    first.metric("Saved movies", len(watchlist))
    second.metric("Average rating", f"{profile['rating']:.1f}")
    third.metric("Favourite genres", ", ".join(profile["top_genres"][:2]) or "None yet")
    fourth.metric("Typical tone", nlp.tone_label(profile["tone"]))

    export = watchlist[["title", "year", "director", "rating", "runtime", "certification"]].copy()
    export["genres"] = watchlist["genres"].apply(", ".join)
    st.download_button("Download watchlist (CSV)", export.to_csv(index=False), "moodreel_watchlist.csv",
                       "text/csv", icon=":material/download:", key="btn_export")
    st.caption("The watchlist lasts for this browser session. Download it to keep a copy.")

    ui.movie_grid(watchlist, key="watchlist")

    st.subheader("Recommended for you")
    st.caption("Ranked by similarity to the plots and genres you saved, closeness to your "
               "typical tone, and vote-adjusted rating.")
    with st.spinner("Building recommendations from your watchlist…"):
        picks = recommender.profile_recommend(watchlist, pipeline.profile_pool(watchlist))
    if picks.empty:
        st.warning("No recommendations could be built from these movies yet. Try saving one more.")
        return
    ui.movie_grid(picks.head(8), key="for_you", why_column="why", score_column="score")


# ==========================================================================
# Trending
# ==========================================================================

@safe_page
def trending_page() -> None:
    ui.page_header("Trending now", "What TMDB users are watching, refreshed every 30 minutes.")
    ui.require_api_key()
    choice = st.segmented_control("Period", ["Today", "This week"], key="f_trend_window")
    window = "day" if choice == "Today" else "week"
    with st.spinner("Loading trending movies…"):
        frame = pipeline.trending_frame(window)
    if frame.empty:
        st.warning("TMDB returned no trending movies. Please try again shortly.")
        return
    ui.movie_grid(frame, key="trending", numbered=True)


# ==========================================================================
# Insights
# ==========================================================================

def _chart(figure) -> None:
    figure.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend_title_text="")
    st.plotly_chart(figure, width="stretch")


@safe_page
def insights_page() -> None:
    ui.page_header("Insights", "Patterns in a set of movies: ratings, genres, eras, tone and box office.")
    ui.require_api_key()
    source = st.selectbox("Movies to analyse",
                          ["Latest Discover results", "My watchlist", "Trending this week"],
                          key="f_insight_source")
    if source == "Latest Discover results":
        frame = st.session_state["results"]
        if frame is None or frame.empty:
            st.info("Run a search on the Discover page first, then come back here.")
            return
    elif source == "My watchlist":
        frame = state.watchlist_frame()
        if frame.empty:
            st.info("Your watchlist is empty. Save a few movies first.")
            return
    else:
        frame = pipeline.trending_frame("week", limit=20)
    if len(frame) < 3:
        st.info("At least three movies are needed to draw meaningful charts.")
        return

    frame = frame.copy()
    frame["tone_label"] = frame["tone"].apply(nlp.tone_label)
    first, second, third, fourth = st.columns(4)
    first.metric("Movies", len(frame))
    second.metric("Average rating", f"{frame['rating'].mean():.1f}")
    third.metric("Median runtime", f"{frame['runtime'].median():.0f} min")
    fourth.metric("Most common tone", frame["tone_label"].mode().iat[0])

    tone_colours = {"Uplifting": "#F5B841", "Light-hearted": "#8FD694", "Neutral": "#9AA3B5",
                    "Tense": "#6CA6E8", "Dark": "#C76B8F"}
    ratings, genres, timeline, tone, money = st.tabs(
        ["Rating vs popularity", "Genres", "Timeline", "Tone", "Box office"]
    )
    with ratings:
        _chart(px.scatter(
            frame, x="popularity", y="rating", size="votes", color="tone_label",
            color_discrete_map=tone_colours, hover_name="title", log_x=True, size_max=40,
            labels={"popularity": "TMDB popularity (log scale)", "rating": "Rating"},
            title="Are the popular movies also the well-rated ones? (bubble size = votes)",
        ))
    with genres:
        exploded = frame.explode("genres").dropna(subset=["genres"])
        by_genre = exploded.groupby("genres").agg(movies=("id", "count"), rating=("rating", "mean"))
        by_genre = by_genre.sort_values("movies").reset_index()
        _chart(px.bar(
            by_genre, x="movies", y="genres", orientation="h", color="rating",
            color_continuous_scale="YlOrBr",
            labels={"movies": "Number of movies", "genres": "", "rating": "Avg rating"},
            title="Genre mix, coloured by average rating",
        ))
    with timeline:
        dated = frame.dropna(subset=["year"]).copy()
        dated["decade"] = (dated["year"] // 10 * 10).astype(int).astype(str) + "s"
        by_decade = dated.groupby("decade").agg(movies=("id", "count"), rating=("rating", "mean")).reset_index()
        _chart(px.bar(
            by_decade, x="decade", y="movies", color="rating", color_continuous_scale="YlOrBr",
            labels={"decade": "Decade", "movies": "Number of movies", "rating": "Avg rating"},
            title="Movies by decade, coloured by average rating",
        ))
    with tone:
        exploded = frame.explode("genres").dropna(subset=["genres"])
        by_genre = exploded.groupby("genres")["tone"].mean().sort_values().reset_index()
        _chart(px.bar(
            by_genre, x="tone", y="genres", orientation="h", color="tone",
            color_continuous_scale="RdYlGn", range_color=[-1, 1],
            labels={"tone": "Average plot sentiment (−1 dark, +1 uplifting)", "genres": ""},
            title="Which genres have the darkest and brightest plot summaries?",
        ))
        st.caption("Tone is the VADER sentiment score of each plot summary, the same "
                   "signal used for mood matching.")
    with money:
        reported = frame[(frame["budget"] > 0) & (frame["revenue"] > 0)]
        if len(reported) < 3:
            st.info("Too few of these movies report both a budget and box office takings.")
        else:
            figure = px.scatter(
                reported, x="budget", y="revenue", color="rating", hover_name="title",
                color_continuous_scale="YlOrBr", log_x=True, log_y=True,
                labels={"budget": "Budget (USD, log scale)", "revenue": "Box office (USD, log scale)",
                        "rating": "Rating"},
                title="Budget against box office (above the line = earned back its budget)",
            )
            low, high = reported["budget"].min(), reported["budget"].max()
            figure.add_shape(type="line", x0=low, y0=low, x1=high, y1=high,
                             line=dict(dash="dash", color="#9AA3B5"))
            _chart(figure)


# ==========================================================================
# Compare
# ==========================================================================

def _winner(first: dict, second: dict, field: str, unit: str, higher: str, lower_is: str | None = None) -> str | None:
    """One sentence saying which movie leads on a numeric field."""
    a, b = first.get(field), second.get(field)
    if not a or not b or pd.isna(a) or pd.isna(b) or a == b:
        return None
    leader, trailer = (first, second) if a > b else (second, first)
    shown = "{:.1f}" if field == "rating" else "{:,.0f}"
    high, low = shown.format(max(a, b)), shown.format(min(a, b))
    if lower_is:
        return f"**{trailer['title']}** is {lower_is} ({low}{unit} against {high}{unit})."
    return f"**{leader['title']}** {higher} ({high}{unit} against {low}{unit})."


@safe_page
def compare_page() -> None:
    ui.page_header("Compare two movies", "Can't decide between two? Put them side by side.")
    ui.require_api_key()
    left, right = st.columns(2)
    with left:
        first_id = ui.movie_picker("First movie", key="pick_compare_a")
    with right:
        second_id = ui.movie_picker("Second movie", key="pick_compare_b")
    if not first_id or not second_id:
        st.info("Choose two movies to compare them.")
        return
    if first_id == second_id:
        st.warning("Both boxes hold the same movie. Pick a different one on one side.")
        return
    details = tmdb.movies_details([first_id, second_id])
    if first_id not in details or second_id not in details:
        st.error("One of these movies could not be loaded from TMDB. Please pick another.")
        return
    first, second = features.to_record(details[first_id]), features.to_record(details[second_id])
    names = [ui.movie_label(first), ui.movie_label(second)]
    if names[0] == names[1]:
        names = [f"{names[0]} [A]", f"{names[1]} [B]"]

    for column, movie, key in zip(st.columns(2), (first, second), ("compare_a", "compare_b")):
        with column:
            ui.movie_card(movie, key=key)

    def profit(movie: dict) -> str:
        if not movie["budget"] or not movie["revenue"]:
            return "Not reported"
        return f"{movie['revenue'] / movie['budget']:.1f}x budget"

    rows = {
        "Release date": lambda m: m["release_date"] or "Unknown",
        "Runtime": lambda m: f"{ui.as_int(m['runtime'])} min" if ui.as_int(m["runtime"]) else "Unknown",
        "Certification": lambda m: m["certification"] or "Unknown",
        "Rating": lambda m: f"{m['rating']:.1f} ({m['votes']:,} votes)",
        "Genres": lambda m: ", ".join(m["genres"]) or "Unknown",
        "Director": lambda m: m["director"] or "Unknown",
        "Lead cast": lambda m: ", ".join(m["cast"][:4]) or "Unknown",
        "Plot tone": lambda m: f"{nlp.tone_label(m['tone'])} ({m['tone']:+.2f})",
        "Budget": lambda m: features.format_money(m["budget"]),
        "Box office": lambda m: features.format_money(m["revenue"]),
        "Return": profit,
    }
    table = pd.DataFrame(
        {names[0]: [f(first) for f in rows.values()], names[1]: [f(second) for f in rows.values()]},
        index=list(rows),
    )
    st.subheader("Side by side")
    st.dataframe(table, width="stretch")

    # Each measure is shown relative to the larger of the two values, so
    # measures with very different units can share one chart.
    measures = {"Rating": "rating", "Votes": "votes", "Popularity": "popularity",
                "Runtime": "runtime", "Box office": "revenue"}
    bars = []
    for label, field in measures.items():
        values = [float(first.get(field) or 0), float(second.get(field) or 0)]
        top = max(values)
        if top > 0:
            bars += [{"Measure": label, "Movie": name, "Relative value": value / top}
                     for name, value in zip(names, values)]
    if bars:
        _chart(px.bar(pd.DataFrame(bars), x="Measure", y="Relative value", color="Movie", barmode="group",
                      color_discrete_sequence=["#F5B841", "#6CA6E8"],
                      title="How they measure up (each measure scaled to the higher movie)"))

    st.subheader("Verdict")
    verdicts = [
        _winner(first, second, "rating", "", "is rated higher"),
        _winner(first, second, "votes", " votes", "has been rated by more people"),
        _winner(first, second, "runtime", " min", "", lower_is="shorter"),
    ]
    for line in filter(None, verdicts):
        st.markdown(f"- {line}")
    similarity, reasons = recommender.compare_pair(first, second)
    st.markdown(f"**How alike are they?** Similarity {similarity:.0%}")
    st.markdown("\n".join(f"- {reason}" for reason in reasons))
