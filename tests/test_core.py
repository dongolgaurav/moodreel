"""Offline tests for the parser, the filters and the recommenders.

No network or API key is needed. Run with:  python -m pytest
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import features  # noqa: E402
import nlp  # noqa: E402
import recommender  # noqa: E402
from filters import Filters  # noqa: E402

TODAY = date(2026, 11, 14)


def movie(movie_id, title, year, genres, overview, **extra):
    """Build a TMDB-style detail payload for tests."""
    return {
        "id": movie_id, "title": title, "release_date": f"{year}-06-15",
        "runtime": extra.get("runtime", 110),
        "genres": [{"name": g} for g in genres],
        "overview": overview,
        "vote_average": extra.get("rating", 7.0), "vote_count": extra.get("votes", 1000),
        "popularity": 10, "original_language": extra.get("language", "en"),
        "keywords": {"keywords": [{"name": k} for k in extra.get("keywords", [])]},
        "credits": {
            "cast": [{"name": n, "order": i} for i, n in enumerate(extra.get("cast", []))],
            "crew": [{"name": extra.get("director", "Nobody"), "job": "Director"}],
        },
        "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [
            {"certification": ""}, {"certification": extra.get("cert", "PG-13")}]}]},
    }


FRAME = features.build_frame([
    movie(1, "Vault", 1995, ["Crime", "Thriller"],
          "A crew of thieves plans a daring bank heist that goes terribly wrong.",
          keywords=["heist", "bank robbery"], cast=["Ann Lee", "Bo Cruz"], director="Dee Park",
          cert="R", runtime=105, rating=7.8, votes=4000),
    movie(2, "Second Score", 1998, ["Crime", "Thriller"],
          "Veteran robbers attempt one last heist on an armoured truck.",
          keywords=["heist"], cast=["Ann Lee"], director="Dee Park", cert="R", runtime=125),
    movie(3, "Sunny Days", 2015, ["Comedy", "Romance"],
          "Two best friends fall in love during a joyful, wonderful summer of laughter.",
          keywords=["friendship"], cast=["Cy Vance"], cert="PG", rating=6.5, votes=300),
    movie(4, "Night Terror", 2019, ["Horror"],
          "A murderous evil haunts a family, killing them one by one in terror.",
          cert="R", rating=6.0, votes=150, language="ko", runtime=92),
])


# ---------------------------------------------------------------- parser

def test_parser_combined_request():
    q = nlp.parse_query("90s thrillers under two hours with denzel washington", TODAY)
    assert (q.year_min, q.year_max) == (1990, 1999)
    assert q.genres_include == ["Thriller"]
    assert q.runtime_max == 120
    assert q.actors == ["denzel washington"]
    assert q.free_text == ""


def test_parser_genre_logic():
    assert nlp.parse_query("comedy or romance", TODAY).genre_mode == "any"
    q = nlp.parse_query("funny sci-fi but no horror", TODAY)
    assert q.genres_include == ["Comedy", "Science Fiction"] and q.genre_mode == "all"
    assert q.genres_exclude == ["Horror"]
    assert nlp.parse_query("a romcom", TODAY).genres_include == ["Romance", "Comedy"]


def test_parser_time_expressions():
    assert nlp.parse_query("movies after 2010", TODAY).year_min == 2011
    assert nlp.parse_query("between 1980 and 1989", TODAY).year_max == 1989
    late = nlp.parse_query("late 80s action", TODAY)
    assert (late.year_min, late.year_max) == (1987, 1989)
    assert nlp.parse_query("something from the last 5 years", TODAY).year_min == 2021
    assert nlp.parse_query("recent drama", TODAY).year_min == 2024
    assert nlp.parse_query("a 2014 film", TODAY).year_min == 2014


def test_parser_quality_runtime_cert_language():
    q = nlp.parse_query("highly rated korean thriller rated R over 90 minutes", TODAY)
    assert q.min_rating == 7.5 and q.language == "ko"
    assert q.certifications == ["R"] and q.runtime_min == 90
    assert nlp.parse_query("rated above 8", TODAY).min_rating == 8.0
    assert nlp.parse_query("7.5+ animated", TODAY).min_rating == 7.5
    assert nlp.parse_query("family friendly adventure", TODAY).certifications == ["G", "PG"]
    assert nlp.parse_query("an hour and a half max", TODAY).runtime_max == 90
    assert nlp.parse_query("under 2 and a half hours", TODAY).runtime_max == 150


def test_parser_mood_people_seed_and_theme():
    q = nlp.parse_query("feel-good heist movie directed by steven soderbergh", TODAY)
    assert q.mood == "Uplifting" and q.director == "steven soderbergh"
    assert q.free_text == "heist"
    assert nlp.parse_query("something like inception but funny", TODAY).similar_to == "inception"
    assert nlp.parse_query("highly rated sci-fi like Inception", TODAY).similar_to == "inception"
    assert nlp.parse_query("I'd like a comedy", TODAY).similar_to is None
    assert nlp.parse_query("i like heist movies", TODAY).similar_to is None
    assert nlp.parse_query("a thriller with a twist", TODAY).actors == []
    assert nlp.parse_query("starring tom hanks and meg ryan", TODAY).actors == ["tom hanks", "meg ryan"]
    assert nlp.parse_query("not too dark", TODAY).mood == "Light-hearted"


def test_parser_handles_nonsense():
    assert nlp.parse_query("", TODAY).notes() == []
    assert nlp.parse_query("!!! ???", TODAY).free_text == ""


# ---------------------------------------------------------------- sentiment

def test_tone_separates_light_from_dark():
    tones = dict(zip(FRAME["title"], FRAME["tone"]))
    assert tones["Sunny Days"] > 0.5 > -0.5 > tones["Night Terror"]
    assert nlp.mood_match(0.75, 0.75) == 1.0
    assert nlp.mood_match(-0.75, 0.75) == 0.0


# ---------------------------------------------------------------- filters

def test_filters_individually_and_combined():
    assert list(Filters(genres_include=["Crime", "Thriller"]).apply(FRAME)["id"]) == [1, 2]
    assert list(Filters(genres_include=["Horror", "Comedy"], genre_mode="any").apply(FRAME)["id"]) == [3, 4]
    assert list(Filters(genres_exclude=["Crime"]).apply(FRAME)["id"]) == [3, 4]
    assert list(Filters(certifications=["R"], runtime_max=110).apply(FRAME)["id"]) == [1, 4]
    assert list(Filters(language="ko").apply(FRAME)["id"]) == [4]
    combined = Filters(date_from="1990-01-01", date_to="1999-12-31", min_rating=7.5,
                       min_votes=1000, certifications=["R"], genres_include=["Crime"],
                       director_name="dee park", runtime_max=120)
    assert list(combined.apply(FRAME)["id"]) == [1]
    assert Filters(min_rating=9.9).apply(FRAME).empty


def test_discover_params_use_tmdb_operators():
    ids = {"Crime": 80, "Thriller": 53, "Horror": 27}
    params = Filters(genres_include=["Crime", "Thriller"], genres_exclude=["Horror"],
                     certifications=["PG", "PG-13"], cast_ids=[5, 6]).to_discover_params(ids)
    assert params["with_genres"] == "80,53" and params["without_genres"] == "27"
    assert params["certification"] == "PG|PG-13" and params["with_cast"] == "5,6"
    any_mode = Filters(genres_include=["Crime", "Horror"], genre_mode="any").to_discover_params(ids)
    assert any_mode["with_genres"] == "80|27"


# ---------------------------------------------------------------- recommenders

def test_content_ranking_finds_theme_and_explains_it():
    scores, terms = recommender.text_similarity(FRAME, "bank heists")
    ranked, used = recommender.rank(FRAME, method="Content match", content=scores, terms=terms)
    assert ranked.iloc[0]["title"] == "Vault" and used == ["content"]
    assert "heist" in ranked.iloc[0]["why"]


def test_mood_ranking_and_fallback():
    ranked, used = recommender.rank(FRAME, method="Mood match", mood="Uplifting")
    assert ranked.iloc[0]["title"] == "Sunny Days" and used == ["mood"]
    fallback, used = recommender.rank(FRAME, method="Content match")  # nothing to match on
    assert used == ["quality"] and fallback.iloc[0]["title"] == "Vault"


def test_hybrid_drops_missing_signals():
    _, used = recommender.rank(FRAME, method="Hybrid", mood="Dark")
    assert set(used) == {"mood", "quality"}


def test_similar_movies_and_profile():
    seed = FRAME.iloc[0].to_dict()
    similar = recommender.similar_movies(seed, FRAME)
    assert similar.iloc[0]["title"] == "Second Score" and 1 not in set(similar["id"])
    assert any("Same director" in r for r in similar.iloc[0]["reasons"])
    picks = recommender.profile_recommend(FRAME.iloc[[0]], FRAME)
    assert picks.iloc[0]["title"] == "Second Score"
    score, reasons = recommender.compare_pair(seed, FRAME.iloc[1].to_dict())
    assert 0 < score <= 1 and reasons
