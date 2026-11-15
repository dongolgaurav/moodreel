# MoodReel

**Tell it your mood. Get tonight's movie.**

MoodReel is a movie recommendation app built on the TMDB API. It is aimed at
people who spend longer choosing a movie than watching one: you describe what
you feel like in a sentence, and it returns a short ranked list with the
reason for each pick.

## What it does

| Area | Details |
|---|---|
| Recommendation approaches | Content-based (TF-IDF and cosine similarity), sentiment-based (VADER tone of the plot against a chosen mood), and a hybrid that blends both with a vote-adjusted rating |
| Filters | Release years, decade or exact dates; minimum rating and vote count; runtime, US certification, language; actors and director; genres with AND / OR / NOT |
| Natural language search | "feel-good 90s comedy under two hours with Tom Hanks" is parsed into filters, a mood and themes |
| More like this | Similar movies with the reasons: shared genres, themes, plot language, cast, director, tone |
| Watchlist | Saved movies build a taste profile that drives personal recommendations |
| Trending | What is popular on TMDB today or this week |
| Insights | Charts of ratings, genres, eras, tone and box office for any result set |
| Compare | Two movies side by side, with a verdict and a similarity score |

## Run it locally

```bash
pip install -r requirements.txt
```

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste
your TMDB API key into it (a free key is available at themoviedb.org). Then:

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. On share.streamlit.io choose "Create app", pick the repository and set the
   main file to `app.py`.
3. Under "Advanced settings", add the secret `TMDB_API_KEY = "your key"`.

The key is read from Streamlit secrets or the `TMDB_API_KEY` environment
variable. It is never stored in the code or in the repository.

## Code structure

| File | Responsibility |
|---|---|
| `app.py` | Entry point, branding and navigation |
| `tmdb_client.py` | TMDB API calls, caching, retries and error handling |
| `features.py` | Flattens TMDB JSON into a feature table; vote-adjusted rating |
| `filters.py` | The filter object: TMDB query parameters and exact re-checking |
| `nlp.py` | Tokenising, stemming, TF-IDF, VADER sentiment, query parser |
| `recommender.py` | Content-based, sentiment-based and hybrid ranking, with explanations |
| `pipeline.py` | Candidate generation: which movies each recommender ranks |
| `state.py` | Session state: filters, parsed search, watchlist |
| `ui.py` | Shared components: movie card, movie search box |
| `views.py` | The six pages |
| `tests/test_core.py` | Offline tests for the parser, filters and recommenders |

## How a recommendation is made

1. **Candidates.** The filters are sent to TMDB's discover endpoint. Popular
   matches, acclaimed matches and matches tagged with the request's themes
   are merged into a pool of up to 140 movies.
2. **Features.** Each movie's details, keywords, credits and certification
   are fetched in one request, cached, and flattened into a table. The plot
   summary is scored for sentiment.
3. **Filtering.** Every filter is re-applied to the detailed records, so
   combined filters always hold.
4. **Ranking.** Content match, mood match and vote-adjusted rating are
   blended with adjustable weights. Signals that do not apply to a request
   are left out and the remaining weights rescaled.
5. **Explanation.** Each card shows the terms and attributes that earned the
   movie its place.

## Tests

```bash
python -m pytest
```

The tests need no network access or API key.

## Attribution

This product uses the TMDB API but is not endorsed or certified by TMDB.
