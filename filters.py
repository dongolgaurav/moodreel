"""The filtering system: one object that holds every filter the user set.

``Filters`` does two jobs:

1. ``to_discover_params`` translates the filters into TMDB discover query
   parameters, so TMDB does the heavy filtering across its whole catalogue.
2. ``apply`` re-checks the same rules against the detailed movie records.
   This guarantees correct results when filters are combined, covers rules
   TMDB cannot express exactly (a specific director), and lets the filters be
   applied to candidates that did not come from discover.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CERTIFICATIONS = ["G", "PG", "PG-13", "R", "NC-17"]

# Display name -> ISO 639-1 code used by TMDB.
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de",
    "Italian": "it", "Portuguese": "pt", "Japanese": "ja", "Korean": "ko",
    "Chinese": "zh", "Hindi": "hi", "Nepali": "ne", "Russian": "ru",
    "Swedish": "sv", "Danish": "da", "Turkish": "tr", "Persian": "fa",
}

RUNTIME_FLOOR = 40    # slider minimum, treated as "no lower limit"
RUNTIME_CEILING = 240  # slider maximum, treated as "no upper limit"


@dataclass
class Filters:
    """Every filter in the app. Fields left at their default mean "no limit"."""

    # Temporal
    date_from: str | None = None  # ISO date, inclusive
    date_to: str | None = None    # ISO date, inclusive
    # Quality
    min_rating: float = 0.0
    min_votes: int = 0
    # Content
    runtime_min: int | None = None
    runtime_max: int | None = None
    certifications: list[str] = field(default_factory=list)
    language: str | None = None  # ISO 639-1 code
    # Personnel
    cast_ids: list[int] = field(default_factory=list)
    director_id: int | None = None
    director_name: str | None = None
    # Genres with logical operators
    genres_include: list[str] = field(default_factory=list)
    genre_mode: str = "all"  # "all" = AND, "any" = OR
    genres_exclude: list[str] = field(default_factory=list)  # NOT

    def to_discover_params(self, genre_ids: dict[str, int]) -> dict:
        """Translate the filters into TMDB discover parameters.

        ``genre_ids`` maps genre names to TMDB ids. TMDB joins ids with a
        comma for AND and with a pipe for OR.
        """
        params: dict = {}
        if self.date_from:
            params["primary_release_date.gte"] = self.date_from
        if self.date_to:
            params["primary_release_date.lte"] = self.date_to
        if self.min_rating > 0:
            params["vote_average.gte"] = self.min_rating
        if self.min_votes > 0:
            params["vote_count.gte"] = self.min_votes
        if self.runtime_min:
            params["with_runtime.gte"] = self.runtime_min
        if self.runtime_max:
            params["with_runtime.lte"] = self.runtime_max
        if self.certifications:
            params["certification_country"] = "US"
            params["certification"] = "|".join(self.certifications)
        if self.language:
            params["with_original_language"] = self.language
        if self.cast_ids:
            params["with_cast"] = ",".join(str(i) for i in self.cast_ids)
        if self.director_id:
            # TMDB can only say "this person was on the crew"; apply() then
            # keeps the movies where the person is credited as director.
            params["with_crew"] = str(self.director_id)

        include = [str(genre_ids[g]) for g in self.genres_include if g in genre_ids]
        if include:
            params["with_genres"] = ("," if self.genre_mode == "all" else "|").join(include)
        exclude = [str(genre_ids[g]) for g in self.genres_exclude if g in genre_ids]
        if exclude:
            params["without_genres"] = ",".join(exclude)
        return params

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return only the rows of ``frame`` that satisfy every filter.

        The check is strict: a movie with no recorded runtime or certification
        is dropped when that filter is active, because it cannot be shown to
        meet it.
        """
        if frame.empty:
            return frame
        keep = pd.Series(True, index=frame.index)

        if self.date_from:
            keep &= frame["release_date"] >= self.date_from
        if self.date_to:
            keep &= (frame["release_date"] <= self.date_to) & (frame["release_date"] != "")
        if self.min_rating > 0:
            keep &= frame["rating"] >= self.min_rating
        if self.min_votes > 0:
            keep &= frame["votes"] >= self.min_votes
        if self.runtime_min:
            keep &= frame["runtime"].fillna(0) >= self.runtime_min
        if self.runtime_max:
            keep &= frame["runtime"].fillna(float("inf")) <= self.runtime_max
        if self.certifications:
            keep &= frame["certification"].isin(self.certifications)
        if self.language:
            keep &= frame["language"] == self.language
        if self.director_name:
            wanted = self.director_name.lower()
            keep &= frame["directors"].apply(lambda names: wanted in [n.lower() for n in names])

        include = set(self.genres_include)
        if include:
            if self.genre_mode == "all":
                keep &= frame["genres"].apply(lambda g: include.issubset(g))
            else:
                keep &= frame["genres"].apply(lambda g: bool(include.intersection(g)))
        exclude = set(self.genres_exclude)
        if exclude:
            keep &= frame["genres"].apply(lambda g: not exclude.intersection(g))

        return frame[keep].reset_index(drop=True)

    def describe(self) -> list[str]:
        """Human-readable summary of the active filters, for the results header."""
        parts: list[str] = []
        if self.date_from or self.date_to:
            parts.append(f"Released {self.date_from or 'any'} to {self.date_to or 'today'}")
        if self.min_rating > 0:
            parts.append(f"Rating {self.min_rating:g}+")
        if self.min_votes > 0:
            parts.append(f"{self.min_votes:,}+ votes")
        if self.runtime_min and self.runtime_max:
            parts.append(f"Runtime {self.runtime_min} to {self.runtime_max} min")
        elif self.runtime_max:
            parts.append(f"Runtime up to {self.runtime_max} min")
        elif self.runtime_min:
            parts.append(f"Runtime at least {self.runtime_min} min")
        if self.certifications:
            parts.append("Certified " + " / ".join(self.certifications))
        if self.language:
            names = {code: name for name, code in LANGUAGES.items()}
            parts.append(names.get(self.language, self.language))
        if self.genres_include:
            joiner = " and " if self.genre_mode == "all" else " or "
            parts.append(joiner.join(self.genres_include))
        if self.genres_exclude:
            parts.append("Not " + ", ".join(self.genres_exclude))
        if self.director_name:
            parts.append(f"Directed by {self.director_name}")
        return parts
