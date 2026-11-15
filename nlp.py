"""Natural language processing: text features, sentiment and query parsing.

Three techniques are implemented here.

1. Text vectorisation. Plot summaries, keywords and genres are tokenised,
   stripped of stop words, stemmed with the Porter stemmer and weighted with
   TF-IDF (scikit-learn). Stemming lets "heists" match "heist"; TF-IDF makes
   distinctive words such as "heist" count for more than common ones such as
   "man" or "life".
2. Sentiment analysis. NLTK's VADER model scores each plot summary from -1
   (dark) to +1 (uplifting). The app calls this the movie's tone and matches
   it against the mood the user asks for.
3. Query understanding. A rule-based parser turns a sentence such as
   "90s thrillers under two hours with Denzel Washington" into structured
   filters. Rules were chosen over a statistical model because the vocabulary
   of movie requests is small and every decision can be shown to the user.

Nothing in this module touches the network or Streamlit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# --------------------------------------------------------------------------
# 1. Tokenising and TF-IDF
# --------------------------------------------------------------------------

_stemmer = PorterStemmer()
_WORD = re.compile(r"[a-z]{3,}")
# Stem -> shortest original word seen, so explanations can show "robbery"
# to the user instead of the stem "robberi".
_display_word: dict[str, str] = {}


@lru_cache(maxsize=50_000)
def _stem(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    """Lower-case, drop stop words and stem. Used as the TF-IDF tokeniser."""
    tokens = []
    for word in _WORD.findall(text.lower()):
        if word in ENGLISH_STOP_WORDS:
            continue
        stem = _stem(word)
        known = _display_word.get(stem)
        if known is None or len(word) < len(known):
            _display_word[stem] = word
        tokens.append(stem)
    return tokens


def display_term(feature: str) -> str:
    """Turn a stemmed TF-IDF feature back into readable words."""
    return " ".join(_display_word.get(stem, stem) for stem in feature.split())


def make_vectorizer() -> TfidfVectorizer:
    """TF-IDF over single words and two-word phrases.

    Sub-linear term frequency stops a word repeated many times in one summary
    from dominating, and bigrams capture phrases such as "serial killer".
    """
    return TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=20_000,
    )


# --------------------------------------------------------------------------
# 2. Sentiment (tone)
# --------------------------------------------------------------------------

# Mood the user can ask for -> the tone score that mood aims at.
MOODS = {
    "Uplifting": 0.75,
    "Light-hearted": 0.30,
    "Tense": -0.30,
    "Dark": -0.75,
}


@lru_cache(maxsize=1)
def _analyzer():
    """Load VADER once, downloading its lexicon on first use if necessary."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except Exception:
        return None  # tone falls back to neutral rather than crashing the app


@lru_cache(maxsize=20_000)
def tone_score(text: str) -> float:
    """VADER compound sentiment of a plot summary, from -1 to +1."""
    analyzer = _analyzer()
    if not text or analyzer is None:
        return 0.0
    return float(analyzer.polarity_scores(text)["compound"])


def tone_label(score: float) -> str:
    """Name the tone band a score falls in."""
    if score >= 0.5:
        return "Uplifting"
    if score >= 0.05:
        return "Light-hearted"
    if score > -0.05:
        return "Neutral"
    if score > -0.5:
        return "Tense"
    return "Dark"


def mood_match(tone: float, target: float) -> float:
    """How close a movie's tone is to the requested mood, from 0 to 1."""
    return max(0.0, 1.0 - abs(tone - target) / 1.5)


# --------------------------------------------------------------------------
# 3. Natural language query parsing
# --------------------------------------------------------------------------

# Phrase a user might type -> TMDB genre names it stands for.
GENRE_SYNONYMS: dict[str, list[str]] = {
    "science fiction": ["Science Fiction"], "sci-fi": ["Science Fiction"],
    "sci fi": ["Science Fiction"], "scifi": ["Science Fiction"],
    "romantic comedy": ["Romance", "Comedy"], "romantic comedies": ["Romance", "Comedy"],
    "rom-com": ["Romance", "Comedy"], "rom-coms": ["Romance", "Comedy"],
    "romcom": ["Romance", "Comedy"], "romcoms": ["Romance", "Comedy"],
    "action": ["Action"], "adventure": ["Adventure"], "adventures": ["Adventure"],
    "animation": ["Animation"], "animated": ["Animation"], "cartoon": ["Animation"],
    "cartoons": ["Animation"], "anime": ["Animation"],
    "comedy": ["Comedy"], "comedies": ["Comedy"], "funny": ["Comedy"], "hilarious": ["Comedy"],
    "crime": ["Crime"], "gangster": ["Crime"], "mob": ["Crime"],
    "documentary": ["Documentary"], "documentaries": ["Documentary"],
    "drama": ["Drama"], "dramas": ["Drama"],
    "family": ["Family"], "fantasy": ["Fantasy"],
    "history": ["History"], "historical": ["History"],
    "horror": ["Horror"], "scary": ["Horror"], "slasher": ["Horror"],
    "musical": ["Music"], "musicals": ["Music"], "music": ["Music"],
    "mystery": ["Mystery"], "mysteries": ["Mystery"], "whodunit": ["Mystery"],
    "romance": ["Romance"], "romantic": ["Romance"], "love story": ["Romance"],
    "thriller": ["Thriller"], "thrillers": ["Thriller"], "suspense": ["Thriller"],
    "war": ["War"], "western": ["Western"], "westerns": ["Western"],
}

MOOD_SYNONYMS: dict[str, str] = {
    "feel-good": "Uplifting", "feel good": "Uplifting", "feelgood": "Uplifting",
    "uplifting": "Uplifting", "heartwarming": "Uplifting", "heart-warming": "Uplifting",
    "wholesome": "Uplifting", "happy": "Uplifting", "cheerful": "Uplifting",
    "inspiring": "Uplifting", "inspirational": "Uplifting", "optimistic": "Uplifting",
    "lighthearted": "Light-hearted", "light-hearted": "Light-hearted", "light": "Light-hearted",
    "fun": "Light-hearted", "easy": "Light-hearted", "cozy": "Light-hearted",
    "cosy": "Light-hearted", "breezy": "Light-hearted", "chill": "Light-hearted",
    "relaxing": "Light-hearted",
    "tense": "Tense", "suspenseful": "Tense", "intense": "Tense", "gripping": "Tense",
    "nail-biting": "Tense", "thrilling": "Tense",
    "dark": "Dark", "gritty": "Dark", "disturbing": "Dark", "bleak": "Dark",
    "grim": "Dark", "sad": "Dark", "depressing": "Dark", "tearjerker": "Dark",
    "melancholy": "Dark", "heavy": "Dark", "brutal": "Dark", "violent": "Dark",
}

LANGUAGE_WORDS: dict[str, str] = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "japanese": "ja", "korean": "ko",
    "chinese": "zh", "mandarin": "zh", "hindi": "hi", "bollywood": "hi",
    "nepali": "ne", "russian": "ru", "swedish": "sv", "danish": "da",
    "turkish": "tr", "persian": "fa", "iranian": "fa",
}

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "five": 5, "ten": 10, "few": 3}

# Words that end a person's name when scanning after "with" or "directed by".
_NAME_STOP = {
    "from", "in", "under", "over", "about", "rated", "that", "which", "but",
    "before", "after", "since", "between", "less", "more", "at", "no", "not",
    "without", "longer", "shorter", "released", "set", "movie", "movies",
    "film", "films", "who", "is", "are", "for", "to", "on", "as", "like",
    "tonight", "please", "i", "we", "it",
}
# First words that show the phrase is not a person ("with a twist").
_NOT_A_NAME = {
    "a", "an", "the", "some", "lots", "lot", "plenty", "great", "good",
    "strong", "any", "my", "our", "me", "us", "friends", "family", "kids",
    "subtitles", "high", "low", "big", "little",
}
# Words ignored when deciding what the free-text "theme" of a query is.
_FILLER = {
    "movie", "movies", "film", "films", "flick", "flicks", "watch", "watching",
    "want", "wanna", "looking", "look", "find", "recommend", "recommendation",
    "recommendations", "suggest", "suggestion", "suggestions", "please", "give",
    "show", "need", "something", "anything", "tonight", "today", "night",
    "weekend", "feel", "feeling", "mood", "kind", "sort", "type", "like",
    "similar", "really", "maybe", "ones", "good", "best", "great", "made",
    "starring", "featuring", "directed", "director", "actor", "actress",
    "rated", "rating", "released", "release", "based", "story", "plot",
    "year", "years", "minute", "minutes", "hour", "hours", "language",
    "genre", "im", "id", "lets", "think", "pick", "picks", "stuff", "thing",
    "things", "bit", "little", "lot", "pretty", "quite", "time", "perhaps",
    "cool", "nice", "just", "also", "kids", "ve", "ll", "don",
}


def _alternation(phrases) -> str:
    """Regex alternation of phrases, longest first so "sci-fi" beats "sci"."""
    return "(?:" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + ")"


_GENRE = _alternation(GENRE_SYNONYMS)
_MOOD = _alternation(MOOD_SYNONYMS)
_LANGUAGE = _alternation(LANGUAGE_WORDS)
_YEAR = r"((?:19|20)\d{2})"
_DURATION = (
    r"(?P<n>\d+(?:\.\d+)?|one|two|three|an?)\s*(?P<h1>and a half\s+)?"
    r"(?P<unit>hours?|hrs?|h|minutes?|mins?|m)\b(?P<h2>\s+and a half)?"
)


@dataclass
class ParsedQuery:
    """Structured meaning extracted from a free-text request."""

    raw: str = ""
    year_min: int | None = None
    year_max: int | None = None
    runtime_min: int | None = None
    runtime_max: int | None = None
    min_rating: float | None = None
    min_votes: int | None = None
    certifications: list[str] = field(default_factory=list)
    language: str | None = None
    genres_include: list[str] = field(default_factory=list)
    genres_exclude: list[str] = field(default_factory=list)
    genre_mode: str = "all"
    mood: str | None = None
    actors: list[str] = field(default_factory=list)
    director: str | None = None
    similar_to: str | None = None
    free_text: str = ""

    def notes(self) -> list[str]:
        """Plain-English list of what was understood, shown back to the user."""
        out: list[str] = []
        if self.year_min and self.year_max:
            out.append(str(self.year_min) if self.year_min == self.year_max
                       else f"{self.year_min} to {self.year_max}")
        elif self.year_min:
            out.append(f"{self.year_min} or later")
        elif self.year_max:
            out.append(f"{self.year_max} or earlier")
        if self.genres_include:
            out.append((" and " if self.genre_mode == "all" else " or ").join(self.genres_include))
        if self.genres_exclude:
            out.append("Not " + ", ".join(self.genres_exclude))
        if self.runtime_min and self.runtime_max:
            out.append(f"{self.runtime_min} to {self.runtime_max} min")
        elif self.runtime_max:
            out.append(f"Up to {self.runtime_max} min")
        elif self.runtime_min:
            out.append(f"At least {self.runtime_min} min")
        if self.min_rating:
            out.append(f"Rating {self.min_rating:g}+")
        if self.min_votes:
            out.append(f"{self.min_votes:,}+ votes")
        if self.certifications:
            out.append("Certified " + " / ".join(self.certifications))
        if self.language:
            names = {code: word.title() for word, code in LANGUAGE_WORDS.items()}
            out.append(f"Language: {names.get(self.language, self.language)}")
        if self.mood:
            out.append(f"Mood: {self.mood}")
        out.extend(f"Actor: {name.title()}" for name in self.actors)
        if self.director:
            out.append(f"Director: {self.director.title()}")
        if self.similar_to:
            out.append(f"Similar to: {self.similar_to.title()}")
        if self.free_text:
            out.append(f"Themes: {self.free_text}")
        return out


def _minutes(match: re.Match) -> int:
    """Convert a matched duration such as "two and a half hours" to minutes."""
    number = match.group("n")
    value = float(NUMBER_WORDS[number]) if number in NUMBER_WORDS else float(number)
    if match.group("unit").startswith("h"):
        if match.group("h1") or match.group("h2"):
            value += 0.5
        value *= 60
    return int(round(value))


def parse_query(text: str, today: date | None = None) -> ParsedQuery:
    """Extract filters, mood, people and themes from a free-text request.

    The parser works by elimination: each rule looks for one kind of phrase,
    records what it means and blanks it out of the working string. Whatever
    survives every rule is treated as the theme of the request ("heist",
    "time travel") and is later matched against plot text with TF-IDF.
    """
    today = today or date.today()
    parsed = ParsedQuery(raw=text.strip())
    work = " " + text.lower().replace("’", "'").strip() + " "

    def take(pattern: str) -> re.Match | None:
        """Find a pattern, remove it from the working string, return the match."""
        nonlocal work
        match = re.search(pattern, work)
        if match:
            work = work[: match.start()] + " " + work[match.end():]
        return match

    def take_people(trigger: str) -> list[str]:
        """Read one or more names following a trigger such as "starring"."""
        nonlocal work
        boundary = _NAME_STOP | set(MOOD_SYNONYMS) | set(LANGUAGE_WORDS) | {
            w for w in GENRE_SYNONYMS if " " not in w
        }
        for match in re.finditer(rf"\b(?:{trigger})\s+", work):
            rest = work[match.end():]
            names: list[str] = []
            current: list[str] = []
            consumed = 0
            for token in re.finditer(r"\S+", rest):
                word = token.group().strip(",.;!?")
                if word in ("and", "&", "or"):
                    if not current:
                        break
                    names.append(" ".join(current))
                    current = []
                    consumed = token.end()
                    continue
                if not word or word in boundary or any(c.isdigit() for c in word) or len(current) >= 4:
                    break
                current.append(word)
                consumed = token.end()
                if token.group().endswith(","):
                    names.append(" ".join(current))
                    current = []
            if current:
                names.append(" ".join(current))
            names = [n for n in names if n.split()[0] not in _NOT_A_NAME]
            if names:
                work = work[: match.start()] + " " + rest[consumed:]
                return names
        return []

    # --- "like Inception": a seed movie for content-based ranking ----------
    seed = take(
        r"\b(?:similar to|in the style of|(?<!would )(?<!i'd )(?<!feel )(?<! i )(?<! we )like)\s+"
        r"(?!a\b|an\b|to\b|watching\b|something\b|some\b|that\b|it\b|this\b)"
        r"(.+?)(?=\s+(?:but|with|from|in the|under|over|rated|starring|directed|before|after|since)\b"
        r"|[,.;!?]|\s*$)"
    )
    if seed:
        parsed.similar_to = seed.group(1).strip()

    # --- People -------------------------------------------------------------
    directors = take_people("directed by|from director|by director|director")
    if directors:
        parsed.director = directors[0]
    parsed.actors = take_people("starring|featuring|stars|with")

    # --- Certification -------------------------------------------------------
    if take(r"\b(?:family[- ]friendly|kid[- ]friendly|child[- ]friendly|for (?:the |my |our )?(?:kids|children))\b"):
        parsed.certifications += ["G", "PG"]
    for pattern, label in [
        (r"\bpg[- ]?13\b", "PG-13"),
        (r"\bnc[- ]?17\b", "NC-17"),
        (r"\b(?:r[- ]rated|rated[- ]r)\b", "R"),
        (r"\b(?:rated[- ]pg|pg[- ]rated|pg)\b", "PG"),
        (r"\b(?:g[- ]rated|rated[- ]g)\b", "G"),
    ]:
        if take(pattern) and label not in parsed.certifications:
            parsed.certifications.append(label)

    # --- Years and decades ---------------------------------------------------
    span = take(rf"\b(?:between|from)\s+{_YEAR}\s*(?:and|to|-|–)\s*{_YEAR}\b") or take(
        rf"\b{_YEAR}\s*(?:-|–|to)\s*{_YEAR}\b"
    )
    if span:
        first, second = int(span.group(1)), int(span.group(2))
        parsed.year_min, parsed.year_max = min(first, second), max(first, second)

    decade_starts: list[int] = []
    decade_ends: list[int] = []
    while True:
        decade = take(r"(?:\b(early|mid|late)[- ]+)?(?<![\d.])((?:19|20)\d0|\d0)'?s\b")
        if not decade:
            break
        digits = int(decade.group(2))
        if digits < 100:  # two-digit decade: "90s" is 1990s, "10s" is 2010s
            digits += 2000 if digits <= 20 else 1900
        offsets = {"early": (0, 3), "mid": (4, 6), "late": (7, 9)}.get(decade.group(1), (0, 9))
        decade_starts.append(digits + offsets[0])
        decade_ends.append(digits + offsets[1])
    if decade_starts and parsed.year_min is None:
        parsed.year_min, parsed.year_max = min(decade_starts), max(decade_ends)

    recent = take(r"\b(?:last|past)\s+(\d+|two|three|five|ten|few)?\s*(years?|decade)\b")
    if recent:
        amount = recent.group(1)
        count = 10 if recent.group(2) == "decade" else (
            NUMBER_WORDS.get(amount) or int(amount) if amount else 1
        )
        parsed.year_min = today.year - int(count)
    after = take(rf"\b(after|since|from|post)[\s-]+{_YEAR}\b")
    if after:
        parsed.year_min = int(after.group(2)) + (1 if after.group(1) == "after" else 0)
    before = take(rf"\b(?:before|prior to|older than|pre)[\s-]+{_YEAR}\b")
    if before:
        parsed.year_max = int(before.group(1)) - 1
    single = take(rf"\b(?:in\s+|from\s+)?{_YEAR}\b")
    if single and parsed.year_min is None and parsed.year_max is None:
        parsed.year_min = parsed.year_max = int(single.group(1))
    if take(r"\b(?:recent|new|latest|newest)\b(?!\s+york)") and parsed.year_min is None:
        parsed.year_min = today.year - 2
    if take(r"\b(?:classics?|old|vintage|oldies|golden age)\b") and parsed.year_max is None:
        parsed.year_max = 1979

    # --- Runtime -------------------------------------------------------------
    around = take(r"\b(?:about|around|roughly|approximately)\s+" + _DURATION)
    if around:
        parsed.runtime_min, parsed.runtime_max = _minutes(around) - 15, _minutes(around) + 15
    longest = take(
        r"\b(?:under|less than|shorter than|below|at most|no longer than|no more than|within|up to"
        r"|max(?:imum)?(?: of)?)\s+" + _DURATION
    ) or take(_DURATION + r"\s+(?:or less|or shorter|max|tops)\b")
    if longest:
        parsed.runtime_max = _minutes(longest)
    shortest = take(
        r"\b(?:over|more than|longer than|at least|above|min(?:imum)?(?: of)?)\s+" + _DURATION
    )
    if shortest:
        parsed.runtime_min = _minutes(shortest)
    if take(r"\bnot (?:too |very |so )?long\b") and parsed.runtime_max is None:
        parsed.runtime_max = 120
    if take(r"\b(?:short|quick)\b") and parsed.runtime_max is None:
        parsed.runtime_max = 100
    if take(r"\b(?:long|lengthy|epic)\b") and parsed.runtime_min is None:
        parsed.runtime_min = 150

    # --- Quality -------------------------------------------------------------
    score = (
        take(r"\b(?:rated|ratings?|score|scored)\s*(?:of\s+)?(?:above|over|at least|higher than|more than)?"
             r"\s*(\d(?:\.\d)?)\b(?:\s*\+|\s*(?:or|and) (?:higher|above|more|up|better))?")
        or take(r"(?<![\d.])(\d(?:\.\d)?)\s*\+(?:\s*(?:ratings?|stars?))?")
        or take(r"\b(?:above|over|at least|minimum)\s+(?:a\s+)?(\d(?:\.\d)?)\b"
                r"(?:\s*(?:stars?|/\s*10|out of 10|ratings?))?")
    )
    if score:
        parsed.min_rating = float(score.group(1))
    if take(r"\b(?:highly[- ]rated|top[- ]rated|critically acclaimed|acclaimed|well[- ]reviewed"
            r"|masterpieces?|award[- ]winning)\b") and parsed.min_rating is None:
        parsed.min_rating = 7.5
    if take(r"\b(?:hidden gems?|underrated|overlooked)\b"):
        parsed.min_rating = parsed.min_rating or 7.0
        parsed.min_votes = 100
    if take(r"\b(?:popular|well[- ]known|blockbusters?|famous|mainstream)\b"):
        parsed.min_votes = 5000

    # --- Language ------------------------------------------------------------
    language = take(rf"\b(?:in\s+)?({_LANGUAGE})\b(?:[- ]language)?")
    if language:
        parsed.language = LANGUAGE_WORDS[language.group(1)]

    # --- Mood (before genres so "not too dark" is read as a mood) -----------
    if take(rf"\b(?:not|nothing|no)\s+(?:too\s+|so\s+|very\s+)?(?:dark|sad|heavy|depressing|grim|bleak|violent|disturbing)\b"):
        parsed.mood = "Light-hearted"
    while True:
        mood = take(rf"(?<![\w-])({_MOOD})(?![\w-])")
        if not mood:
            break
        parsed.mood = parsed.mood or MOOD_SYNONYMS[mood.group(1)]

    # --- Genres with AND / OR / NOT -----------------------------------------
    while True:
        excluded = take(
            rf"\b(?:no|not|without|except|excluding|non|nothing|avoid|skip)[\s-]+"
            rf"(?:a |an |any |too |really |very )*({_GENRE}(?:\s*(?:,|or|and|nor)\s*{_GENRE})*)\b"
        )
        if not excluded:
            break
        for phrase in re.findall(_GENRE, excluded.group(1)):
            parsed.genres_exclude += [g for g in GENRE_SYNONYMS[phrase] if g not in parsed.genres_exclude]
    if re.search(rf"\b{_GENRE}\s*,?\s+or\s+{_GENRE}\b", work):
        parsed.genre_mode = "any"
    while True:
        included = take(rf"(?<![\w-])({_GENRE})(?![\w-])")
        if not included:
            break
        for genre in GENRE_SYNONYMS[included.group(1)]:
            if genre not in parsed.genres_include and genre not in parsed.genres_exclude:
                parsed.genres_include.append(genre)

    # --- Whatever is left is the theme --------------------------------------
    leftovers = [
        word for word in re.findall(r"[a-z]{2,}", work)
        if word not in ENGLISH_STOP_WORDS and word not in _FILLER
    ]
    parsed.free_text = " ".join(dict.fromkeys(leftovers))
    return parsed
