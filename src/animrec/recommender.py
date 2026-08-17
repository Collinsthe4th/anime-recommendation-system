"""Content-based recommender built from synopses and genres.

Each title is described two ways:

* **Synopsis**: TF-IDF vectors (word and two-word phrases) of the cleaned
  synopsis, compared with cosine similarity.
* **Genres**: a multi-hot genre vector, compared with Jaccard similarity
  (shared genres divided by all genres of the two titles).

The final similarity is a weighted mix of the two. A small bonus for the
community score breaks ties in favour of better-rated shows.
"""

from __future__ import annotations

import difflib
import html
import re
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REQUIRED_COLUMNS = {"title", "synopsis", "genres", "episodes", "score"}


@dataclass(frozen=True)
class Recommendation:
    """One recommended title and why it was picked."""

    title: str
    similarity: float
    synopsis_similarity: float
    genre_similarity: float
    shared_genres: list[str]
    genres: list[str]
    score: float
    episodes: int | None
    synopsis: str

    def to_dict(self) -> dict:
        return asdict(self)


def clean_synopsis(text: str) -> str:
    """Remove HTML line breaks, source notes and extra spaces from a synopsis."""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"\[(Written|Source)[^\]]*\]|\((Source|Written)[^)]*\)", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def split_genres(value: str) -> list[str]:
    return [g.strip() for g in str(value).split(",") if g.strip()]


def jaccard_matrix(multi_hot: np.ndarray) -> np.ndarray:
    """Pairwise Jaccard similarity of the rows of a 0/1 matrix."""
    m = multi_hot.astype(float)
    inter = m @ m.T
    sizes = m.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, inter / union, 0.0)
    return out


class AnimeRecommender:
    """Recommend similar anime from a catalogue.

    Args:
        catalogue: frame with title, synopsis, genres, episodes and score.
        synopsis_weight: share of the similarity that comes from the synopsis
            (the rest comes from genres). 0.8 was chosen in scripts/evaluate.py:
            the highest franchise MRR whose genre coherence stays within 0.01
            of the best possible.
        score_weight: size of the community-score bonus.
    """

    def __init__(self, catalogue: pd.DataFrame, synopsis_weight: float = 0.8, score_weight: float = 0.05) -> None:
        missing = REQUIRED_COLUMNS - set(catalogue.columns)
        if missing:
            raise ValueError(f"Catalogue is missing columns: {sorted(missing)}")
        if not 0 <= synopsis_weight <= 1:
            raise ValueError("synopsis_weight must be between 0 and 1")

        df = catalogue.drop_duplicates("title").reset_index(drop=True).copy()
        df["synopsis_clean"] = df["synopsis"].map(clean_synopsis)
        df["genre_list"] = df["genres"].map(split_genres)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(df["score"].median() if len(df) else 0)
        self.df = df
        self.synopsis_weight = synopsis_weight
        self.score_weight = score_weight

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(df["synopsis_clean"])
        self.genre_names = sorted({g for gs in df["genre_list"] for g in gs})
        index = {g: i for i, g in enumerate(self.genre_names)}
        self.genre_matrix = np.zeros((len(df), len(self.genre_names)), dtype=int)
        for row, gs in enumerate(df["genre_list"]):
            for g in gs:
                self.genre_matrix[row, index[g]] = 1

        self.synopsis_sim = cosine_similarity(self.tfidf)
        self.genre_sim = jaccard_matrix(self.genre_matrix)
        s = df["score"].to_numpy(float)
        self.score_norm = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.zeros_like(s)
        self._lower_titles = {t.lower(): i for i, t in enumerate(df["title"])}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_csv(cls, path: str | Path | None = None, **kwargs) -> "AnimeRecommender":
        """Build the recommender from a CSV (defaults to the bundled catalogue)."""
        if path is None:
            with resources.files("animrec").joinpath("data/anime.csv").open("r", encoding="utf-8") as fh:
                return cls(pd.read_csv(fh), **kwargs)
        return cls(pd.read_csv(path), **kwargs)

    # ------------------------------------------------------------------ lookup
    @property
    def titles(self) -> list[str]:
        return self.df["title"].tolist()

    def find_title(self, query: str) -> str | None:
        """Return the catalogue title that best matches ``query`` (case and typo tolerant)."""
        q = query.strip().lower()
        if not q:
            return None
        if q in self._lower_titles:
            return self.df["title"].iloc[self._lower_titles[q]]
        contains = [t for t in self.titles if q in t.lower()]
        if contains:
            return min(contains, key=len)
        close = difflib.get_close_matches(q, list(self._lower_titles), n=1, cutoff=0.6)
        return self.df["title"].iloc[self._lower_titles[close[0]]] if close else None

    def search(self, query: str, limit: int = 8) -> list[str]:
        """Titles for an autocomplete box."""
        q = query.strip().lower()
        if not q:
            return []
        starts = [t for t in self.titles if t.lower().startswith(q)]
        contains = [t for t in self.titles if q in t.lower() and t not in starts]
        return (starts + contains)[:limit]

    # ------------------------------------------------------------------ scoring
    def similarity_row(self, idx: int, synopsis_weight: float | None = None) -> np.ndarray:
        w = self.synopsis_weight if synopsis_weight is None else synopsis_weight
        return w * self.synopsis_sim[idx] + (1 - w) * self.genre_sim[idx]

    def _build(self, idx: int, scores: np.ndarray, k: int, exclude: set[int], genre: str | None,
               source_genres: set[str]) -> list[Recommendation]:
        ranked = scores + self.score_weight * self.score_norm
        order = np.argsort(-ranked)
        out: list[Recommendation] = []
        for j in order:
            if j in exclude:
                continue
            row = self.df.iloc[j]
            if genre and genre not in row["genre_list"]:
                continue
            ep = row["episodes"]
            out.append(
                Recommendation(
                    title=row["title"],
                    similarity=round(float(scores[j]), 4),
                    synopsis_similarity=round(float(self.synopsis_sim[idx, j]) if idx >= 0 else 0.0, 4),
                    genre_similarity=round(float(self.genre_sim[idx, j]) if idx >= 0 else 0.0, 4),
                    shared_genres=sorted(source_genres & set(row["genre_list"])),
                    genres=row["genre_list"],
                    score=float(row["score"]),
                    episodes=None if pd.isna(ep) else int(ep),
                    synopsis=row["synopsis_clean"],
                )
            )
            if len(out) == k:
                break
        return out

    def recommend(self, title: str, k: int = 10, genre: str | None = None) -> list[Recommendation]:
        """Titles most similar to ``title``.

        Args:
            title: a catalogue title (fuzzy matched).
            k: number of results.
            genre: if given, only return titles in this genre.

        Raises:
            KeyError: if no catalogue title matches.
        """
        match = self.find_title(title)
        if match is None:
            raise KeyError(f"'{title}' is not in the catalogue")
        idx = self._lower_titles[match.lower()]
        return self._build(idx, self.similarity_row(idx), k, {idx}, genre, set(self.df["genre_list"].iloc[idx]))

    def recommend_from_text(self, text: str, k: int = 10, genres: list[str] | None = None) -> list[Recommendation]:
        """Recommend from a free-text description such as "space bounty hunters with jazz"."""
        q = self.vectorizer.transform([clean_synopsis(text)])
        syn = cosine_similarity(q, self.tfidf)[0]
        wanted = set(genres or [])
        if wanted:
            vec = np.array([[1 if g in wanted else 0 for g in self.genre_names]])
            gen = jaccard_matrix(np.vstack([vec, self.genre_matrix]))[0, 1:]
            scores = self.synopsis_weight * syn + (1 - self.synopsis_weight) * gen
        else:
            scores = syn
        return self._build(-1, scores, k, set(), None, wanted)
