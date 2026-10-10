"""Content-based movie recommender (TF-IDF on genres + cosine similarity)."""
import difflib
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import DEFAULT_PATH, clean_movies, load_movies

_YEAR = re.compile(r"\s*\(\d{4}\)\s*$")
_ARTICLE = re.compile(r"^(the|a|an)\s+(.+)$")


class MovieNotFoundError(ValueError):
    """The requested title is not in the dataset."""


class NoGenreInfoError(ValueError):
    """The movie has no genre information, so nothing can be compared."""


class MovieRecommender:
    def __init__(self, movies: pd.DataFrame, chunk_size: int = 1000):
        """`movies` must be the output of preprocessing.clean_movies()."""
        self.movies = movies.reset_index(drop=True)

        # Convert the genre strings into TF-IDF vectors.
        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.tfidf = self.vectorizer.fit_transform(self.movies["genre_text"])

        # Pairwise cosine similarity matrix (float32, computed in chunks to save memory).
        self.similarity = self._pairwise_similarity(chunk_size)

        # Lookup helpers for matching user-typed titles to rows.
        self._title_lower = self.movies["title"].str.lower()
        self._title_no_year = self._title_lower.str.replace(_YEAR, "", regex=True)

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_csv(cls, path=DEFAULT_PATH):
        return cls(clean_movies(load_movies(path)))

    def _pairwise_similarity(self, chunk_size: int) -> np.ndarray:
        n = self.tfidf.shape[0]
        sim = np.empty((n, n), dtype=np.float32)
        for start in range(0, n, chunk_size):
            stop = start + chunk_size
            sim[start:stop] = cosine_similarity(self.tfidf[start:stop], self.tfidf)
        return sim

    # --------------------------------------------------------- title matching
    def find_movie(self, title: str):
        """Return the row index for a title, or None.

        Tries: exact title, title without year, MovieLens-style articles
        ("The Matrix" -> "Matrix, The"), then a partial match.
        """
        q = title.strip().lower()
        if not q:
            return None
        variants = [q]
        m = _ARTICLE.match(q)
        if m:
            variants.append(f"{m.group(2)}, {m.group(1)}")

        for series in (self._title_lower, self._title_no_year):
            for v in variants:
                hits = self.movies.index[series == v]
                if len(hits):
                    return int(hits[0])
        for v in variants:
            hits = self.movies.index[self._title_lower.str.contains(v, regex=False)]
            if len(hits):
                return int(hits[0])
        return None

    def suggest_titles(self, query: str, n: int = 5) -> list:
        """Closest titles to a (possibly misspelled) query."""
        names = self._title_no_year.tolist()
        close = difflib.get_close_matches(query.strip().lower(), names, n=n, cutoff=0.6)
        lookup = dict(zip(reversed(names), reversed(self.movies["title"].tolist())))
        return [lookup[c] for c in close]

    def genres(self) -> list:
        return sorted({g for lst in self.movies["genre_list"] for g in lst})

    # --------------------------------------------------------- recommendations
    def recommend_movies(self, title: str, n: int = 5, genre: str = None) -> pd.DataFrame:
        """Return the top-n most similar movies as a DataFrame.

        Columns: title, genres, similarity, shared_genres.
        The selected movie is never included. `genre` optionally restricts the results.
        Raises MovieNotFoundError / NoGenreInfoError / ValueError with clear messages.
        """
        idx = self.find_movie(title)
        if idx is None:
            hint = self.suggest_titles(title)
            msg = f"Movie '{title}' was not found."
            if hint:
                msg += " Did you mean: " + "; ".join(hint) + "?"
            raise MovieNotFoundError(msg)

        selected = self.movies.loc[idx]
        if not selected["genre_list"]:
            raise NoGenreInfoError(
                f"'{selected['title']}' has no genre information, so it can't be compared."
            )

        sims = self.similarity[idx].astype(float).copy()
        sims[idx] = -1.0  # exclude the selected movie itself

        if genre is not None:
            wanted = {g.lower(): g for g in self.genres()}.get(genre.strip().lower())
            if wanted is None:
                raise ValueError(f"Unknown genre '{genre}'. Choose from: {', '.join(self.genres())}")
            has_genre = self.movies["genre_list"].apply(lambda g: wanted in g).to_numpy()
            sims[~has_genre] = -1.0

        order = np.argsort(-sims, kind="stable")  # stable => deterministic ties
        order = [i for i in order if sims[i] > 0][:n]

        result = self.movies.loc[order, ["title", "genres"]].copy()
        result["similarity"] = sims[order].round(3)
        result["shared_genres"] = [
            ", ".join(g for g in selected["genre_list"] if g in self.movies.at[i, "genre_list"])
            for i in order
        ]
        return result.reset_index(drop=True)


def print_recommendations(rec: MovieRecommender, title: str, n: int = 5, genre: str = None):
    """Friendly console output; catches the 'expected' errors."""
    try:
        out = rec.recommend_movies(title, n=n, genre=genre)
    except ValueError as err:
        print(err)
        return
    shown = rec.movies.loc[rec.find_movie(title), "title"]
    suffix = f" (genre filter: {genre})" if genre else ""
    print(f"Because you picked '{shown}'{suffix}:")
    if out.empty:
        print("  No similar movies found.")
    for i, row in out.iterrows():
        print(f"{i + 1}. {row['title']}  [{row['genres']}]")
        print(f"   similarity {row['similarity']:.2f} | shared genres: {row['shared_genres']}")


def main():
    """Simple command-line menu:  python -m src.recommender"""
    rec = MovieRecommender.from_csv()
    menu = "\n1) Recommend  2) Recommend with genre filter  3) List genres  4) Quit"
    while True:
        print(menu)
        choice = input("Choose: ").strip()
        if choice == "1":
            print_recommendations(rec, input("Movie title: "))
        elif choice == "2":
            print_recommendations(rec, input("Movie title: "), genre=input("Genre: "))
        elif choice == "3":
            print(", ".join(rec.genres()))
        elif choice == "4":
            break
        else:
            print("Please enter 1-4.")


if __name__ == "__main__":
    main()
