"""Loading, inspecting and cleaning the MovieLens movies table."""
from pathlib import Path

import pandas as pd

NO_GENRES = "(no genres listed)"
DEFAULT_PATH = Path("data/ml-latest-small/movies.csv")


def load_movies(path=DEFAULT_PATH) -> pd.DataFrame:
    """Read movies.csv (columns: movieId, title, genres)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the MovieLens Latest Small dataset "
            "from https://grouplens.org/datasets/movielens/latest/ and extract it "
            "into the data/ folder (see data/README.md)."
        )
    return pd.read_csv(path)


def data_quality_report(movies: pd.DataFrame) -> dict:
    """Quick checks: size, missing values, duplicates, movies without genres."""
    return {
        "rows": len(movies),
        "missing_values": movies.isnull().sum().to_dict(),
        "duplicate_movie_ids": int(movies["movieId"].duplicated().sum()),
        "duplicate_titles": int(movies["title"].duplicated().sum()),
        "movies_without_genres": int((movies["genres"] == NO_GENRES).sum()),
    }


def clean_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with helper columns used by the recommender.

    Added columns:
      genre_list : list of genres (empty list when none are listed)
      genre_text : genres separated by spaces (the text fed to TF-IDF)
      year       : release year extracted from the title (may be missing)
    """
    df = movies.drop_duplicates(subset="movieId").dropna(subset=["title"]).copy()
    df["title"] = df["title"].str.strip()
    df["genres"] = df["genres"].fillna(NO_GENRES)

    df["genre_list"] = df["genres"].apply(
        lambda g: [] if g == NO_GENRES else g.split("|")
    )
    # Step 1 of the roadmap: replace the pipe characters with spaces.
    df["genre_text"] = df["genre_list"].apply(" ".join)
    df["year"] = df["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    return df.reset_index(drop=True)
