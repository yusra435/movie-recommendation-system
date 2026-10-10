# Movie Recommendation System

A small recommender system I built in Python with pandas and scikit-learn. You give it a movie you like, and it suggests similar ones and shows which genres they share.

You can try it in Google Colab: [open the notebook](https://colab.research.google.com/github/yusra435/movie-recommendation-system/blob/main/notebooks/movie_recommendation_colab.ipynb)

## How it works

Each movie has genre labels, like "Adventure|Animation|Children". I turn those into numbers with TF-IDF, then use cosine similarity to score how close every pair of movies is. For a chosen movie, the highest-scoring others are the recommendations.

The notebook also has a second approach using user ratings (collaborative filtering), so the two can be compared.

## Results

Recommendations for Toy Story using the MovieLens Small dataset:

| # | From ratings | From genres only |
|---|--------------|------------------|
| 1 | Toy Story 2 | Antz |
| 2 | Aladdin | Toy Story 2 |
| 3 | Toy Story 3 | Rocky and Bullwinkle |
| 4 | Wallace & Gromit: The Wrong Trousers | Emperor's New Groove |
| 5 | The Incredibles | Monsters, Inc. |

The genre-based list gives every movie a score of 1.0, because they all have exactly the same genres as Toy Story, so the order doesn't mean much. The ratings-based list is more meaningful.

## Running it

In Colab, upload the notebook in `notebooks/` and run all cells. It downloads the dataset itself.

On your own computer:

```
pip install -r requirements.txt
```

Download MovieLens Latest Small from https://grouplens.org/datasets/movielens/latest/ and extract it so that `data/ml-latest-small/movies.csv` exists. Then run:

```
python -m src.recommender
streamlit run app.py
```

The first gives a simple menu in the terminal, and the second opens a web app.

## Files

- `notebooks/` - the Colab notebook
- `src/preprocessing.py` - loads and cleans the data
- `src/recommender.py` - the recommendation code
- `app.py` - Streamlit web app

## Limitations

- **Only genres are used.** The genre-based model has no information about plot, cast, director or tags, so it can't tell how two movies actually differ.
- **Many movies tie.** A lot of movies have exactly the same genre labels, so they get the same similarity score and their order is arbitrary, as the Toy Story results show.
- **No user history.** The genre-based model gives the same recommendations to everyone who picks the same movie.
- **Ratings need enough data.** The collaborative approach only covers movies with enough ratings, so it can't recommend new or rarely rated movies.
- **Small dataset.** MovieLens Small has about 100,000 ratings from 610 users, so the results may not hold for a larger or different audience.
- **No formal evaluation yet.** I checked the results by eye but haven't measured accuracy with a metric like Precision@K.

## Next steps

- Add plot descriptions or tags to improve the genre-based model
- Combine both methods into a hybrid
- Evaluate the models with Precision@K

## Data

MovieLens, from GroupLens (Harper and Konstan, 2015). The dataset isn't included in this repository.
