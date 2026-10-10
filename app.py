"""Optional Streamlit interface.  Run from the project root:  streamlit run app.py"""
import streamlit as st

from src.preprocessing import DEFAULT_PATH
from src.recommender import MovieRecommender

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommendation System")
st.caption("Content-based filtering: TF-IDF on genres + cosine similarity (MovieLens)")


@st.cache_resource
def get_recommender():
    return MovieRecommender.from_csv()


if not DEFAULT_PATH.exists():
    st.error(f"Dataset not found at {DEFAULT_PATH}. See data/README.md for download steps.")
    st.stop()

rec = get_recommender()
titles = sorted(rec.movies.loc[rec.movies["genre_list"].str.len() > 0, "title"])

title = st.selectbox("Pick a movie you like", titles, index=None, placeholder="Type to search...")
col1, col2 = st.columns(2)
n = col1.slider("Number of recommendations", 1, 20, 5)
genre = col2.selectbox("Only show genre", ["Any"] + rec.genres())

if title:
    try:
        result = rec.recommend_movies(title, n=n, genre=None if genre == "Any" else genre)
    except ValueError as err:
        st.warning(str(err))
    else:
        if result.empty:
            st.info("No similar movies found with these settings.")
        else:
            result.index = result.index + 1
            st.dataframe(result, use_container_width=True)
