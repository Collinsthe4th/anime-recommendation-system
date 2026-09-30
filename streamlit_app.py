"""AnimRec in Streamlit: the live demo of the shared anime recommender.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from animrec import AnimeRecommender  # noqa: E402

st.set_page_config(page_title="Anime Recommendation System", layout="wide")

HEADER = "https://images.unsplash.com/photo-1613376023733-0a73315d9b06?auto=format&fit=crop&w=1600&h=340&q=80"
st.image(HEADER, use_container_width=True)


@st.cache_resource
def get_recommender() -> AnimeRecommender:
    return AnimeRecommender.from_csv()


rec = get_recommender()

st.title("Anime Recommendation System")
st.caption(f"I built this recommender from {len(rec.titles)} anime titles. It compares synopses with TF-IDF (80% of the score) and shared genres (20%), and it shows why each title was picked.")

mode = st.radio("Search by", ["A title I liked", "A description"], horizontal=True)
c1, c2, c3 = st.columns([3, 1.4, 1])
with c1:
    if mode == "A title I liked":
        title = st.selectbox("Anime title", rec.titles, index=rec.titles.index("Cowboy Bebop") if "Cowboy Bebop" in rec.titles else 0)
        text = ""
    else:
        text = st.text_input("Describe what you want to watch", "high school students playing basketball")
        title = ""
with c2:
    genre = st.selectbox("Only this genre", ["Any genre", *rec.genre_names])
with c3:
    k = st.slider("Results", 3, 20, 8)

genre_filter = None if genre == "Any genre" else genre
if mode == "A title I liked":
    results = rec.recommend(title, k=k, genre=genre_filter)
    st.subheader(f"Because you liked {title}")
else:
    results = rec.recommend_from_text(text, k=k, genres=[genre_filter] if genre_filter else None) if text.strip() else []
    st.subheader(f"Matches for “{text}”" if text.strip() else "Type a description above")

if not results and (title or text.strip()):
    st.info("Nothing matches that combination. Try 'Any genre'.")

cols = st.columns(2)
for i, r in enumerate(results):
    with cols[i % 2].container(border=True):
        top = st.columns([4, 1])
        top[0].markdown(f"**{i + 1}. {r.title}**")
        top[1].markdown(f"★ {r.score:.0f}")
        st.progress(min(max(r.similarity, 0.0), 1.0), text=f"Similarity {r.similarity:.2f}")
        why = []
        if r.shared_genres:
            why.append("Shares " + ", ".join(r.shared_genres))
        if mode == "A title I liked":
            why.append(f"story match {r.synopsis_similarity:.2f}, genre match {r.genre_similarity:.2f}")
        st.caption(" · ".join(why))
        st.write(r.synopsis[:320] + ("…" if len(r.synopsis) > 320 else ""))
        st.caption(" · ".join(r.genres) + (f" · {r.episodes} episodes" if r.episodes else ""))

with st.expander("How the model was evaluated"):
    st.write(
        "There are no user ratings, so I checked whether the model finds sequels and spin-offs it was never told about. "
        "With the chosen 80/20 blend, a title from the same franchise appears in the top 10 for 96.9% of the 64 titles "
        "that have one, while the genres of the top 10 stay almost as close as a genre-only model."
    )
    st.image(str(ROOT / "results" / "figures" / "weight_sweep.png"))

st.markdown(
    "[Model source](https://github.com/Collinsthe4th/anime-recommendation-system) · "
    "[Django version of this app](https://github.com/Collinsthe4th/animrec-django) · Header photo from Unsplash."
)
