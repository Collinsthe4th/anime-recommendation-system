import numpy as np
import pandas as pd
import pytest

from animrec import AnimeRecommender
from animrec.cli import main as cli_main
from animrec.recommender import clean_synopsis, jaccard_matrix


@pytest.fixture(scope="module")
def tiny():
    df = pd.DataFrame(
        {
            "title": ["Space Hunters", "Space Hunters II", "Kitchen Days", "Robot War"],
            "synopsis": [
                "Bounty hunters chase criminals across space on their ship.<br>",
                "The bounty hunters return to chase a new criminal across space.",
                "A young chef cooks with friends in a small kitchen.",
                "Giant robots fight a war to protect the colony in space.",
            ],
            "genres": ["Action, Sci-Fi", "Action, Sci-Fi", "Comedy, Slice of Life", "Action, Mecha, Sci-Fi"],
            "episodes": [26, 12, None, 50],
            "score": [85, 78, 70, 72],
        }
    )
    return AnimeRecommender(df)


def test_clean_synopsis_strips_html_and_source():
    assert clean_synopsis("Hello<br><br>world [Written by MAL Rewrite]") == "Hello world"


def test_jaccard_matrix():
    m = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]])
    j = jaccard_matrix(m)
    assert j[0, 1] == pytest.approx(0.5)
    assert j[0, 2] == 0
    assert np.allclose(np.diag(j), 1)


def test_sequel_is_top_recommendation(tiny):
    recs = tiny.recommend("space hunters", k=3)
    assert recs[0].title == "Space Hunters II"
    assert "Space Hunters" not in [r.title for r in recs]
    assert recs[0].shared_genres == ["Action", "Sci-Fi"]


def test_genre_filter_and_missing_episodes(tiny):
    recs = tiny.recommend("Space Hunters", k=5, genre="Slice of Life")
    assert [r.title for r in recs] == ["Kitchen Days"]
    assert recs[0].episodes is None


def test_fuzzy_lookup_and_errors(tiny):
    assert tiny.find_title("robot wor") == "Robot War"
    assert tiny.search("space") == ["Space Hunters", "Space Hunters II"]
    with pytest.raises(KeyError):
        tiny.recommend("Nonexistent Show")


def test_free_text_query(tiny):
    recs = tiny.recommend_from_text("a chef who cooks", k=1)
    assert recs[0].title == "Kitchen Days"


def test_bundled_catalogue_and_cli(capsys):
    rec = AnimeRecommender.from_csv()
    assert len(rec.titles) == 200
    assert cli_main(["cowboy bebop", "-k", "3"]) == 0
    assert "Because you liked Cowboy Bebop" in capsys.readouterr().out


def test_invalid_weight():
    with pytest.raises(ValueError):
        AnimeRecommender(pd.DataFrame(columns=["title", "synopsis", "genres", "episodes", "score"]), synopsis_weight=2)
