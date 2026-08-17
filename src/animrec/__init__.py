"""AnimRec: a content-based anime recommender.

This package holds the model shared by the notebook analysis in this repository
and the AnimRec Django web app (github.com/Collinsthe4th/animrec-django).

Example:
    >>> from animrec import AnimeRecommender
    >>> rec = AnimeRecommender.from_csv()
    >>> rec.recommend("Cowboy Bebop", k=5)
"""

from .recommender import AnimeRecommender, Recommendation

__all__ = ["AnimeRecommender", "Recommendation"]
__version__ = "1.0.0"
