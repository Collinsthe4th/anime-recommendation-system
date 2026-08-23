"""Command-line interface.

Examples:
    animrec "Cowboy Bebop"
    animrec "neon genesis" -k 5 --genre Mecha
    animrec --describe "a detective solving murders in a quiet town"
"""

from __future__ import annotations

import argparse
import sys

from .recommender import AnimeRecommender


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recommend anime similar to a title or a description.")
    parser.add_argument("title", nargs="?", help="an anime title (typos are fine)")
    parser.add_argument("--describe", help="free-text description instead of a title")
    parser.add_argument("-k", type=int, default=10, help="number of recommendations")
    parser.add_argument("--genre", help="only recommend titles in this genre")
    args = parser.parse_args(argv)

    if not args.title and not args.describe:
        parser.error("give a title or --describe")

    rec = AnimeRecommender.from_csv()
    try:
        results = rec.recommend_from_text(args.describe, k=args.k) if args.describe else rec.recommend(args.title, k=args.k, genre=args.genre)
    except KeyError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.title:
        print(f"Because you liked {rec.find_title(args.title)}:\n")
    for i, r in enumerate(results, 1):
        print(f"{i:2d}. {r.title:45s} similarity {r.similarity:.3f}  score {r.score:.0f}  [{', '.join(r.genres)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
