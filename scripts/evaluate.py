"""Offline evaluation of the recommender, with no user ratings needed.

Two checks:

1. **Franchise retrieval.** Titles that share their first two title words
   (for example the Gundam or Evangelion entries) belong to the same franchise.
   The franchise is never given to the model, so a good recommender should still
   rank the other entries near the top. Reported as hit rate @5 / @10 and MRR
   (mean reciprocal rank, 1 if the first sibling is ranked first, 0.5 if second).
2. **Genre coherence.** Mean Jaccard genre overlap between a title and its top
   10 recommendations, compared with random picks.

The synopsis weight is swept from 0 (genres only) to 1 (synopsis only).

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from animrec import AnimeRecommender  # noqa: E402

RESULTS = ROOT / "results"


def franchise_key(title: str) -> str:
    import re

    words = re.sub(r"[^a-z0-9 ]", " ", title.lower()).split()
    return " ".join(words[:2])


def evaluate(rec: AnimeRecommender, weight: float) -> dict:
    keys = np.array([franchise_key(t) for t in rec.titles])
    n = len(keys)
    hits5 = hits10 = 0
    rr = []
    coherence = []
    for i in range(n):
        scores = rec.similarity_row(i, synopsis_weight=weight) + rec.score_weight * rec.score_norm
        scores[i] = -np.inf
        order = np.argsort(-scores)
        coherence.append(rec.genre_sim[i, order[:10]].mean())
        siblings = np.where((keys == keys[i]) & (np.arange(n) != i))[0]
        if len(siblings) == 0:
            continue
        rank = int(np.min([np.where(order == s)[0][0] for s in siblings])) + 1
        hits5 += rank <= 5
        hits10 += rank <= 10
        rr.append(1 / rank)
    m = len(rr)
    return {
        "synopsis_weight": weight,
        "titles_with_franchise_siblings": m,
        "hit_rate_at_5": hits5 / m,
        "hit_rate_at_10": hits10 / m,
        "mrr": float(np.mean(rr)),
        "genre_coherence_top10": float(np.mean(coherence)),
    }


def main() -> None:
    rec = AnimeRecommender.from_csv()
    sweep = [evaluate(rec, w) for w in np.round(np.arange(0, 1.01, 0.1), 1)]
    rng = np.random.default_rng(42)
    n = len(rec.titles)
    random_coh = float(np.mean([rec.genre_sim[i, rng.choice(np.delete(np.arange(n), i), 10, replace=False)].mean() for i in range(n)]))
    random_hit10 = float(np.mean([1 - np.prod([(n - 1 - s - j) / (n - 1 - j) for j in range(10)]) for s in [1]]))

    best = max(sweep, key=lambda r: (round(r["mrr"], 3), r["hit_rate_at_10"]))
    # Selection rule: highest MRR among weights whose genre coherence is within
    # 0.01 of the maximum, so recommendations stay on-genre.
    top_coh = max(r["genre_coherence_top10"] for r in sweep)
    eligible = [r for r in sweep if r["genre_coherence_top10"] >= top_coh - 0.01]
    selected = max(eligible, key=lambda r: r["mrr"])
    chosen = next(r for r in sweep if r["synopsis_weight"] == rec.synopsis_weight)
    out = {
        "catalogue_size": n,
        "chosen_synopsis_weight": rec.synopsis_weight,
        "chosen": {k: round(v, 4) if isinstance(v, float) else v for k, v in chosen.items()},
        "selected_by_rule": {k: round(v, 4) if isinstance(v, float) else v for k, v in selected.items()},
        "best_by_mrr": {k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
        "random_baseline": {"genre_coherence_top10": round(random_coh, 4), "hit_rate_at_10_one_sibling": round(random_hit10, 4)},
        "sweep": [{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()} for r in sweep],
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))

    w = [r["synopsis_weight"] for r in sweep]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(w, [r["hit_rate_at_10"] for r in sweep], marker="o", label="Franchise hit rate @10")
    ax.plot(w, [r["mrr"] for r in sweep], marker="o", label="Franchise MRR")
    ax.plot(w, [r["genre_coherence_top10"] for r in sweep], marker="o", label="Genre coherence (top 10)")
    ax.axhline(random_coh, color="grey", ls=":", label="Genre coherence, random picks")
    ax.axvline(rec.synopsis_weight, color="black", ls="--", lw=1, label=f"Chosen weight {rec.synopsis_weight}")
    ax.set_xlabel("Synopsis weight (0 = genres only, 1 = synopsis only)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Offline evaluation across the synopsis/genre mix")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "weight_sweep.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: out[k] for k in ("chosen", "best_by_mrr", "random_baseline")}, indent=2))
    for r in sweep:
        print(r["synopsis_weight"], round(r["hit_rate_at_10"], 3), round(r["mrr"], 3), round(r["genre_coherence_top10"], 3))


if __name__ == "__main__":
    main()
