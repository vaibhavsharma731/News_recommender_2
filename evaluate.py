"""
Compare the four retrieval & reranking search methods.

Run with: python evaluate.py
"""
import json
import math
import random
import sys

import config
from src.data_loader import load_articles
from src.recommender import METHODS, Recommender

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

DEMO_QUERIES = ["flight crash", "actress Leony dies", "Ujjain mosque demolition",
                "Iran US war", "cricket world cup India", "health insurance claim"]


def demo(rec):
    print("\n=== PART 1: DEMO QUERIES (HYBRID+RERANK) ===")
    for query in DEMO_QUERIES:
        result = rec.recommend(query=query, top_k=3)
        flag = "  <-- no close match" if result.low_confidence else ""
        print(f"\n'{query}'  (best relevance {result.best_relevance:.2f}){flag}")
        for r in result.recommendations:
            print(f"   {r.score:.2f}  {r.article['title'][:90]}")


def section_precision(rec, ids, method, k=5):
    hits = total = 0
    for article_id in ids:
        section = rec.get_article(article_id)["section"]
        for r in rec.recommend(article_id=article_id, top_k=k, method=method, use_boosts=False).recommendations:
            hits += r.article["section"] == section
            total += 1
    return hits / max(total, 1)


def evaluate_linked_labels(rec, method):
    """Evaluate against editor-chosen 'ALSO READ' linked articles extracted during cleaning."""
    df = rec.articles
    valid = df[df["linked_ids"].map(len) > 0]
    if valid.empty:
        return 0.0, 0.0
    valid = valid.sample(min(len(valid), 150), random_state=0)
    
    p5, ndcg = [], []
    for _, row in valid.iterrows():
        article_id = row["id"]
        relevant = set(row["linked_ids"])
        ranked = [r.article["id"] for r in
                  rec.recommend(article_id=article_id, top_k=10, method=method, use_boosts=False).recommendations]
        p5.append(sum(i in relevant for i in ranked[:5]) / 5)
        dcg = sum(1 / math.log2(rank + 2) for rank, i in enumerate(ranked) if i in relevant)
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
        ndcg.append(dcg / ideal if ideal else 0)
    return (sum(p5) / len(p5)) if p5 else 0.0, (sum(ndcg) / len(ndcg)) if ndcg else 0.0


def main():
    rec = Recommender(load_articles())
    print("Engine:", rec.engine_summary)
    demo(rec)

    random.seed(0)
    ids = random.sample(list(rec.articles["id"]), 60)
    print("\n=== PART 2: SAME-SECTION PRECISION@5 (PROXY) ===")
    for method in METHODS:
        print(f"  {method:15s} {section_precision(rec, ids, method):.3f}")

    print("\n=== PART 3: EDITOR-LINKED 'ALSO READ' GROUND TRUTH ===")
    for method in METHODS:
        p5, ndcg = evaluate_linked_labels(rec, method)
        print(f"  {method:15s} P@5={p5:.3f}  nDCG@10={ndcg:.3f}")

    labels_path = config.ROOT / "data" / "eval_labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        print("\n=== PART 4: MANUAL EVAL LABELS ===")
        # optional manual labels
    else:
        print("\n(Optional) Add data/eval_labels.json for custom ground-truth labels.")


if __name__ == "__main__":
    main()
