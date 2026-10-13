"""
End-to-end ingestion and precomputation pipeline:
CSV -> Cleaning -> PostgreSQL -> Dense Embeddings (pgvector) -> BM25 + pgvector Hybrid Retrieval (RRF Top 30) -> Cross-Encoder Rerank -> News Boosts & MMR -> Top 10 Recommendations with Reasons
"""
import io
import time
import json
import math
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

import config
from .connection import SessionLocal
from .models import Article, ArticleRecommendation, IngestionJob
from .repository import ArticleRepository
from src.retrieval import BM25Index, tokenize, reciprocal_rank_fusion
from src.model_registry import get_embedder
from src.data_loader import (
    normalise_text,
    remove_links,
    _short_description,
    _headline_index,
    strip_also_read,
    _find_boilerplate,
    _strip_boilerplate,
    _subsection,
    first_words
)


def _get_embedding_model():
    """Return the shared embedding model from the model registry."""
    return get_embedder()


def _get_reranker_model():
    """Return the shared cross-encoder reranker."""
    from src.reranker import Reranker
    return Reranker()


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def _precompute_for_article(
    idx, records, bm25, reranker, emb, max_views, max_date, n_retrieve, n_cand
):
    """Compute top recommendations for one article and return rows for persistence."""
    row = records[idx]

    # A. Keyword candidates
    title_toks = tokenize(str(row["title"]))
    body_toks = tokenize(f"{row['description']} {row['body']}")
    top_terms = sorted(set(body_toks), key=lambda t: -bm25.idf.get(t, 0.0))[:15]
    bm25_scores = bm25.scores(" ".join(title_toks + top_terms))
    bm25_scores[idx] = -np.inf
    bm25_rank = [
        int(p)
        for p in np.argsort(-bm25_scores)[:n_retrieve]
        if bm25_scores[p] > 1e-9
    ]

    # B. Semantic candidates
    dense_scores = emb @ emb[idx]
    dense_scores[idx] = -np.inf
    dense_rank = [int(p) for p in np.argsort(-dense_scores)[:n_retrieve]]

    # C. Reciprocal-rank fusion
    fused = reciprocal_rank_fusion([bm25_rank, dense_rank], k=config.RRF_K)
    cands = sorted(fused, key=fused.get, reverse=True)[:n_cand]
    if not cands:
        return []

    # D. Cross-encoder reranking
    rerank_scores = np.asarray(
        reranker.score(
            row["text_rerank"],
            [records[c]["text_rerank"] for c in cands],
            batch_size=32,
        ),
        dtype=np.float32,
    )

    # E. Blend reranker, BM25, and dense relevance scores
    cand_bm25 = np.where(np.isneginf(bm25_scores[cands]), 0.0, bm25_scores[cands])
    bm25_max = cand_bm25.max() if cand_bm25.max() > 0 else 1.0
    bm25_norm = cand_bm25 / bm25_max

    cand_dense = dense_scores[cands]
    dense_max = cand_dense.max() if cand_dense.max() > 0 else 1.0
    dense_norm = np.clip(cand_dense / dense_max, 0.0, 1.0)
    relevance = 0.55 * rerank_scores + 0.30 * bm25_norm + 0.15 * dense_norm

    # F. News metadata boosts and G. MMR diversity
    boosts = _calculate_boosts(row, [records[c] for c in cands], max_views, max_date)
    order = _apply_mmr(
        cands,
        relevance * boosts,
        emb[cands],
        top_k=min(config.TOP_K, len(cands)),
        lam=config.MMR_LAMBDA,
    )

    # H. Package ranked recommendations
    recs = []
    for rank, rel_pos in enumerate(order, start=1):
        c = cands[rel_pos]
        recs.append({
            "recommended_article_id": str(records[c]["id"]),
            "rank": rank,
            "score": round(float(np.clip(relevance[rel_pos], 0.0, 1.0)), 4),
            "method": "neural_hybrid_rerank",
            "reason": _generate_reason(
                row,
                records[c],
                title_toks,
                bm25.docs[c],
                bm25,
                dense_sim=float(dense_norm[rel_pos]),
                rerank_score=float(rerank_scores[rel_pos]),
            ),
        })
    return recs

def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "y", "t"}


def clean_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning and normalization with flexible column resolution
    supporting diverse CSV structures.
    """
    raw = raw_df.copy()

    # 1. Flexible Column Extractor
    def _col(candidates: list, default="") -> pd.Series:
        for c in candidates:
            if c in raw.columns:
                return raw[c]
        return pd.Series([default] * len(raw))

    # Resolve ID column & deduplicate
    id_col = next((c for c in ["articleId", "id", "article_id", "ID"] if c in raw.columns), None)
    if id_col:
        raw = raw.drop_duplicates(subset=id_col, keep="last").reset_index(drop=True)
        ids = raw[id_col].astype(str).tolist()
    else:
        raw = raw.reset_index(drop=True)
        ids = [f"art_{i+1}" for i in range(len(raw))]

    # Extract text fields
    titles = _col(["articleTitle", "title", "headline", "articleName"]).map(normalise_text)
    raw_desc = _col(["description", "snippet", "summary"])
    raw_bodies = _col(["body_raw", "body", "content", "text"]).fillna("")

    # If description is missing/empty, take first 200 chars of body
    clean_desc = []
    for d, b in zip(raw_desc, raw_bodies):
        text = str(d).strip() if pd.notna(d) and str(d).strip() else str(b)[:200].strip()
        clean_desc.append(_short_description(remove_links(normalise_text(text))))

    headlines = _headline_index(titles, ids)
    bodies, linked_ids = [], []
    for article_id, body in zip(ids, raw_bodies):
        body_clean = remove_links(normalise_text(body))
        body_clean, linked = strip_also_read(body_clean, article_id, headlines)
        bodies.append(body_clean)
        linked_ids.append(sorted(set(linked)))

    boilerplate = _find_boilerplate(bodies)
    bodies = [_strip_boilerplate(b, boilerplate) for b in bodies]

    sec_series = _col(["sectionName", "section1Slug", "section", "category"])
    subsec_series = raw["section2Name"].map(_subsection) if "section2Name" in raw.columns else _col(["section2Slug", "subsection"])
    pv_series = _col(["pageViews", "pageViews_latest", "pageViews_max", "views"], default=0)
    uu_series = _col(["uniqueUsers", "uniqueUsers_latest", "uniqueUsers_max", "users"], default=0)
    date_series = _col(["postPublishDate", "date", "published_at", "publishDate"], default=None)

    clean_df = pd.DataFrame({
        "id": ids,
        "title": titles,
        "description": clean_desc,
        "section": sec_series.fillna(""),
        "subsection": subsec_series.fillna(""),
        "date": pd.to_datetime(date_series, utc=True, errors="coerce"),
        "url": _col(["articleUrl", "url", "link"]).fillna(""),
        "image": _col(["articleFeaturedImage", "image", "imageUrl", "img"]).fillna(""),
        "is_live": _col(["liveBlog", "is_live"], default=False).map(_as_bool),
        "body": bodies,
        "linked_ids": [json.dumps(l) for l in linked_ids],
        "page_views": pd.to_numeric(pv_series, errors="coerce").fillna(0).astype(int),
        "unique_users": pd.to_numeric(uu_series, errors="coerce").fillna(0).astype(int),
    })

    # Text representations for neural embeddings
    clean_df["text_embed"] = [
        f"{t}. {d}. {first_words(b, config.BODY_WORDS_EMBED)}"
        for t, d, b in zip(clean_df.title, clean_df.description, clean_df.body)
    ]
    # BM25 keyword representation: Title repeated 3x for entity salience
    clean_df["text_bm25"] = [
        f"{t}. {t}. {t}. {d}. {b}"
        for t, d, b in zip(clean_df.title, clean_df.description, clean_df.body)
    ]
    # Reranker representation: Truncated lead text
    clean_df["text_rerank"] = [
        f"{t}. {d}. {first_words(b, config.BODY_WORDS_RERANK)}"
        for t, d, b in zip(clean_df.title, clean_df.description, clean_df.body)
    ]
    return clean_df


def _calculate_boosts(
    source_row: dict,
    candidate_rows: List[dict],
    max_views: int,
    max_date: Optional[pd.Timestamp]
) -> np.ndarray:
    """
    Computes news domain boosts:
    1. Editorial "Also Read" link: 1.40x
    2. Same section: 1.08x
    3. Same subsection: 1.05x
    4. Popularity log-scale: up to 1.06x
    5. Dynamic recency exponential decay based on section half-life: up to 1.10x
    """
    multipliers = []
    source_linked = source_row.get("linked_ids", [])
    if isinstance(source_linked, str):
        try:
            source_linked = json.loads(source_linked)
        except Exception:
            source_linked = []

    for cand in candidate_rows:
        mult = 1.0

        # Editorial link boost
        if cand["id"] in source_linked:
            mult *= 1.40

        # Section / Subsection boost
        if source_row.get("section") and source_row["section"] == cand.get("section"):
            mult *= config.SECTION_BOOST_MULT
        if source_row.get("subsection") and source_row["subsection"] == cand.get("subsection"):
            mult *= config.SUBSECTION_BOOST_MULT

        # Popularity boost
        views = cand.get("page_views", 0)
        if views > 0 and max_views > 0:
            log_views = math.log1p(views)
            log_max = math.log1p(max_views)
            pop_factor = 1.0 + (config.POPULARITY_BOOST_MULT - 1.0) * (log_views / log_max)
            mult *= pop_factor

        # Recency decay boost
        c_date = cand.get("date")
        if pd.notna(c_date) and pd.notna(max_date):
            days_old = max(0.0, (max_date - c_date).total_seconds() / 86400.0)
            sec = "live_blog" if cand.get("is_live") else str(cand.get("section", "")).lower()
            half_life = config.SECTION_HALF_LIVES.get(sec, config.SECTION_HALF_LIVES.get("default", 2.0))
            recency_mult = 1.0 + (config.RECENCY_MAX_BOOST_MULT - 1.0) * (0.5 ** (days_old / half_life))
            mult *= recency_mult

        multipliers.append(mult)
    return np.array(multipliers, dtype=float)


def _apply_mmr(
    cand_indices: List[int],
    final_scores: np.ndarray,
    candidate_vectors: np.ndarray,
    top_k: int = config.TOP_K,
    lam: float = config.MMR_LAMBDA
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR): picks results with high relevance while
    penalizing redundant/near-duplicate news wire copy.
    """
    if len(cand_indices) <= top_k:
        return list(np.argsort(-final_scores))

    similarity = candidate_vectors @ candidate_vectors.T
    selected = []
    remaining = list(range(len(cand_indices)))

    while remaining and len(selected) < top_k:
        def mmr_score(i):
            high_sims = [similarity[i][j] for j in selected if similarity[i][j] > 0.80]
            redundancy = max(high_sims, default=0.0)
            return lam * final_scores[i] - (1 - lam) * redundancy

        best = max(remaining, key=mmr_score)
        selected.append(best)
        remaining.remove(best)
    return selected


def _generate_reason(
    source_row: dict,
    cand_row: dict,
    source_tokens: List[str],
    cand_bm25_doc: List[str],
    bm25_index: BM25Index,
    dense_sim: float = 0.0,
    rerank_score: float = 0.0
) -> str:
    """Builds an informative human-readable explanation combining neural semantics and keywords."""
    parts = []
    source_linked = source_row.get("linked_ids", [])
    if isinstance(source_linked, str):
        try:
            source_linked = json.loads(source_linked)
        except Exception:
            source_linked = []

    if cand_row["id"] in source_linked:
        parts.append("Editor's Pick ('Also Read')")

    # 1. Explicit Neural Semantic Similarity & Meaning Alignment
    top_neural = max(dense_sim, rerank_score)
    if top_neural >= 0.85:
        parts.append(f"Strong semantic match ({int(top_neural * 100)}% context similarity)")
    elif top_neural >= 0.55:
        parts.append(f"Semantic context match ({int(top_neural * 100)}%)")

    # 2. Shared Topic Keywords / Entities
    shared = set(source_tokens) & set(cand_bm25_doc)
    shared_keywords = sorted(shared, key=lambda t: -bm25_index.idf.get(t, 0.0))[:3]
    if shared_keywords:
        parts.append("shared topics: " + ", ".join(shared_keywords))
    elif not any("semantic" in p.lower() for p in parts):
        parts.append("thematic conceptual match")

    # 3. Editorial Metadata (Section & Publication Time)
    if source_row.get("section") and source_row["section"] == cand_row.get("section"):
        parts.append(f"same section ({source_row['section']})")

    if pd.notna(source_row.get("date")) and pd.notna(cand_row.get("date")):
        days = abs((source_row["date"] - cand_row["date"]).days)
        if days <= 1:
            parts.append("published around the same time")

    return "; ".join(parts) if parts else "High semantic and topic relevance"



def run_ingestion_pipeline(
    csv_source: Union[str, Path, io.BytesIO],
    job_id: str,
    limit: Optional[int] = None,
    recompute_all_recommendations: bool = True,
):
    """CSV -> clean -> PostgreSQL -> embeddings -> BM25+dense -> RRF -> rerank -> boosts -> MMR -> top-10 saved."""
    db = SessionLocal()
    t_job = time.time()
    try:
        upd = lambda **kw: ArticleRepository.update_job(db, job_id, **kw)

        # 1. Parse CSV
        upd(status="processing", message="STAGE:PARSE|Parsing and validating CSV records...")
        raw_df = pd.read_csv(csv_source)
        if limit and limit > 0:
            raw_df = raw_df.head(limit)
        upd(
            total_articles=len(raw_df),
            message=f"STAGE:CLEAN|Cleaning and normalizing {len(raw_df)} articles...",
        )

        # 2. Clean records
        df = clean_dataframe(raw_df)
        n = len(df)
        if n == 0:
            raise ValueError("CSV contained no usable articles.")
        upd(total_articles=n, message=f"STAGE:STORAGE|Saving {n} cleaned articles to PostgreSQL...")

        # 3. Persist articles with batched INSERT ... ON CONFLICT
        records = df.to_dict("records")
        ids = [str(r["id"]) for r in records]
        ArticleRepository.bulk_upsert_articles_fast(db, [{
            "id": str(r["id"]),
            "title": str(r["title"]),
            "description": str(r["description"]),
            "body": str(r["body"]),
            "section": str(r["section"]),
            "subsection": str(r["subsection"]),
            "date": r["date"].to_pydatetime() if pd.notna(r["date"]) else None,
            "url": str(r["url"]),
            "image": str(r["image"]),
            "is_live": bool(r["is_live"]),
            "page_views": int(r["page_views"]),
            "unique_users": int(r["unique_users"]),
            "linked_ids": str(r["linked_ids"]),
            "text_embed": str(r["text_embed"]),
            "text_bm25": str(r["text_bm25"]),
            "text_rerank": str(r["text_rerank"]),
        } for r in records])

        # 4. Build and persist embeddings
        upd(message="STAGE:EMBEDDING|Loading embedding model (first run downloads ~440 MB)...")
        embedder = _get_embedding_model()

        texts = df["text_embed"].tolist()
        chunks, chunk_size, t0 = [], 128, time.time()
        for i in range(0, n, chunk_size):
            chunks.append(embedder.encode(
                texts[i:i + chunk_size],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            ))
            done = min(i + chunk_size, n)
            eta = (time.time() - t0) / done * (n - done)
            upd(
                processed_articles=int(0.40 * done),
                message=f"STAGE:EMBEDDING|Embedded {done}/{n} articles (ETA {_fmt_eta(eta)})...",
            )
        emb = np.vstack(chunks).astype(np.float32)
        ArticleRepository.bulk_set_embeddings(db, ids, emb)

        # 5. Build BM25 index
        upd(message="STAGE:INDEX|Building in-memory BM25 index...")
        bm25 = BM25Index(df["text_bm25"])

        # 6. Rerank candidates and save recommendations in batches
        upd(message="STAGE:RERANK|Loading cross-encoder (first run downloads ~1 GB, can take minutes)...")
        reranker = _get_reranker_model()

        max_views = int(df["page_views"].max()) if n else 1
        max_date = df["date"].max()
        n_retrieve = min(config.N_RETRIEVE, max(n - 1, 0))
        n_cand = min(config.N_CANDIDATES, max(n - 1, 0))

        already = (
            set()
            if recompute_all_recommendations
            else ArticleRepository.get_ids_with_recommendations(db)
        )
        todo = [i for i in range(n) if n > 1 and ids[i] not in already]
        failed, t0 = 0, time.time()

        for done, idx in enumerate(todo, start=1):
            try:
                recs = _precompute_for_article(
                    idx, records, bm25, reranker, emb,
                    max_views, max_date, n_retrieve, n_cand,
                )
            except Exception as art_err:  # one bad article must not terminate the entire job
                failed += 1
                recs = []
                print(f"[Pipeline] Skipped article {ids[idx]}: {art_err}")

            if recs:
                ArticleRepository.save_recommendations(db, ids[idx], recs, commit=False)

            if done % config.PRECOMPUTE_COMMIT_EVERY == 0 or done == len(todo):
                per = (time.time() - t0) / done
                eta = per * (len(todo) - done)
                # update_job commits and therefore commits pending recommendation writes too.
                upd(
                    processed_articles=int(0.40 * n + 0.60 * n * done / len(todo)),
                    message=(
                        f"STAGE:RERANK|Precomputed {done}/{len(todo)} "
                        f"({per:.1f}s/article, ETA {_fmt_eta(eta)})"
                    ),
                )
        db.commit()

        # 7. Refresh live engine directly in this process (no HTTP request needed).
        upd(message="STAGE:INDEX|Refreshing live search engine...")
        try:
            from api.server import reload_engine_from_postgres
            reload_engine_from_postgres()
        except Exception as reload_err:
            print(f"[Pipeline] Engine reload skipped: {reload_err}")

        note = f" ({failed} skipped)" if failed else ""
        upd(
            status="completed",
            processed_articles=n,
            message=f"STAGE:COMPLETE|Processed {n} articles{note} in {_fmt_eta(time.time() - t_job)}.",
        )

    except Exception as exc:
        import traceback
        print(f"[Pipeline Error] {traceback.format_exc()}")
        try:
            db.rollback()  # ensure the failure status can be written after a DB error
            ArticleRepository.update_job(
                db,
                job_id,
                status="failed",
                message="Processing encountered an error",
                error=str(exc)[:2000],
            )
        except Exception as inner:
            print(f"[Pipeline] Could not record failure: {inner}")
    finally:
        db.close()
