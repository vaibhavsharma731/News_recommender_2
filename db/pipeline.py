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
    """Lazily load the SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(config.EMBEDDING_MODEL, device=device)


def _get_reranker_model():
    """Lazily load the CrossEncoder model."""
    from src.reranker import Reranker
    return Reranker()


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
        "is_live": _col(["liveBlog", "is_live"], default=False).fillna(False).astype(bool),
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
    recompute_all_recommendations: bool = True
):
    """
    Complete PostgreSQL Ingestion & Precomputation Pipeline:
    1. Parse CSV
    2. Clean & Normalize Text
    3. Save Articles & Metadata in PostgreSQL
    4. Compute Neural Embeddings (bge-base-en-v1.5) & Save to pgvector
    5. Build In-Corpus BM25 Keyword Index
    6. Hybrid Retrieval (BM25 + pgvector) -> Reciprocal Rank Fusion (RRF Top 30)
    7. Cross-Encoder Reranker on Top 30 Candidates
    8. Multi-Factor News Boosts & MMR Diversity Filtering
    9. Store Top 10 Precomputed Recommendations with Reasons in PostgreSQL
    """
    db = SessionLocal()
    try:
        ArticleRepository.update_job(
            db, job_id, status="processing", message="STAGE:PARSE|Parsing and validating CSV records..."
        )

        # 1. Read CSV
        if isinstance(csv_source, (str, Path)):
            raw_df = pd.read_csv(csv_source)
        else:
            raw_df = pd.read_csv(csv_source)

        if limit and limit > 0:
            raw_df = raw_df.head(limit)

        total_rows = len(raw_df)
        ArticleRepository.update_job(
            db, job_id, total_articles=total_rows, message=f"STAGE:CLEAN|Cleaning and normalizing {total_rows} articles..."
        )

        # 2. Clean & Normalize
        df = clean_dataframe(raw_df)
        n_articles = len(df)
        ArticleRepository.update_job(
            db, job_id, total_articles=n_articles, message=f"STAGE:STORAGE|Saving {n_articles} cleaned articles to PostgreSQL..."
        )

        # 3. Upsert articles into PostgreSQL
        for _, row in df.iterrows():
            art_dict = {
                "id": str(row["id"]),
                "title": str(row["title"]),
                "description": str(row["description"]),
                "body": str(row["body"]),
                "section": str(row["section"]),
                "subsection": str(row["subsection"]),
                "date": row["date"] if pd.notna(row["date"]) else None,
                "url": str(row["url"]),
                "image": str(row["image"]),
                "is_live": bool(row["is_live"]),
                "page_views": int(row["page_views"]),
                "unique_users": int(row["unique_users"]),
                "linked_ids": str(row["linked_ids"]),
                "text_embed": str(row["text_embed"]),
                "text_bm25": str(row["text_bm25"]),
                "text_rerank": str(row["text_rerank"]),
            }
            ArticleRepository.upsert_article(db, art_dict)
        db.commit()

        # 4. Neural Embeddings Generation
        ArticleRepository.update_job(
            db, job_id, message="STAGE:EMBEDDING|Loading neural embedding model (bge-base-en-v1.5)..."
        )
        embed_model = _get_embedding_model()

        texts_to_embed = df["text_embed"].tolist()
        batch_size = 32
        all_embeddings = []

        ArticleRepository.update_job(
            db, job_id, message=f"STAGE:EMBEDDING|Computing 768d dense vectors for {n_articles} articles..."
        )

        for i in range(0, n_articles, batch_size):
            batch_texts = texts_to_embed[i : i + batch_size]
            batch_vecs = embed_model.encode(batch_texts, normalize_embeddings=True, show_progress_bar=False)
            all_embeddings.extend(batch_vecs)

            current_count = min(i + batch_size, n_articles)
            ArticleRepository.update_job(
                db,
                job_id,
                processed_articles=int(current_count * 0.40), # 40% allocated to embeddings
                message=f"STAGE:EMBEDDING|Embedded {current_count}/{n_articles} articles into pgvector column..."
            )

        all_embeddings_np = np.array(all_embeddings, dtype=float)

        # Save embeddings in PostgreSQL
        for idx, row in df.iterrows():
            art = ArticleRepository.get_article(db, str(row["id"]))
            if art:
                art.embedding = all_embeddings_np[idx].tolist()
        db.commit()

        # 5. Hybrid Index Construction: BM25 Keyword Engine
        ArticleRepository.update_job(
            db, job_id, message="STAGE:INDEX|Building in-memory BM25 index with Indian entity fuzzy vocabulary..."
        )
        bm25 = BM25Index(df["text_bm25"])

        # 6. Precompute Recommendations (BM25 + pgvector -> RRF Top 30 -> Cross-Encoder -> Boosts -> MMR)
        ArticleRepository.update_job(
            db, job_id, message="STAGE:RERANK|Loading cross-encoder reranker model (bge-reranker-base)..."
        )
        reranker = _get_reranker_model()

        text_rerank_map = dict(zip(df["id"].astype(str), df["text_rerank"]))
        max_views = int(df["page_views"].max()) if len(df) else 1
        max_date = df["date"].max() if "date" in df.columns else None

        ArticleRepository.update_job(
            db, job_id, message=f"STAGE:RERANK|Precomputing hybrid recommendations (Top 30 RRF fusion -> Rerank -> MMR)..."
        )

        n_retrieve = min(config.N_RETRIEVE, n_articles - 1) if n_articles > 1 else 1
        n_candidates = min(config.N_CANDIDATES, n_articles - 1) if n_articles > 1 else 1

        for idx, row in df.iterrows():
            art_id = str(row["id"])
            source_row = row.to_dict()
            source_rerank_text = text_rerank_map.get(art_id, str(row["title"]))

            if n_articles <= 1:
                continue

            # Step 6A: BM25 focused query
            title_toks = tokenize(str(row["title"]))
            body_toks = tokenize(f"{row['description']} {row['body']}")
            top_body_terms = sorted(set(body_toks), key=lambda t: -bm25.idf.get(t, 0.0))[:15]
            focused_query = " ".join(title_toks + top_body_terms)
            bm25_scores = bm25.scores(focused_query)
            bm25_scores[idx] = -np.inf  # Exclude self

            bm25_rank = [int(p) for p in np.argsort(-bm25_scores)[:n_retrieve] if bm25_scores[p] > 1e-9]

            # Step 6B: Dense semantic vector scores (cosine similarity)
            dense_scores = all_embeddings_np @ all_embeddings_np[idx]
            dense_scores[idx] = -np.inf  # Exclude self
            dense_rank = [int(p) for p in np.argsort(-dense_scores)[:n_retrieve]]

            # Step 6C: Reciprocal Rank Fusion (RRF) -> Top 30 candidates
            fused = reciprocal_rank_fusion([bm25_rank, dense_rank], k=config.RRF_K)
            candidate_indices = sorted(fused, key=fused.get, reverse=True)[:n_candidates]

            if not candidate_indices:
                continue

            # Step 6D: Cross-Encoder Reranking on Top 30 candidates
            cand_rerank_texts = [
                text_rerank_map.get(str(df.iloc[c]["id"]), str(df.iloc[c]["title"]))
                for c in candidate_indices
            ]
            rerank_scores = reranker.score(source_rerank_text, cand_rerank_texts, batch_size=32)

            # Step 6E: Score Blending (Cross-Encoder 55% + BM25 30% + Dense 15%)
            cand_bm25 = bm25_scores[candidate_indices]
            cand_bm25 = np.where(np.isneginf(cand_bm25), 0.0, cand_bm25)
            bm25_max = cand_bm25.max() if len(cand_bm25) and cand_bm25.max() > 0 else 1.0
            bm25_norm = cand_bm25 / bm25_max

            cand_dense = dense_scores[candidate_indices]
            dense_max = cand_dense.max() if len(cand_dense) and cand_dense.max() > 0 else 1.0
            dense_norm = np.clip(cand_dense / dense_max, 0.0, 1.0)

            relevance = 0.55 * rerank_scores + 0.30 * bm25_norm + 0.15 * dense_norm

            # Step 6F: Editorial & News Multiplier Boosts
            candidate_rows = [df.iloc[c].to_dict() for c in candidate_indices]
            boost_mults = _calculate_boosts(source_row, candidate_rows, max_views, max_date)
            boosted_scores = relevance * boost_mults

            # Step 6G: MMR Diversity Selection (Top 10)
            cand_vecs = all_embeddings_np[candidate_indices]
            final_order = _apply_mmr(
                candidate_indices,
                boosted_scores,
                cand_vecs,
                top_k=min(config.TOP_K, len(candidate_indices)),
                lam=config.MMR_LAMBDA
            )

            # Step 6H: Build Recommendation Objects with Reasons
            top_10_recommendations = []
            for rank_pos, rel_pos in enumerate(final_order, start=1):
                actual_df_idx = candidate_indices[rel_pos]
                cand_dict = df.iloc[actual_df_idx].to_dict()
                c_dense_sim = float(dense_norm[rel_pos]) if len(dense_norm) > rel_pos else 0.0
                c_rerank_score = float(rerank_scores[rel_pos]) if len(rerank_scores) > rel_pos else 0.0

                reason = _generate_reason(
                    source_row,
                    cand_dict,
                    title_toks,
                    cand_bm25_doc,
                    bm25,
                    dense_sim=c_dense_sim,
                    rerank_score=c_rerank_score
                )

                top_10_recommendations.append({
                    "recommended_article_id": str(cand_dict["id"]),
                    "rank": rank_pos,
                    "score": round(float(np.clip(relevance[rel_pos], 0.0, 1.0)), 4),
                    "method": "neural_hybrid_rerank",
                    "reason": reason
                })

            # Save to PostgreSQL article_recommendations table
            ArticleRepository.save_recommendations(db, art_id, top_10_recommendations)

            # Frequent progress update: every 2 articles or on final article
            if (idx + 1) % 2 == 0 or (idx + 1) == n_articles:
                processed_recs = int(n_articles * 0.40) + int(((idx + 1) / n_articles) * (n_articles * 0.60))
                pct = int(((idx + 1) / n_articles) * 100)
                ArticleRepository.update_job(
                    db,
                    job_id,
                    processed_articles=processed_recs,
                    message=f"STAGE:RERANK|Precomputing recommendations: {idx + 1}/{n_articles} articles ({pct}% complete)..."
                )

        # Notify active server engine to reload memory index if available
        try:
            import requests
            requests.post("http://localhost:5000/api/db/reload", timeout=1)
        except Exception:
            pass

        # Completed!
        ArticleRepository.update_job(
            db,
            job_id,
            status="completed",
            processed_articles=n_articles,
            message=f"STAGE:COMPLETE|Successfully processed {n_articles} articles and generated top-10 precomputed recommendations!"
        )

    except Exception as exc:
        import traceback
        err_msg = traceback.format_exc()
        print(f"[Pipeline Error] {err_msg}")
        ArticleRepository.update_job(
            db,
            job_id,
            status="failed",
            message="Processing encountered an error",
            error=str(exc)
        )
    finally:
        db.close()
