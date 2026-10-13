"""
End-to-end ingestion and precomputation pipeline:
CSV -> Cleaning -> PostgreSQL -> Dense Embeddings (pgvector) -> BM25 + pgvector Hybrid Retrieval (RRF Top 30) -> Cross-Encoder Rerank -> News Boosts & MMR -> Top 10 Recommendations with Reasons
"""
import io
import time
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

import config
from .connection import SessionLocal, init_db
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


def _focused_query(row, bm25) -> str:
    """
    The query used to represent an article when searching for its neighbours:
    its title tokens plus its 15 most distinctive (highest-IDF) body terms.
    Shared by single-article precompute and by the reverse-push so both use
    exactly the same notion of "what this article is about".
    """
    title_toks = tokenize(str(row["title"]))
    body_toks = tokenize(f"{row['description']} {row['body']}")
    top_terms = sorted(set(body_toks), key=lambda t: -bm25.idf.get(t, 0.0))[:15]
    return " ".join(title_toks + top_terms)


def _load_full_corpus(db):
    """
    Load EVERY stored article (already-existing + just-ingested) into pipeline
    records plus an aligned 768-d embedding matrix.

    Doing this is what lets a newly uploaded article retrieve neighbours from the
    whole database instead of only from the batch it arrived in.
    """
    arts = db.query(Article).order_by(Article.date.desc().nulls_last()).all()
    records, vectors, id_to_pos = [], [], {}
    for a in arts:
        try:
            linked = json.loads(a.linked_ids) if a.linked_ids else []
        except Exception:
            linked = []
        pos = len(records)
        records.append({
            "id": str(a.id),
            "title": str(a.title),
            "description": str(a.description or ""),
            "body": str(a.body or ""),
            "section": str(a.section or ""),
            "subsection": str(a.subsection or ""),
            "date": a.date,
            "url": str(a.url or ""),
            "image": str(a.image or ""),
            "is_live": bool(a.is_live),
            "page_views": int(a.page_views or 0),
            "unique_users": int(a.unique_users or 0),
            "linked_ids": linked,
            "text_embed": a.text_embed or f"{a.title}. {a.description or ''}",
            "text_bm25": a.text_bm25 or f"{a.title}. {a.title}. {a.title}. {a.description or ''}. {a.body or ''}",
            "text_rerank": a.text_rerank or f"{a.title}. {a.description or ''}.",
        })
        id_to_pos[str(a.id)] = pos
        vectors.append(a.embedding if a.embedding is not None else [0.0] * 768)
    emb = np.asarray(vectors, dtype=np.float32) if vectors else np.zeros((0, 768), dtype=np.float32)
    return records, emb, id_to_pos


def _precompute_for_article(
    idx, records, bm25, reranker, emb, max_views, max_date, n_retrieve, n_cand
):
    """
    Compute the top recommendations for ONE article against the full corpus.

    Returns a tuple ``(recs, candidates)``:
      * recs       -> rows ready for persistence (each carries a raw rerank_score)
      * candidates -> the fused candidate positions, reused by the reverse-push
                      so we know which existing articles this one is close to.
    """
    row = records[idx]

    # A. Keyword candidates
    bm25_scores = bm25.scores(_focused_query(row, bm25))
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
        return [], []

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
    title_toks = tokenize(str(row["title"]))
    recs = []
    for rank, rel_pos in enumerate(order, start=1):
        c = cands[rel_pos]
        recs.append({
            "recommended_article_id": str(records[c]["id"]),
            "rank": rank,
            "score": round(float(np.clip(relevance[rel_pos], 0.0, 1.0)), 4),
            "rerank_score": round(float(rerank_scores[rel_pos]), 6),
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
    return recs, cands


def _reverse_push_for_old(
    db, new_positions, candidate_map, records, emb, bm25, reranker,
    max_views, max_date, new_id_set, id_to_pos, progress_cb=None,
):
    """
    Let newly ingested articles flow into the recommendation lists of the EXISTING
    articles they are genuinely close to.

    Why this is correct (and cheap):
      * For each new article N we already know its full-corpus candidates
        (``candidate_map``). By (approximate) similarity symmetry, an old article O
        can only be displaced by N if N is one of O's close neighbours too — so the
        candidate set found from N's side is exactly the set worth checking.
      * We reuse O's *stored* rerank scores for its current entries (no cross-encoder
        needed for the old list) and only run the cross-encoder for the genuinely new
        candidate pairs, with O as the query (O-centric scoring).
      * We then merge, re-apply boosts + MMR, and keep the top-K.

    Old articles that no new article matches are never touched.

    ``progress_cb(done, total, title)`` (optional) is called as each existing article
    is processed so the caller can stream progress to the Editor log.

    Returns a list of dicts ``{"id", "title"}`` describing every existing article whose
    list was refreshed — so the caller can report exactly which old articles changed.
    """
    # 1. Reverse adjacency: old article position -> new articles close to it.
    reverse = defaultdict(list)
    for npos in new_positions:
        for cpos in candidate_map.get(npos, []):
            if str(records[cpos]["id"]) not in new_id_set:   # keep only OLD articles
                reverse[cpos].append(npos)
    if not reverse:
        return []

    total = len(reverse)
    report_every = max(1, total // 12)   # lively, but no more than ~12 log lines
    refreshed: List[Dict[str, Any]] = []

    for done, (opos, new_neighbors) in enumerate(reverse.items(), start=1):
        if progress_cb and (done % report_every == 0 or done == total):
            progress_cb(done, total, str(records[opos].get("title", "")))

        old_id = str(records[opos]["id"])
        stored = ArticleRepository.get_precomputed_rows(db, old_id, limit=config.TOP_K)

        # 2. Seed the merged pool with O's current entries (reusing their rerank scores).
        cand_positions, rerank_scores, seen = [], [], set()
        for r in stored:
            p = id_to_pos.get(str(r["recommended_article_id"]))
            if p is None or p in seen:
                continue
            seen.add(p)
            cand_positions.append(p)
            rerank_scores.append(r.get("rerank_score"))

        # 3. Add the new neighbours (deduped); they still need an O-centric score.
        for npos in dict.fromkeys(new_neighbors):
            if npos != opos and npos not in seen:
                seen.add(npos)
                cand_positions.append(npos)
                rerank_scores.append(None)
        if not cand_positions:
            continue

        # 4. Score only the candidates that have no reusable score (new + legacy rows).
        missing = [i for i, s in enumerate(rerank_scores) if s is None]
        if missing:
            scored = reranker.score(
                records[opos]["text_rerank"],
                [records[cand_positions[i]]["text_rerank"] for i in missing],
                batch_size=32,
            )
            for k, i in enumerate(missing):
                rerank_scores[i] = float(scored[k])
        rerank_vec = np.asarray(rerank_scores, dtype=np.float32)

        # 5. Recompute BM25 + dense O-centrically for the merged candidate set.
        bm25_all = bm25.scores(_focused_query(records[opos], bm25))
        dense_all = emb @ emb[opos]
        bm25_c = np.array([bm25_all[p] for p in cand_positions])
        dense_c = np.array([dense_all[p] for p in cand_positions])
        bm25_max = bm25_c.max() if bm25_c.max() > 0 else 1.0
        dense_max = dense_c.max() if dense_c.max() > 0 else 1.0
        bm25_norm = bm25_c / bm25_max
        dense_norm = np.clip(dense_c / dense_max, 0.0, 1.0)
        relevance = 0.55 * rerank_vec + 0.30 * bm25_norm + 0.15 * dense_norm

        # 6. Boosts + diversity, then persist.
        boosts = _calculate_boosts(records[opos], [records[p] for p in cand_positions], max_views, max_date)
        order = _apply_mmr(
            cand_positions,
            relevance * boosts,
            emb[cand_positions],
            top_k=min(config.TOP_K, len(cand_positions)),
            lam=config.MMR_LAMBDA,
        )

        title_toks = tokenize(str(records[opos]["title"]))
        recs = []
        for rank, rel_pos in enumerate(order, start=1):
            c = cand_positions[rel_pos]
            recs.append({
                "recommended_article_id": str(records[c]["id"]),
                "rank": rank,
                "score": round(float(np.clip(relevance[rel_pos], 0.0, 1.0)), 4),
                "rerank_score": round(float(rerank_vec[rel_pos]), 6),
                "method": "neural_hybrid_rerank",
                "reason": _generate_reason(
                    records[opos], records[c], title_toks, bm25.docs[c], bm25,
                    dense_sim=float(dense_norm[rel_pos]),
                    rerank_score=float(rerank_vec[rel_pos]),
                ),
            })
        ArticleRepository.save_recommendations(db, old_id, recs, commit=False)
        refreshed.append({"id": old_id, "title": str(records[opos].get("title", ""))})
    return refreshed

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
    recompute_all_recommendations: bool = False,
):
    """
    CSV -> clean -> PostgreSQL -> embeddings -> full-corpus BM25+dense -> RRF -> rerank
        -> boosts -> MMR -> top-10 saved.

    Recommendation maintenance is INCREMENTAL by default:
      * the uploaded (new) articles are precomputed against the WHOLE database, and
      * existing articles that a new article is close to get a reverse-push update.

    Pass ``recompute_all_recommendations=True`` to rebuild every article's list
    from scratch (correct but slower — the "Option A" full rebuild).
    """
    db = SessionLocal()
    t_job = time.time()
    try:
        # Idempotent: ensures the pgvector extension, tables and the rerank_score
        # column all exist before we start writing (safe on a fresh database).
        init_db()
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

        # 5. Build a FULL-corpus index (already-stored articles + the new ones).
        #    New articles must be able to retrieve neighbours from the entire DB,
        #    not just from the batch that was uploaded this time.
        upd(message="STAGE:INDEX|Building full-corpus BM25 index...")
        full_records, full_emb, id_to_pos = _load_full_corpus(db)
        n_total = len(full_records)
        if n_total == 1:
            print("[Pipeline] Only one article in the corpus; no neighbours to precompute.")
        bm25 = BM25Index([r["text_bm25"] for r in full_records])

        # 6. Rerank candidates and save recommendations in batches
        upd(message="STAGE:RERANK|Loading cross-encoder (first run downloads ~1 GB, can take minutes)...")
        reranker = _get_reranker_model()

        max_views = int(max((r["page_views"] for r in full_records), default=0)) or 1
        max_date = max((r["date"] for r in full_records if pd.notna(r["date"])), default=None)
        n_retrieve = min(config.N_RETRIEVE, max(n_total - 1, 0))
        n_cand = min(config.N_CANDIDATES, max(n_total - 1, 0))

        # Which articles are "new" (i.e. arrived in THIS upload)?
        new_id_set = set(ids)
        if recompute_all_recommendations:
            target_positions = list(range(n_total))                       # Option A: full rebuild
        else:
            target_positions = [id_to_pos[i] for i in ids if i in id_to_pos]  # just this upload

        candidate_map: Dict[int, List[int]] = {}
        failed, t0 = 0, time.time()
        total_targets = max(len(target_positions), 1)

        for done, pos in enumerate(target_positions, start=1):
            try:
                recs, cands = _precompute_for_article(
                    pos, full_records, bm25, reranker, full_emb,
                    max_views, max_date, n_retrieve, n_cand,
                )
            except Exception as art_err:  # one bad article must not terminate the entire job
                failed += 1
                recs, cands = [], []
                print(f"[Pipeline] Skipped article {full_records[pos]['id']}: {art_err}")

            if recs:
                ArticleRepository.save_recommendations(
                    db, str(full_records[pos]["id"]), recs, commit=False
                )
            candidate_map[pos] = cands

            if done % config.PRECOMPUTE_COMMIT_EVERY == 0 or done == total_targets:
                per = (time.time() - t0) / done
                eta = per * (total_targets - done)
                # update_job commits and therefore commits pending recommendation writes too.
                # Reserve the last 5% of the bar for the reverse-update pass (step 6b).
                upd(
                    processed_articles=int(0.40 * n + 0.55 * n * done / total_targets),
                    message=(
                        f"STAGE:RERANK|Precomputed {done}/{total_targets} "
                        f"({per:.1f}s/article, ETA {_fmt_eta(eta)})"
                    ),
                )
        db.commit()

        # 6b. Reverse pass: let the new articles flow into the lists of the EXISTING
        #     articles they are close to (skipped during a full rebuild).
        refreshed_infos: List[Dict[str, Any]] = []
        if not recompute_all_recommendations and config.INCREMENTAL_RECOMMENDATIONS:
            upd(message="STAGE:REVERSE|Finding existing articles affected by this upload...")

            def _rev_progress(done, total, title):
                snippet = (title[:48] + "…") if len(title) > 48 else title
                upd(message=(
                    f"STAGE:REVERSE|Re-scoring existing articles ({done}/{total}) — "
                    f"checking “{snippet}”..."
                ))

            refreshed_infos = _reverse_push_for_old(
                db, target_positions, candidate_map, full_records, full_emb, bm25,
                reranker, max_views, max_date, new_id_set, id_to_pos,
                progress_cb=_rev_progress,
            )
            db.commit()

            count = len(refreshed_infos)
            upd(refreshed_articles=count)
            if refreshed_infos:
                titles = [a["title"] for a in refreshed_infos[:4] if a.get("title")]
                examples = ", ".join(f"“{t[:45]}”" for t in titles)
                more = f" (+{count - len(titles)} more)" if count > len(titles) else ""
                upd(message=(
                    f"STAGE:REVERSE|Updated precomputed recommendations for {count} "
                    f"existing article(s): {examples}{more}"
                ))
            else:
                upd(message="STAGE:REVERSE|No existing articles needed refreshing for this upload.")
            print(f"[Pipeline] Reverse pass refreshed {count} existing article(s).")

        # 7. Refresh live engine directly in this process (no HTTP request needed).
        upd(message="STAGE:INDEX|Refreshing live search engine...")
        try:
            from api.server import reload_engine_from_postgres
            reload_engine_from_postgres()
        except Exception as reload_err:
            print(f"[Pipeline] Engine reload skipped: {reload_err}")

        note = f" ({failed} skipped)" if failed else ""
        extra = ""
        if refreshed_infos:
            titles = [a["title"] for a in refreshed_infos[:4] if a.get("title")]
            examples = ", ".join(f"“{t[:45]}”" for t in titles)
            more = f" (+{len(refreshed_infos) - len(titles)} more)" if len(refreshed_infos) > len(titles) else ""
            extra = (
                f" Reverse-updated {len(refreshed_infos)} existing article(s): {examples}{more}."
            )
        upd(
            status="completed",
            processed_articles=n,
            refreshed_articles=len(refreshed_infos),
            message=f"STAGE:COMPLETE|Processed {n} articles{note} in {_fmt_eta(time.time() - t_job)}.{extra}",
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
