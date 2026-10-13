"""
FastAPI REST API — News Recommender Backend
Exposes all recommender functionality as JSON endpoints for the frontend.
Run:  python api/server.py
"""
import sys
import os
import math
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import config
from src.data_loader import load_articles
from src.recommender import METHODS, Recommender
from api.editor_routes import router as editor_router

from db.connection import SessionLocal, ensure_schema
from db.repository import ArticleRepository
from db.models import Article

# ─── App lifecycle and setup ──────────────────────────────────────────────────
_reload_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        ensure_schema()  # idempotent: adds new columns to existing tables if needed
    except Exception as e:
        print(f"[Startup] Schema check skipped: {e}")

    try:
        with SessionLocal() as db:
            closed = ArticleRepository.fail_orphaned_jobs(db)
            if closed:
                print(f"[Startup] Marked {closed} interrupted ingestion job(s) as failed.")
    except Exception as e:
        print(f"[Startup] Could not clean up old jobs: {e}")

    # Initial loading is heavy; don't block the event loop.
    await run_in_threadpool(reload_engine_from_postgres)
    yield


app = FastAPI(
    title="News Recommender API",
    description="News recommendation engine built on Indian Express articles.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Database and Editor Router
app.include_router(editor_router)

# ─── Globals & PostgreSQL Sync ────────────────────────────────────────────────
rec: Optional[Recommender] = None
articles: pd.DataFrame = pd.DataFrame()


def reload_engine_from_postgres():
    """Build the live search engine from PostgreSQL, reusing stored texts and vectors."""
    global rec, articles
    with _reload_lock:
        try:
            with SessionLocal() as db:
                if ArticleRepository.count_articles(db) == 0:
                    print("[INFO] PostgreSQL articles table is empty. Ready for Editor upload.")
                    articles, rec = pd.DataFrame(), None
                    return

                arts = db.query(Article).order_by(Article.date.desc().nulls_last()).all()
                rows, vecs, missing_vec = [], [], False
                for a in arts:
                    title, desc, body = str(a.title), str(a.description or ""), str(a.body or "")
                    try:
                        linked = json.loads(a.linked_ids) if a.linked_ids else []
                    except Exception:
                        linked = []

                    rows.append({
                        "id": str(a.id),
                        "title": title,
                        "description": desc,
                        "section": str(a.section or ""),
                        "subsection": str(a.subsection or ""),
                        "date": a.date,
                        "url": str(a.url or ""),
                        "image": str(a.image or ""),
                        "is_live": bool(a.is_live),
                        "body": body,
                        "pageViews": int(a.page_views or 0),
                        "uniqueUsers": int(a.unique_users or 0),
                        "linked_ids": linked,
                        # Reuse the exact text representations saved by ingestion.
                        "text_embed": a.text_embed or f"{title}. {desc}",
                        "text_bm25": a.text_bm25 or f"{title}. {title}. {title}. {desc}. {body}",
                        "text_rerank": a.text_rerank or f"{title}. {desc}.",
                    })
                    if a.embedding is None:
                        missing_vec = True
                    else:
                        vecs.append(a.embedding)

                df_db = pd.DataFrame(rows)
                df_db["date"] = pd.to_datetime(df_db["date"], utc=True, errors="coerce")
                vectors = None if missing_vec else np.asarray(vecs, dtype=np.float32)

                new_rec = Recommender(df_db, vectors=vectors, use_disk_cache=False)
                rec, articles = new_rec, new_rec.articles  # swap only after the new engine builds
                print(f"[READY] Engine initialized from PostgreSQL: {len(articles)} articles.")
        except Exception as e:
            print(f"[DB Warning] Could not load from PostgreSQL: {e}")
            articles, rec = pd.DataFrame(), None


# ─── Pydantic request/response models ──────────────────────────────────────
class RecommendRequest(BaseModel):
    query: Optional[str] = None
    article_id: Optional[str] = None
    top_k: int = config.TOP_K
    method: str = "hybrid+rerank"
    use_boosts: bool = True
    diversify: bool = True
    section: Optional[str] = None


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _clean_str(val) -> str:
    if val is None or pd.isna(val):
        return ""
    return str(val).strip()


def _clean_float(val, default: float = 0.0) -> float:
    try:
        if val is None or pd.isna(val) or math.isnan(float(val)):
            return default
        return round(float(val), 4)
    except (TypeError, ValueError):
        return default


def _article_dict(a: dict) -> dict:
    """Serialise one article dict to a safe JSON-able dict with no NaNs."""
    return {
        "id":         _clean_str(a.get("id")),
        "title":      _clean_str(a.get("title")),
        "snippet":    _clean_str(a.get("snippet") or a.get("description")),
        "section":    _clean_str(a.get("section")),
        "subsection": _clean_str(a.get("subsection")),
        "date":       _clean_str(a.get("date")),
        "url":        _clean_str(a.get("url")),
        "image":      _clean_str(a.get("image")),
        "is_live":    bool(a.get("is_live", False)),
    }


def _row_to_dict(row) -> dict:
    """Convert a DataFrame row to a serialisable article dict."""
    desc = _clean_str(row.get("description", ""))
    body = _clean_str(row.get("body", ""))
    snippet = desc if desc else body[:200]
    
    date_val = row.get("date")
    if hasattr(date_val, "strftime") and pd.notna(date_val):
        date_str = date_val.strftime("%d %b %Y")
    else:
        date_str = _clean_str(date_val)

    return {
        "id":         _clean_str(row.get("id")),
        "title":      _clean_str(row.get("title")),
        "snippet":    snippet,
        "section":    _clean_str(row.get("section")),
        "subsection": _clean_str(row.get("subsection")),
        "date":       date_str,
        "url":        _clean_str(row.get("url")),
        "image":      _clean_str(row.get("image")),
        "is_live":    bool(row.get("is_live", False)),
    }


def _result_json(result, method: str) -> dict:
    return {
        "method":         method,
        "engine":         rec.engine_summary if rec else "PostgreSQL Engine",
        # "postgresql" recommendations are served from the precomputed table (see
        # /api/recommend); anything that reaches here was computed live in memory.
        "source":         "live",
        "low_confidence": bool(result.low_confidence),
        "best_relevance": _clean_float(result.best_relevance),
        "notes":          list(result.notes) if result.notes else [],
        "recommendations": [
            {
                "article": _article_dict(r.article),
                "score":   _clean_float(r.score),
                "reason":  _clean_str(r.reason),
            }
            for r in result.recommendations
        ],
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/api/status")
def status():
    """Engine health-check + metadata directly from PostgreSQL."""
    total = int(len(articles)) if not articles.empty else 0
    secs = sorted(articles["section"].dropna().unique().tolist()) if not articles.empty else []
    return {
        "engine":         rec.engine_summary if rec else "PostgreSQL Storage (Awaiting Dataset Ingestion)",
        "total_articles": total,
        "database_empty": total == 0,
        "methods":        METHODS,
        "sections":       secs,
        "notes":          rec.notes if rec else ["PostgreSQL is active. Upload a dataset in Editor Studio to index articles."],
        "config": {
            "top_k":        config.TOP_K,
            "n_retrieve":   config.N_RETRIEVE,
            "n_candidates": config.N_CANDIDATES,
            "mmr_lambda":   config.MMR_LAMBDA,
            "use_neural":   config.USE_NEURAL_MODELS,
        },
    }


@app.get("/api/articles")
def get_articles(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    section: str = Query(""),
    sort: str = Query("date"),
):
    """Return a paginated list of articles from PostgreSQL."""
    try:
        with SessionLocal() as db:
            total = ArticleRepository.count_articles(db)
            if total == 0:
                return {
                    "total": 0,
                    "page": page,
                    "per_page": per_page,
                    "articles": [],
                    "database_empty": True
                }
            start = (page - 1) * per_page
            arts = ArticleRepository.list_articles(
                db, limit=per_page, offset=start, section=section or None
            )
            return {
                "total": total,
                "page": page,
                "per_page": per_page,
                "articles": [a.to_dict() for a in arts],
                "database_empty": False
            }
    except Exception as e:
        print(f"Error listing articles: {e}")
        return {"total": 0, "page": page, "per_page": per_page, "articles": [], "error": str(e)}


@app.get("/api/articles/trending")
def trending(n: int = Query(6, ge=1, le=20)):
    """Return top-N most recent articles directly from PostgreSQL."""
    try:
        with SessionLocal() as db:
            arts = ArticleRepository.list_articles(db, limit=n)
            return {"articles": [a.to_dict() for a in arts]}
    except Exception as e:
        return {"articles": []}


@app.get("/api/articles/{article_id}")
def get_article(article_id: str):
    """Fetch a single article by ID directly from PostgreSQL."""
    try:
        with SessionLocal() as db:
            art = ArticleRepository.get_article(db, str(article_id))
            if not art:
                raise HTTPException(status_code=404, detail="Article not found")
            return art.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/recommend")
def recommend(body: RecommendRequest):
    if not body.query and not body.article_id:
        raise HTTPException(
            status_code=400,
            detail="Please provide either 'query' (for search) or 'article_id' (for similar articles).",
        )

    # Precomputed recommendations only represent the default similar-article request.
    use_precomputed = bool(
        body.article_id
        and not body.query
        and not body.section
        and body.method == "hybrid+rerank"
        and body.use_boosts
        and body.diversify
        and body.top_k <= config.TOP_K
    )
    if use_precomputed:
        try:
            with SessionLocal() as db:
                precomputed = ArticleRepository.get_precomputed_recommendations(
                    db, str(body.article_id), limit=body.top_k
                )
            if precomputed:
                return {
                    "method": "postgresql_precomputed_rerank",
                    "engine": "PostgreSQL pgvector + Precomputed Cross-Encoder",
                    "source": "postgresql",
                    "low_confidence": False,
                    "best_relevance": precomputed[0].get("score", 0.0),
                    "notes": ["⚡ Served directly from PostgreSQL precomputed table (sub-5ms)"],
                    "recommendations": [{
                        "article": _article_dict(r),
                        "score": r.get("score", 0.0),
                        "reason": r.get("reason") or f"Rank #{r.get('rank', 1)} precomputed",
                    } for r in precomputed],
                }
        except Exception as e:
            print(f"Error checking precomputed recommendations: {e}")

    if rec is None:
        raise HTTPException(
            status_code=400,
            detail="Database has no articles indexed yet. Please upload a dataset in the Editor Studio.",
        )

    try:
        result = rec.recommend(
            query=body.query,
            article_id=str(body.article_id) if body.article_id else None,
            top_k=body.top_k,
            method=body.method,
            use_boosts=body.use_boosts,
            diversify=body.diversify,
            section=body.section,
        )
        return _result_json(result, body.method)
    except KeyError:
        raise HTTPException(status_code=404, detail="Article not found in the search index.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sections")
def sections():
    """Return all distinct section names directly from PostgreSQL."""
    try:
        with SessionLocal() as db:
            secs = ArticleRepository.get_sections(db)
            return {"sections": secs}
    except Exception:
        secs = sorted(articles["section"].dropna().unique().tolist()) if not articles.empty and "section" in articles.columns else []
        return {"sections": secs}


# ─── Serve Frontend Static Files ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/")
def serve_index():
    """Serve the frontend index.html."""
    return FileResponse(FRONTEND_DIR / "index.html")


def _icon_response():
    """Serve the SVG app icon, or a silent 204 if it is missing."""
    icon = FRONTEND_DIR / "favicon.svg"
    if icon.exists():
        return FileResponse(icon, media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/favicon.svg", include_in_schema=False)
def serve_favicon_svg():
    return _icon_response()


@app.get("/favicon.ico", include_in_schema=False)
def serve_favicon():
    """
    Browsers request /favicon.ico automatically. Without a handler this logs a
    noisy 404 on every page load, so we answer it with the SVG icon (or a 204).
    """
    return _icon_response()


# Mount static dirs for CSS and JS
app.mount("/css", StaticFiles(directory=FRONTEND_DIR / "css"), name="css")
app.mount("/js",  StaticFiles(directory=FRONTEND_DIR / "js"),  name="js")


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=5000, reload=False)
