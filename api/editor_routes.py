"""
Editor and PostgreSQL API routes:
- Database health check & table setup
- CSV upload & background ingestion task triggering
- Ingestion progress polling
- Instant precomputed recommendations endpoint
"""
import os
import uuid
import shutil
import threading
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.connection import get_db, check_db_health, init_db
from db.models import Article, ArticleRecommendation, IngestionJob
from db.repository import ArticleRepository
from db.pipeline import run_ingestion_pipeline

router = APIRouter(prefix="/api/db", tags=["Database & Editor"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health")
def db_health():
    """Returns PostgreSQL connection status, pgvector extension readiness, and article counts."""
    health = check_db_health()
    if health["connected"]:
        try:
            from db.connection import SessionLocal
            with SessionLocal() as db:
                health["total_articles"] = ArticleRepository.count_articles(db)
                health["embedded_articles"] = ArticleRepository.count_embedded_articles(db)
                health["total_recommendations"] = ArticleRepository.count_total_recommendations(db)
        except Exception as e:
            health["stats_error"] = str(e)
    return health


@router.post("/init")
def trigger_db_init():
    """Initializes pgvector extension and creates all database tables."""
    success = init_db()
    if not success:
        raise HTTPException(status_code=500, detail="Database initialization failed. Check logs.")
    return {"status": "success", "message": "PostgreSQL tables and extensions verified and ready."}


MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB security limit


@router.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Accepts CSV upload with security validation (path sanitization & size limit),
    launches background pipeline (Cleaning -> Postgres -> Embeddings -> pgvector Top 50 -> Reranker Top 10).
    """
    # Guard against concurrent background jobs competing for CPU
    active_job = ArticleRepository.get_active_job(db)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"Ingestion job '{active_job.id}' ({active_job.filename}) is currently running. Please wait for it to finish."
        )

    safe_filename = Path(file.filename or "articles.csv").name
    if not safe_filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    saved_path = UPLOAD_DIR / f"{job_id}_{safe_filename}"

    # Safe chunked write with max size enforcement
    size = 0
    with open(saved_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB chunks
            size += len(chunk)
            if size > MAX_FILE_SIZE:
                buffer.close()
                saved_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 50MB.")
            buffer.write(chunk)

    ArticleRepository.create_job(db, job_id=job_id, filename=safe_filename)
    threading.Thread(
        target=run_ingestion_pipeline,
        args=(saved_path, job_id),
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "filename": safe_filename,
        "status": "processing",
        "message": "File validated and uploaded. Ingestion pipeline running in background."
    }



@router.post("/ingest-default")
def ingest_default_dataset(
    limit: Optional[int] = Query(None, description="Max rows to process (e.g. 100 for fast demo)"),
    db: Session = Depends(get_db)
):
    """
    One-click ingestion of the existing sample dataset into PostgreSQL.
    """
    active_job = ArticleRepository.get_active_job(db)
    if active_job:
        raise HTTPException(
            status_code=409,
            detail=f"Ingestion job '{active_job.id}' ({active_job.filename}) is currently running. Please wait for it to finish."
        )

    default_csv = Path(__file__).resolve().parent.parent / "data" / "articles_clean_train_80.csv"
    if not default_csv.exists():
        raise HTTPException(status_code=404, detail="Default dataset file not found on server.")

    job_id = f"job_demo_{uuid.uuid4().hex[:6]}"
    ArticleRepository.create_job(db, job_id=job_id, filename="articles_clean_train_80.csv")
    threading.Thread(
        target=run_ingestion_pipeline,
        args=(default_csv, job_id),
        kwargs={"limit": limit},
        daemon=True,
    ).start()

    return {
        "job_id": job_id,
        "filename": "articles_clean_train_80.csv",
        "status": "processing",
        "message": f"Started ingestion of sample dataset into PostgreSQL (limit={limit or 'all'})."
    }



@router.get("/jobs/{job_id}")
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Poll progress of an ingestion job for the Editor progress bar."""
    job = ArticleRepository.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@router.get("/jobs")
def list_jobs(limit: int = 10, db: Session = Depends(get_db)):
    """List recent ingestion jobs for the Editor dashboard."""
    jobs = ArticleRepository.list_recent_jobs(db, limit=limit)
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/recommendations/{article_id}")
def get_precomputed_recs(article_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """
    Ultra-fast sub-5ms endpoint: retrieves precomputed top-10 recommendations directly
    from PostgreSQL article_recommendations table.
    """
    recs = ArticleRepository.get_precomputed_recommendations(db, article_id, limit=limit)
    return {
        "article_id": article_id,
        "source": "postgresql_precomputed",
        "count": len(recs),
        "recommendations": recs
    }


@router.post("/clear")
def clear_database(db: Session = Depends(get_db)):
    """
    Clears all stored articles, precomputed recommendations, and ingestion jobs from PostgreSQL.
    Allows editor to reset and start completely fresh from scratch.
    """
    try:
        stats = ArticleRepository.clear_all_data(db)
        # Notify in-memory engine if running
        try:
            from api.server import reload_engine_from_postgres
            reload_engine_from_postgres()
        except Exception:
            pass

        return {
            "status": "success",
            "message": "Database cleared successfully. All tables reset.",
            "deleted": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@router.post("/reload")
def reload_engine():
    """Trigger reload of in-memory engine from PostgreSQL articles."""
    try:
        from api.server import reload_engine_from_postgres
        reload_engine_from_postgres()
        return {"status": "success", "message": "In-memory engine synchronized with PostgreSQL."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
