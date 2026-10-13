"""
Database Repository providing clean CRUD, pgvector similarity queries,
recommendation persistence, and ingestion job management.
"""
from typing import List, Optional, Dict, Any
import json
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from pgvector.sqlalchemy import Vector

from .models import Article, ArticleRecommendation, IngestionJob


class ArticleRepository:
    """Encapsulates all PostgreSQL + pgvector queries."""

    # ──────────────────────────────────────────────────────────────────────────
    # Article CRUD
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_article(db: Session, article_id: str) -> Optional[Article]:
        return db.query(Article).filter(Article.id == article_id).first()

    @staticmethod
    def count_articles(db: Session) -> int:
        return db.query(func.count(Article.id)).scalar() or 0

    @staticmethod
    def count_embedded_articles(db: Session) -> int:
        return db.query(func.count(Article.id)).filter(Article.embedding.isnot(None)).scalar() or 0

    @staticmethod
    def list_articles(
        db: Session,
        limit: int = 50,
        offset: int = 0,
        section: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[Article]:
        query = db.query(Article)
        if section:
            query = query.filter(Article.section.ilike(f"%{section}%"))
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    Article.title.ilike(search_pattern),
                    Article.description.ilike(search_pattern)
                )
            )
        return query.order_by(desc(Article.date).nulls_last()).offset(offset).limit(limit).all()

    @staticmethod
    def get_sections(db: Session) -> List[str]:
        results = db.query(Article.section).distinct().filter(Article.section.isnot(None)).all()
        return sorted([r[0] for r in results if r[0]])

    @staticmethod
    def upsert_article(db: Session, data: dict) -> Article:
        existing = db.query(Article).filter(Article.id == data["id"]).first()
        if existing:
            for k, v in data.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            return existing
        else:
            new_art = Article(**data)
            db.add(new_art)
            return new_art

    @staticmethod
    def bulk_upsert_articles(db: Session, articles_data: List[dict]):
        for item in articles_data:
            ArticleRepository.upsert_article(db, item)
        db.commit()

    @staticmethod
    def bulk_upsert_articles_fast(db: Session, rows: List[dict], chunk: int = 500):
        """One INSERT ... ON CONFLICT per chunk instead of a SELECT per article."""
        if not rows:
            return
        update_cols = [column for column in rows[0].keys() if column != "id"]
        for i in range(0, len(rows), chunk):
            stmt = pg_insert(Article).values(rows[i:i + chunk])
            set_ = {column: stmt.excluded[column] for column in update_cols}
            set_["updated_at"] = func.now()
            db.execute(stmt.on_conflict_do_update(index_elements=[Article.id], set_=set_))
        db.commit()

    @staticmethod
    def bulk_set_embeddings(db: Session, ids: List[str], vectors, chunk: int = 500):
        for i in range(0, len(ids), chunk):
            db.bulk_update_mappings(Article, [
                {"id": article_id, "embedding": vector.tolist()}
                for article_id, vector in zip(ids[i:i + chunk], vectors[i:i + chunk])
            ])
        db.commit()

    # ──────────────────────────────────────────────────────────────────────────
    # pgvector Similarity Search
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def search_similar_by_vector(
        db: Session,
        query_vector: List[float],
        top_k: int = 50,
        exclude_id: Optional[str] = None,
        section: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most similar articles to query_vector using pgvector's
        cosine distance operator (<=>).
        Cosine similarity = 1 - cosine_distance
        """
        query = db.query(
            Article,
            Article.embedding.cosine_distance(query_vector).label("distance")
        ).filter(Article.embedding.isnot(None))

        if exclude_id:
            query = query.filter(Article.id != exclude_id)
        if section:
            query = query.filter(Article.section.ilike(f"%{section}%"))

        results = query.order_by("distance").limit(top_k).all()

        formatted = []
        for article, distance in results:
            sim = 1.0 - float(distance) if distance is not None else 0.0
            formatted.append({
                "article": article,
                "similarity": round(max(0.0, sim), 4),
            })
        return formatted

    # ──────────────────────────────────────────────────────────────────────────
    # Precomputed Recommendations
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def save_recommendations(
        db: Session,
        article_id: str,
        recommendations: List[Dict[str, Any]],
        commit: bool = True,
    ):
        """
        Atomically saves precomputed recommendations for an article.
        recommendations: list of dicts with keys:
            'recommended_article_id', 'rank', 'score', 'rerank_score', 'method', 'reason'
        'rerank_score' is the raw cross-encoder probability for this (source -> recommended)
        pair, computed with the source article as the query. It is used by the incremental
        reverse-push to compare a new article against an existing article's stored entries.
        """
        db.query(ArticleRecommendation).filter(
            ArticleRecommendation.article_id == article_id
        ).delete(synchronize_session=False)
        db.flush()

        for item in recommendations:
            rec = ArticleRecommendation(
                article_id=article_id,
                recommended_article_id=item["recommended_article_id"],
                rank=item["rank"],
                score=item.get("score", 0.0),
                rerank_score=item.get("rerank_score"),
                method=item.get("method", "neural_hybrid_rerank"),
                reason=item.get("reason", "High topic and semantic relevance"),
            )
            db.add(rec)
        if commit:
            db.commit()

    @staticmethod
    def get_precomputed_rows(
        db: Session,
        article_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Lightweight read of an article's own stored recommendations (no join).

        Returns dicts with the ids, scores AND the raw rerank_score, which the
        incremental reverse-push needs in order to merge new candidates into an
        existing article's list.
        """
        rows = db.query(ArticleRecommendation).filter(
            ArticleRecommendation.article_id == article_id
        ).order_by(ArticleRecommendation.rank).limit(limit).all()
        return [{
            "recommended_article_id": r.recommended_article_id,
            "rank": r.rank,
            "score": r.score,
            "rerank_score": r.rerank_score,
            "method": r.method,
            "reason": r.reason,
        } for r in rows]

    @staticmethod
    def get_precomputed_recommendations(
        db: Session,
        article_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Fast lookup: Retrieves precomputed top recommendations for an article
        in sub-5ms by joining with the articles table.
        """
        recs = db.query(ArticleRecommendation, Article).join(
            Article, ArticleRecommendation.recommended_article_id == Article.id
        ).filter(
            ArticleRecommendation.article_id == article_id
        ).order_by(
            ArticleRecommendation.rank
        ).limit(limit).all()

        results = []
        for rec, art in recs:
            art_dict = art.to_dict()
            art_dict["rank"] = rec.rank
            art_dict["score"] = rec.score
            art_dict["rerank_score"] = rec.rerank_score
            art_dict["method"] = rec.method
            art_dict["reason"] = rec.reason or "High topic and semantic relevance"
            results.append(art_dict)
        return results

    @staticmethod
    def count_total_recommendations(db: Session) -> int:
        return db.query(func.count(ArticleRecommendation.id)).scalar() or 0

    @staticmethod
    def clear_all_data(db: Session) -> Dict[str, int]:
        """
        Clears all articles, precomputed recommendations, and ingestion jobs from PostgreSQL.
        """
        n_recs = db.query(ArticleRecommendation).delete()
        n_articles = db.query(Article).delete()
        n_jobs = db.query(IngestionJob).delete()
        db.commit()
        return {
            "articles_deleted": n_articles,
            "recommendations_deleted": n_recs,
            "jobs_deleted": n_jobs
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Ingestion Jobs (Editor Side Tracking)
    # ──────────────────────────────────────────────────────────────────────────
    @staticmethod
    def create_job(db: Session, job_id: str, filename: str, total_articles: int = 0) -> IngestionJob:
        job = IngestionJob(
            id=job_id,
            filename=filename,
            total_articles=total_articles,
            processed_articles=0,
            status="pending",
            message="Job created, awaiting worker..."
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        return job

    @staticmethod
    def update_job(
        db: Session,
        job_id: str,
        status: Optional[str] = None,
        total_articles: Optional[int] = None,
        processed_articles: Optional[int] = None,
        refreshed_articles: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None
    ) -> Optional[IngestionJob]:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if not job:
            return None
        if status:
            job.status = status
        if total_articles is not None:
            job.total_articles = total_articles
        if processed_articles is not None:
            job.processed_articles = processed_articles
        if refreshed_articles is not None:
            job.refreshed_articles = refreshed_articles
        if message:
            job.message = message
        if error:
            job.error = error
        db.commit()
        db.refresh(job)
        return job


    @staticmethod
    def get_job(db: Session, job_id: str) -> Optional[IngestionJob]:
        return db.query(IngestionJob).filter(IngestionJob.id == job_id).first()

    @staticmethod
    def list_recent_jobs(db: Session, limit: int = 10) -> List[IngestionJob]:
        return db.query(IngestionJob).order_by(desc(IngestionJob.created_at)).limit(limit).all()

    @staticmethod
    def get_ingestion_summary(db: Session) -> Dict[str, Any]:
        """Aggregate upload history: how many jobs and how many articles/recommendations."""
        total_jobs = db.query(func.count(IngestionJob.id)).scalar() or 0
        completed = db.query(func.count(IngestionJob.id)).filter(
            IngestionJob.status == "completed"
        ).scalar() or 0
        failed = db.query(func.count(IngestionJob.id)).filter(
            IngestionJob.status == "failed"
        ).scalar() or 0
        articles_ingested = db.query(func.coalesce(func.sum(IngestionJob.total_articles), 0)).filter(
            IngestionJob.status == "completed"
        ).scalar() or 0
        refreshed_total = db.query(func.coalesce(func.sum(IngestionJob.refreshed_articles), 0)).filter(
            IngestionJob.status == "completed"
        ).scalar() or 0
        return {
            "total_jobs": int(total_jobs),
            "completed_jobs": int(completed),
            "failed_jobs": int(failed),
            "articles_ingested": int(articles_ingested),
            "refreshed_articles": int(refreshed_total),
        }

    @staticmethod
    def get_active_job(db: Session) -> Optional[IngestionJob]:
        return (
            db.query(IngestionJob)
            .filter(IngestionJob.status.in_(["processing", "pending"]))
            .order_by(desc(IngestionJob.created_at))
            .first()
        )

    @staticmethod
    def fail_orphaned_jobs(db: Session) -> int:
        """Mark jobs left in pending/processing after a server restart as failed."""
        jobs = (
            db.query(IngestionJob)
            .filter(IngestionJob.status.in_(["processing", "pending"]))
            .all()
        )
        for job in jobs:
            job.status = "failed"
            job.error = "Interrupted because the API server restarted."
            job.message = "Processing encountered an error"
        db.commit()
        return len(jobs)

    @staticmethod
    def get_ids_with_recommendations(db: Session) -> set:
        rows = db.query(ArticleRecommendation.article_id).distinct().all()
        return {row[0] for row in rows}
