"""
Database Repository providing clean CRUD, pgvector similarity queries,
recommendation persistence, and ingestion job management.
"""
from typing import List, Optional, Dict, Any
import json
from sqlalchemy.orm import Session
from sqlalchemy import desc, func, or_
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
        return query.order_by(desc(Article.date)).offset(offset).limit(limit).all()

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
        recommendations: List[Dict[str, Any]]
    ):
        """
        Atomically saves precomputed recommendations for an article.
        recommendations: list of dicts with keys: 'recommended_article_id', 'rank', 'score', 'method', 'reason'
        """
        # Remove existing recommendations for this article
        db.query(ArticleRecommendation).filter(
            ArticleRecommendation.article_id == article_id
        ).delete()

        # Add new recommendations
        for item in recommendations:
            rec = ArticleRecommendation(
                article_id=article_id,
                recommended_article_id=item["recommended_article_id"],
                rank=item["rank"],
                score=item.get("score", 0.0),
                method=item.get("method", "neural_hybrid_rerank"),
                reason=item.get("reason", "High topic and semantic relevance"),
            )
            db.add(rec)
        db.commit()

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
