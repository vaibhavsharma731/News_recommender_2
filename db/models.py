"""
SQLAlchemy ORM models for news articles, precomputed recommendations, and ingestion jobs.
"""
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from .connection import Base


class Article(Base):
    """
    Core storage for news articles and their neural vector embeddings.
    """
    __tablename__ = "articles"

    id = Column(String(100), primary_key=True, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    body = Column(Text, nullable=True)
    section = Column(String(100), index=True, nullable=True)
    subsection = Column(String(100), nullable=True)
    date = Column(DateTime(timezone=True), index=True, nullable=True)
    url = Column(Text, nullable=True)
    image = Column(Text, nullable=True)
    is_live = Column(Boolean, default=False)
    page_views = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    linked_ids = Column(Text, nullable=True)    # JSON string of editor "Also Read" article IDs
    text_embed = Column(Text, nullable=True)    # Preprocessed text passed to embedding model
    text_bm25 = Column(Text, nullable=True)     # Weighted text for BM25 keyword matching
    text_rerank = Column(Text, nullable=True)   # Truncated lead text for Cross-Encoder reranker

    # BAAI/bge-base-en-v1.5 produces 768-dimensional dense vector embeddings
    embedding = Column(Vector(768), nullable=True)

    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Precomputed recommendations where this article is the source
    recommendations = relationship(
        "ArticleRecommendation",
        foreign_keys="ArticleRecommendation.article_id",
        back_populates="source_article",
        cascade="all, delete-orphan",
        order_by="ArticleRecommendation.rank",
    )

    def to_dict(self, include_embedding: bool = False) -> dict:
        description = self.description or ""
        body = self.body or ""
        if self.date:
            try:
                date_str = self.date.strftime("%d %b %Y")
            except Exception:
                date_str = str(self.date)
        else:
            date_str = ""
        data = {
            "id": self.id,
            "title": self.title,
            "description": description,
            "snippet": (description or body)[:200],
            "section": self.section or "",
            "subsection": self.subsection or "",
            "date": date_str,
            "url": self.url or "",
            "image": self.image or "",
            "is_live": bool(self.is_live),
            "pageViews": self.page_views or 0,
            "uniqueUsers": self.unique_users or 0,
            "has_embedding": self.embedding is not None,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }
        if include_embedding and self.embedding is not None:
            data["embedding"] = list(self.embedding)
        return data


class ArticleRecommendation(Base):
    """
    Precomputed recommendations for fast sub-5ms reader lookups.
    Stores the Top 10 recommendations for each article generated via Vector Search + Reranker.
    """
    __tablename__ = "article_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    article_id = Column(
        String(100),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    recommended_article_id = Column(
        String(100),
        ForeignKey("articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    rank = Column(Integer, nullable=False)   # 1 to 10
    score = Column(Float, nullable=True)     # Reranker probability score (0.0 - 1.0)
    method = Column(String(50), default="neural_hybrid_rerank")
    reason = Column(Text, nullable=True)     # Human-readable explanation why recommended
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    source_article = relationship("Article", foreign_keys=[article_id], back_populates="recommendations")
    recommended_article = relationship("Article", foreign_keys=[recommended_article_id])

    __table_args__ = (
        UniqueConstraint("article_id", "rank", name="uq_source_article_rank"),
        Index("idx_rec_source_rank", "article_id", "rank"),
    )


class IngestionJob(Base):
    """
    Tracks CSV upload and recommendation precomputation jobs for the Editor side.
    """
    __tablename__ = "ingestion_jobs"

    id = Column(String(64), primary_key=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    total_articles = Column(Integer, default=0)
    processed_articles = Column(Integer, default=0)
    message = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        progress = 0.0
        if self.total_articles and self.total_articles > 0:
            progress = round((self.processed_articles / self.total_articles) * 100, 1)

        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "total_articles": self.total_articles,
            "processed_articles": self.processed_articles,
            "progress_percent": progress,
            "message": self.message or "",
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
        }
