"""
Database package for PostgreSQL and pgvector storage.
Provides modular connection, schema models, repository CRUD, and background ingestion pipeline.
"""
from .connection import get_db, engine, SessionLocal, init_db, check_db_health, ensure_schema
from .models import Article, ArticleRecommendation, IngestionJob
from .repository import ArticleRepository

__all__ = [
    "get_db",
    "engine",
    "SessionLocal",
    "init_db",
    "check_db_health",
    "ensure_schema",
    "Article",
    "ArticleRecommendation",
    "IngestionJob",
    "ArticleRepository",
]
