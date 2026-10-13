"""
Database connection, session management, and extension initialization.
"""
import os
import sys
from typing import Dict, Any, Generator
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, ProgrammingError
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Read Postgres connection details from environment variables
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "news_recommender")

# Construct URL (or use direct DATABASE_URL if provided)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLAlchemy Base and Session Setup
Base = declarative_base()

try:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # checks liveness before giving connections
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    # Try fallback to standard postgresql://
    try:
        alt_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(alt_url, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e2:
        engine = None
        SessionLocal = None
        print(f"[DB Warning] Could not initialize engine: {e2}")



def get_db() -> Generator:
    """FastAPI dependency for obtaining a database session."""
    if SessionLocal is None:
        raise RuntimeError("Database engine is not configured.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db_health() -> Dict[str, Any]:
    """
    Check if PostgreSQL is reachable and whether the pgvector extension is installed.
    """
    health = {
        "connected": False,
        "database": DB_NAME,
        "host": DB_HOST,
        "port": DB_PORT,
        "user": DB_USER,
        "pgvector_ready": False,
        "tables_ready": False,
        "error": None,
    }
    if not engine:
        health["error"] = "SQLAlchemy engine not initialized"
        return health

    try:
        with engine.connect() as conn:
            # 1. Test basic connectivity
            conn.execute(text("SELECT 1"))
            health["connected"] = True

            # 2. Check if pgvector extension is available or installed
            try:
                result = conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                ).fetchone()
                if result:
                    health["pgvector_ready"] = True
                else:
                    # Try to create extension if user has superuser privileges
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    health["pgvector_ready"] = True
            except Exception as ext_err:
                conn.rollback()
                health["pgvector_ready"] = False
                health["vector_warning"] = (
                    f"pgvector extension is not installed in PostgreSQL: {str(ext_err)}"
                )

            # 3. Check if articles table exists
            check_table = conn.execute(
                text(
                    "SELECT to_regclass('public.articles')"
                )
            ).scalar()
            health["tables_ready"] = bool(check_table)

    except OperationalError as op_err:
        health["error"] = f"Cannot connect to PostgreSQL at {DB_HOST}:{DB_PORT}. Is PostgreSQL running? Error: {op_err.orig if hasattr(op_err, 'orig') else str(op_err)}"
    except Exception as err:
        health["error"] = str(err)

    return health


def ensure_schema() -> bool:
    """
    Idempotent, lightweight migrations for columns added after the first release.

    ``Base.metadata.create_all`` only creates *missing tables*; it never adds a
    column to a table that already exists. So new columns must be added here with
    ``ADD COLUMN IF NOT EXISTS`` (safe to run on every startup / ingestion).
    """
    if not engine:
        return False
    try:
        with engine.connect() as conn:
            # Store the raw cross-encoder score per recommendation pair so the
            # incremental reverse-push can compare fairly without re-scoring old lists.
            conn.execute(text(
                "ALTER TABLE article_recommendations "
                "ADD COLUMN IF NOT EXISTS rerank_score DOUBLE PRECISION"
            ))
            # Track how many EXISTING articles each upload refreshed via reverse-push,
            # so the Editor dashboard can show per-upload history.
            conn.execute(text(
                "ALTER TABLE ingestion_jobs "
                "ADD COLUMN IF NOT EXISTS refreshed_articles INTEGER DEFAULT 0"
            ))
            conn.commit()
        return True
    except Exception as e:
        print(f"[DB Schema] Could not ensure schema: {e}")
        return False


def init_db():
    """
    Ensures the pgvector extension is created and all declared tables exist.
    """
    health = check_db_health()
    if not health["connected"]:
        print(f"[DB Init Error] Database connection failed: {health['error']}")
        return False

    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            print("[DB Init] 'vector' extension verified/created.")
        except Exception as e:
            print(f"[DB Init Notice] Note on pgvector: {e}")

    # Import models so Base has them registered before create_all
    from . import models  # noqa
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    print("[DB Init] All database tables created successfully.")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing PostgreSQL Connection & pgvector Setup...")
    print("=" * 60)
    res = check_db_health()
    for k, v in res.items():
        print(f"{k.ljust(18)}: {v}")
    print("=" * 60)
    if res["connected"]:
        print("Running table initialization...")
        init_db()
        print("Done!")
    else:
        print("Please check your .env settings or make sure PostgreSQL is running.")
