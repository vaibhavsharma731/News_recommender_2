"""
Diagnostic & Setup Script:
Verifies PostgreSQL connection, checks/creates pgvector extension, and initializes database tables.
Run:
    python test_postgres_setup.py
"""
import os
import sys

# Configure UTF-8 encoding on Windows terminal safely
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

from db.connection import (
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DATABASE_URL,
    check_db_health,
    init_db
)


def print_banner(text):
    print("\n" + "=" * 65)
    print(f" {text}")
    print("=" * 65)


def run_checks():
    print_banner("1. Checking PostgreSQL Configuration (.env)")
    print(f"  Host:     {DB_HOST}")
    print(f"  Port:     {DB_PORT}")
    print(f"  User:     {DB_USER}")
    print(f"  Database: {DB_NAME}")
    masked_pw = '*' * len(DB_PASSWORD) if DB_PASSWORD else '(none)'
    print(f"  URL:      postgresql://{DB_USER}:{masked_pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    print_banner("2. Testing PostgreSQL Connectivity & pgvector Extension")
    health = check_db_health()

    if not health["connected"]:
        print("  [FAIL] PostgreSQL Connection Failed!")
        print(f"  Details: {health['error']}")
        print("\n  👉 Action Required:")
        print("  1. Verify PostgreSQL service is running on your machine.")
        print("  2. In your .env file, ensure POSTGRES_PASSWORD matches your postgres password.")
        print(f"  3. In pgAdmin, ensure database '{DB_NAME}' exists (or run: CREATE DATABASE {DB_NAME};).")
        return False

    print("  [OK] Successfully connected to PostgreSQL!")

    if health["pgvector_ready"]:
        print("  [OK] pgvector extension is INSTALLED and ACTIVE!")
    else:
        print("  [WARNING] pgvector extension is NOT YET INSTALLED in PostgreSQL.")
        print("\n  👉 To install pgvector on Windows:")
        print("     Option A (Precompiled binary for Windows):")
        print("       1. Go to: https://github.com/pgvector/pgvector/releases")
        print("       2. Download the zip matching your PostgreSQL version (e.g. pg16, pg17).")
        print("       3. Copy vector.dll to C:\\Program Files\\PostgreSQL\\<version>\\lib")
        print("       4. Copy vector.control and vector--*.sql to C:\\Program Files\\PostgreSQL\\<version>\\share\\extension")
        print("       5. In pgAdmin query tool, run: CREATE EXTENSION vector;")
        print("     Option B (Using C Compiler / MSVC):")
        print("       set PGROOT=C:\\Program Files\\PostgreSQL\\<version>")
        print("       nmake /F Makefile.win")
        print("       nmake /F Makefile.win install")
        return False

    print_banner("3. Initializing Database Tables")
    success = init_db()
    if success:
        print("  [OK] Tables created / verified successfully:")
        print("       - articles (with 768d vector column)")
        print("       - article_recommendations (precomputed top-10)")
        print("       - ingestion_jobs (editor tracking)")
        print_banner("All checks passed! Your database is ready for the news recommender.")
        return True
    else:
        print("  [FAIL] Failed to create tables. Review logs above.")
        return False


if __name__ == "__main__":
    run_checks()
