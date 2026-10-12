"""
Central settings. Every number you may want to tune lives here,
so you never have to hunt through the code.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# Dataset sources to load and merge:
# Loads both articles.csv and articles_2.csv by default if present.
# Can also be customized via DATA_CSV environment variable (comma-separated).
_env_csvs = os.getenv("DATA_CSV", "")
if _env_csvs:
    DATA_PATHS = [DATA_DIR / name.strip() for name in _env_csvs.split(",") if name.strip()]
else:
    _default_files = ["articles.csv", "articles_2.csv", "articles_clean_train_80.csv"]
    DATA_PATHS = [DATA_DIR / f for f in _default_files if (DATA_DIR / f).exists()]
    if not DATA_PATHS:
        DATA_PATHS = [p for p in DATA_DIR.glob("*.csv") if "uploads" not in str(p)]
    if not DATA_PATHS:
        DATA_PATHS = [DATA_DIR / "articles.csv"]

DATA_PATH = DATA_PATHS[0]                    # primary path for backward compatibility
CACHE_DIR = ROOT / "data" / "cache"          # embeddings are saved here after the first run

# --- Environment & HuggingFace authentication ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Token is now loaded from .env if present.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Set the environment variable NEWSREC_NEURAL=0 to skip them and use the light TF-IDF mode.
USE_NEURAL_MODELS = os.getenv("NEWSREC_NEURAL", "1") == "1"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "BAAI/bge-reranker-base"
# bge models work better when a SHORT search query gets this prefix (articles do not need it)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SHORT_QUERY_WORDS = 12                        # fewer words than this = "short query"

# --- How much of each article is used ---
BODY_WORDS_EMBED = 350      # first N body words (after title + description) for embeddings
BODY_WORDS_RERANK = 150     # first N body words for cross-encoder reranking
BOILERPLATE_MIN_DOCS = 4    # a sentence repeated in >= N articles is treated as boilerplate

# --- Retrieval pipeline ---
N_RETRIEVE = 50             # candidates taken from each search (BM25, semantic)
N_CANDIDATES = 20           # candidates kept after fusion, sent to the reranker
RRF_K = 60                  # standard Reciprocal Rank Fusion constant
TOP_K = 10                  # default number of recommendations

# --- Diversity (MMR): 1.0 = pure relevance, lower = more variety ---
MMR_LAMBDA = 0.75

# --- Personalization Settings ---
MAX_USER_HISTORY = 10       # track last N read articles
USER_PROFILE_WEIGHT = 0.35  # blend weight for user history profile vector

# --- News Metadata Boosts (Calibrated Multipliers) ---
SECTION_BOOST_MULT = 1.08    # 8% multiplicative boost for same section
SUBSECTION_BOOST_MULT = 1.05 # 5% multiplicative boost for same subsection
POPULARITY_BOOST_MULT = 1.06 # up to 6% boost for popular articles (log-scaled pageViews)

# --- Dynamic Recency Half-lives (in days) ---
SECTION_HALF_LIVES = {
    "live_blog": 0.25,      # 6 hours
    "sports": 1.0,          # 1 day
    "entertainment": 2.0,   # 2 days
    "business": 3.0,        # 3 days
    "brand-story": 4.0,     # 4 days
    "default": 2.0
}
RECENCY_MAX_BOOST_MULT = 1.10 # up to 10% boost for fresh articles

# --- "No good match" detection ---
# If the best relevance score is below this, the UI says "no close match".
# These are starting points: run `python evaluate.py` and adjust them to your data.
MIN_RELEVANCE = {
    "reranker": 0.05,   # reranker probability (0-1)
    "dense": 0.50,      # cosine similarity of bge embeddings
    "tfidf": 0.08,      # cosine similarity of TF-IDF vectors (light mode)
}

