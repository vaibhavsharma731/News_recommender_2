"""
Step 2 - The search engines.

  BM25Index    keyword search (great for names and exact terms)
  DenseIndex   semantic search with neural embeddings (understands meaning)
  TfidfIndex   light fallback for semantic search if the neural model is unavailable

Both semantic indexes offer the same three methods, so the rest of the code
does not care which one is in use:
  scores_for_text(text, short)   similarity of every article to a text query
  scores_for_doc(position)       similarity of every article to one article
  similarity_matrix(positions)   article-to-article similarity (used for diversity)
"""
import hashlib
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import torch

# Use all available CPU cores for PyTorch matrix operations
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
    torch.set_num_threads(max(1, os.cpu_count() or 4))
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

import config

import difflib

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list:
    """lowercase -> words -> drop stop words. Preserves Indian names without naive 's' stripping."""
    tokens = []
    for word in TOKEN_RE.findall(text.lower()):
        if len(word) < 2 or word in ENGLISH_STOP_WORDS:
            continue
        tokens.append(word)
    return tokens


def _normalize_repeats(w: str) -> str:
    """Compress repeated characters: 'akkhhlesh' -> 'akhlesh', 'modiii' -> 'modi'."""
    return re.sub(r"(.)\1+", r"\1", w)


# ----------------------------------------------------------------------------
class BM25Index:
    """
    BM25 keyword ranking with built-in Indian entity fuzzy query expansion.

    Handles:
      - Character typos & dropped letters: 'akhiles'   -> 'akhilesh'
      - Repeated characters & typing drag: 'akkhhlesh' -> 'akhilesh'
      - Split tokens & fragmented names:   'akkh lesh' -> 'akhilesh'
      - Phonetic transliteration variants: 'skhilesh'  -> 'akhilesh'
      - Partial names:                     'virat koli'-> 'virat kohli'
    """

    FUZZY_MIN_RATIO = 0.70       # SequenceMatcher similarity cutoff for corrections
    FUZZY_WEIGHT = 0.85          # Score weight for fuzzy matched terms

    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [tokenize(t) for t in texts]
        self.n = len(self.docs)
        self.lengths = np.array([len(d) for d in self.docs])
        self.avg_len = self.lengths.mean() if self.n else 1.0

        # inverted index: word -> [(article position, tf), ...]
        self.postings = defaultdict(list)
        for pos, doc in enumerate(self.docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((pos, tf))

        # rare words get a high IDF weight, common words a low one
        self.idf = {
            term: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

        # Fast character 2-gram index over vocabulary for rapid candidate retrieval
        self._vocab = set(self.postings.keys())
        self._ngram_to_vocab: dict[str, set] = defaultdict(set)
        for word in self._vocab:
            w_pad = f"#{word}#"
            for i in range(len(w_pad) - 1):
                self._ngram_to_vocab[w_pad[i:i+2]].add(word)

    def find_best_vocab_match(self, token: str, cutoff: float = 0.70) -> tuple[str | None, float]:
        """Find the closest vocabulary term using 2-gram candidate filtering + SequenceMatcher."""
        if token in self._vocab:
            return token, 1.0
        norm = _normalize_repeats(token)
        if norm in self._vocab:
            return norm, 0.95

        t_pad = f"#{token}#"
        n_pad = f"#{norm}#"
        ngrams = {t_pad[i:i+2] for i in range(len(t_pad) - 1)} | {n_pad[i:i+2] for i in range(len(n_pad) - 1)}
        candidates: Counter = Counter()
        for ng in ngrams:
            for w in self._ngram_to_vocab.get(ng, ()):
                if abs(len(w) - len(token)) <= 3 or abs(len(w) - len(norm)) <= 3:
                    candidates[w] += 1

        if not candidates:
            return None, 0.0

        best_word, best_ratio, best_count = None, 0.0, 0
        for cand in [w for w, _ in candidates.most_common(60)]:
            r1 = difflib.SequenceMatcher(None, token, cand).ratio()
            r2 = difflib.SequenceMatcher(None, norm, cand).ratio()
            r = max(r1, r2)
            if r >= cutoff:
                cnt = len(self.postings.get(cand, []))
                # Prefer significantly higher ratio, or if within 0.03 ratio, prefer much more frequent term
                if (r > best_ratio + 0.03) or (abs(r - best_ratio) <= 0.03 and cnt > best_count) or (r > best_ratio and best_word is None):
                    best_ratio = r
                    best_word = cand
                    best_count = cnt
        return best_word, best_ratio

    def normalize_query(self, query: str) -> tuple[str, list[tuple[str, str]], list[tuple[str, float]]]:
        """
        Processes raw user query into:
          - clean_query: rewritten string with typos corrected (for Dense Embeddings & Reranker)
          - corrections: list of (original, corrected) for UI display ('Did you mean...')
          - expanded_terms: list of (vocab_term, weight) for BM25 multi-term scoring
        """
        raw_tokens = tokenize(query)
        if not raw_tokens:
            return query, [], []

        corrected_tokens: list[str] = []
        corrections: list[tuple[str, str]] = []
        expanded_terms: list[tuple[str, float]] = []

        i = 0
        while i < len(raw_tokens):
            # Check if adjacent pair of tokens should merge (e.g. 'akkh' + 'lesh' -> 'akhilesh')
            if i + 1 < len(raw_tokens):
                t1, t2 = raw_tokens[i], raw_tokens[i+1]
                # Only attempt merge if at least one token is unknown or weak
                if t1 not in self._vocab or t2 not in self._vocab:
                    merged = t1 + t2
                    match_word, score = self.find_best_vocab_match(merged, cutoff=0.74)
                    if match_word and score >= 0.80:
                        corrected_tokens.append(match_word)
                        corrections.append((f"{t1} {t2}", match_word))
                        expanded_terms.append((match_word, self.FUZZY_WEIGHT * score))
                        i += 2
                        continue

            token = raw_tokens[i]
            if token in self._vocab:
                corrected_tokens.append(token)
                expanded_terms.append((token, 1.0))
            else:
                match_word, score = self.find_best_vocab_match(token, cutoff=self.FUZZY_MIN_RATIO)
                if match_word:
                    corrected_tokens.append(match_word)
                    corrections.append((token, match_word))
                    expanded_terms.append((match_word, self.FUZZY_WEIGHT * score))
                else:
                    corrected_tokens.append(token)
                    expanded_terms.append((token, 1.0))
            i += 1

        clean_query = " ".join(corrected_tokens)
        return clean_query, corrections, expanded_terms

    def scores(self, query: str) -> np.ndarray:
        """BM25 score for every article with fuzzy entity matching."""
        result = np.zeros(self.n)
        _, _, expanded_terms = self.normalize_query(query)
        for term, weight in expanded_terms:
            idf = self.idf.get(term, 0.0)
            for pos, tf in self.postings.get(term, []):
                length_norm = 1 - self.b + self.b * self.lengths[pos] / self.avg_len
                result[pos] += weight * idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        return result

    def query_expand(self, query: str) -> list[str]:
        """
        Return the list of corrected vocabulary terms.
        Used by UI to display 'Auto-corrected: akkhhlesh -> akhilesh'.
        """
        _, corrections, _ = self.normalize_query(query)
        return [f"{orig} -> {corr}" for orig, corr in corrections]




# ----------------------------------------------------------------------------
class DenseIndex:
    """Neural embeddings: every article becomes a vector; similar meaning = close vectors."""

    name = "dense"

    def __init__(self, texts, model_name=config.EMBEDDING_MODEL, vectors=None, use_disk_cache=True):
        from src.model_registry import get_embedder

        texts = list(texts)
        self.model = get_embedder(model_name)
        if vectors is not None:
            stored = np.asarray(vectors, dtype=np.float32)
            if stored.ndim == 2 and stored.shape[0] == len(texts):
                self.vectors = stored
                return
            print("[DenseIndex] Stored embeddings do not match article count; rebuilding.")
        self.vectors = self._load_or_build(texts, model_name, use_disk_cache=use_disk_cache)

    def _load_or_build(self, texts, model_name, use_disk_cache=True) -> np.ndarray:
        data_paths = getattr(config, "DATA_PATHS", [config.DATA_PATH])
        csv_mtime = sum(int(p.stat().st_mtime) for p in data_paths if p.exists())
        key = hashlib.md5(f"{model_name}|{csv_mtime}|{len(texts)}".encode()).hexdigest()[:12]
        path = config.CACHE_DIR / f"embeddings_{key}.npy"
        if use_disk_cache and path.exists():
            return np.load(path)
        
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        master_cache_file = config.CACHE_DIR / "text_embedding_cache.npz"

        # Load persistent incremental hash cache if available
        cache_map = {}
        if master_cache_file.exists():
            try:
                data = np.load(master_cache_file)
                cached_hashes = data["hashes"]
                cached_vecs = data["vectors"]
                for h, v in zip(cached_hashes, cached_vecs):
                    cache_map[h] = v
            except Exception as e:
                print(f"[DenseIndex] Warning: Failed to read master cache ({e}), rebuilding.")

        text_hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        vectors = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        for idx, (t, h) in enumerate(zip(texts, text_hashes)):
            if h in cache_map:
                vectors[idx] = cache_map[h]
            else:
                missing_indices.append(idx)
                missing_texts.append(t)

        if missing_texts:
            print(f"[DenseIndex] {len(texts) - len(missing_texts)}/{len(texts)} articles found in cache.")
            print(f"[DenseIndex] Encoding {len(missing_texts)} new articles (model={model_name})...")
            self.model.max_seq_length = getattr(config, "EMBED_MAX_SEQ_LEN", 256)
            new_vecs = self.model.encode(
                missing_texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            for idx, vec in zip(missing_indices, new_vecs):
                vectors[idx] = vec
                cache_map[text_hashes[idx]] = vec

            # Update persistent master cache
            try:
                all_hashes = np.array(list(cache_map.keys()))
                all_vecs = np.array(list(cache_map.values()))
                np.savez_compressed(master_cache_file, hashes=all_hashes, vectors=all_vecs)
            except Exception as e:
                print(f"[DenseIndex] Warning: Could not save master cache: {e}")
        else:
            print(f"[DenseIndex] All {len(texts)} articles found in incremental cache.")

        vectors = np.array(vectors)
        if use_disk_cache:
            np.save(path, vectors)
            print(f"[DenseIndex] Cache saved to {path}")
        return vectors


    def scores_for_text(self, text: str, short: bool = False) -> np.ndarray:
        if short:
            text = config.QUERY_PREFIX + text
        query_vec = self.model.encode(text, normalize_embeddings=True)
        return self.vectors @ query_vec

    def scores_for_doc(self, pos: int) -> np.ndarray:
        return self.vectors @ self.vectors[pos]

    def get_vector(self, pos: int) -> np.ndarray:
        return self.vectors[pos]

    def scores_for_vector(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        return self.vectors @ vec

    def similarity_matrix(self, positions) -> np.ndarray:
        sub = self.vectors[positions]
        return sub @ sub.T


# ----------------------------------------------------------------------------
class TfidfIndex:
    """Light fallback: no downloads, no GPU. Matches words, not meaning."""

    name = "tfidf"

    def __init__(self, texts):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(texts)   # rows are already L2-normalised

    def get_vector(self, pos: int):
        return self.matrix[pos]

    def scores_for_text(self, text: str, short: bool = False) -> np.ndarray:
        query_vec = self.vectorizer.transform([text])
        return (self.matrix @ query_vec.T).toarray().ravel()

    def scores_for_doc(self, pos: int) -> np.ndarray:
        return (self.matrix @ self.matrix[pos].T).toarray().ravel()

    def scores_for_vector(self, vec) -> np.ndarray:
        return (self.matrix @ vec.T).toarray().ravel()

    def similarity_matrix(self, positions) -> np.ndarray:
        sub = self.matrix[positions]
        return (sub @ sub.T).toarray()



# ----------------------------------------------------------------------------
def reciprocal_rank_fusion(rankings, k: int = config.RRF_K) -> dict:
    """
    Merge several ranked lists into one.
    An article near the top of ANY list gets points: 1 / (k + rank).
    Articles that appear high in BOTH lists win. Works without comparing raw scores.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, pos in enumerate(ranking, start=1):
            fused[pos] += 1.0 / (k + rank)
    return dict(fused)
