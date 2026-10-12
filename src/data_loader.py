"""
Step 1 - Load the CSV and clean it. Every cleaning rule here comes from the EDA findings.

Cleaning rules:
  1. drop duplicate articles (same articleId)
  2. normalise text: curly quotes -> plain quotes, \\r\\n and extra spaces -> one space
  3. remove links and e-mail addresses
  4. remove inline "ALSO READ <headline of another article>" blocks
       (they paste unrelated headlines into the body and confuse search models)
       -> we also extract linked article IDs: editor choices become ground truth labels for evaluate.py
  5. remove boilerplate sentences repeated in many articles (disclaimers, quiz lines, etc.)
  6. shorten very long descriptions (video descriptions can be 1,000+ characters)
"""
import json
from pathlib import Path
import re
import unicodedata
from collections import Counter
import pandas as pd
import config

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
ALSO_READ_RE = re.compile(r"\balso\s+read\b[\s:|\-\u2013\u2014]*", re.I)
ALSO_READ_FALLBACK_WORDS = 14       # headline length used when the linked article is not in dataset
ALSO_READ_KEY_WORDS = 4             # first N words of a headline are used to match it
QUOTES = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b"} | {ord(c): '"' for c in "\u201c\u201d\u201e"}


def first_words(text: str, n: int) -> str:
    return " ".join(text.split()[:n])


def normalise_text(text) -> str:
    """Unicode clean-up: smart quotes become plain quotes, whitespace collapses."""
    text = unicodedata.normalize("NFKC", str(text) if pd.notna(text) else "")
    text = text.translate(QUOTES).replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()


def remove_links(text: str) -> str:
    return re.sub(r"\s+", " ", EMAIL_RE.sub(" ", URL_RE.sub(" ", text))).strip()


def _word(token: str) -> str:
    """Comparable form of a word: lowercase alphanumeric characters only."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


def _headline_index(titles, ids) -> dict:
    """first 4 words of every title -> (article id, number of words in the title)"""
    index = {}
    for article_id, title in zip(ids, titles):
        words = [w for w in map(_word, title.split()) if w]
        if len(words) >= ALSO_READ_KEY_WORDS:
            index[tuple(words[:ALSO_READ_KEY_WORDS])] = (article_id, len(words))
    return index


def strip_also_read(body: str, own_id, headlines: dict):
    """Cut every 'ALSO READ <headline>' block out of the body. Returns (clean body, linked article ids)."""
    pieces, linked, cursor = [], [], 0
    for match in ALSO_READ_RE.finditer(body):
        if match.start() < cursor:
            continue                                    # already inside a removed block
        pieces.append(body[cursor:match.start()])
        segment = body[match.end(): match.end() + 400]
        tokens = [(w, m.end()) for m in re.finditer(r"\S+", segment) if (w := _word(m.group()))]
        key = tuple(w for w, _ in tokens[:ALSO_READ_KEY_WORDS])
        article_id, n_words = headlines.get(key, (None, ALSO_READ_FALLBACK_WORDS))
        if article_id is not None and article_id != own_id:
            linked.append(article_id)
        cut = tokens[min(n_words, len(tokens)) - 1][1] if tokens else 0
        cursor = match.end() + cut
    pieces.append(body[cursor:])
    return re.sub(r"\s+", " ", " ".join(pieces)).strip(), linked


def _find_boilerplate(bodies) -> set:
    """Sentences that appear in many different articles are not about any specific topic."""
    counts = Counter()
    for body in bodies:
        counts.update({s.strip() for s in SENTENCE_SPLIT.split(body) if len(s.strip()) > 25})
    return {s for s, c in counts.items() if c >= config.BOILERPLATE_MIN_DOCS}


def _strip_boilerplate(body: str, boilerplate: set) -> str:
    return " ".join(s for s in SENTENCE_SPLIT.split(body) if s.strip() not in boilerplate).strip()


def _subsection(value) -> str:
    """'["Sports","Cricket"]' -> 'Cricket'   |   '["India"]' -> ''"""
    try:
        parts = json.loads(value)
    except (TypeError, ValueError):
        return ""
    return parts[-1] if len(parts) > 1 else ""


def _short_description(text: str, limit: int = 300) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def load_articles(paths=None) -> pd.DataFrame:
    if paths is None:
        paths = getattr(config, "DATA_PATHS", [config.DATA_PATH])
    elif isinstance(paths, (list, tuple)):
        paths = paths
    else:
        paths = [paths]

    dfs = []
    for p in paths:
        p_obj = Path(p)
        if p_obj.exists():
            print(f"[DataLoader] Loading {p_obj.name}...")
            dfs.append(pd.read_csv(p_obj))
        else:
            print(f"[DataLoader] Warning: {p_obj} does not exist, skipping.")

    if not dfs:
        raise FileNotFoundError(f"No valid CSV dataset files found in {paths}")

    raw = pd.concat(dfs, ignore_index=True)
    n_raw = len(raw)
    raw = raw.drop_duplicates(subset="articleId", keep="last").reset_index(drop=True)
    print(f"[DataLoader] Cleaned: {n_raw} total rows merged -> {len(raw)} unique articles.")

    ids = raw["articleId"].tolist()
    titles = raw["articleTitle"].map(normalise_text)
    descriptions = raw["description"].map(normalise_text).map(remove_links).map(_short_description)

    headlines = _headline_index(titles, ids)
    bodies, linked_ids = [], []
    for article_id, body in zip(ids, raw["body_raw"].fillna("")):
        body_clean = remove_links(normalise_text(body))
        body_clean, linked = strip_also_read(body_clean, article_id, headlines)
        bodies.append(body_clean)
        linked_ids.append(sorted(set(linked)))
    
    boilerplate = _find_boilerplate(bodies)
    bodies = [_strip_boilerplate(b, boilerplate) for b in bodies]

    # Flexible column resolution across different dataset formats
    sec_series = raw["sectionName"] if "sectionName" in raw.columns else (raw["section1Slug"] if "section1Slug" in raw.columns else pd.Series([""] * len(raw)))
    subsec_series = raw["section2Name"].map(_subsection) if "section2Name" in raw.columns else (raw["section2Slug"] if "section2Slug" in raw.columns else pd.Series([""] * len(raw)))
    pv_series = raw["pageViews"] if "pageViews" in raw.columns else (raw["pageViews_latest"] if "pageViews_latest" in raw.columns else (raw["pageViews_max"] if "pageViews_max" in raw.columns else pd.Series([0] * len(raw))))
    uu_series = raw["uniqueUsers"] if "uniqueUsers" in raw.columns else (raw["uniqueUsers_latest"] if "uniqueUsers_latest" in raw.columns else (raw["uniqueUsers_max"] if "uniqueUsers_max" in raw.columns else pd.Series([0] * len(raw))))

    df = pd.DataFrame({
        "id": ids,
        "title": titles,
        "description": descriptions,
        "section": sec_series.fillna(""),
        "subsection": subsec_series.fillna(""),
        "date": pd.to_datetime(raw["postPublishDate"], utc=True, errors="coerce") if "postPublishDate" in raw.columns else pd.Series([None] * len(raw)),
        "url": raw["articleUrl"].fillna("") if "articleUrl" in raw.columns else "",
        "image": raw["articleFeaturedImage"].fillna("") if "articleFeaturedImage" in raw.columns else "",
        "is_live": raw["liveBlog"].fillna(False).astype(bool) if "liveBlog" in raw.columns else False,
        "body": bodies,
        "linked_ids": linked_ids,      # editor-chosen "also read" articles (used for evaluation)
        "pageViews": pv_series.fillna(0).astype(int),
        "uniqueUsers": uu_series.fillna(0).astype(int),
    })


    # Text used by the semantic search and as the "query" when an article is the input:
    df["text_embed"] = [
        f"{t}. {d}. {first_words(b, config.BODY_WORDS_EMBED)}"
        for t, d, b in zip(df.title, df.description, df.body)
    ]
    # Text used by BM25 keyword search: the title is repeated 3x so it counts more.
    df["text_bm25"] = [
        f"{t}. {t}. {t}. {d}. {b}" for t, d, b in zip(df.title, df.description, df.body)
    ]
    # Shorter text shown to the reranker.
    df["text_rerank"] = [
        f"{t}. {d}. {first_words(b, config.BODY_WORDS_RERANK)}"
        for t, d, b in zip(df.title, df.description, df.body)
    ]
    return df
