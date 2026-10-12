/**
 * api.js — All HTTP calls to the FastAPI backend.
 * Centralised so the URL is changed in exactly one place.
 */
const API_BASE = (window.location.protocol.startsWith('http'))
  ? `${window.location.origin}/api`
  : 'http://localhost:5000/api';

const Api = (() => {

  async function _fetch(path, opts = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      const msg = (typeof err.detail === 'string' ? err.detail : null) || err.error || err.message || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return res.json();
  }

  /** Engine status + config */
  async function status() {
    return _fetch('/status');
  }

  /** Trending / hero articles */
  async function trending(n = 6) {
    return _fetch(`/articles/trending?n=${n}`);
  }

  /** Paginated article list */
  async function articles({ page = 1, perPage = 20, section = '', sort = 'date' } = {}) {
    const params = new URLSearchParams({ page, per_page: perPage, section, sort });
    return _fetch(`/articles?${params}`);
  }

  /** All sections */
  async function sections() {
    return _fetch('/sections');
  }

  /** Single article */
  async function getArticle(id) {
    return _fetch(`/articles/${encodeURIComponent(id)}`);
  }

  /**
   * Recommend endpoint: supports query or articleId.
   */
  async function recommend({
    query      = null,
    articleId  = null,
    topK       = 10,
    method     = 'hybrid+rerank',
    useBoosts  = true,
    diversify  = true,
    section    = null,
  } = {}) {
    return _fetch('/recommend', {
      method: 'POST',
      body: JSON.stringify({
        query,
        article_id:   articleId,
        top_k:        topK,
        method,
        use_boosts:   useBoosts,
        diversify,
        section,
      }),
    });
  }

  /** PostgreSQL & Editor Endpoints */
  async function dbHealth() {
    return _fetch('/db/health');
  }

  async function dbInit() {
    const res = await fetch(`${API_BASE}/db/init`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'DB Init failed');
    }
    return res.json();
  }

  async function uploadCsv(file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/db/upload-csv`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'CSV upload failed');
    }
    return res.json();
  }

  async function jobStatus(jobId) {
    return _fetch(`/db/jobs/${encodeURIComponent(jobId)}`);
  }

  async function recentJobs() {
    return _fetch('/db/jobs');
  }

  async function ingestDefault(limit = null) {
    const query = limit ? `?limit=${limit}` : '';
    const res = await fetch(`${API_BASE}/db/ingest-default${query}`, { method: 'POST' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Default dataset ingest failed');
    }
    return res.json();
  }

  /** Precomputed recommendations lookup — sub-5ms from PostgreSQL */
  async function precomputedRecs(articleId, limit = 10) {
    return _fetch(`/db/recommendations/${encodeURIComponent(articleId)}?limit=${limit}`);
  }

  return {
    status,
    trending,
    articles,
    sections,
    getArticle,
    recommend,
    dbHealth,
    dbInit,
    uploadCsv,
    ingestDefault,
    jobStatus,
    recentJobs,
    precomputedRecs
  };
})();


