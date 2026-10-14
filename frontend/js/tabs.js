/**
 * tabs.js — Tab switching and recommendation flows.
 *
 * "Similar Articles" renders a scrolling FEED of "halo" widgets: each widget has
 * ONE source story in the centre with its recommendations arranged around it
 * (top / left / right / bottom). Widgets load lazily as you scroll.
 * "Semantic Search" stays a query → results grid.
 *
 * Every widget carries a source chip so it is obvious whether the answer came
 * from the precomputed PostgreSQL table (⚡ hybrid+rerank) or the live in-memory
 * engine (🛰️ bm25 / semantic / hybrid).
 *
 * The Recommendations Count slider (1–10) is sent as top_k, so every widget shows
 * exactly that many recommendations — served from precompute by default.
 */

const Tabs = (() => {
  let allArticlesCache = [];
  let feedObserver = null;   // lazy-loads a widget's recommendations near viewport
  let scrollObserver = null; // loads the NEXT batch when the bottom sentinel shows
  let feedQueue = [];        // source articles still waiting to be rendered
  let feedRendered = 0;      // how many halo widgets are currently in the DOM

  const FEED_CHUNK = 15;     // widgets appended per batch as you scroll
  const FEED_PAGE  = 100;    // API page size when pulling the full article list
  const SLOT_ORDER = ['top', 'bottom', 'left', 'right'];  // assignment order

  // ─── TAB NAVIGATION ─────────────────────────────────────────────────────────
  function switchTab(tabName) {
    const nav = document.getElementById('tabs-nav');
    if (!nav) return;

    nav.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });

    document.querySelectorAll('.tab-panel').forEach(panel => {
      panel.classList.toggle('active', panel.id === `tab-${tabName}`);
    });

    State.set({ activeTab: tabName });

    if (tabName === 'similar') {
      buildFeed();
    } else if (tabName === 'editor') {
      if (window.Editor) {
        Editor.refreshDbHealth();
      }
    }
  }


  // ─── SIMILAR ARTICLES FEED ──────────────────────────────────────────────────

  /** A coloured chip telling the reader where a result came from. */
  function sourceChip(source) {
    const pre = source === 'postgresql';
    const label = pre ? '⚡ Precomputed' : '🛰️ Live';
    const title = pre
      ? 'Served from the precomputed PostgreSQL table (sub-5ms)'
      : 'Computed live by the in-memory retrieval + rerank engine';
    return `<span class="source-chip source-chip-${pre ? 'pre' : 'live'}" title="${title}">${label}</span>`;
  }

  /**
   * Pull EVERY article, not just the first page. The feed is unlimited, so we
   * page through the API (100 at a time) until the whole table is loaded, then
   * render it progressively as the reader scrolls.
   *
   * sort='confidence' ranks stories by the average score of their precomputed
   * recommendations, so widgets whose recommendations are the strongest
   * (70%, 80% …) appear before weak ones (10%, 20% …).
   */
  async function loadAllArticles(force = false) {
    if (allArticlesCache.length && !force) return allArticlesCache;

    const acc = [];
    let page = 1;
    let total = Infinity;
    while (acc.length < total && page <= 500) {
      const data = await Api.articles({ page, perPage: FEED_PAGE, sort: 'confidence' });
      total = data.total || 0;
      const arts = data.articles || [];
      if (!arts.length) break;
      acc.push(...arts);
      if (arts.length < FEED_PAGE) break;   // last page reached
      page += 1;
    }
    allArticlesCache = acc;
    return allArticlesCache;
  }

  const skelTile = () => '<div class="skeleton orbit-tile orbit-tile-skel"></div>';

  /** Format an article's upload (ingest) time, e.g. "14 Oct, 13:04". */
  function uploadStamp(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '';
    return d.toLocaleString(undefined, {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit',
    });
  }

  /** Build a halo widget: centre story + 4 empty recommendation slots. */
  function buildFeedWidget(article) {
    const widget = document.createElement('article');
    widget.className = 'feed-widget';
    widget.dataset.id = article.id;

    const color = sectionColor(article.section);
    const emoji = sectionEmoji(article.section);
    const thumb = article.image
      ? `<img src="${article.image}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'orbit-main-fallback\\'>${emoji}</div>'" />`
      : `<div class="orbit-main-fallback">${emoji}</div>`;
    const liveBadge = article.is_live ? '<span class="badge badge-live">🔴 LIVE</span>' : '';
    const readLink = article.url
      ? `<a class="feed-read-link" href="${article.url}" target="_blank" rel="noopener">📖 Read story</a>`
      : '';

    widget.innerHTML = `
      <div class="feed-widget-head">
        <span class="feed-recs-title">✦ Around this story</span>
        <span class="feed-head-right">
          <span class="feed-conf-slot" data-slot="conf"></span>
          <span class="feed-source-slot" data-slot="source"></span>
        </span>
      </div>
      <div class="orbit-stage">
        <div class="orbit-slot orbit-slot-top" data-slot="top">${skelTile()}</div>
        <div class="orbit-mid">
          <div class="orbit-slot orbit-slot-left" data-slot="left">${skelTile()}</div>
          <div class="orbit-main">
            <div class="orbit-main-thumb">${thumb}</div>
            <div class="orbit-main-body">
              <div class="orbit-main-section" style="color:${color}">${(article.section || 'General').toUpperCase()} ${liveBadge}</div>
              <h3 class="orbit-main-title">${article.title}</h3>
              <div class="orbit-main-meta">
                <span>📅 ${article.date || 'Recent'}</span>
                ${article.created_at ? `<span class="feed-upload" title="Uploaded to the system ${uploadStamp(article.created_at)}">🕘 ${uploadStamp(article.created_at)}</span>` : ''}
                ${readLink}
              </div>
            </div>
          </div>
          <div class="orbit-slot orbit-slot-right" data-slot="right">${skelTile()}</div>
        </div>
        <div class="orbit-slot orbit-slot-bottom" data-slot="bottom">${skelTile()}</div>
      </div>`;

    return widget;
  }

  /** One recommendation tile that sits around the centre story. */
  function buildOrbitTile(rec) {
    const art   = rec.article || rec;
    const emoji = sectionEmoji(art.section);
    const thumb = art.image
      ? `<img src="${art.image}" alt="" loading="lazy" onerror="this.parentElement.innerHTML='<div class=\\'orbit-tile-fallback\\'>${emoji}</div>'" />`
      : `<div class="orbit-tile-fallback">${emoji}</div>`;
    const score = rec.score != null
      ? `<span class="orbit-score">${Math.round(rec.score * 100)}%</span>` : '';

    const tile = document.createElement('div');
    tile.className = 'orbit-tile';
    tile.innerHTML = `
      <div class="orbit-tile-thumb">${thumb}${score}</div>
      <div class="orbit-tile-body">
        <div class="orbit-tile-section">${(art.section || 'General')}</div>
        <div class="orbit-tile-title">${art.title}</div>
        ${art.created_at ? `<div class="orbit-tile-meta" title="Uploaded to the system ${uploadStamp(art.created_at)}">🕘 ${uploadStamp(art.created_at)}</div>` : ''}
      </div>`;

    if (art.url) {
      tile.classList.add('clickable');
      tile.title = 'Open full article';
      tile.addEventListener('click', () => window.open(art.url, '_blank', 'noopener'));
    }
    return tile;
  }

  /**
   * Decide how many recommendation tiles go on each side of the centre story.
   *
   * Few tiles  -> all on top/bottom rows (a tidy 2-row cluster, avoiding the
   *               half-empty side columns that used to leave big blank gaps).
   * Many tiles -> left/right get balanced PAIRS (always equal, so the halo stays
   *               symmetric) and the remainder fills the top/bottom rows.
   */
  function planHaloSlots(n) {
    if (n <= 0) return { top: 0, bottom: 0, left: 0, right: 0 };
    if (n <= 6) {
      const top = Math.ceil(n / 2);
      return { top, bottom: n - top, left: 0, right: 0 };
    }
    const side = n >= 9 ? 2 : 1;
    const rest = n - 2 * side;
    const top = Math.ceil(rest / 2);
    return { top, bottom: rest - top, left: side, right: side };
  }

  /** Distribute the returned recommendations around the centre. */
  function renderWidgetRecs(widget, data) {
    const chipSlot = widget.querySelector('[data-slot="source"]');
    if (chipSlot) chipSlot.innerHTML = sourceChip(data.source);

    // Average confidence across this story's recommendations — the value the feed
    // is ordered by, shown so readers can see why a widget ranks where it does.
    const confSlot = widget.querySelector('[data-slot="conf"]');
    if (confSlot) {
      const scores = (data.recommendations || [])
        .map(r => r.score)
        .filter(s => typeof s === 'number' && !Number.isNaN(s));
      if (scores.length) {
        const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
        confSlot.innerHTML = `<span class="feed-conf-chip" title="Average confidence across this story's recommendations">${Math.round(avg * 100)}% avg</span>`;
      } else {
        confSlot.innerHTML = '';
      }
    }

    const slots = {
      top:    widget.querySelector('[data-slot="top"]'),
      bottom: widget.querySelector('[data-slot="bottom"]'),
      left:   widget.querySelector('[data-slot="left"]'),
      right:  widget.querySelector('[data-slot="right"]'),
    };
    Object.values(slots).forEach(s => { if (s) s.innerHTML = ''; });

    const recs = (data.recommendations || [])
      .filter(r => String((r.article || r).id) !== String(widget.dataset.id));

    if (!recs.length) {
      const stage = widget.querySelector('.orbit-stage');
      if (stage) stage.insertAdjacentHTML('beforeend',
        '<div class="feed-recs-empty">No close matches found for this story.</div>');
      return;
    }

    // Spread the recommendations around the centre using the plan above.
    const plan = planHaloSlots(recs.length);
    let assign = 0;
    SLOT_ORDER.forEach(name => {
      const slot = slots[name];
      if (!slot) return;
      for (let k = 0; k < plan[name] && assign < recs.length; k++) {
        slot.appendChild(buildOrbitTile(recs[assign++]));
      }
    });
  }

  /** Fetch + fill a single halo widget (called when it scrolls into view). */
  async function loadWidgetRecommendations(widget) {
    if (widget.dataset.loaded === 'true') return;
    widget.dataset.loaded = 'true';

    const articleId = widget.dataset.id;
    const { topK, method, useBoosts, diversify } = State.get();
    try {
      // No section is sent on purpose: keeping the default similar-article request
      // means method=hybrid+rerank is served from the precomputed table. The source
      // list itself is filtered by category client-side.
      const data = await Api.recommend({ articleId, topK, method, useBoosts, diversify });
      renderWidgetRecs(widget, data);
    } catch (err) {
      const stage = widget.querySelector('.orbit-stage');
      if (stage) {
        stage.querySelectorAll('.orbit-slot').forEach(s => { s.innerHTML = ''; });
        const empty = document.createElement('div');
        empty.className = 'feed-recs-empty';
        empty.textContent = `⚠️ ${err.message}`;
        stage.appendChild(empty);
      }
    }
  }

  /** Lazy-load widgets only as they approach the viewport. Reuses one observer. */
  function observeWidgets(widgets) {
    if (!feedObserver) {
      feedObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            feedObserver.unobserve(entry.target);
            loadWidgetRecommendations(entry.target);
          }
        });
      }, { root: null, rootMargin: '400px 0px', threshold: 0.01 });
    }
    const nodes = widgets || document.querySelectorAll('#article-feed .feed-widget');
    nodes.forEach(w => feedObserver.observe(w));
  }

  function skeletonFeed(n = 2) {
    return Array(n).fill(`
      <div class="feed-widget feed-widget-skeleton">
        <div class="skeleton" style="height:18px;width:190px;border-radius:8px;margin-bottom:16px"></div>
        <div style="display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center">
          <div class="skeleton orbit-tile-skel" style="width:168px"></div>
          <div class="skeleton" style="height:250px;width:310px;border-radius:16px"></div>
          <div class="skeleton orbit-tile-skel" style="width:168px"></div>
        </div>
      </div>`).join('');
  }

  /** Sentinel state: "more" while batches remain, "done" at the end. */
  function setSentinel(state) {
    const sentinel = document.getElementById('article-feed-sentinel');
    if (!sentinel) return;
    if (state === 'more') {
      sentinel.classList.remove('done');
      sentinel.innerHTML = '<span class="feed-sentinel-text">Scroll for more stories…</span>';
    }
  }

  function markFeedEnd() {
    if (scrollObserver) { scrollObserver.disconnect(); scrollObserver = null; }
    const sentinel = document.getElementById('article-feed-sentinel');
    if (!sentinel) return;
    sentinel.classList.add('done');
    sentinel.innerHTML = feedRendered
      ? `<span class="feed-sentinel-text">✓ You've reached the end — ${feedRendered} stories shown.</span>`
      : '';
  }

  /** Append the next batch of halo widgets (infinite scroll). */
  function appendFeedChunk() {
    const feed = document.getElementById('article-feed');
    if (!feed) return;
    if (!feedQueue.length) { markFeedEnd(); return; }

    const batch = feedQueue.splice(0, FEED_CHUNK);
    const added = [];
    const frag = document.createDocumentFragment();
    batch.forEach(art => {
      const w = buildFeedWidget(art);
      frag.appendChild(w);
      added.push(w);
    });
    feed.appendChild(frag);
    feedRendered += batch.length;

    observeWidgets(added);
    if (feedQueue.length) setSentinel('more');
    else markFeedEnd();
  }

  /** Watch the bottom sentinel and pull the next batch when it scrolls into view. */
  function observeSentinel() {
    const sentinel = document.getElementById('article-feed-sentinel');
    if (!sentinel || !feedQueue.length) return;
    if (scrollObserver) scrollObserver.disconnect();
    scrollObserver = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) appendFeedChunk();
    }, { root: null, rootMargin: '600px 0px', threshold: 0 });
    scrollObserver.observe(sentinel);
  }

  /** (Re)build the whole feed from the current controls/state — unlimited. */
  async function buildFeed() {
    const feed = document.getElementById('article-feed');
    if (!feed) return;

    if (feedObserver) { feedObserver.disconnect(); feedObserver = null; }
    if (scrollObserver) { scrollObserver.disconnect(); scrollObserver = null; }
    feedQueue = [];
    feedRendered = 0;

    feed.innerHTML = skeletonFeed(2);
    setSentinel('more');

    let articles;
    try {
      articles = await loadAllArticles(false);
    } catch (err) {
      feed.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">⚠️</div>
          <h3>Could not load articles</h3>
          <p>${err.message}</p>
        </div>`;
      markFeedEnd();
      return;
    }

    const { activeSection } = State.get();
    let list = articles;
    if (activeSection) {
      list = articles.filter(a => (a.section || '') === activeSection);
    }
    // No cap — every matching story is queued and rendered as the reader scrolls.

    if (!list.length) {
      feed.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📰</div>
          <h3>No articles in this category</h3>
          <p>Try clearing the category filter.</p>
        </div>`;
      markFeedEnd();
      return;
    }

    feed.innerHTML = '';
    feedQueue = list.slice();
    appendFeedChunk();
    observeSentinel();
  }

  // Kept for backward compatibility: app.js calls this after ingestion / role switch.
  async function populateArticlePicker() {
    allArticlesCache = [];
    return buildFeed();
  }


  // ─── SEARCH TAB ─────────────────────────────────────────────────────────────
  async function performSearch(query) {
    if (!query || !query.trim()) return;
    query = query.trim();

    switchTab('search');

    const searchInput = document.getElementById('search-query-input');
    if (searchInput) searchInput.value = query;

    const globalInput = document.getElementById('global-search-input');
    if (globalInput) globalInput.value = query;

    const resultsContainer = document.getElementById('search-results-container');
    const resultsGrid = document.getElementById('search-results-grid');
    const resultMeta = document.getElementById('search-result-meta');
    const emptyState = document.getElementById('search-empty');

    if (resultsContainer) resultsContainer.classList.remove('hidden');
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsGrid) showSpinner(resultsGrid, 4);

    const { topK, method, useBoosts, diversify, activeSection } = State.get();

    try {
      const data = await Api.recommend({
        query,
        topK,
        method,
        useBoosts,
        diversify,
        section: activeSection || null,
      });

      if (resultMeta) renderResultMeta(resultMeta, data);
      if (resultsGrid) {
        renderNotes(resultsGrid, data.notes || [], data.low_confidence || false);
        const recs = data.recommendations || [];
        if (recs.length === 0) {
          if (resultsContainer) resultsContainer.classList.add('hidden');
          if (emptyState) emptyState.classList.remove('hidden');
        } else {
          renderCards(resultsGrid, recs);
        }
      }
    } catch (err) {
      if (resultsGrid) {
        resultsGrid.innerHTML = `
          <div class="empty-state" style="grid-column: 1 / -1">
            <div class="empty-icon">⚠️</div>
            <h3>Search request failed</h3>
            <p>${err.message}</p>
          </div>`;
      }
    }
  }

  // ─── INITIALIZATION ─────────────────────────────────────────────────────────
  function init() {
    const nav = document.getElementById('tabs-nav');
    if (nav) {
      nav.addEventListener('click', (e) => {
        const btn = e.target.closest('.tab-btn');
        if (btn && btn.dataset.tab) {
          switchTab(btn.dataset.tab);
        }
      });
    }

    const searchBtn = document.getElementById('search-btn');
    const searchInput = document.getElementById('search-query-input');

    if (searchBtn && searchInput) {
      searchBtn.addEventListener('click', () => {
        performSearch(searchInput.value);
      });
      searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          performSearch(searchInput.value);
        }
      });
    }

    const pills = document.querySelectorAll('#quick-pills .pill');
    pills.forEach(pill => {
      pill.addEventListener('click', () => {
        const q = pill.dataset.q;
        if (q) performSearch(q);
      });
    });

    buildFeed();
  }

  function reloadCurrentTab() {
    const { activeTab } = State.get();
    if (activeTab === 'similar') {
      buildFeed();
    } else if (activeTab === 'search') {
      const q = document.getElementById('search-query-input')?.value;
      if (q) performSearch(q);
    }
  }

  return { init, switchTab, performSearch, populateArticlePicker, buildFeed, reloadCurrentTab };
})();

// Expose to window so app.js can detect and call Tabs methods
window.Tabs = Tabs;
