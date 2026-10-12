/**
 * app.js — Main application orchestrator.
 * Coordinates themes, sidebar, controls, initial load, and event listeners.
 */

document.addEventListener('DOMContentLoaded', async () => {

  // ─── 1. THEME TOGGLE ────────────────────────────────────────────────────────
  const themeToggle = document.getElementById('theme-toggle');
  const themeIcon = document.getElementById('theme-icon');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    if (themeIcon) {
      themeIcon.textContent = theme === 'dark' ? '☀️' : '🌙';
    }
    localStorage.setItem('theme', theme);
  }

  const savedTheme = localStorage.getItem('theme') || 'dark';
  applyTheme(savedTheme);

  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'dark';
      const next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      State.set({ theme: next });
    });
  }

  // ─── 2. MOBILE SIDEBAR TOGGLE ───────────────────────────────────────────────
  const sidebarToggle = document.getElementById('sidebar-toggle');
  const sidebar = document.getElementById('sidebar');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('open');
    });

    // Close when clicking outside on mobile
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !sidebarToggle.contains(e.target) && sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
      }
    });
  }

  // ─── 3. RECSYS CONTROLS BINDING ─────────────────────────────────────────────
  const topKSlider = document.getElementById('top-k-slider');
  const topKValue = document.getElementById('top-k-value');
  const sectionFilter = document.getElementById('section-filter');
  const methodSelect = document.getElementById('method-select');
  const useBoostsToggle = document.getElementById('use-boosts');
  const useDiversityToggle = document.getElementById('use-diversity');

  if (topKSlider && topKValue) {
    topKSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      topKValue.textContent = val;
      State.set({ topK: val });
    });
    topKSlider.addEventListener('change', () => {
      triggerActiveTabReload();
    });
  }

  if (sectionFilter) {
    sectionFilter.addEventListener('change', (e) => {
      const sec = e.target.value;
      State.set({ activeSection: sec });
      syncCategoryPills(sec);
      triggerActiveTabReload();
    });
  }

  if (methodSelect) {
    methodSelect.addEventListener('change', (e) => {
      State.set({ method: e.target.value });
      triggerActiveTabReload();
    });
  }

  if (useBoostsToggle) {
    useBoostsToggle.addEventListener('change', (e) => {
      State.set({ useBoosts: e.target.checked });
      triggerActiveTabReload();
    });
  }

  if (useDiversityToggle) {
    useDiversityToggle.addEventListener('change', (e) => {
      State.set({ useDiversity: e.target.checked });
      triggerActiveTabReload();
    });
  }

  // ─── 4. GLOBAL SEARCH BAR IN NAVBAR ────────────────────────────────────────
  const globalSearchInput = document.getElementById('global-search-input');
  const searchClearBtn = document.getElementById('search-clear-btn');

  if (globalSearchInput) {
    globalSearchInput.addEventListener('input', (e) => {
      if (searchClearBtn) {
        searchClearBtn.classList.toggle('hidden', !e.target.value);
      }
    });

    globalSearchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const query = globalSearchInput.value.trim();
        if (query) {
          Tabs.performSearch(query);
        }
      }
    });
  }

  if (searchClearBtn && globalSearchInput) {
    searchClearBtn.addEventListener('click', () => {
      globalSearchInput.value = '';
      searchClearBtn.classList.add('hidden');
      globalSearchInput.focus();
    });
  }

  // ─── 5. CATEGORIES & RIGHT PANEL PILLS ─────────────────────────────────────
  const categoryPillsContainer = document.getElementById('category-pills');
  const viewAllBtn = document.getElementById('rp-view-all');

  function syncCategoryPills(activeSec) {
    if (!categoryPillsContainer) return;
    categoryPillsContainer.querySelectorAll('.cat-pill').forEach(pill => {
      pill.classList.toggle('active', (pill.dataset.section || '') === (activeSec || ''));
    });
  }

  if (viewAllBtn) {
    viewAllBtn.addEventListener('click', () => {
      State.set({ activeSection: '' });
      if (sectionFilter) sectionFilter.value = '';
      syncCategoryPills('');
      triggerActiveTabReload();
    });
  }

  // ─── 6. HELPER: RELOAD ACTIVE TAB ──────────────────────────────────────────
  function triggerActiveTabReload() {
    Tabs.reloadCurrentTab();
  }

  // ─── 7. SYSTEM INITIALIZATION & API BOOTSTRAP ─────────────────────────────
  const loadingOverlay = document.getElementById('loading-overlay');
  const appContainer = document.getElementById('app');
  const loadingText = document.getElementById('loading-text');

  try {
    if (loadingText) loadingText.textContent = 'Connecting to Recommendation Engine...';

    // 1. Fetch Engine Status
    const statusData = await Api.status();
    State.set({ engineInfo: statusData });

    const engineNameEl = document.getElementById('engine-name');
    const totalArticlesEl = document.getElementById('total-articles');
    if (engineNameEl) engineNameEl.textContent = statusData.engine || 'RecSys v1.0';
    if (totalArticlesEl) totalArticlesEl.textContent = `${statusData.total_articles.toLocaleString()} stories`;

    // 2. Fetch Sections & Categories
    if (loadingText) loadingText.textContent = 'Loading news categories...';
    const sectionsData = await Api.sections();
    const sections = sectionsData.sections || [];
    State.set({ allSections: sections });

    // Populate Sidebar Category Dropdown
    if (sectionFilter) {
      sections.forEach(sec => {
        const opt = document.createElement('option');
        opt.value = sec;
        opt.textContent = sec.charAt(0).toUpperCase() + sec.slice(1);
        sectionFilter.appendChild(opt);
      });
    }

    // Populate Right-Panel Category Pills
    if (categoryPillsContainer) {
      buildCategoryPills(sections, categoryPillsContainer, (selectedSection) => {
        State.set({ activeSection: selectedSection });
        if (sectionFilter) sectionFilter.value = selectedSection;
        triggerActiveTabReload();
      });
    }

    // 3. Initialize Hero Carousel
    if (loadingText) loadingText.textContent = 'Loading top trending stories...';
    try { await Hero.init(); } catch(e) { console.warn('Hero init skipped:', e.message); }

    // 4. Initialize Tabs
    Tabs.init();

    // 5. Initialize Editor & PostgreSQL Pipeline Hub (Editor is a const, always available)
    await Editor.init();

    // 6. Role Switcher Controller (Editor vs Reader)
    const roleEditorBtn = document.getElementById('role-btn-editor');
    const roleReaderBtn = document.getElementById('role-btn-reader');
    const editorView = document.getElementById('editor-view');
    const readerView = document.getElementById('reader-view');
    const readerEmptyBanner = document.getElementById('reader-empty-banner');
    const btnEmptyGotoEditor = document.getElementById('btn-empty-goto-editor');

    function checkReaderDatabaseState() {
      const isDbEmpty = !statusData || statusData.database_empty || statusData.total_articles === 0;
      if (readerEmptyBanner) {
        readerEmptyBanner.classList.toggle('hidden', !isDbEmpty);
      }
      const heroSection = document.getElementById('hero-section');
      const tabsNav = document.getElementById('tabs-nav');
      const similarPanel = document.getElementById('tab-similar');
      if (heroSection) heroSection.style.display = isDbEmpty ? 'none' : '';
      if (tabsNav) tabsNav.style.display = isDbEmpty ? 'none' : '';
      if (similarPanel) similarPanel.style.display = isDbEmpty ? 'none' : '';
    }

    function setAppRole(role) {
      const isEditor = role === 'editor';
      if (roleEditorBtn) roleEditorBtn.classList.toggle('active', isEditor);
      if (roleReaderBtn) roleReaderBtn.classList.toggle('active', !isEditor);
      if (editorView) editorView.classList.toggle('hidden', !isEditor);
      if (readerView) readerView.classList.toggle('hidden', isEditor);
      localStorage.setItem('app_role', role);

      if (isEditor) {
        Editor.refreshDbHealth();
      } else {
        checkReaderDatabaseState();
        Tabs.populateArticlePicker();
      }
    }

    window.setAppRole = setAppRole;

    if (roleEditorBtn) roleEditorBtn.addEventListener('click', () => setAppRole('editor'));
    if (roleReaderBtn) roleReaderBtn.addEventListener('click', () => setAppRole('reader'));
    if (btnEmptyGotoEditor) btnEmptyGotoEditor.addEventListener('click', () => setAppRole('editor'));

    // Default to editor if database is empty, otherwise restore saved role
    const initialRole = (!statusData || statusData.database_empty || statusData.total_articles === 0)
      ? 'editor'
      : (localStorage.getItem('app_role') || 'editor');
    setAppRole(initialRole);

    // Reveal App
    if (loadingOverlay) loadingOverlay.classList.add('fade-out');
    if (appContainer) appContainer.classList.remove('hidden');

    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.remove();
    }, 400);


  } catch (err) {
    console.error('Failed to initialize app with backend:', err);
    if (loadingText) {
      loadingText.innerHTML = `
        <span style="color:#ef4444">Could not connect to API backend</span><br/>
        <small style="color:var(--text-muted);font-size:0.8rem">Ensure the server is running at <code>http://localhost:5000</code></small>
      `;
    }

    setTimeout(() => {
      if (loadingOverlay) loadingOverlay.classList.add('fade-out');
      if (appContainer) appContainer.classList.remove('hidden');
      showToast('Backend offline. Please start the API server.', 'warn', 6000);
    }, 1500);
  }
});
