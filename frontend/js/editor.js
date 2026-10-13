/**
 * editor.js — Clean, modular Editor Studio & Animated Pipeline Controller.
 */
const Editor = (() => {
  let selectedFile = null;
  let activeJobId = null;
  let pollInterval = null;
  const logged = new Set();

  // DOM elements cache
  const dom = {};

  function cacheDom() {
    dom.fileInput = document.getElementById('csv-file-input');
    dom.dropzone = document.getElementById('upload-dropzone');
    dom.uploadBtn = document.getElementById('start-ingest-btn');
    dom.demoBtn = document.getElementById('ingest-demo-btn');
    dom.preview = document.getElementById('file-preview');
    dom.previewName = document.getElementById('file-preview-name');
    dom.visualizer = document.getElementById('pipeline-visualizer-card');
    dom.completion = document.getElementById('pipeline-completion-banner');
    dom.progressPct = document.getElementById('visualizer-progress-pct');
    dom.progressFill = document.getElementById('visualizer-progress-fill');
    dom.statusMsg = document.getElementById('visualizer-status-msg');
    dom.logWindow = document.getElementById('pipeline-log-window');
    dom.switchBtn = document.getElementById('btn-switch-to-reader');
    dom.initBtn = document.getElementById('init-db-btn');
    dom.clearBtn = document.getElementById('clear-db-btn');
    dom.inspectorSelect = document.getElementById('inspector-article-select');
    dom.inspectorBody = document.getElementById('inspector-table-body');
    dom.inspectorEmpty = document.getElementById('inspector-empty');
    dom.recentJobsBody = document.getElementById('recent-jobs-body');
  }

  async function init() {
    cacheDom();
    bindEvents();
    await refreshDbHealth();
    loadInspectorArticles();
    loadRecentJobs();
  }

  function bindEvents() {
    // 1. File Selection via Click
    if (dom.dropzone && dom.fileInput) {
      dom.dropzone.addEventListener('click', (e) => {
        // Prevent re-triggering if click was on the input itself
        if (e.target !== dom.fileInput) {
          dom.fileInput.click();
        }
      });

      dom.fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          onFileChosen(e.target.files[0]);
        }
      });

      // 2. Drag & Drop
      ['dragenter', 'dragover'].forEach(ev => {
        dom.dropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dom.dropzone.classList.add('dragover');
        });
      });

      ['dragleave', 'drop'].forEach(ev => {
        dom.dropzone.addEventListener(ev, (e) => {
          e.preventDefault();
          dom.dropzone.classList.remove('dragover');
        });
      });

      dom.dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          onFileChosen(e.dataTransfer.files[0]);
        }
      });
    }

    // 3. Upload Trigger Button
    if (dom.uploadBtn) {
      dom.uploadBtn.addEventListener('click', handleUpload);
    }

    // 4. Demo Dataset Ingest Button
    if (dom.demoBtn) {
      dom.demoBtn.addEventListener('click', handleDemoIngest);
    }

    // 5. DB Verification Button
    if (dom.initBtn) {
      dom.initBtn.addEventListener('click', handleDbInit);
    }

    // 5B. DB Clear Button
    if (dom.clearBtn) {
      dom.clearBtn.addEventListener('click', handleDbClear);
    }

    // 6. Switch to Reader CTA
    if (dom.switchBtn) {
      dom.switchBtn.addEventListener('click', () => {
        if (window.setAppRole) window.setAppRole('reader');
      });
    }

    // 7. Inspector Select
    if (dom.inspectorSelect) {
      dom.inspectorSelect.addEventListener('change', (e) => {
        if (e.target.value) inspectArticle(e.target.value);
      });
    }
  }

  async function handleDbClear() {
    if (!confirm('Are you sure you want to clear all data in PostgreSQL? This will delete all articles, embeddings, and precomputed recommendations so you can start completely fresh.')) {
      return;
    }
    dom.clearBtn.disabled = true;
    dom.clearBtn.textContent = 'Clearing...';
    try {
      const res = await fetch('/api/db/clear', { method: 'POST' });
      const data = await res.json();
      if (res.ok) {
        showToast('PostgreSQL tables cleared. Ready for fresh upload.', 'success');
        await refreshDbHealth();
        loadInspectorArticles();
        loadRecentJobs();
      } else {
        showToast('Clear failed: ' + (data.detail || data.message), 'error');
      }
    } catch (err) {
      showToast('Network error: ' + err.message, 'error');
    } finally {
      dom.clearBtn.disabled = false;
      dom.clearBtn.textContent = '🗑️ Clear Database';
    }
  }

  function onFileChosen(file) {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      showToast('Please select a valid CSV file (.csv)', 'warn');
      return;
    }
    selectedFile = file;
    if (dom.preview && dom.previewName) {
      dom.previewName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      dom.preview.classList.remove('hidden');
    }
    if (dom.uploadBtn) {
      dom.uploadBtn.disabled = false;
      dom.uploadBtn.textContent = `Run Ingestion on ${file.name.slice(0, 20)}...`;
    }
  }

  async function handleUpload() {
    if (!selectedFile) return;
    if (dom.uploadBtn) dom.uploadBtn.disabled = true;
    if (dom.demoBtn) dom.demoBtn.disabled = true;
    showToast('Uploading CSV to PostgreSQL...', 'info');

    try {
      const res = await Api.uploadCsv(selectedFile);
      activeJobId = res.job_id;
      showToast(`Pipeline started: ${res.job_id}`, 'success');
      startPipelineMonitor(res.job_id);
    } catch (err) {
      showToast(`Upload failed: ${err.message}`, 'error');
      if (dom.uploadBtn) dom.uploadBtn.disabled = false;
      if (dom.demoBtn) dom.demoBtn.disabled = false;
    }
  }

  async function handleDemoIngest() {
    if (dom.demoBtn) {
      dom.demoBtn.disabled = true;
      dom.demoBtn.textContent = 'Starting Demo Pipeline...';
    }
    if (dom.uploadBtn) dom.uploadBtn.disabled = true;
    showToast('Starting sample dataset ingestion (100 articles demo)...', 'info');

    try {
      const res = await Api.ingestDefault(100);
      activeJobId = res.job_id;
      showToast(`Demo pipeline active: ${res.job_id}`, 'success');
      startPipelineMonitor(res.job_id);
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
      if (dom.demoBtn) {
        dom.demoBtn.disabled = false;
        dom.demoBtn.textContent = '⚡ Ingest Sample Dataset (100 Articles Demo)';
      }
      if (dom.uploadBtn && selectedFile) dom.uploadBtn.disabled = false;
    }
  }

  async function handleDbInit() {
    dom.initBtn.disabled = true;
    dom.initBtn.textContent = 'Verifying...';
    try {
      const res = await Api.dbInit();
      showToast(res.message, 'success');
      await refreshDbHealth();
    } catch (err) {
      showToast(`Init error: ${err.message}`, 'error');
    } finally {
      dom.initBtn.disabled = false;
      dom.initBtn.textContent = 'Verify PostgreSQL & Extension';
    }
  }

  // ─── Pipeline Monitor & Animations ───────────────────────────────────────
  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = null;
    }
  }

  function startPipelineMonitor(jobId) {
    if (dom.visualizer) dom.visualizer.classList.remove('hidden');
    if (dom.completion) dom.completion.classList.add('hidden');
    if (dom.logWindow) dom.logWindow.innerHTML = '';
    logged.clear();

    resetPipelineNodes();
    appendLog(`[Pipeline Started] Task ${jobId} initiated in PostgreSQL.`);

    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
      try {
        const job = await Api.jobStatus(jobId);

        // Progress bar
        if (dom.progressPct) dom.progressPct.textContent = `${job.progress_percent}%`;
        if (dom.progressFill) dom.progressFill.style.width = `${job.progress_percent}%`;

        // Message & stage
        const rawMsg = job.message || '';
        const stage = rawMsg.startsWith('STAGE:') ? rawMsg.slice(6).split('|')[0] : 'PARSE';
        const cleanMsg = rawMsg.includes('|') ? rawMsg.split('|')[1] : rawMsg;

        if (dom.statusMsg) dom.statusMsg.textContent = cleanMsg;
        if (!logged.has(cleanMsg)) {
          logged.add(cleanMsg);
          appendLog(cleanMsg);
        }

        updatePipelineNodes(stage);

        if (job.status === 'completed') {
          stopPolling();
          setAllNodesDone();
          const refreshed = job.refreshed_articles || 0;
          appendLog('🎉 [Success] All articles and top 10 recommendations stored in PostgreSQL.');
          if (refreshed > 0) {
            appendLog(`🔄 [Reverse Update] Refreshed recommendations for ${refreshed} existing article(s) that this upload affected.`);
          }

          if (dom.completion) {
            dom.completion.classList.remove('hidden');
            const summary = document.getElementById('completion-summary-text');
            if (summary) {
              const refreshText = refreshed > 0
                ? ` Also refreshed recommendations for ${refreshed} previously uploaded article(s).`
                : ' No existing articles needed refreshing.';
              summary.textContent = `Indexed ${job.total_articles} articles in PostgreSQL with 768d vectors and top-10 precomputed recommendations.${refreshText}`;
            }
          }

          showToast('Pipeline Finished! Recommendations are ready in PostgreSQL.', 'success');
          await refreshDbHealth();
          loadInspectorArticles();
          loadRecentJobs();
          if (typeof window.refreshReaderAfterIngest === 'function') {
            await window.refreshReaderAfterIngest();
          }
          if (dom.uploadBtn && selectedFile) dom.uploadBtn.disabled = false;
          if (dom.demoBtn) {
            dom.demoBtn.disabled = false;
            dom.demoBtn.textContent = '⚡ Ingest Sample Dataset (100 Articles Demo)';
          }
        } else if (job.status === 'failed') {
          stopPolling();
          appendLog(`❌ [Failed] ${job.error || 'Pipeline encountered an error.'}`);
          showToast(`Pipeline failed: ${job.error}`, 'error');
          if (dom.uploadBtn && selectedFile) dom.uploadBtn.disabled = false;
          if (dom.demoBtn) {
            dom.demoBtn.disabled = false;
            dom.demoBtn.textContent = '⚡ Ingest Sample Dataset (100 Articles Demo)';
          }
        }
      } catch (e) {
        console.warn('Poll error:', e);
      }
    }, 1200);
  }

  function resetPipelineNodes() {
    ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank', 'node-reverse'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.className = 'pipeline-node';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = 'Waiting';
      }
    });
  }

  function updatePipelineNodes(stage) {
    const stageMap = {
      'PARSE':     { curr: 'node-parse',   done: [] },
      'CLEAN':     { curr: 'node-clean',   done: ['node-parse'] },
      'STORAGE':   { curr: 'node-storage', done: ['node-parse', 'node-clean'] },
      'EMBEDDING': { curr: 'node-embed',   done: ['node-parse', 'node-clean', 'node-storage'] },
      'INDEX':     { curr: 'node-rerank',  done: ['node-parse', 'node-clean', 'node-storage', 'node-embed'] },
      'PGVECTOR':  { curr: 'node-rerank',  done: ['node-parse', 'node-clean', 'node-storage', 'node-embed'] },
      'RERANK':    { curr: 'node-rerank',  done: ['node-parse', 'node-clean', 'node-storage', 'node-embed'] },
      'REVERSE':   { curr: 'node-reverse', done: ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank'] },
      'COMPLETE':  { curr: null,           done: ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank', 'node-reverse'] },
    };

    const conf = stageMap[stage] || stageMap['PARSE'];

    conf.done.forEach(id => {
      const el = document.getElementById(id);
      if (el && !el.classList.contains('done')) {
        el.className = 'pipeline-node done';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = '✓ Done';
      }
    });

    if (conf.curr) {
      const el = document.getElementById(conf.curr);
      if (el) {
        el.className = 'pipeline-node active';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = '⚡ Running...';
      }
    }
  }

  function setAllNodesDone() {
    ['node-parse', 'node-clean', 'node-storage', 'node-embed', 'node-rerank', 'node-reverse'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.className = 'pipeline-node done';
        const st = el.querySelector('.node-status');
        if (st) st.textContent = '✓ Stored';
      }
    });
  }

  function appendLog(msg) {
    if (!dom.logWindow) return;
    const time = new Date().toLocaleTimeString();
    const line = document.createElement('div');
    line.className = 'log-entry';
    line.innerHTML = `<span class="log-time">[${time}]</span> <span>${msg}</span>`;
    dom.logWindow.appendChild(line);
    dom.logWindow.scrollTop = dom.logWindow.scrollHeight;
  }

  // ─── Health & Inspector ───────────────────────────────────────────────────
  async function refreshDbHealth() {
    try {
      const data = await Api.dbHealth();
      const pgBadge = document.getElementById('pg-status-badge');
      const vecBadge = document.getElementById('vector-status-badge');
      const totalArts = document.getElementById('db-total-articles');
      const totalVecs = document.getElementById('db-total-vectors');
      const totalRecs = document.getElementById('db-total-recs');

      if (pgBadge) {
        pgBadge.className = data.connected ? 'stat-badge ready' : 'stat-badge error';
        pgBadge.innerHTML = data.connected ? '🟢 Connected (PostgreSQL)' : '🔴 Disconnected';
      }
      if (vecBadge) {
        vecBadge.className = data.pgvector_ready ? 'stat-badge ready' : 'stat-badge warning';
        vecBadge.innerHTML = data.pgvector_ready ? '⚡ pgvector Ready (768d)' : '⚠️ Extension Missing';
      }
      if (totalArts) totalArts.textContent = (data.total_articles || 0).toLocaleString();
      if (totalVecs) totalVecs.textContent = (data.embedded_articles || 0).toLocaleString();
      if (totalRecs) totalRecs.textContent = (data.total_recommendations || 0).toLocaleString();
    } catch (e) {
      console.warn('DB health check warning:', e);
    }
  }

  async function loadInspectorArticles() {
    if (!dom.inspectorSelect) return;
    try {
      dom.inspectorSelect.innerHTML = '<option value="">-- Loading all articles… --</option>';

      const perPage = 100;
      const placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = '-- Choose an article to inspect precomputed recs --';
      dom.inspectorSelect.innerHTML = '';
      dom.inspectorSelect.appendChild(placeholder);

      // Page through the WHOLE article table so the dropdown is not capped.
      let page = 1;
      let total = Infinity;
      let count = 0;
      while (count < total && page <= 500) {
        const data = await Api.articles({ page, perPage, sort: 'date' });
        total = data.total || 0;
        const arts = data.articles || [];
        arts.forEach(art => {
          const opt = document.createElement('option');
          opt.value = art.id;
          opt.textContent = `[${art.section || 'General'}] ${art.title.slice(0, 80)}`;
          dom.inspectorSelect.appendChild(opt);
        });
        count += arts.length;
        if (arts.length < perPage) break;   // last page reached
        page += 1;
      }
      placeholder.textContent = `-- Choose an article to inspect precomputed recs (${count}) --`;
    } catch (e) {
      console.warn('Inspector list warning:', e);
    }
  }

  async function inspectArticle(articleId) {
    if (!dom.inspectorBody) return;
    dom.inspectorBody.innerHTML = '<tr><td colspan="5" style="text-align:center;padding:20px;">Fetching from PostgreSQL in sub-5ms...</td></tr>';

    try {
      const res = await Api.precomputedRecs(articleId);
      const recs = res.recommendations || [];

      if (recs.length === 0) {
        dom.inspectorBody.innerHTML = '';
        if (dom.inspectorEmpty) dom.inspectorEmpty.classList.remove('hidden');
        return;
      }

      if (dom.inspectorEmpty) dom.inspectorEmpty.classList.add('hidden');
      dom.inspectorBody.innerHTML = '';

      recs.forEach(rec => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td><span class="rank-badge">#${rec.rank}</span></td>
          <td style="font-weight:600;color:var(--text-primary);max-width:300px;">
            <a href="${rec.url || '#'}" target="_blank" style="color:inherit;text-decoration:none;">${rec.title}</a>
          </td>
          <td><span class="cat-pill" style="font-size:0.75rem;">${rec.section || 'General'}</span></td>
          <td><span class="score-badge">${(rec.score * 100).toFixed(1)}%</span></td>
          <td style="font-size:0.75rem;color:var(--text-muted);">${rec.method || 'neural_hybrid_rerank'}</td>
        `;
        dom.inspectorBody.appendChild(tr);
      });
    } catch (err) {
      dom.inspectorBody.innerHTML = `<tr><td colspan="5" style="color:var(--color-danger);text-align:center;">Failed: ${err.message}</td></tr>`;
    }
  }

  function formatWhen(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    if (isNaN(d.getTime())) return '—';
    return d.toLocaleString(undefined, {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
    });
  }

  async function loadRecentJobs() {
    const summaryEls = {
      uploads:   document.getElementById('summary-uploads'),
      articles:  document.getElementById('summary-articles'),
      refreshed: document.getElementById('summary-refreshed'),
      failed:    document.getElementById('summary-failed'),
    };

    if (!dom.recentJobsBody) return;
    try {
      const data = await Api.recentJobs();
      const jobs = data.jobs || [];
      const summary = data.summary || {};

      // 1. Aggregate upload-history summary
      if (summaryEls.uploads)   summaryEls.uploads.textContent   = summary.total_jobs ?? jobs.length;
      if (summaryEls.articles)  summaryEls.articles.textContent  = (summary.articles_ingested || 0).toLocaleString();
      if (summaryEls.refreshed) summaryEls.refreshed.textContent = (summary.refreshed_articles || 0).toLocaleString();
      if (summaryEls.failed)    summaryEls.failed.textContent    = summary.failed_jobs || 0;

      const running = jobs.find(j => j.status === 'processing' || j.status === 'pending');
      if (running && !pollInterval) {
        activeJobId = running.id;
        if (dom.demoBtn) dom.demoBtn.disabled = true;
        if (dom.uploadBtn) dom.uploadBtn.disabled = true;
        startPipelineMonitor(running.id);
      }

      if (jobs.length === 0) {
        dom.recentJobsBody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:14px;">No uploads recorded yet.</td></tr>';
        return;
      }

      dom.recentJobsBody.innerHTML = '';
      jobs.forEach(job => {
        const tr = document.createElement('tr');
        const badgeClass = job.status === 'completed' ? 'ready' : (job.status === 'failed' ? 'error' : 'warning');
        const refreshed = job.refreshed_articles || 0;
        const articlesCell = job.status === 'completed'
          ? `${job.total_articles}`
          : `${job.processed_articles}/${job.total_articles || '?'}`;
        tr.innerHTML = `
          <td style="font-size:0.78rem;color:var(--text-secondary);">${formatWhen(job.created_at)}</td>
          <td>
            <div class="job-file">${job.filename || '—'}</div>
            <div class="job-id">${job.id}</div>
          </td>
          <td>${articlesCell}</td>
          <td><span class="job-refresh-count ${refreshed > 0 ? '' : 'zero'}">${refreshed > 0 ? refreshed : '—'}</span></td>
          <td><span class="stat-badge ${badgeClass}">${job.status}</span></td>
        `;
        dom.recentJobsBody.appendChild(tr);
      });
    } catch (e) {
      console.warn('Recent jobs warning:', e);
    }
  }

  return { init, refreshDbHealth, inspectArticle };
})();

// Expose to window so app.js can detect and call Editor.init()
window.Editor = Editor;
