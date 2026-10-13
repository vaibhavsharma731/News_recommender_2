/**
 * hero.js — Hero carousel (top trending stories).
 */
const Hero = (() => {
  let slides   = [];
  let current  = 0;
  let timer    = null;
  let navBound = false;

  const carousel = () => document.getElementById('hero-carousel');
  const dotsEl   = () => document.getElementById('hero-dots');

  function buildSlide(article) {
    const div = document.createElement('div');
    div.className = 'hero-slide';
    const color = sectionColor(article.section);
    const imgHTML = article.image
      ? `<img src="${article.image}" alt="${article.title}" loading="eager" onerror="this.style.display='none'" />`
      : `<div style="background:linear-gradient(135deg,#1a1a2e,#0d0d14);position:absolute;inset:0;"></div>`;

    div.innerHTML = `
      ${imgHTML}
      <div class="hero-slide-content">
        <div class="hero-eyebrow">✦ TOP STORY</div>
        <h2 class="hero-title">${article.title}</h2>
        <p class="hero-snippet">${(article.snippet || '').slice(0, 140)}…</p>
        <div class="hero-meta">
          <div class="hero-meta-item">
            <span style="width:9px;height:9px;border-radius:50%;background:${color};display:inline-block"></span>
            ${article.section || ''}
          </div>
          ${article.date ? `<div class="hero-meta-item">📅 ${article.date}</div>` : ''}
          ${article.is_live ? `<div class="hero-meta-item" style="color:#ef4444">🔴 LIVE</div>` : ''}
        </div>
        <div class="hero-actions">
          <a href="${article.url || '#'}" target="${article.url ? '_blank' : '_self'}"
             rel="noopener" class="hero-read-btn" data-id="${article.id}">
            Read Full Story →
          </a>
        </div>
      </div>`;

    const readBtn = div.querySelector('.hero-read-btn');
    if (readBtn && typeof State.addToHistory === 'function') {
      readBtn.addEventListener('click', () => {
        State.addToHistory(article.id);
        if (typeof State.saveHistory === 'function') State.saveHistory();
      });
    }
    return div;
  }

  function buildDot(index) {
    const dot = document.createElement('div');
    dot.className = 'hero-dot' + (index === 0 ? ' active' : '');
    dot.addEventListener('click', () => goTo(index));
    return dot;
  }

  function goTo(index) {
    const c = carousel(); const d = dotsEl();
    if (!c || !d) return;
    c.querySelectorAll('.hero-slide').forEach((s, i) => s.classList.toggle('active', i === index));
    d.querySelectorAll('.hero-dot').forEach((dot, i) => dot.classList.toggle('active', i === index));
    current = index;
  }

  function next() { goTo((current + 1) % slides.length); }
  function prev() { goTo((current - 1 + slides.length) % slides.length); }

  function startAutoPlay() {
    clearInterval(timer);
    timer = setInterval(next, 5000);
  }

  async function init() {
    try {
      const data = await Api.trending(6);
      slides = data.articles || [];
      if (!slides.length) return;

      const c = carousel(); const d = dotsEl();
      c.innerHTML = '';
      d.innerHTML = '';

      slides.forEach((art, i) => {
        const slide = buildSlide(art);
        if (i === 0) slide.classList.add('active');
        c.appendChild(slide);
        d.appendChild(buildDot(i));
      });

      if (!navBound) {
        const prevBtn = document.getElementById('hero-prev');
        const nextBtn = document.getElementById('hero-next');
        if (prevBtn) prevBtn.addEventListener('click', () => { prev(); clearInterval(timer); startAutoPlay(); });
        if (nextBtn) nextBtn.addEventListener('click', () => { next(); clearInterval(timer); startAutoPlay(); });
        navBound = true;
      }

      startAutoPlay();
    } catch (err) {
      carousel().innerHTML = `<div class="empty-state" style="height:340px">
        <div class="empty-icon">⚠️</div><h3>Could not load stories</h3><p>${err.message}</p>
      </div>`;
    }
  }

  return { init };
})();
