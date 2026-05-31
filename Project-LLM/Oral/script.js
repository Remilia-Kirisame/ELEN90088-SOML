/* Oral deck — keyboard nav + active-section highlighting.
   Navigation: ↑/↓, PgUp/PgDn, j/k (vim), Space (advance), Home/End, 1–6 (jump).
*/

(() => {
  const TOTAL = 6;
  const slidesContainer = document.getElementById('slides-container');
  const navLinks = document.querySelectorAll('.sidenav-list a');
  const slides = Array.from(document.querySelectorAll('.slide'));

  const slideById = (n) => document.getElementById(`s${n}`);

  // Lock active highlight to navigation intent while a programmatic
  // scroll is in flight — otherwise IntersectionObserver fires on every
  // intermediate slide the viewport sweeps past during smooth-scroll
  // and the highlight flickers from origin → intermediates → destination.
  let lockedTarget = null;
  let lockReleaseTimer = null;

  function getCurrent() {
    return parseInt(document.body.dataset.currentSlide || '1', 10);
  }

  function setActive(n) {
    document.body.dataset.currentSlide = String(n);
    navLinks.forEach((link) => {
      link.classList.toggle('active', link.dataset.target === String(n));
    });
  }

  function releaseLock() {
    lockedTarget = null;
    if (lockReleaseTimer) {
      clearTimeout(lockReleaseTimer);
      lockReleaseTimer = null;
    }
  }

  function scrollToSlide(n) {
    if (n < 1 || n > TOTAL) return;
    const target = slideById(n);
    if (!target) return;

    // Lock the highlight to the destination, then animate.
    lockedTarget = n;
    setActive(n);

    target.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Release the lock once the scroll settles. `scrollend` is the
    // accurate signal where supported (Chrome 114+, Safari 17.4+);
    // the timeout is a defensive fallback for older browsers and for
    // the rare case scrollend doesn't fire (e.g. target already in view).
    if (lockReleaseTimer) clearTimeout(lockReleaseTimer);
    lockReleaseTimer = setTimeout(releaseLock, 900);
  }

  // ----- Keyboard nav ------------------------------------------------------
  document.addEventListener('keydown', (e) => {
    // Ignore if user is typing in a form field (defensive — we have none, but)
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;

    const current = getCurrent();

    switch (e.key) {
      case 'ArrowDown':
      case 'PageDown':
      case ' ':
      case 'j':
        e.preventDefault();
        scrollToSlide(Math.min(current + 1, TOTAL));
        break;
      case 'ArrowUp':
      case 'PageUp':
      case 'k':
        e.preventDefault();
        scrollToSlide(Math.max(current - 1, 1));
        break;
      case 'Home':
        e.preventDefault();
        scrollToSlide(1);
        break;
      case 'End':
        e.preventDefault();
        scrollToSlide(TOTAL);
        break;
      default:
        if (/^[1-6]$/.test(e.key)) {
          e.preventDefault();
          scrollToSlide(parseInt(e.key, 10));
        }
    }
  });

  // ----- Click handlers on nav links --------------------------------------
  navLinks.forEach((link) => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const n = parseInt(link.dataset.target, 10);
      scrollToSlide(n);
      // Remove DOM focus from the clicked link — otherwise the browser
      // leaves a focus ring stranded on it after the user switches to
      // arrow-key navigation (which doesn't move DOM focus).
      link.blur();
    });
  });

  // ----- Active section detection on organic scroll -----------------------
  // Observer only updates the highlight when no programmatic scroll is
  // in flight; during programmatic scrolls the lock takes precedence.
  const observer = new IntersectionObserver(
    (entries) => {
      if (lockedTarget !== null) return;

      let best = null;
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        if (!best || entry.intersectionRatio > best.intersectionRatio) {
          best = entry;
        }
      }
      if (best) {
        const id = best.target.id;
        const n = parseInt(id.replace('s', ''), 10);
        if (!Number.isNaN(n)) setActive(n);
      }
    },
    {
      root: slidesContainer,
      threshold: [0.3, 0.5, 0.7],
    }
  );

  slides.forEach((s) => observer.observe(s));

  // ----- scrollend: release the lock as soon as the smooth scroll settles
  // Preferred over the timeout because it's accurate; the timeout above
  // is the fallback for browsers without scrollend support.
  slidesContainer.addEventListener('scrollend', () => {
    releaseLock();
  });

  // ----- Initial state -----------------------------------------------------
  setActive(1);

  // ----- Lightbox: click a fullscreen button to expand a figure -----------
  const lightbox = document.getElementById('lightbox');
  const lightboxImg = lightbox.querySelector('.lightbox-img');
  const lightboxCaption = lightbox.querySelector('.lightbox-caption');
  const lightboxClose = lightbox.querySelector('.lightbox-close');
  let lightboxOpen = false;
  let lastFocusedTrigger = null;

  function openLightbox(figureEl, triggerEl) {
    const img = figureEl.querySelector('img');
    const cap = figureEl.querySelector('figcaption');
    if (!img) return;
    lightboxImg.src = img.src;
    lightboxImg.alt = img.alt || '';
    lightboxCaption.textContent = cap ? cap.textContent.trim() : '';
    lastFocusedTrigger = triggerEl || null;
    lightbox.hidden = false;
    lightbox.setAttribute('aria-hidden', 'false');
    // Force reflow so the .open transition runs from opacity:0 → 1.
    void lightbox.offsetWidth;
    lightbox.classList.add('open');
    lightboxOpen = true;
    lightboxClose.focus();
  }

  function closeLightbox() {
    if (!lightboxOpen) return;
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden', 'true');
    // Wait for fade-out before fully hiding so the transition is visible.
    const onEnd = () => {
      lightbox.hidden = true;
      lightboxImg.src = '';
      lightboxCaption.textContent = '';
      lightbox.removeEventListener('transitionend', onEnd);
    };
    lightbox.addEventListener('transitionend', onEnd);
    lightboxOpen = false;
    if (lastFocusedTrigger) {
      lastFocusedTrigger.focus();
      lastFocusedTrigger = null;
    }
  }

  // Each .fullscreen-btn opens its parent figure.
  document.querySelectorAll('.fullscreen-btn').forEach((btn) => {
    btn.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      const fig = btn.closest('.slide-figure');
      if (fig) openLightbox(fig, btn);
    });
  });

  // Close on X-button click, backdrop click, or Escape.
  lightboxClose.addEventListener('click', closeLightbox);
  lightbox.addEventListener('click', (e) => {
    // Only close if the click is on the backdrop itself, not the image or caption.
    if (e.target === lightbox) closeLightbox();
  });

  // Escape-to-close runs before the slide-nav keydown handler. We also need
  // slide-nav keys (arrows / 1-6 / etc.) to be ignored while the lightbox is
  // open — the existing keydown handler is on `document`, so we hook above it
  // with capture phase and stop propagation when appropriate.
  document.addEventListener('keydown', (e) => {
    if (!lightboxOpen) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      closeLightbox();
    } else {
      // Swallow slide-nav keys while lightbox is open so they don't advance
      // the deck behind the modal.
      const navKeys = ['ArrowDown', 'ArrowUp', 'PageDown', 'PageUp',
                       ' ', 'j', 'k', 'Home', 'End'];
      if (navKeys.includes(e.key) || /^[1-6]$/.test(e.key)) {
        e.stopPropagation();
      }
    }
  }, true); // capture phase — runs before the slide-nav handler
})();

/* ============================================================
   Mode toggle — Presentation ⇄ Verbose
   Independent IIFE; does not interact with the nav / lightbox.
   ============================================================ */
(() => {
  const MODES = ['presentation', 'verbose'];
  const STORAGE_KEY = 'dora-oral-mode';

  const toggleEl = document.querySelector('.mode-toggle');
  if (!toggleEl) return;
  const segments = toggleEl.querySelectorAll('[data-mode]');

  function readPreferredMode() {
    const fromUrl = new URLSearchParams(location.search).get('mode');
    if (MODES.includes(fromUrl)) return fromUrl;
    const fromStore = localStorage.getItem(STORAGE_KEY);
    if (MODES.includes(fromStore)) return fromStore;
    return 'presentation';
  }

  function applyMode(mode) {
    document.body.classList.toggle('presentation-mode', mode === 'presentation');
    document.body.classList.toggle('verbose-mode', mode === 'verbose');
    try { localStorage.setItem(STORAGE_KEY, mode); } catch (_) { /* ignore quota errors */ }
    segments.forEach((seg) => {
      seg.classList.toggle('active', seg.dataset.mode === mode);
    });
  }

  function currentMode() {
    return document.body.classList.contains('verbose-mode') ? 'verbose' : 'presentation';
  }

  function setMode(mode) {
    if (!MODES.includes(mode)) return;
    if (currentMode() === mode) return;
    if (document.startViewTransition) {
      document.startViewTransition(() => applyMode(mode));
    } else {
      applyMode(mode);
    }
  }

  function toggleMode() {
    setMode(currentMode() === 'presentation' ? 'verbose' : 'presentation');
  }

  // Apply initial mode synchronously — no transition for the first paint.
  applyMode(readPreferredMode());

  // Click handler on each segment.
  segments.forEach((seg) => {
    seg.addEventListener('click', () => {
      setMode(seg.dataset.mode);
      seg.blur();
    });
  });

  // Global 'm' key toggles mode. Don't fire inside form fields or with
  // modifiers, and don't fire when the lightbox is open (the existing
  // capture-phase lightbox handler will stopPropagation for nav keys, but
  // 'm' isn't in its swallow list, so we guard here defensively).
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === 'm' || e.key === 'M') {
      const lightbox = document.getElementById('lightbox');
      if (lightbox && !lightbox.hidden) return;
      e.preventDefault();
      toggleMode();
    }
  });
})();
