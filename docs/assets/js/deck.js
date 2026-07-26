/* Deck navigation for the API page. The scroll is still the browser's. This
   decides where to go and keeps the footer in step. */
(() => {
  "use strict";
  const strip = document.querySelector(".api");
  if (!strip) return;

  const slides = [...strip.querySelectorAll(":scope > section")];
  const rail = document.querySelector(".deck-rail");
  const label = document.querySelector(".deck-no");
  const name = document.querySelector(".deck-name");
  const nav = document.querySelector(".deck-nav");
  if (slides.length < 2 || !rail || !label || !nav) return;

  const wide = matchMedia("(min-width:981px)");
  const still = matchMedia("(prefers-reduced-motion:reduce)");
  const pad = (n) => String(n).padStart(2, "0");
  const clamp = (i) => Math.max(0, Math.min(slides.length - 1, i));
  const titleOf = (s, i) => s.dataset.slide || `slide ${i + 1}`;

  let at = 0;

  // The ticks are the progress bar and the jump list at once, so the slide
  // count lives in the DOM instead of a figure typed into the markup.
  const ticks = slides.map((s, i) => {
    s.tabIndex = -1;
    const t = document.createElement("button");
    t.type = "button";
    t.className = "tick";
    t.dataset.to = String(i);
    t.setAttribute("aria-label", `${pad(i + 1)} ${titleOf(s, i)}`);
    rail.append(t);
    return t;
  });

  const back = nav.querySelector('[data-go="-1"]');
  const fwd = nav.querySelector('[data-go="1"]');

  const paint = () => {
    label.innerHTML = `<b>${pad(at + 1)}</b> / ${pad(slides.length)}`;
    if (name) name.textContent = titleOf(slides[at], at);
    ticks.forEach((t, i) => {
      t.classList.toggle("seen", i <= at);
      t.classList.toggle("on", i === at);
      if (i === at) t.setAttribute("aria-current", "true");
      else t.removeAttribute("aria-current");
    });
    if (back) back.disabled = at === 0;
    if (fwd) fwd.disabled = at === slides.length - 1;
  };

  // A slide is its own scroll box on the wide layout, so it can hold more than
  // it shows. The footer fade is the only sign of that.
  const gauge = () => {
    const s = slides[at];
    const more = wide.matches && s.scrollHeight - s.scrollTop - s.clientHeight > 4;
    document.body.classList.toggle("more-below", more);
  };

  // Measured live and in fractional pixels. A cached origin goes stale the
  // moment the width changes.
  const goal = (i) => strip.scrollLeft
    + slides[i].getBoundingClientRect().left - strip.getBoundingClientRect().left;

  let holding = 0;
  const go = (i, opt = {}) => {
    i = clamp(i);
    const moved = i !== at;
    at = i;
    const s = slides[at];
    if (moved) s.scrollTop = 0;
    holding = Date.now() + 900;
    const behavior = opt.jump || still.matches ? "instant" : "smooth";
    if (wide.matches) strip.scrollTo({ left: goal(at), behavior });
    else s.scrollIntoView({ behavior, block: "start" });
    if (moved && s.id) history.replaceState(null, "", "#" + s.id);
    if (opt.focus) s.focus({ preventScroll: true });
    paint();
    gauge();
  };

  nav.addEventListener("click", (e) => {
    const b = e.target.closest("[data-go]");
    if (b) go(at + Number(b.dataset.go), { focus: true });
  });
  rail.addEventListener("click", (e) => {
    const t = e.target.closest("[data-to]");
    if (t) go(Number(t.dataset.to), { focus: true });
  });

  const inField = (t) => t.closest("input, textarea, select, [contenteditable]");
  const activates = (t) => t.closest("button, a[href], summary");

  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || inField(e.target)) return;

    if (e.key === "ArrowRight") { e.preventDefault(); go(at + 1, { focus: true }); return; }
    if (e.key === "ArrowLeft") { e.preventDefault(); go(at - 1, { focus: true }); return; }
    if (e.key === "Home") { e.preventDefault(); go(0, { focus: true }); return; }
    if (e.key === "End") { e.preventDefault(); go(slides.length - 1, { focus: true }); return; }

    const down = e.key === "ArrowDown" || e.key === "PageDown" || e.key === " ";
    const up = e.key === "ArrowUp" || e.key === "PageUp";
    if (!down && !up) return;
    if (e.key === " " && activates(e.target)) return;
    if (!wide.matches) return;

    // The slide gets these keys first. Only once it has nothing left to give
    // do they turn the page.
    const s = slides[at];
    const room = down
      ? s.scrollHeight - s.scrollTop - s.clientHeight > 2
      : s.scrollTop > 2;
    e.preventDefault();
    if (room) {
      const page = e.key === " " || e.key === "PageDown" || e.key === "PageUp";
      const step = Math.round(s.clientHeight * (page ? 0.84 : 0.3));
      s.scrollBy({ top: down ? step : -step, behavior: still.matches ? "instant" : "smooth" });
    } else {
      go(at + (down ? 1 : -1), { focus: true });
    }
  });

  const nearest = () => {
    if (wide.matches) return clamp(Math.round(strip.scrollLeft / strip.clientWidth));
    const line = innerHeight * 0.35;
    let best = 0;
    slides.forEach((s, i) => { if (s.getBoundingClientRect().top <= line) best = i; });
    return best;
  };

  let tick, rid, pinned = null;
  const follow = () => {
    clearTimeout(tick);
    tick = setTimeout(() => {
      if (pinned !== null || Date.now() < holding) return;
      const i = nearest();
      if (i !== at) {
        at = i;
        if (slides[at].id) history.replaceState(null, "", "#" + slides[at].id);
        paint();
      }
      gauge();
    }, 90);
  };
  strip.addEventListener("scroll", follow, { passive: true });
  addEventListener("scroll", follow, { passive: true });
  slides.forEach((s) => s.addEventListener("scroll", gauge, { passive: true }));
  if ("onscrollend" in window) {
    const done = () => { if (pinned === null) holding = 0; follow(); };
    strip.addEventListener("scrollend", done);
    addEventListener("scrollend", done);
  }

  // A width change moves the slide boundaries but not scrollLeft, so without
  // this the strip ends up parked between two. The index is frozen on the
  // first resize of the burst because the browser re-snaps on its own in the
  // meantime, and the frame of margin is there because the measurements of
  // the current frame are still the old layout's.
  const repin = () => {
    if (pinned === null) pinned = at;
    holding = Date.now() + 1200;
    clearTimeout(rid);
    rid = setTimeout(() => requestAnimationFrame(() => {
      const i = pinned;
      pinned = null;
      go(i, { jump: true });
    }), 140);
  };
  addEventListener("resize", repin);
  wide.addEventListener("change", repin);

  const fromHash = slides.findIndex((s) => s.id && "#" + s.id === location.hash);
  go(fromHash > 0 ? fromHash : nearest(), { jump: true });
})();
