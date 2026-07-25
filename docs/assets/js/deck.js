/* geozl — navegacion de la baraja horizontal. El scroll sigue siendo el del
   navegador, aqui solo se le dice a donde ir y se refleja donde estamos. */
/* Navegacion tipo deck sobre las secciones. El scroll sigue siendo el del
   navegador, aqui solo se le dice a donde ir y se refleja donde estamos. */
(() => {
  "use strict";
  const slides = [...document.querySelectorAll(".api section")];
  const bar = document.querySelector(".deck-bar i");
  const label = document.querySelector(".deck-no");
  const nav = document.querySelector(".deck-nav");
  if (slides.length < 2 || !bar) return;

  const pad = (n) => String(n).padStart(2, "0");
  let at = 0;

  // La tira es horizontal en escritorio y vertical en movil, asi que el eje
  // se lee del layout en vez de duplicar la condicion del CSS.
  const strip = document.querySelector(".api");
  const across = () => {
    const c = getComputedStyle(strip);
    return c.display === "flex" && c.flexDirection === "row";
  };

  const nearest = () => {
    let best = 0, dist = Infinity;
    slides.forEach((s, i) => {
      const b = s.getBoundingClientRect();
      const d = Math.abs(across() ? b.left - strip.getBoundingClientRect().left : b.top);
      if (d < dist) { dist = d; best = i; }
    });
    return best;
  };

  const paint = () => {
    bar.style.width = (100 * (at + 1) / slides.length).toFixed(2) + "%";
    label.innerHTML = "<b>" + pad(at + 1) + "</b> / " + slides.length;
    nav.querySelector('[data-go="-1"]').disabled = at === 0;
    nav.querySelector('[data-go="1"]').disabled = at === slides.length - 1;
  };

  let moving = 0;
  const go = (i) => {
    at = Math.max(0, Math.min(slides.length - 1, i));
    moving = Date.now() + 600;
    slides[at].scrollIntoView(across()
      ? { behavior: "smooth", inline: "start", block: "nearest" }
      : { behavior: "smooth", block: "start" });
    const id = slides[at].id;
    if (id) history.replaceState(null, "", "#" + id);
    paint();
  };

  nav.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-go]");
    if (btn) go(at + Number(btn.dataset.go));
  });

  // Typing in a field or riding a slider must not page the deck.
  const typing = (t) => t.closest("input, textarea, select, [contenteditable]");

  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return;
    const fwd = ["ArrowRight", "ArrowDown", "PageDown", " "];
    const back = ["ArrowLeft", "ArrowUp", "PageUp"];
    if (fwd.includes(e.key)) { e.preventDefault(); go(at + 1); }
    else if (back.includes(e.key)) { e.preventDefault(); go(at - 1); }
    else if (e.key === "Home") { e.preventDefault(); go(0); }
    else if (e.key === "End") { e.preventDefault(); go(slides.length - 1); }
  });

  // Scrolling by hand, or landing on a hash, still has to move the counter.
  let tick;
  const follow = () => {
    clearTimeout(tick);
    tick = setTimeout(() => {
      if (Date.now() < moving) return;
      at = nearest();
      paint();
    }, 90);
  };
  strip.addEventListener("scroll", follow, { passive: true });
  addEventListener("scroll", follow, { passive: true });
  addEventListener("resize", follow);

  const fromHash = slides.findIndex((s) => "#" + s.id === location.hash);
  if (fromHash > 0) {
    // Llegar con un hash tiene que mover la tira, no solo el contador.
    at = fromHash;
    slides[at].scrollIntoView({ behavior: "instant", inline: "start", block: "nearest" });
  } else {
    at = nearest();
  }
  paint();
})();
