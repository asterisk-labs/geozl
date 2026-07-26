/* Chart tooltip. It attaches to any .plot holding data-r points, so it serves
   the scatter and the curve alike, and it picks the point nearest the cursor
   rather than trusting the event target, since the squares are 8px and the
   markers sit on top of them. */
(() => {
  "use strict";
  const plots = [...document.querySelectorAll(".plot")];
  if (!plots.length) return;

  const tip = document.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "status");
  document.body.appendChild(tip);

  const RADIUS = 30;
  const recipe = (g) => g.split(">").map((t, i) => (i ? "<i> &gt; </i>" : "") + t).join("");

  const body = (d) => d.g
    ? `<span class="t-g">${recipe(d.g)}</span><dl>
         <dt>ratio</dt><dd class="hot">${d.r}&times;</dd>
         <dt>encode</dt><dd>${d.e} MB/s</dd>
         <dt>decode</dt><dd>${d.d} MB/s</dd>
         <dt>shannon</dt><dd>${d.s}%</dd>
       </dl>`
    : `<span class="t-g">max_error &plusmn;${d.err} m</span><dl>
         <dt>frame</dt><dd>${d.kb} kB</dd>
         <dt>ratio</dt><dd class="hot">${d.r}&times;</dd>
         <dt>quant step</dt><dd>${d.step}</dd>
       </dl>`;

  /* No cache. Measuring on every move costs a fraction of a millisecond and
     keeps the panel's horizontal scroll from putting the points out of step. */
  const pts = plots.map((svg) => [...svg.querySelectorAll("[data-r]")]);

  let current = null;
  let markers = () => {};

  let hide = () => {
    if (current) current.classList.remove("hi");
    current = null;
    tip.classList.remove("on");
    markers(null);
  };

  let show = (el, x, y) => {
    if (current !== el) {
      if (current) current.classList.remove("hi");
      current = el;
      el.classList.add("hi");
      tip.innerHTML = body(el.dataset);
    }
    const w = tip.offsetWidth, h = tip.offsetHeight, pad = 16;
    let left = x + pad, top = y - h - pad;
    if (left + w > innerWidth - 8) left = x - w - pad;
    if (top < 8) top = y + pad;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = top + "px";
    tip.dataset.owner = "plot";
    tip.classList.add("on");
    markers(el);
  };

  plots.forEach((svg, i) => {
    svg.addEventListener("pointermove", (e) => {
      let best = null, bx = 0, by = 0, bd = RADIUS * RADIUS;
      for (const el of pts[i]) {
        const b = el.getBoundingClientRect();
        const x = b.left + b.width / 2, y = b.top + b.height / 2;
        const dx = x - e.clientX, dy = y - e.clientY, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = el; bx = x; by = y; }
      }
      if (best) show(best, bx, by); else hide();
    });
    svg.addEventListener("pointerleave", hide);
    svg.addEventListener("focusin", (e) => {
      const el = e.target.closest("[data-r]");
      if (!el) return;
      const b = el.getBoundingClientRect();
      show(el, b.left + b.width / 2, b.top + b.height / 2);
    });
    svg.addEventListener("focusout", hide);
  });

  addEventListener("resize", hide);

  /* Cross-highlight with the table, plus guides out to the axes. */
  const scatter = plots[0];
  const halo = scatter.querySelector("#halo");
  const gx = scatter.querySelector("#gx"), gy = scatter.querySelector("#gy");
  let litRow = null;

  markers = (el) => {
    if (!el || !el.dataset.g) {
      [halo, gx, gy].forEach((m) => m.classList.remove("on"));
      if (litRow) { litRow.classList.remove("hi"); litRow = null; }
      return;
    }
    const x = parseFloat(el.getAttribute("x")) + 4, y = parseFloat(el.getAttribute("y")) + 4;
    halo.setAttribute("x", x - 10); halo.setAttribute("y", y - 10);
    gx.setAttribute("x1", 76); gx.setAttribute("y1", y); gx.setAttribute("x2", x); gx.setAttribute("y2", y);
    gy.setAttribute("x1", x); gy.setAttribute("y1", y); gy.setAttribute("x2", x); gy.setAttribute("y2", 330);
    [halo, gx, gy].forEach((m) => m.classList.add("on"));
    if (litRow) litRow.classList.remove("hi");
    litRow = document.querySelector('.grow[data-g="' + el.dataset.g + '"]');
    if (litRow) litRow.classList.add("hi");
  };

  document.querySelectorAll(".grow[data-g]").forEach((row) => {
    row.addEventListener("pointerenter", () => {
      const dot = scatter.querySelector('[data-g="' + row.dataset.g + '"]');
      if (!dot) return;
      const b = dot.getBoundingClientRect();
      show(dot, b.left + b.width / 2, b.top + b.height / 2);
    });
    row.addEventListener("pointerleave", hide);
  });
})();
