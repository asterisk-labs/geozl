/* Codec pager, the only thing this file does. The active nav link is marked in
   the HTML of each page, which is why there is no runtime pass for it here.

   CODECS is the catalog order and has to match the cards in docs.html, the
   table in the README and the codec pages. test_catalog.py checks the copies. */
(() => {
  "use strict";

  const CODECS = [
    { no: "01", name: "delta_w",      file: "delta-w.html",      fam: "purple" },
    { no: "02", name: "delta_n",      file: "delta-n.html",      fam: "purple" },
    { no: "03", name: "planar",       file: "planar.html",       fam: "purple" },
    { no: "04", name: "med",          file: "med.html",          fam: "purple" },
    { no: "05", name: "average",      file: "average.html",      fam: "purple" },
    { no: "06", name: "wp_static",    file: "wp-static.html",    fam: "purple" },
    { no: "07", name: "deinterleave", file: "deinterleave.html", fam: "green"  },
    { no: "08", name: "nodata",       file: "nodata.html",       fam: "green"  },
    { no: "09", name: "pfor",         file: "pfor.html",         fam: "green"  },
    { no: "10", name: "quant_linear", file: "quant-linear.html", fam: "orange" },
    { no: "11", name: "quant_log",    file: "quant-log.html",    fam: "orange" },
    { no: "12", name: "quant_sqrt",   file: "quant-sqrt.html",   fam: "orange" }
  ];

  const pager = document.querySelector(".pager");
  if (!pager) return;

  const here = location.pathname.split("/").pop();
  const i = CODECS.findIndex((c) => c.file === here);
  if (i < 0) return;

  const prev = CODECS.at(i - 1);
  const next = CODECS[(i + 1) % CODECS.length];

  const arrow = (dir) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${
    dir === "prev"
      ? '<path d="M19 12H5"/><path d="m12 19-7-7 7-7"/>'
      : '<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>'
  }</svg>`;

  const link = (dir, c) => {
    const a = document.createElement("a");
    a.className = `pg ${dir}`;
    a.href = c.file;
    a.style.setProperty("--fam", `var(--${c.fam})`);
    a.setAttribute("aria-label", `${dir === "prev" ? "Previous" : "Next"} codec, ${c.name}`);

    const label = `<span class="pg-lab"><span class="pg-k">${dir} &middot; ${c.no}</span><span class="pg-n">${c.name}</span></span>`;
    const swatch = '<span class="pg-fam"></span>';
    a.innerHTML = dir === "prev" ? arrow(dir) + swatch + label : label + swatch + arrow(dir);
    return a;
  };

  pager.prepend(link("prev", prev));
  pager.append(link("next", next));

  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "ArrowLeft") location.href = prev.file;
    if (e.key === "ArrowRight") location.href = next.file;
  });
})();
