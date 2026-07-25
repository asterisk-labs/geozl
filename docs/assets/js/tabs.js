/* geozl — pestanas del selector de funciones. Sin JS los paneles quedan
   todos visibles, que es el estado correcto sin script. */
/* Pestanas de la seccion api. Sin JS los tres paneles quedan visibles. */
(() => {
  "use strict";
  const list = document.querySelector(".fnpick");
  if (!list) return;
  const tabs = [...list.querySelectorAll('[role="tab"]')];

  const select = (tab, focus) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) tab.focus();
  };

  list.addEventListener("click", (e) => {
    const t = e.target.closest('[role="tab"]');
    if (t) select(t, false);
  });

  list.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    if (step) { e.preventDefault(); select(tabs[(i + step + tabs.length) % tabs.length], true); }
    if (e.key === "Home") { e.preventDefault(); select(tabs[0], true); }
    if (e.key === "End") { e.preventDefault(); select(tabs[tabs.length - 1], true); }
  });
})();
