/* Tabs for the function picker. The markup ships the first panel open and the
   other two hidden, so that is also the unscripted state. */
(() => {
  "use strict";
  const list = document.querySelector(".fnpick");
  if (!list) return;
  const tabs = [...list.querySelectorAll('[role="tab"]')];

  const still = matchMedia("(prefers-reduced-motion:reduce)");

  const swap = (tab) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
  };

  const select = (tab, focus) => {
    if (document.startViewTransition && !still.matches) document.startViewTransition(() => swap(tab));
    else swap(tab);
    if (focus) tab.focus();
  };

  list.addEventListener("click", (e) => {
    const t = e.target.closest('[role="tab"]');
    if (t) select(t, false);
  });

  // The tablist eats its own keys. Without this they also reach the window,
  // and on api-high that pages the deck at the same time.
  list.addEventListener("keydown", (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
    const mine = step !== 0 || e.key === "Home" || e.key === "End";
    if (!mine) return;
    e.preventDefault();
    e.stopPropagation();
    if (step) select(tabs[(i + step + tabs.length) % tabs.length], true);
    else if (e.key === "Home") select(tabs[0], true);
    else select(tabs[tabs.length - 1], true);
  });
})();
