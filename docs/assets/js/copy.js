/* Copy button in the label bar above each code block. */
(() => {
  "use strict";
  const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12"/><path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/></svg>';
  const TICK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

  document.querySelectorAll(".code").forEach((block) => {
    const bar = block.previousElementSibling;
    const pre = block.querySelector("pre");
    if (!bar || !bar.classList.contains("lbl") || !pre) return;

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "cp";
    btn.innerHTML = ICON + "<span>copy</span>";
    btn.setAttribute("aria-label", "Copy this snippet");
    bar.appendChild(btn);

    let timer;
    btn.addEventListener("click", async () => {
      const text = pre.innerText;
      try {
        if (navigator.clipboard && isSecureContext) await navigator.clipboard.writeText(text);
        else {
          const ta = document.createElement("textarea");
          ta.value = text; ta.setAttribute("readonly", "");
          ta.style.cssText = "position:fixed;top:-9999px";
          document.body.appendChild(ta); ta.select();
          document.execCommand("copy"); ta.remove();
        }
        btn.classList.add("done");
        btn.innerHTML = TICK + "<span>copied</span>";
      } catch {
        btn.innerHTML = ICON + "<span>press ctrl c</span>";
      }
      clearTimeout(timer);
      timer = setTimeout(() => {
        btn.classList.remove("done");
        btn.innerHTML = ICON + "<span>copy</span>";
      }, 1800);
    });
  });
})();
