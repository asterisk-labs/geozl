/* engine.js, slide navigation, keyboard/touch input, viewport scaling */

(function () {
  var deck = document.getElementById('deck');
  var bar = document.getElementById('progress');
  var num = document.getElementById('slide-num');
  var btnP = document.getElementById('btn-prev');
  var btnN = document.getElementById('btn-next');

  var slides, cur = 0;

  // build-up within a slide: elements marked data-step="n" appear on the nth press
  // data-on="n" never hides; it gains class "on" from the nth press, for moves and recolours
  function steps(slide) { return Array.from(slide.querySelectorAll('[data-step]')); }
  function ons(slide) { return Array.from(slide.querySelectorAll('[data-on]')); }
  function lastStep(slide) {
    var m = steps(slide).reduce(function (m, e) { return Math.max(m, +e.dataset.step || 1); }, 0);
    return ons(slide).reduce(function (m, e) { return Math.max(m, +e.dataset.on || 1); }, m);
  }
  function showSteps(slide, k) {
    steps(slide).forEach(function (e) { e.classList.toggle('shown', (+e.dataset.step || 1) <= k); });
    ons(slide).forEach(function (e) { e.classList.toggle('on', (+e.dataset.on || 1) <= k); });
    slide.dataset.k = k;
  }

  // arriving forward starts a slide empty, arriving backward shows it complete
  function goTo(i, complete) {
    cur = Math.max(0, Math.min(i, slides.length - 1));
    slides.forEach(function (s, idx) { s.classList.toggle('active', idx === cur); });
    showSteps(slides[cur], complete ? lastStep(slides[cur]) : 0);
    num.textContent = (cur + 1) + ' / ' + slides.length;
    btnP.disabled = cur === 0;
    btnN.disabled = cur === slides.length - 1;
    bar.style.width = ((cur + 1) / slides.length * 100) + '%';
  }

  function next() {
    var s = slides[cur], k = +(s.dataset.k || 0);
    if (k < lastStep(s)) showSteps(s, k + 1);
    else if (cur < slides.length - 1) goTo(cur + 1, false);
  }
  function prev() {
    var s = slides[cur], k = +(s.dataset.k || 0);
    if (k > 0) showSteps(s, k - 1);
    else if (cur > 0) goTo(cur - 1, true);
  }

  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); next(); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
    if (e.key === 'Home') { e.preventDefault(); goTo(0, false); }
    if (e.key === 'End') { e.preventDefault(); goTo(slides.length - 1, true); }
  });

  btnP.addEventListener('click', prev);
  btnN.addEventListener('click', next);

  var touchX = 0;
  deck.addEventListener('touchstart', function (e) { touchX = e.touches[0].clientX; }, { passive: true });
  deck.addEventListener('touchend', function (e) {
    var dx = e.changedTouches[0].clientX - touchX;
    if (Math.abs(dx) > 50) dx < 0 ? next() : prev();
  }, { passive: true });

  function resize() {
    var s = Math.min(window.innerWidth / 960, window.innerHeight / 540);
    deck.style.transform = 'translate(-50%,-50%) scale(' + s + ')';
  }
  window.addEventListener('resize', resize);

  // house style furniture: the mark bottom-left and the corner triangles.
  // added here so every slide carries it without repeating markup.
  function corner(where) {
    var el = document.createElement('div');
    el.className = 'corner corner-' + where;
    el.innerHTML = '<i class="t1"></i><i class="t2"></i><i class="t3"></i>';
    return el;
  }

  function dressSlide(slide) {
    if (slide.dataset.dressed) return;
    slide.dataset.dressed = '1';

    // a bare slide carries one thing and no furniture at all
    if (slide.hasAttribute('data-bare')) return;

    var isCover = /cover-/.test(slide.className);
    if (!isCover && !slide.querySelector('.deck-footer')) {
      var foot = document.createElement('div');
      foot.className = 'deck-footer';
      foot.innerHTML = '<img src="assets/logo.svg" alt=""><span>asterisk.coop</span>';
      slide.appendChild(foot);
    }

    // four corners frame the deck: cover and closing. one corner elsewhere.
    var frame = isCover || slide.hasAttribute('data-frame');
    (frame ? ['tl', 'tr', 'bl', 'br'] : ['br']).forEach(function (w) {
      slide.appendChild(corner(w));
    });
  }

  // exposed globally so fetch callback in index.html can re-init
  window.initDeck = function () {
    slides = Array.from(deck.querySelectorAll('.slide'));
    if (!slides.length) {
      resize();
      return;
    }
    slides.forEach(dressSlide);
    cur = 0;
    goTo(0, false);
    resize();
  };
})();
