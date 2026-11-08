// Step-through diagrams, in two kinds.
//   <figure class="stepper" data-steps="N" data-src="...step-#.svg">
//     one standalone SVG per step, swapped by the controls.
//   <figure class="stepper" data-beats="2.9,5.7,..."> around an animated <img>
//     the SVG is loaded inline and its animation plays one beat at a time,
//     pausing at each time listed in data-beats.
// The controls are the arrows, the dots, a click on the picture and the arrow keys.
(function () {
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var arrowPrev = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5"/></svg>';
  var arrowNext = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3l5 5-5 5"/></svg>';

  function buildBar(figure, total) {
    var bar = document.createElement('div');
    bar.className = 'stepper-bar';
    var dots = '';
    for (var i = 1; i <= total; i += 1) dots += '<li><button type="button" data-go="' + i + '" aria-label="Step ' + i + '"></button></li>';
    bar.innerHTML = '<button type="button" class="stepper-prev" aria-label="Previous step">' + arrowPrev + '</button>' +
      '<ol class="stepper-dots">' + dots + '</ol>' +
      '<span class="stepper-count" aria-live="polite"></span>' +
      '<button type="button" class="stepper-next">Next' + arrowNext + '</button>';
    var caption = figure.querySelector('figcaption');
    figure.insertBefore(bar, caption);
    return bar;
  }

  function wire(figure, total, view, go) {
    var bar = figure.querySelector('.stepper-bar') || buildBar(figure, total);
    var prev = bar.querySelector('.stepper-prev');
    var next = bar.querySelector('.stepper-next');
    var count = bar.querySelector('.stepper-count');
    var dots = bar.querySelectorAll('[data-go]');
    var step = 0;

    function show(value, back) {
      step = Math.min(Math.max(value, 1), total);
      go(step, back);
      dots.forEach(function (dot) {
        var at = Number(dot.getAttribute('data-go'));
        dot.classList.toggle('done', at < step);
        if (at === step) dot.setAttribute('aria-current', 'step');
        else dot.removeAttribute('aria-current');
      });
      prev.disabled = step === 1;
      next.firstChild.nodeValue = step === total ? 'Again' : 'Next';
      count.textContent = step + ' / ' + total;
    }
    function forward() { show(step === total ? 1 : step + 1); }

    prev.addEventListener('click', function () { show(step - 1, true); });
    next.addEventListener('click', forward);
    view.addEventListener('click', forward);
    dots.forEach(function (dot) {
      dot.addEventListener('click', function () { show(Number(dot.getAttribute('data-go')), true); });
    });
    view.setAttribute('tabindex', '0');
    view.classList.add('stepper-view');
    view.addEventListener('keydown', function (event) {
      if (event.key === 'ArrowRight' || event.key === ' ') forward();
      else if (event.key === 'ArrowLeft') show(step - 1, true);
      else if (event.key === 'Home') show(1, true);
      else if (event.key === 'End') show(total, true);
      else return;
      event.preventDefault();
    });
    figure.classList.add('ready');
    return show;
  }

  function swapper(figure) {
    var total = Number(figure.getAttribute('data-steps')) || 1;
    var pattern = figure.getAttribute('data-src');
    var view = figure.querySelector('img');
    for (var i = 1; i <= total; i += 1) new Image().src = pattern.replace('#', i);
    var show = wire(figure, total, view, function (step) { view.src = pattern.replace('#', step); });
    show(1);
  }

  function player(figure) {
    var img = figure.querySelector('img');
    var beats = figure.getAttribute('data-beats').split(',').map(Number);
    fetch(img.getAttribute('src')).then(function (response) { return response.text(); }).then(function (text) {
      var host = document.createElement('div');
      host.className = 'player-view';
      host.setAttribute('role', 'img');
      host.setAttribute('aria-label', img.getAttribute('alt') || '');
      var root = host.attachShadow({ mode: 'open' });
      root.innerHTML = '<style>:host{display:block}svg{display:block;width:100%;height:auto}</style>' + text;
      img.replaceWith(host);
      var anims = root.getAnimations();
      var frame = 0;

      function seek(t) {
        window.cancelAnimationFrame(frame);
        anims.forEach(function (a) { a.pause(); a.currentTime = t * 1000; });
      }
      function playTo(from, to) {
        seek(from);
        if (reduce) { seek(to); return; }
        anims.forEach(function (a) { a.play(); });
        var start = performance.now();
        (function tick(now) {
          if (from + (now - start) / 1000 >= to) { seek(to); return; }
          frame = window.requestAnimationFrame(tick);
        })(start);
      }

      var show = wire(figure, beats.length, host, function (step, back) {
        var target = beats[step - 1];
        var current = anims.length ? anims[0].currentTime / 1000 : 0;
        var from = step === 1 ? 0 : beats[step - 2];
        // Going forward continues from where the picture is; going back or
        // jumping replays the chosen step from its start.
        if (!back && Math.abs(current - from) < 0.05) playTo(current, target);
        else playTo(from, target);
      });
      seek(0);
      var started = false;
      var watch = new IntersectionObserver(function (entries) {
        if (started || !entries[0].isIntersecting) return;
        started = true;
        watch.disconnect();
        show(1);
      }, { threshold: 0.4 });
      watch.observe(host);
    }).catch(function () { /* keep the looping image */ });
  }

  document.querySelectorAll('.stepper').forEach(function (figure) {
    if (figure.hasAttribute('data-beats')) player(figure);
    else swapper(figure);
  });
})();
