/* cover.js, reads DECK config, injects cover slide into #cover-slot */

var LOGO = 'assets/logo.svg';
var ICON = 'assets/icon.svg';

function img(src, cls, h) {
  return '<img src="' + src + '" class="' + (cls || '') + '" style="height:' + (h || 28) + 'px" alt="">';
}

function triBar() {
  return '<span></span><span></span><span></span>';
}

// the cover's right half: the three datasets of the "how it works" slide, each
// with its own graph. A small pulse runs from the tile through the codecs into
// the file, tinting each codec as it passes; on IMERG it splits into both
// branches. Kept quiet on purpose: the title is what people should read first.
function startDatasetGraph(svg) {
  var NS = 'http://www.w3.org/2000/svg';
  var VIOLET = '#492ae8', INK = '#17161C', WIRE = '#D6D3D1';
  var GEO = { med: 1, wp_static: 1, planar: 1, pfor: 1 };
  // outlined chips, tinted when the pulse is inside them
  var EDGE = { zigzag: '#D9B300', entropy: INK, split: '#9CA3AF' };
  var TINT = { zigzag: '#FEF3C2', entropy: '#EEF0F2', split: '#EEF0F2' };
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var W = 84, H = 26, COLX = [118, 218, 318], BRANCH = 40, OUTX = 372;
  var rows = [
    { name: 'Sentinel-2', type: 'optical', tile: 'assets/img/tiles/t01.jpg', h: 112,
      nodes: [['a', 'med', 0, 0], ['b', 'zigzag', 1, 0], ['c', 'entropy', 2, 0]],
      edges: [['a', 'b'], ['b', 'c']], first: 'a', last: 'c', routes: [['a', 'b', 'c']] },
    { name: 'Copernicus DEM', type: 'elevation', tile: 'assets/img/tiles/t03.jpg', h: 112,
      nodes: [['a', 'wp_static', 0, 0], ['b', 'zigzag', 1, 0], ['c', 'entropy', 2, 0]],
      edges: [['a', 'b'], ['b', 'c']], first: 'a', last: 'c', routes: [['a', 'b', 'c']] },
    { name: 'GPM IMERG', type: 'rain', tile: 'assets/img/tiles/t13.jpg', h: 150,
      nodes: [['s', 'split', 0, 0], ['m', 'med', 1, 1], ['e', 'entropy', 2, 0]],
      edges: [['s', 'e'], ['s', 'm'], ['m', 'e']], first: 's', last: 'e', routes: [['s', 'e'], ['s', 'm', 'e']] }
  ];

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  // a smooth step between two points, flat where they share a row
  function seg(a, b) {
    if (a.y === b.y) return ' L ' + b.x + ' ' + b.y;
    var mx = (a.x + b.x) / 2;
    return ' C ' + mx + ' ' + a.y + ' ' + mx + ' ' + b.y + ' ' + b.x + ' ' + b.y;
  }

  var defs = el('defs', {}, svg);
  var mk = el('marker', { id: 'cg-arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto' }, defs);
  el('path', { d: 'M 0 0 L 10 5 L 0 10 z', fill: WIRE }, mk);

  var top = 0, live = [];
  rows.forEach(function (row, ri) {
    var g = el('g', {}, svg), y = top + 54, pos = {};
    var title = el('text', { x: 0, y: top + 16, 'font-size': 16, 'font-weight': 700, fill: INK }, g);
    title.textContent = row.name;
    var sub = el('tspan', { dx: 8, 'font-size': 12.5, 'font-weight': 400, fill: '#6B7280' }, title);
    sub.textContent = row.type;

    // the tile, cut to our hexagon
    var cx = 32, r = 30, clip = 'cg-hex' + ri;
    var cp = el('clipPath', { id: clip }, defs);
    el('polygon', { points: [[cx, y - r], [cx + 26, y - 15], [cx + 26, y + 15], [cx, y + r], [cx - 26, y + 15], [cx - 26, y - 15]].map(function (p) { return p.join(','); }).join(' ') }, cp);
    el('image', { href: row.tile, x: cx - 30, y: y - 30, width: 60, height: 60, preserveAspectRatio: 'xMidYMid slice', 'clip-path': 'url(#' + clip + ')' }, g);

    row.nodes.forEach(function (n) { pos[n[0]] = { x: COLX[n[2]], y: y + n[3] * BRANCH, name: n[1] }; });
    var L = function (id) { return { x: pos[id].x - W / 2, y: pos[id].y }; };
    var R = function (id) { return { x: pos[id].x + W / 2, y: pos[id].y }; };
    var tileOut = { x: cx + 30, y: y }, out = { x: OUTX, y: y };

    // wires under the chips
    var wire = { stroke: WIRE, 'stroke-width': 1.4, fill: 'none' };
    el('path', Object.assign({ d: 'M ' + tileOut.x + ' ' + y + seg(tileOut, L(row.first)) }, wire), g);
    row.edges.forEach(function (e) { var a = R(e[0]), b = L(e[1]); el('path', Object.assign({ d: 'M ' + a.x + ' ' + a.y + seg(a, b) }, wire), g); });
    el('path', Object.assign({ d: 'M ' + R(row.last).x + ' ' + y + ' L ' + (out.x - 2) + ' ' + y, 'marker-end': 'url(#cg-arrow)' }, wire), g);

    // pulses travel under the chips, so a codec glows while the pulse is inside it
    var under = el('g', {}, g);

    // chips: white with a coloured edge, and a tint the pulse switches on
    var glows = {};
    row.nodes.forEach(function (n) {
      var p = pos[n[0]], geo = GEO[n[1]], box = { x: p.x - W / 2, y: p.y - H / 2, width: W, height: H, rx: 13 };
      el('rect', Object.assign({ fill: '#FFFFFF' }, box), g);
      glows[n[0]] = el('rect', Object.assign({ fill: geo ? '#E9E5FD' : (TINT[n[1]] || '#EEF0F2'), opacity: 0 }, box), g);
      el('rect', Object.assign({ fill: 'none', stroke: geo ? VIOLET : (EDGE[n[1]] || '#9CA3AF'), 'stroke-width': 1.5 }, box), g);
      var t = el('text', { x: p.x, y: p.y + 5, 'text-anchor': 'middle', 'font-size': 14, 'font-weight': 600, fill: geo ? VIOLET : INK }, g);
      t.textContent = n[1];
    });

    // the compressed file the wire ends in
    var fx = out.x + 4, fy = y - 16;
    el('rect', { x: fx, y: fy, width: 26, height: 32, rx: 4, fill: '#FFFFFF', stroke: '#A8A29E', 'stroke-width': 1.4 }, g);
    for (var bar = 0; bar < 3; bar++) el('rect', { x: fx + 6, y: fy + 8 + bar * 6.5, width: 14, height: 2.6, rx: 1, fill: '#D6D3D1' }, g);

    // one route per branch, from the tile to the arrow head
    var routes = row.routes.map(function (ids) {
      var pts = [tileOut], d;
      ids.forEach(function (id) { pts.push(L(id), R(id)); });
      pts.push(out);
      d = 'M ' + pts[0].x + ' ' + pts[0].y;
      for (var i = 1; i < pts.length; i++) d += seg(pts[i - 1], pts[i]);
      var path = el('path', { d: d, fill: 'none', stroke: 'none' }, under);
      var dot = el('circle', { r: 3.2, fill: VIOLET, opacity: 0 }, under);
      return { path: path, len: path.getTotalLength(), dot: dot, ids: ids };
    });
    live.push({ routes: routes, glows: glows, pos: pos, offset: ri * 2 });
    top += row.h;
  });
  svg.setAttribute('viewBox', '-4 0 452 ' + (top - 30));

  var CYCLE = 6, TRAVEL = 3.6;
  function ease(p) { return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2; }
  function draw(t) {
    live.forEach(function (row) {
      var local = ((t - row.offset) % CYCLE + CYCLE) % CYCLE;
      var p = local < TRAVEL ? ease(local / TRAVEL) : 1;
      var moving = local < TRAVEL;
      var lit = {};
      row.routes.forEach(function (rt) {
        var pt = rt.path.getPointAtLength(p * rt.len);
        var a = moving ? 0.85 : Math.max(0, 0.85 - (local - TRAVEL) / 0.4);
        rt.dot.setAttribute('cx', pt.x); rt.dot.setAttribute('cy', pt.y); rt.dot.setAttribute('opacity', a.toFixed(3));
        rt.ids.forEach(function (id) {
          var c = row.pos[id], dist = Math.hypot(pt.x - c.x, (pt.y - c.y) * 2);
          lit[id] = Math.max(lit[id] || 0, moving ? Math.max(0, 1 - dist / 46) : 0);
        });
      });
      for (var id in row.glows) row.glows[id].setAttribute('opacity', (lit[id] || 0).toFixed(3));
    });
  }

  if (reduceMotion) { draw(TRAVEL + 0.01); return; }
  var slide = svg.closest('.slide'), raf = null, t0 = 0;
  function frame(now) { draw((now - t0) / 1000); raf = requestAnimationFrame(frame); }
  function sync() {
    var on = !slide || slide.classList.contains('active');
    if (on && !raf) { t0 = performance.now(); raf = requestAnimationFrame(frame); }
    if (!on && raf) { cancelAnimationFrame(raf); raf = null; }
  }
  if (slide) new MutationObserver(sync).observe(slide, { attributes: true, attributeFilter: ['class'] });
  draw(0);
  sync();
}

function buildCover(cfg) {
  var slot = document.getElementById('cover-slot');
  var mode = cfg.cover || 'brand';
  var logos = (cfg.logos || []).map(function (l) { return '<img src="' + l + '" alt="">'; }).join('');
  // each name stays on one line; the list only breaks between names
  var people = (cfg.speaker
    ? ['<strong class="author-speaker">' + cfg.speaker + '</strong>'].concat(cfg.authors || [])
    : (cfg.authors || [])).map(function (n) { return '<span class="author">' + n + '</span>'; });
  var affiliationLine = (cfg.affiliations || []).concat(cfg.date ? [cfg.date] : []).join(' &middot; ');
  var plogo = cfg.project_visual === 'dataset-graphs'
    ? '<div class="project-badge project-badge--graph"><svg class="cover-dataset-graph" role="img" aria-label="Three datasets, three compression graphs: Sentinel-2 through med, zigzag and entropy, 2.03 times smaller; Copernicus DEM through wp_static, zigzag and entropy, 6.13 times smaller; GPM IMERG split into two branches, 14.1 times smaller."></svg></div>'
    : (cfg.project_logo
      ? '<div class="project-badge"><img src="' + cfg.project_logo + '" alt=""></div>'
      : '');
  var html = '';

  switch (mode) {
    case 'brand':
      html = '<section class="slide cover-brand">' +
        '<div class="logo-wrap">' + img(LOGO, '', 60) + '</div>' +
        '</section>';
      break;

    case 'conference':
      html = '<section class="slide cover-conference">' +
        '<div class="top-bar">' + triBar() + '</div>' +
        '<div class="text-block">' +
          '<div class="event-line">' + (cfg.event || '') + '</div>' +
          '<h1>' + cfg.title + '</h1>' +
          '<div class="subtitle">' + (cfg.subtitle || '') + '</div>' +
          '<div class="authors">' + people.join(' &middot; ') + '</div>' +
          '<div class="affil">' + affiliationLine + '</div>' +
        '</div>' +
        plogo +
        '<div class="bottom-logos">' +
          img(LOGO, '', 28) +
          '<div class="extra-logos">' + logos + '</div>' +
        '</div>' +
        '</section>';
      break;

    case 'talk':
      html = '<section class="slide cover-talk">' +
        '<div class="logo-mark">' + img(ICON, '', 48) + '</div>' +
        '<h1>' + cfg.title + '</h1>' +
        '<div class="subtitle">' + (cfg.subtitle || '') + '</div>' +
        '<div class="bottom-accent">' + triBar() + '</div>' +
        '</section>';
      break;

    case 'project':
      var projSrc = cfg.project_logo || ICON;
      html = '<section class="slide cover-project">' +
        '<div class="project-logo"><img src="' + projSrc + '" alt=""></div>' +
        '<h1>' + cfg.title + '</h1>' +
        '<div class="subtitle">' + (cfg.subtitle || '') + '</div>' +
        '<div class="powered-by">powered by ' + img(LOGO, '', 18) + '</div>' +
        '</section>';
      break;
  }

  slot.innerHTML = html;
  var graph = slot.querySelector('.cover-dataset-graph');
  if (graph) startDatasetGraph(graph);
}
