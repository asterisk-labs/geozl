/* cover.js, reads DECK config, injects cover slide into #cover-slot */

var LOGO = 'assets/logo.svg';
var ICON = 'assets/icon.svg';

function img(src, cls, h) {
  return '<img src="' + src + '" class="' + (cls || '') + '" style="height:' + (h || 28) + 'px" alt="">';
}

function triBar() {
  return '<span></span><span></span><span></span>';
}

function startDatasetGraph(canvas, imageSource) {
  var ctx = canvas.getContext('2d');
  var image = new Image();
  var width = 400;
  var height = 400;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var colors = {
    ink: '#1A1A1A',
    edge: '#8B94A1',
    rule: '#E5E7EB',
    dim: '#6B7280',
    violet: '#4E04EB',
    gold: '#D3AC00',
    mint: '#4FAE78',
    ember: '#D9684E'
  };
  var datasets = [
    {
      name: 'SENTINEL-1', type: 'complex radar', crop: [1, 2],
      nodes: [
        ['split', 112, 54, 46, 18, 'SPLIT', colors.mint],
        ['real', 194, 36, 48, 18, 'PLANAR', colors.violet],
        ['imag', 194, 72, 48, 18, 'PLANAR', colors.violet],
        ['realE', 278, 36, 48, 18, 'ENTROPY', colors.gold],
        ['imagE', 278, 72, 48, 18, 'ENTROPY', colors.gold],
        ['out', 365, 54, 28, 44, '', colors.dim, 'output']
      ],
      paths: [
        [['input', 'split', 'real', 'realE', 'outTop'], colors.mint],
        [['input', 'split', 'imag', 'imagE', 'outBottom'], colors.violet]
      ]
    },
    {
      name: 'ERA5', type: 'climate grid', crop: [2, 1],
      nodes: [
        ['nodata', 112, 54, 48, 18, 'NODATA', colors.mint],
        ['planar', 194, 36, 48, 18, 'PLANAR', colors.violet],
        ['generic', 220, 75, 52, 18, 'GENERIC', colors.ember],
        ['entropy', 284, 36, 48, 18, 'ENTROPY', colors.gold],
        ['out', 365, 54, 28, 40, '', colors.dim, 'output']
      ],
      paths: [
        [['input', 'nodata', 'planar', 'entropy', 'outTop'], colors.violet],
        [['input', 'nodata', 'generic', 'outBottom'], colors.ember]
      ]
    },
    {
      name: 'COPERNICUS DEM', type: 'elevation', crop: [0, 1],
      nodes: [
        ['planar', 116, 54, 50, 18, 'PLANAR', colors.violet],
        ['zigzag', 205, 54, 50, 18, 'ZIGZAG', colors.dim],
        ['entropy', 294, 54, 50, 18, 'ENTROPY', colors.gold],
        ['out', 365, 54, 28, 36, '', colors.dim, 'output']
      ],
      paths: [
        [['input', 'planar', 'zigzag', 'entropy', 'out'], colors.violet]
      ]
    },
    {
      name: 'ESA WORLDCOVER', type: 'categorical', crop: [2, 0],
      nodes: [
        ['entropy', 205, 54, 54, 18, 'ENTROPY', colors.gold],
        ['out', 365, 54, 28, 36, '', colors.dim, 'output']
      ],
      paths: [
        [['input', 'entropy', 'out'], colors.gold]
      ]
    }
  ];

  function configureCanvas() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function nodeMap(dataset) {
    var map = { input: {x: 58, y: 54} };
    dataset.nodes.forEach(function (node) {
      map[node[0]] = {x: node[1], y: node[2]};
      if (node[0] === 'out') {
        map.outTop = {x: node[1], y: node[2] - 9};
        map.outBottom = {x: node[1], y: node[2] + 9};
      }
    });
    return map;
  }

  function curveControls(a, b) {
    var middle = (a.x + b.x) / 2;
    return [{x: middle, y: a.y}, {x: middle, y: b.y}];
  }

  function curvePoint(a, b, progress) {
    var controls = curveControls(a, b);
    var inverse = 1 - progress;
    return {
      x: inverse * inverse * inverse * a.x + 3 * inverse * inverse * progress * controls[0].x + 3 * inverse * progress * progress * controls[1].x + progress * progress * progress * b.x,
      y: inverse * inverse * inverse * a.y + 3 * inverse * inverse * progress * controls[0].y + 3 * inverse * progress * progress * controls[1].y + progress * progress * progress * b.y
    };
  }

  function drawPath(points, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.62;
    ctx.lineWidth = 1.35;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (var index = 1; index < points.length; index += 1) {
      var controls = curveControls(points[index - 1], points[index]);
      ctx.bezierCurveTo(controls[0].x, controls[0].y, controls[1].x, controls[1].y, points[index].x, points[index].y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function pointOnPath(points, progress) {
    var scaled = progress * (points.length - 1);
    var segment = Math.min(points.length - 2, Math.floor(scaled));
    return curvePoint(points[segment], points[segment + 1], scaled - segment);
  }

  function drawTile(dataset, rowTop) {
    var sourceSize = Math.floor(Math.min(image.naturalWidth, image.naturalHeight) / 3);
    var step = (Math.min(image.naturalWidth, image.naturalHeight) - sourceSize) / 2;
    var sourceX = dataset.crop[0] * step;
    var sourceY = dataset.crop[1] * step;
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(8, rowTop + 29, 42, 42, 3);
    ctx.clip();
    ctx.drawImage(image, sourceX, sourceY, sourceSize, sourceSize, 8, rowTop + 29, 42, 42);
    ctx.restore();
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(8.5, rowTop + 29.5, 41, 41);
  }

  function drawNode(node, rowTop) {
    var x = node[1] - node[3] / 2;
    var y = rowTop + node[2] - node[4] / 2;
    var output = node[7] === 'output';
    ctx.save();
    ctx.fillStyle = output ? '#FFFFFF' : '#FAFAFB';
    ctx.strokeStyle = node[6];
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ctx.roundRect(x, y, node[3], node[4], output ? 4 : 3);
    ctx.fill();
    ctx.stroke();
    if (output) {
      ctx.fillStyle = '#C7CED7';
      for (var bar = 0; bar < 3; bar += 1) ctx.fillRect(x + 7, y + 9 + bar * 7, node[3] - 14, 3);
    } else {
      ctx.fillStyle = node[6];
      ctx.font = '600 6.5px "SFMono-Regular", Consolas, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(node[5], node[1], rowTop + node[2] + 0.3);
    }
    ctx.restore();
  }

  function drawDataset(dataset, datasetIndex, time) {
    var rowTop = datasetIndex * 100;
    var map = nodeMap(dataset);
    ctx.save();
    ctx.fillStyle = colors.ink;
    ctx.font = '700 10.5px "SFMono-Regular", Consolas, monospace';
    ctx.textBaseline = 'middle';
    ctx.fillText(dataset.name, 8, rowTop + 12);
    var nameWidth = ctx.measureText(dataset.name).width;
    ctx.fillStyle = colors.dim;
    ctx.font = '500 8px "SFMono-Regular", Consolas, monospace';
    ctx.fillText(dataset.type, 8 + nameWidth + 10, rowTop + 12);
    ctx.restore();

    dataset.paths.forEach(function (path) {
      drawPath(path[0].map(function (id) { return {x: map[id].x, y: rowTop + map[id].y}; }), path[1]);
    });
    drawTile(dataset, rowTop);
    dataset.nodes.forEach(function (node) { drawNode(node, rowTop); });

    dataset.paths.forEach(function (path, pathIndex) {
      var points = path[0].map(function (id) { return {x: map[id].x, y: rowTop + map[id].y}; });
      var progress = reduceMotion ? 0.64 : ((time * 0.00018) + datasetIndex * 0.19 + pathIndex * 0.34) % 1;
      var token = pointOnPath(points, progress);
      ctx.save();
      ctx.fillStyle = path[1];
      ctx.shadowColor = path[1];
      ctx.shadowBlur = 7;
      ctx.beginPath();
      ctx.arc(token.x, token.y, 2.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    });
  }

  function render(time) {
    ctx.clearRect(0, 0, width, height);
    for (var row = 1; row < 4; row += 1) {
      ctx.strokeStyle = colors.rule;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(8, row * 100);
      ctx.lineTo(392, row * 100);
      ctx.stroke();
    }
    datasets.forEach(function (dataset, index) { drawDataset(dataset, index, time || 0); });
    if (!reduceMotion && canvas.isConnected) window.requestAnimationFrame(render);
  }

  configureCanvas();
  image.addEventListener('load', function () { render(0); }, { once: true });
  image.src = imageSource;
}

function buildCover(cfg) {
  var slot = document.getElementById('cover-slot');
  var mode = cfg.cover || 'brand';
  var logos = (cfg.logos || []).map(function (l) { return '<img src="' + l + '" alt="">'; }).join('');
  var people = cfg.speaker
    ? ['<strong class="author-speaker">' + cfg.speaker + '</strong>'].concat(cfg.authors || [])
    : (cfg.authors || []);
  var affiliationLine = (cfg.affiliations || []).concat(cfg.date ? [cfg.date] : []).join(' &middot; ');
  var plogo = cfg.project_visual === 'dataset-graphs'
    ? '<div class="project-badge project-badge--graph"><canvas class="cover-dataset-graph" width="400" height="400" role="img" aria-label="Animated dataset-specific compression graphs for Sentinel-1, ERA5, Copernicus DEM and ESA WorldCover."></canvas></div>'
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
  if (graph) startDatasetGraph(graph, cfg.project_visual_data || 'assets/img/cover-tiles.jpg');
}
