// docs/js/viewer.js — interactive annotation viewer
(() => {
const WIN_SEC = 18;            // visible window
const state = {
  track: null, t0: 0,          // window start (sec)
  hiddenGroups: new Set(),
  cqtOn: false, cqtManifest: null,
};
const els = {};
const cqtTiles = new Map();   // "trackId_idx" -> Image

function sizeCanvas(c, cssH) {
  const W = Math.round(c.clientWidth * devicePixelRatio), H = Math.round(cssH * devicePixelRatio);
  if (c.width !== W) c.width = W;
  if (c.height !== H) c.height = H;
  return [W, H];
}

function buildDOM() {
  const v = document.querySelector('#viewer');
  v.innerHTML = `
    <div class="row">
      <select id="v-select"></select>
      <button id="v-play">▶ Play</button>
      <span id="v-time" class="meta">0:00 / 0:00</span>
      <label><input type="checkbox" id="v-cqt"> CQT background</label>
    </div>
    <canvas id="v-overview" height="46"></canvas>
    <canvas id="v-lyrics" height="34"></canvas>
    <canvas id="v-main" height="300"></canvas>
    <div class="legend" id="v-legend"></div>
    <audio id="v-audio" preload="auto"></audio>
    <div id="tooltip"></div>`;
  ['select','play','time','cqt','overview','lyrics','main','audio','legend']
    .forEach(k => els[k] = v.querySelector('#v-' + k) || document.querySelector('#v-' + k));
  els.tooltip = document.querySelector('#tooltip');
}

const groupOf = kr => window.DEMO.ontology.types[kr]?.group ?? 'onset';
const colorOf = kr => window.DEMO.ontology.group_colors[groupOf(kr)];

// --- coordinate mapping -------------------------------------------------
function xOf(sec, w) { return (sec - state.t0) / WIN_SEC * w; }
// log-frequency y mapping shared with CQT tiles: fmin C2, 6 octaves
const FMIN = 65.41, N_OCT = 6;
function yOf(hz, h) {
  const oct = Math.log2(hz / FMIN);
  return h * (1 - Math.min(Math.max(oct / N_OCT, 0), 1));
}

// --- main canvas --------------------------------------------------------
function drawMain() {
  const c = els.main, ctx = c.getContext('2d');
  const [w, h] = sizeCanvas(c, 300);
  ctx.clearRect(0, 0, w, h);
  const t = state.track; if (!t) return;
  const t1 = state.t0 + WIN_SEC;

  if (state.cqtOn) drawCQT(ctx, w, h);          // Task 11 fills this in

  // sigimsae region spans
  for (const r of t.sigimsae_regions) {
    if (r.end < state.t0 || r.start > t1) continue;
    const groups = r.types.map(groupOf);
    if (groups.every(g => state.hiddenGroups.has(g))) continue;
    const x0 = xOf(r.start, w), x1 = xOf(r.end, w);
    const bandH = h / r.types.length;           // multi-label: stacked bands
    r.types.forEach((kr, i) => {
      ctx.fillStyle = colorOf(kr) + '55';
      ctx.fillRect(x0, i * bandH, x1 - x0, bandH);
    });
    ctx.fillStyle = colorOf(r.types[0]);
    ctx.fillRect(x0, h - 6 * devicePixelRatio, x1 - x0, 6 * devicePixelRatio);
  }

  // F0 contour
  const hop = t.f0.hop_sec, hz = t.f0.hz;
  ctx.strokeStyle = '#1c1f24'; ctx.lineWidth = 1.6 * devicePixelRatio;
  ctx.beginPath(); let pen = false;
  const i0 = Math.max(0, Math.floor(state.t0 / hop)), i1 = Math.min(hz.length, Math.ceil(t1 / hop));
  for (let i = i0; i < i1; i++) {
    if (hz[i] == null) { pen = false; continue; }
    const x = xOf(i * hop, w), y = yOf(hz[i], h);
    pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y); pen = true;
  }
  ctx.stroke();

  // playhead
  const now = els.audio.currentTime;
  if (now >= state.t0 && now <= t1) {
    ctx.strokeStyle = '#d33'; ctx.lineWidth = 2 * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(xOf(now, w), 0); ctx.lineTo(xOf(now, w), h); ctx.stroke();
  }
}

// --- lyrics strip -------------------------------------------------------
function drawLyrics() {
  const c = els.lyrics, ctx = c.getContext('2d');
  const [w, h] = sizeCanvas(c, 34);
  ctx.clearRect(0, 0, w, h);
  const t = state.track; if (!t) return;
  ctx.font = `${12 * devicePixelRatio}px sans-serif`; ctx.textBaseline = 'middle';
  for (const r of t.lyrics_regions) {
    if (r.end < state.t0 || r.start > state.t0 + WIN_SEC) continue;
    const x0 = xOf(r.start, w), x1 = xOf(r.end, w);
    ctx.fillStyle = '#e9edf3'; ctx.fillRect(x0 + 1, 2, x1 - x0 - 2, h - 4);
    ctx.fillStyle = '#1c1f24';
    ctx.save(); ctx.beginPath(); ctx.rect(x0 + 4, 0, x1 - x0 - 8, h); ctx.clip();
    ctx.fillText(r.text, Math.max(x0 + 6, 6), h / 2); ctx.restore();
  }
}

// --- overview strip -----------------------------------------------------
function drawOverview() {
  const c = els.overview, ctx = c.getContext('2d');
  const [w, h] = sizeCanvas(c, 46);
  ctx.clearRect(0, 0, w, h);
  const t = state.track; if (!t) return;
  ctx.fillStyle = '#eef1f5'; ctx.fillRect(0, 0, w, h);
  const dur = t.duration_sec;
  for (const r of t.sigimsae_regions) {          // density: one thin bar per region
    ctx.fillStyle = colorOf(r.types[0]) + '99';
    ctx.fillRect(r.start / dur * w, h * 0.2, Math.max(1, (r.end - r.start) / dur * w), h * 0.6);
  }
  ctx.fillStyle = '#4C72B033';                    // current window
  ctx.fillRect(state.t0 / dur * w, 0, WIN_SEC / dur * w, h);
  const now = els.audio.currentTime;
  ctx.fillStyle = '#d33'; ctx.fillRect(now / dur * w - 1, 0, 2, h);
}

function drawCQT(ctx, w, h) {
  const m = state.cqtManifest?.[state.track.id];
  if (!m) return;
  const t1 = state.t0 + WIN_SEC;
  const first = Math.floor(state.t0 / m.tile_sec), last = Math.floor(t1 / m.tile_sec);
  for (let i = Math.max(0, first); i <= Math.min(last, m.n_tiles - 1); i++) {
    const key = `${state.track.id}_${i}`;
    let img = cqtTiles.get(key);
    if (!img) {
      img = new Image();
      img.onload = redraw;
      img.src = `assets/cqt/${state.track.id}_${String(i).padStart(3, '0')}.png`;
      cqtTiles.set(key, img);
    }
    if (!img.complete || !img.naturalWidth) continue;
    const x0 = xOf(i * m.tile_sec, w);
    const tileW = m.tile_sec / WIN_SEC * w;
    ctx.globalAlpha = 0.5;
    ctx.drawImage(img, x0, 0, tileW * (img.naturalWidth / (m.px_per_sec * m.tile_sec)), h);
    ctx.globalAlpha = 1;
  }
}

function redraw() { drawOverview(); drawLyrics(); drawMain(); updateTime(); }

function updateTime() {
  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  els.time.textContent = state.track
    ? `${fmt(els.audio.currentTime)} / ${fmt(state.track.duration_sec)}` : '';
}

// --- legend -------------------------------------------------------------
function buildLegend() {
  const o = window.DEMO.ontology;
  els.legend.innerHTML = o.groups.map(g =>
    `<span data-g="${g}"><span class="sw" style="background:${o.group_colors[g]}"></span>${o.group_labels[g]}</span>`
  ).join('');
  els.legend.querySelectorAll('span[data-g]').forEach(el => el.addEventListener('click', () => {
    const g = el.dataset.g;
    state.hiddenGroups.has(g) ? state.hiddenGroups.delete(g) : state.hiddenGroups.add(g);
    el.classList.toggle('off');
    redraw();
  }));
}

// --- track loading ------------------------------------------------------
function loadTrack(id) {
  state.track = window.DEMO.tracks[id];
  state.t0 = 0; state.hiddenGroups.clear();
  els.audio.src = `assets/audio/${id}.mp3`;
  els.select.value = id;
  els.legend.querySelectorAll('.off').forEach(e => e.classList.remove('off'));
  redraw();
}

function init() {
  buildDOM(); buildLegend();
  const ids = Object.keys(window.DEMO.tracks);
  els.select.innerHTML = ids.map(id =>
    `<option value="${id}">${window.DEMO.tracks[id].title}</option>`).join('');
  els.select.addEventListener('change', () => loadTrack(els.select.value));
  document.addEventListener('load-track', e => loadTrack(e.detail));
  window.addEventListener('resize', redraw);
  wirePlayback();      // Task 10
  wirePointer();       // Task 10
  wireCQT();           // Task 11
  loadTrack(ids[0]);
}
function wirePlayback() {
  els.play.addEventListener('click', () => {
    els.audio.paused ? els.audio.play() : els.audio.pause();
  });
  els.audio.addEventListener('play',  () => { els.play.textContent = '⏸ Pause'; tick(); });
  els.audio.addEventListener('pause', () => { els.play.textContent = '▶ Play'; });
  function tick() {
    if (els.audio.paused) return;
    const now = els.audio.currentTime;
    // auto-scroll: keep playhead inside the window, jump-scroll at 85%
    if (now > state.t0 + WIN_SEC * 0.85 || now < state.t0) {
      state.t0 = Math.max(0, now - WIN_SEC * 0.15);
    }
    redraw();
    requestAnimationFrame(tick);
  }
}

function wirePointer() {
  // overview: click = seek
  els.overview.addEventListener('click', (e) => {
    const frac = e.offsetX / els.overview.clientWidth;
    seekTo(frac * state.track.duration_sec);
  });
  // main canvas: click region = play from region start; hover = tooltip
  els.main.addEventListener('click', (e) => {
    els.tooltip.style.display = 'none';
    const r = regionAt(e);
    seekTo(r ? r.start : state.t0 + e.offsetX / els.main.clientWidth * WIN_SEC);
  });
  els.main.addEventListener('mousemove', (e) => {
    const r = regionAt(e);
    if (!r) { els.tooltip.style.display = 'none'; return; }
    const o = window.DEMO.ontology;
    els.tooltip.innerHTML = r.types.map(kr =>
      `<b>${kr}</b> · ${o.types[kr]?.roman ?? ''} · ${o.types[kr]?.en ?? ''}`).join('<br>');
    els.tooltip.style.display = 'block';
    els.tooltip.style.left = (e.clientX + 12) + 'px';
    els.tooltip.style.top  = (e.clientY + 12) + 'px';
  });
  els.main.addEventListener('mouseleave', () => els.tooltip.style.display = 'none');

  function regionAt(e) {
    const sec = state.t0 + e.offsetX / els.main.clientWidth * WIN_SEC;
    return state.track.sigimsae_regions.find(r =>
      r.start <= sec && sec <= r.end &&
      !r.types.every(kr => state.hiddenGroups.has(groupOf(kr))));
  }
  function seekTo(sec) {
    els.audio.currentTime = Math.max(0, Math.min(sec, state.track.duration_sec - 0.1));
    state.t0 = Math.max(0, els.audio.currentTime - WIN_SEC * 0.15);
    if (els.audio.paused) els.audio.play();
    redraw();
  }
}
function wireCQT() {
  els.cqt.addEventListener('change', async () => {
    state.cqtOn = els.cqt.checked;
    if (state.cqtOn && !state.cqtManifest) {
      try { state.cqtManifest = await fetch('assets/cqt/manifest.json').then(r => r.json()); }
      catch { state.cqtManifest = {}; }
    }
    redraw();
  });
}
window.VIEWER = { state, els, redraw, drawCQT, wirePlayback, wirePointer, wireCQT, xOf, WIN_SEC };

document.addEventListener('tracks-ready', init);
})();
