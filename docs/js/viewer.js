// docs/js/viewer.js — interactive annotation viewer
(() => {
const WIN_SEC = 18;            // visible window
const state = {
  track: null, t0: 0,          // window start (sec)
  hiddenGroups: new Set(),
  cqtOn: false, cqtManifest: null,
  playing: false,
};
const els = {};

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
  const w = c.width = c.clientWidth * devicePixelRatio;
  const h = c.height = 300 * devicePixelRatio;
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
  const w = c.width = c.clientWidth * devicePixelRatio;
  const h = c.height = 34 * devicePixelRatio;
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
  const w = c.width = c.clientWidth * devicePixelRatio;
  const h = c.height = 46 * devicePixelRatio;
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

function drawCQT() {}   // placeholder body — implemented in Task 11

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
function wirePlayback() {}   // placeholder — Task 10
function wirePointer() {}    // placeholder — Task 10
function wireCQT() {}        // placeholder — Task 11
window.VIEWER = { state, els, redraw, drawCQT, wirePlayback, wirePointer, wireCQT, xOf, WIN_SEC };

document.addEventListener('tracks-ready', init);
})();
