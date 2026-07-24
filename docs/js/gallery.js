// docs/js/gallery.js — 17-type sigimsae gallery
(() => {
let playingSnippet = null;   // {audio, btn} of the currently playing chip

function miniContour(canvas, snip, color) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width = 120 * devicePixelRatio, h = canvas.height = 44 * devicePixelRatio;
  const hz = snip.f0.filter(v => v != null);
  if (!hz.length) return;
  const lo = Math.min(...hz) * 0.97, hi = Math.max(...hz) * 1.03;
  const y = v => h - (Math.log2(v / lo) / Math.log2(hi / lo)) * h;
  const n = snip.f0.length;
  // shaded band = the annotated region inside the padded clip
  const dur = n * snip.f0_hop;
  ctx.fillStyle = color + '33';
  ctx.fillRect(snip.start_in_clip / dur * w, 0, (snip.end_in_clip - snip.start_in_clip) / dur * w, h);
  ctx.strokeStyle = color; ctx.lineWidth = 1.5 * devicePixelRatio;
  ctx.beginPath(); let pen = false;
  snip.f0.forEach((v, i) => {
    if (v == null) { pen = false; return; }
    const x = i / n * w;
    pen ? ctx.lineTo(x, y(v)) : ctx.moveTo(x, y(v)); pen = true;
  });
  ctx.stroke();
}

async function initGallery() {
  const holder = document.querySelector('#gallery');
  let data;
  try { data = await fetch('assets/gallery/gallery.json').then(r => r.json()); }
  catch { holder.innerHTML = '<p class="meta">Gallery assets unavailable.</p>'; return; }
  const o = window.DEMO.ontology;
  for (const g of o.groups) {
    const block = document.createElement('div');
    block.className = 'group-block';
    block.innerHTML = `<h3><span class="sw" style="background:${o.group_colors[g]}"></span> ${o.group_labels[g]}</h3>`;
    for (const [kr, t] of Object.entries(o.types)) {
      if (t.group !== g) continue;
      const entry = data[kr] ?? { count: 0, snippets: [] };
      const card = document.createElement('div');
      card.className = 'type-card';
      card.style.borderLeftColor = o.group_colors[g];
      card.innerHTML = `
        <div class="names"><span class="count">${entry.count.toLocaleString()} instances</span>
          <b>${kr}</b><span class="roman">${t.roman}</span>${t.en}</div>
        <div class="snips"></div>`;
      const snips = card.querySelector('.snips');
      for (const s of entry.snippets) {
        const chip = document.createElement('div');
        chip.className = 'snip';
        chip.innerHTML = `<button title="play">▶</button><canvas></canvas>
          <span class="meta" style="font-size:.72rem">${s.track.slice(-7)}</span>`;
        let audio = null;   // lazy — no network cost until first play
        chip.querySelector('button').addEventListener('click', (e) => {
          audio = audio || new Audio(`assets/gallery/${s.file}`);
          if (audio.paused) {
            if (playingSnippet && playingSnippet.audio !== audio) {
              playingSnippet.audio.pause();
              playingSnippet.btn.textContent = '▶';
            }
            playingSnippet = { audio, btn: e.target };
            audio.currentTime = 0; audio.play(); e.target.textContent = '⏸';
            audio.onended = () => { e.target.textContent = '▶'; };
          } else {
            audio.pause(); e.target.textContent = '▶';
          }
        });
        snips.appendChild(chip);
        miniContour(chip.querySelector('canvas'), s, o.group_colors[g]);
      }
      block.appendChild(card);
    }
    holder.appendChild(block);
  }
}
document.addEventListener('tracks-ready', initGallery);
})();
