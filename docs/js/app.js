// docs/js/app.js — Listen section + shared helpers
const $ = (sel, el = document) => el.querySelector(sel);

async function fetchJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

window.DEMO = { tracks: {}, ontology: null };   // shared state for viewer/gallery

function genrePath(t) {
  const parts = [t.genre, t.subgenre, t.subsub].filter(Boolean);
  return parts.join(' › ');
}

const mmss = (s) => {
  const r = Math.round(s);
  return `${Math.floor(r / 60)}:${String(r % 60).padStart(2, '0')}`;
};

function trackCard(t) {
  const card = document.createElement('div');
  // audio is cut to preview_sec pending release clearance; the card still states the true length
  const preview = t.preview_sec && t.preview_sec < t.duration_sec
    ? ` · <span class="preview-tag">${Math.round(t.preview_sec)} s preview</span>` : '';
  card.className = 'card';
  card.innerHTML = `
    <h3>${t.title}</h3>
    <div class="meta">${genrePath(t)}<br>
      ${t.singer} (${t.gender}) · ${t.jangdan} · ${t.key} · ${Math.round(t.tempo)} BPM ·
      ${mmss(t.duration_sec)}${preview}</div>
    <audio controls preload="none" src="assets/audio/${t.id}.mp3"></audio>
    <details><summary>Caption <span class="kr">캡션</span></summary>
      <p class="cap-en">${t.caption_en}</p>
      <p class="cap-ko" hidden>${t.caption_ko}</p>
      <button class="cap-toggle">한국어로 보기</button></details>
    <details><summary>Lyrics <span class="kr">가사</span></summary>
      <p>${t.lyrics_regions.map(r => r.text).join('<br>')}</p></details>
    <div class="actions"><button class="explore-btn">Explore annotations →</button></div>`;
  card.querySelector('.cap-toggle').addEventListener('click', (e) => {
    const en = card.querySelector('.cap-en'), ko = card.querySelector('.cap-ko');
    en.hidden = !en.hidden; ko.hidden = !ko.hidden;
    e.target.textContent = en.hidden ? 'Show English' : '한국어로 보기';
  });
  card.querySelector('.explore-btn').addEventListener('click', () => {
    document.dispatchEvent(new CustomEvent('load-track', { detail: t.id }));
    $('#explore').scrollIntoView({ behavior: 'smooth' });
  });
  return card;
}

async function initListen() {
  const holder = $('#track-cards');
  try {
    const ids = await fetchJSON('assets/tracks/manifest.json');
    window.DEMO.ontology = await fetchJSON('assets/ontology.json');
    for (const id of ids) {
      const t = await fetchJSON(`assets/tracks/${id}.json`);
      window.DEMO.tracks[id] = t;
      holder.appendChild(trackCard(t));
    }
    document.dispatchEvent(new CustomEvent('tracks-ready'));
  } catch (err) {
    holder.innerHTML = `<p class="meta">Demo assets unavailable (${err.message}).</p>`;
  }
}
initListen();
