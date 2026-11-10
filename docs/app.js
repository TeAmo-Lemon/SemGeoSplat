'use strict';

const $ = (id) => document.getElementById(id);
const featuredIds = ['truck-133', 'garden-77', 'bonsai-87', 'flower-19'];
const names = { bonsai: 'Bonsai', counter: 'Counter', flower: 'Flower', garden: 'Garden', horns: 'Horns', ignatius: 'Ignatius', kitchen: 'Kitchen', m60: 'M60', orchids: 'Orchids', train: 'Train', trex: 'T-Rex', truck: 'Truck' };
const paths = {
  video: (item) => `assets/results/${item.id}/semgeo.mp4`,
  poster: (item) => `assets/posters/${item.id}.jpg`,
  original: (item) => `assets/gt/${item.scene}.mp4`,
  originalPoster: (item) => `assets/posters/${item.scene}-original.jpg`,
  style: (item) => `assets/styles/${item.styleId}.jpg`,
};
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const canPreview = window.matchMedia('(hover: hover) and (pointer: fine)');
const state = { items: [], filtered: [], scene: 'all', limit: 12, featured: null, viewerItems: [], viewerIndex: 0, heroWanted: !reducedMotion.matches && !navigator.connection?.saveData, heroVisible: true };
const hero = $('hero-video');
const dialog = $('result-dialog');
let activePreview = null;
let opener = null;

function safePlay(video) {
  const promise = video.play();
  if (promise) promise.catch(() => {});
}
function unload(video) {
  video.pause();
  video.removeAttribute('src');
  video.load();
}
function stopPreview() {
  if (!activePreview) return;
  activePreview.parentElement.classList.remove('is-playing');
  unload(activePreview);
  activePreview = null;
}
function updateHeroButton() {
  $('hero-toggle').textContent = hero.paused ? 'Play ▷' : 'Pause Ⅱ';
  $('hero-toggle').setAttribute('aria-label', hero.paused ? 'Play featured animation' : 'Pause featured animation');
}
hero.addEventListener('play', updateHeroButton);
hero.addEventListener('pause', updateHeroButton);
hero.addEventListener('error', () => { $('hero-toggle').textContent = 'Retry ▷'; });
$('hero-toggle').addEventListener('click', () => {
  state.heroWanted = hero.paused;
  if (hero.paused) { if (hero.error) hero.load(); safePlay(hero); } else hero.pause();
});
new IntersectionObserver(([entry]) => {
  state.heroVisible = entry.isIntersecting;
  if (!entry.isIntersecting) hero.pause();
  else if (state.heroWanted && !dialog.open && !document.hidden) safePlay(hero);
}, { threshold: 0.15 }).observe(hero);

function selectFeatured(item) {
  state.featured = item;
  hero.poster = paths.poster(item);
  hero.src = paths.video(item);
  hero.setAttribute('aria-label', `SemGeo result: ${names[item.scene]}, style ${item.styleId}`);
  $('hero-caption').replaceChildren(document.createTextNode(names[item.scene] + ' '));
  const style = document.createElement('span');
  style.textContent = `/ Style ${item.styleId}`;
  $('hero-caption').append(style);
  $('hero-original').src = paths.originalPoster(item);
  $('hero-original').alt = `Original ${names[item.scene]} scene`;
  $('hero-style').src = paths.style(item);
  $('hero-style').alt = `Style reference ${item.styleId}`;
  $('hero-style-id').textContent = item.styleId;
  for (const button of $('featured-options').children) button.setAttribute('aria-pressed', String(button.dataset.id === item.id));
  if (state.heroWanted && state.heroVisible && !document.hidden) safePlay(hero);
}

function makeCard(item) {
  const article = document.createElement('article');
  article.className = 'result-card';
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'card-button';
  button.setAttribute('aria-label', `View ${names[item.scene]}, style ${item.styleId}`);
  const media = document.createElement('div');
  media.className = 'card-media';
  const poster = document.createElement('img');
  poster.src = paths.poster(item);
  poster.alt = `${names[item.scene]} stylized with reference ${item.styleId}`;
  poster.loading = 'lazy';
  poster.decoding = 'async';
  poster.width = 960;
  poster.height = 720;
  const video = document.createElement('video');
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  video.preload = 'none';
  video.setAttribute('aria-hidden', 'true');
  const style = document.createElement('img');
  style.src = paths.style(item);
  style.alt = '';
  style.className = 'card-style';
  style.loading = 'lazy';
  const expand = document.createElement('span');
  expand.className = 'card-open';
  expand.textContent = '↗';
  expand.setAttribute('aria-hidden', 'true');
  media.append(poster, video, style, expand);
  const meta = document.createElement('div');
  meta.className = 'card-meta';
  const title = document.createElement('h3');
  title.textContent = names[item.scene];
  const subtitle = document.createElement('span');
  subtitle.textContent = `Style ${item.styleId}`;
  meta.append(title, subtitle);
  button.append(media, meta);
  article.append(button);
  const preview = () => {
    if (!canPreview.matches || reducedMotion.matches || navigator.connection?.saveData || dialog.open) return;
    stopPreview();
    activePreview = video;
    video.src = paths.video(item);
    safePlay(video);
  };
  video.addEventListener('playing', () => { if (activePreview === video) media.classList.add('is-playing'); });
  button.addEventListener('pointerenter', preview);
  button.addEventListener('pointerleave', () => { if (activePreview === video) stopPreview(); });
  button.addEventListener('focus', preview);
  button.addEventListener('blur', () => { if (activePreview === video) stopPreview(); });
  button.addEventListener('click', () => openViewer(item, state.filtered, button));
  return article;
}
function renderCards(append = false) {
  const grid = $('results-grid');
  const start = append ? grid.children.length : 0;
  if (!append) { stopPreview(); grid.replaceChildren(); }
  const fragment = document.createDocumentFragment();
  for (const item of state.filtered.slice(start, state.limit)) fragment.append(makeCard(item));
  grid.append(fragment);
  $('empty-state').hidden = state.filtered.length > 0;
  $('load-more').hidden = state.limit >= state.filtered.length;
  const visible = Math.min(state.limit, state.filtered.length);
  $('filter-status').textContent = `${state.filtered.length} result${state.filtered.length === 1 ? '' : 's'}${state.scene === 'all' ? ' across all scenes' : ' · ' + names[state.scene]} · Select a result to explore`;
  $('shown-count').textContent = state.filtered.length ? `Showing ${visible} of ${state.filtered.length}` : '';
}
function filterResults() {
  const query = $('search').value.trim().toLowerCase();
  const style = $('style-filter').value;
  state.filtered = state.items.filter((item) =>
    (state.scene === 'all' || item.scene === state.scene) &&
    (style === 'all' || item.styleId === style) &&
    (!query || `${item.scene} ${names[item.scene]} ${item.sceneLabel} style ${item.styleId} ${item.id}`.toLowerCase().includes(query))
  );
  state.limit = 12;
  renderCards();
}
function resetFilters() {
  state.scene = 'all';
  $('search').value = '';
  $('style-filter').value = 'all';
  for (const button of $('scene-filters').children) button.setAttribute('aria-pressed', String(button.dataset.scene === 'all'));
  filterResults();
}
$('search').addEventListener('input', filterResults);
$('style-filter').addEventListener('change', filterResults);
$('reset-filters').addEventListener('click', resetFilters);
$('load-more').addEventListener('click', () => {
  const firstNew = $('results-grid').children.length;
  state.limit += 12;
  renderCards(true);
  $('results-grid').children[firstNew]?.querySelector('button').focus({ preventScroll: true });
});

function renderViewer() {
  const item = state.viewerItems[state.viewerIndex];
  $('dialog-title').textContent = `${names[item.scene]} / Style ${item.styleId}`;
  const video = $('dialog-video');
  const original = $('dialog-original');
  video.poster = paths.poster(item);
  video.src = paths.video(item);
  video.load();
  video.setAttribute('aria-label', `SemGeo result: ${names[item.scene]}, style ${item.styleId}`);
  original.poster = paths.originalPoster(item);
  original.src = paths.original(item);
  original.load();
  original.setAttribute('aria-label', `Original ${names[item.scene]} scene`);
  $('dialog-style').src = paths.style(item);
  $('dialog-style').alt = `Style reference ${item.styleId}`;
  $('open-video').href = paths.video(item);
  $('dialog-position').textContent = `${state.viewerIndex + 1} / ${state.viewerItems.length} results`;
  $('prev-result').disabled = state.viewerItems.length < 2;
  $('next-result').disabled = state.viewerItems.length < 2;
  if (!reducedMotion.matches) safePlay(video);
}
function openViewer(item, collection, source) {
  opener = source;
  stopPreview();
  hero.pause();
  state.viewerItems = collection;
  state.viewerIndex = collection.findIndex((result) => result.id === item.id);
  if (state.viewerIndex < 0) { state.viewerItems = [item]; state.viewerIndex = 0; }
  document.body.classList.add('dialog-open');
  dialog.showModal();
  renderViewer();
  $('close-dialog').focus();
}
function stepViewer(direction) {
  state.viewerIndex = (state.viewerIndex + direction + state.viewerItems.length) % state.viewerItems.length;
  renderViewer();
}
$('hero-expand').addEventListener('click', () => { if (state.featured) openViewer(state.featured, state.items, $('hero-expand')); });
$('close-dialog').addEventListener('click', () => dialog.close());
$('prev-result').addEventListener('click', () => stepViewer(-1));
$('next-result').addEventListener('click', () => stepViewer(1));
dialog.addEventListener('click', (event) => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); } });
dialog.addEventListener('close', () => {
  unload($('dialog-video'));
  unload($('dialog-original'));
  document.body.classList.remove('dialog-open');
  opener?.focus({ preventScroll: true });
  if (state.heroWanted && state.heroVisible && !document.hidden) safePlay(hero);
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    hero.pause(); stopPreview(); $('dialog-video').pause(); $('dialog-original').pause();
  } else if (state.heroWanted && state.heroVisible && !dialog.open) safePlay(hero);
});
reducedMotion.addEventListener('change', () => {
  if (reducedMotion.matches) { state.heroWanted = false; hero.pause(); stopPreview(); }
});

async function init() {
  const response = await fetch('data/results.json');
  if (!response.ok) throw new Error(`Manifest: ${response.status}`);
  const items = await response.json();
  const scenes = [...new Set(items.map((item) => item.scene))];
  const styles = [...new Set(items.map((item) => item.styleId))].sort((a, b) => Number(a) - Number(b));
  // Lead with one example per scene before showing the remaining collection.
  const featured = featuredIds.map((id) => items.find((item) => item.id === id)).filter(Boolean);
  const leads = scenes.map((scene) => featured.find((item) => item.scene === scene) || items.find((item) => item.scene === scene));
  const leadingIds = new Set(leads.map((item) => item.id));
  state.items = [...leads, ...items.filter((item) => !leadingIds.has(item.id))];
  $('collection-stats').textContent = `${scenes.length} scenes · ${styles.length} styles · ${items.length} results`;
  for (const scene of ['all', ...scenes]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.scene = scene;
    button.textContent = scene === 'all' ? 'All scenes' : names[scene];
    button.setAttribute('aria-pressed', String(scene === 'all'));
    button.addEventListener('click', () => {
      state.scene = scene;
      for (const other of $('scene-filters').children) other.setAttribute('aria-pressed', String(other === button));
      filterResults();
    });
    $('scene-filters').append(button);
  }
  for (const style of styles) {
    const option = document.createElement('option');
    option.value = style;
    option.textContent = `Style ${style}`;
    $('style-filter').append(option);
  }
  for (const [index, item] of featured.entries()) {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.id = item.id;
    button.textContent = `0${index + 1} ${names[item.scene]}`;
    button.addEventListener('click', () => selectFeatured(item));
    $('featured-options').append(button);
  }
  selectFeatured(featured[0] || items[0]);
  filterResults();
}
init().catch((error) => {
  console.error(error);
  $('filter-status').textContent = 'The collection could not load. Please refresh the page to try again.';
});
