(() => {
  const select = document.querySelector('#scene-choice');
  const picker = document.querySelector('#style-options');
  const reference = document.querySelector('#reference-image');
  const original = document.querySelector('#original-video');
  const styled = document.querySelector('#styled-video');
  const videos = [original, styled];
  const comparison = document.querySelector('#comparison');
  const clips = [...document.querySelectorAll('img[data-animation]')];
  const toggles = [...document.querySelectorAll('.motion')];
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const visible = new Set();
  let motion = !preference.matches;

  function updateVideos() {
    if (!motion || !visible.has(comparison) || document.hidden) {
      videos.forEach(video => video.pause());
      return;
    }
    for (const video of videos) video.preload = 'auto';
    // Start the two trajectories together, including after a reference switch.
    if (videos.some(video => video.readyState < HTMLMediaElement.HAVE_FUTURE_DATA)) {
      videos.forEach(video => video.pause());
      return;
    }
    for (const video of videos) {
      if (video.paused) video.play().catch(error => {
        if (error.name !== 'AbortError') video.controls = true;
      });
    }
  }

  function updateMotion() {
    for (const clip of clips) {
      const animate = motion && visible.has(clip) && !document.hidden;
      const source = animate ? clip.dataset.animation : clip.dataset.poster;
      if (clip.getAttribute('src') !== source) clip.setAttribute('src', source);
    }
    for (const toggle of toggles) toggle.textContent = motion ? 'Pause motion' : 'Play motion';
    updateVideos();
  }

  for (const video of videos) {
    video.controls = false;
    video.addEventListener('canplay', updateVideos);
    video.addEventListener('waiting', updateVideos);
  }
  original.addEventListener('timeupdate', () => {
    if (!original.paused && !styled.paused && Math.abs(original.currentTime - styled.currentTime) > 0.12) {
      styled.currentTime = original.currentTime;
    }
  });

  function setLink(id, href, text) {
    const link = document.getElementById(id);
    link.href = href;
    link.textContent = text;
  }

  function showStyle(style, updateUrl) {
    const scene = select.selectedOptions[0];
    const directory = 'media/gallery/' + scene.value;
    for (const video of videos) {
      const isOriginal = video === original;
      const base = directory + '/' + (isOriginal ? 'original' : style);
      video.pause();
      video.preload = 'none';
      video.poster = base + '.jpg';
      video.src = base + '.mp4';
      video.setAttribute('aria-label', isOriginal ? 'Original ' + scene.text : scene.text + ' rendered with reference ' + style);
      video.load();
    }
    const referencePath = 'media/gallery/styles/' + style + '.jpg';
    reference.src = referencePath;
    reference.alt = 'Style reference ' + style;
    document.querySelector('#reference-link').href = referencePath;
    document.querySelector('#reference-label').textContent = 'Reference ' + style;

    const originalGif = Boolean(scene.dataset.gifStyle);
    const styleGif = scene.dataset.gifStyle === style;
    setLink('original-link', originalGif ? 'media/' + scene.value + '-original.gif' : directory + '/original.mp4', originalGif ? 'Open GIF' : 'Open clip');
    setLink('styled-link', styleGif ? 'media/' + scene.value + '-scene.gif' : directory + '/' + style + '.mp4', styleGif ? 'Open GIF' : 'Open clip');
    if (updateUrl && location.protocol !== 'file:') {
      const url = new URL(location.href);
      url.searchParams.set('scene', scene.value);
      url.searchParams.set('style', style);
      url.hash = 'scenes';
      history.replaceState(null, '', url);
    }
    updateVideos();
  }

  function chooseScene(requestedStyle, updateUrl) {
    const scene = select.selectedOptions[0];
    const styles = scene.dataset.styles.split(' ');
    const selected = styles.includes(requestedStyle) ? requestedStyle : scene.dataset.default;
    const choices = document.createDocumentFragment();
    for (const id of styles) {
      const label = document.createElement('label');
      const radio = document.createElement('input');
      radio.type = 'radio';
      radio.name = 'reference';
      radio.value = id;
      radio.checked = id === selected;
      radio.setAttribute('aria-label', 'Reference ' + id);
      const image = document.createElement('img');
      image.src = 'media/gallery/styles/' + id + '.jpg';
      image.alt = '';
      const number = document.createElement('span');
      number.textContent = id;
      label.append(radio, image, number);
      choices.append(label);
    }
    picker.replaceChildren(choices);
    document.querySelector('#dataset').textContent = scene.dataset.dataset;
    showStyle(selected, updateUrl);
  }

  picker.addEventListener('change', event => showStyle(event.target.value, true));
  select.addEventListener('change', () => chooseScene(null, true));
  for (const toggle of toggles) {
    toggle.hidden = false;
    toggle.addEventListener('click', () => {
      motion = !motion;
      updateMotion();
    });
  }
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      }
      updateMotion();
    });
    [...clips, comparison].forEach(element => observer.observe(element));
  } else {
    [...clips, comparison].forEach(element => visible.add(element));
  }
  preference.addEventListener('change', event => {
    motion = !event.matches;
    updateMotion();
  });
  document.addEventListener('visibilitychange', updateMotion);

  const query = new URLSearchParams(location.search);
  if ([...select.options].some(option => option.value === query.get('scene'))) select.value = query.get('scene');
  chooseScene(query.get('style'), false);
  document.querySelector('.scene-picker').hidden = false;
  document.querySelector('.style-picker').hidden = false;
  updateMotion();
})();
