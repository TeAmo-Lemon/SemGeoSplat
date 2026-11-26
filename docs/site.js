(() => {
  const select = document.querySelector('#scene-choice');
  const picker = document.querySelector('#style-options');
  const reference = document.querySelector('#reference-image');
  const original = document.querySelector('#original-video');
  const styled = document.querySelector('#styled-video');
  const videos = [original, styled];
  const comparison = document.querySelector('#comparison');
  const preview = document.querySelector('#media-preview');
  const clips = [...document.querySelectorAll('img[data-animation]')];
  const toggles = [...document.querySelectorAll('.motion')];
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const visible = new Set();
  let motion = !preference.matches;
  const onPages = location.hostname.endsWith('.github.io');
  const videoRoot = onPages
    ? 'https://raw.githubusercontent.com/TeAmo-Lemon/SemGeoSplat/9c50d278e4641bb691e9c90503314c7447c1ef64/docs/media/gallery'
    : '../gallery-videos';
  const sources = new Map();

  function clearSource(video) {
    const previous = sources.get(video);
    if (!previous) return;
    previous.controller?.abort();
    if (previous.blobUrl) URL.revokeObjectURL(previous.blobUrl);
    sources.delete(video);
  }

  async function retryWithVideoMime(video) {
    const source = sources.get(video);
    if (!onPages || !source || source.controller || !video.error) return;
    if (video.error.code !== MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED && video.error.code !== MediaError.MEDIA_ERR_DECODE) return;
    // GitHub serves raw MP4s as octet-stream. Retry only when native loading rejects it.
    source.controller = new AbortController();
    try {
      const response = await fetch(source.url, { signal: source.controller.signal });
      if (!response.ok) throw new Error('Could not load video: ' + response.status);
      const bytes = await response.blob();
      if (sources.get(video) !== source) return;
      source.blobUrl = URL.createObjectURL(bytes.slice(0, bytes.size, 'video/mp4'));
      video.src = source.blobUrl;
      video.load();
    } catch (error) {
      if (error.name !== 'AbortError' && sources.get(video) === source) video.controls = true;
    }
  }


  function updateVideos() {
    if (!motion || preview?.open || !visible.has(comparison) || document.hidden) {
      videos.forEach(video => video.pause());
      return;
    }
    for (const video of videos) {
      video.preload = 'auto';
      if (video.error) retryWithVideoMime(video);
    }
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
      const animate = motion && !preview?.open && visible.has(clip) && !document.hidden;
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
    video.addEventListener('error', updateVideos);
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
    const videoDirectory = videoRoot + '/' + scene.value;
    for (const video of videos) {
      const isOriginal = video === original;
      const name = isOriginal ? 'original' : style;
      const url = videoDirectory + '/' + name + '.mp4';
      clearSource(video);
      sources.set(video, {url, controller: null, blobUrl: null});
      video.pause();
      video.preload = 'none';
      video.poster = directory + '/' + name + '.jpg';
      video.src = url;
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
    setLink('original-link', originalGif ? 'media/' + scene.value + '-original.gif' : videoDirectory + '/original.mp4', originalGif ? 'Open GIF' : 'Open clip');
    setLink('styled-link', styleGif ? 'media/' + scene.value + '-scene.gif' : videoDirectory + '/' + style + '.mp4', styleGif ? 'Open GIF' : 'Open clip');
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

  function enablePreview() {
    if (!preview || typeof preview.showModal !== 'function') return;

    const links = [...document.querySelectorAll('main a.media-box, .control-meta a, #original-link, #styled-link')];
    const title = preview.querySelector('#preview-title');
    const content = preview.querySelector('.preview-content');
    let opener = null;
    let openingScroll = null;

    function show(link) {
      if (preview.open) return;
      const isVideo = new URL(link.href).pathname.endsWith('.mp4');
      const figureVideo = link.closest('figure')?.querySelector('video');
      const image = link.querySelector('img') || link.closest('article')?.querySelector('.media-box img');
      title.textContent = image?.alt || figureVideo?.getAttribute('aria-label') || 'Media preview';

      const media = document.createElement(isVideo ? 'video' : 'img');
      if (isVideo) {
        media.controls = true;
        media.loop = true;
        media.autoplay = motion;
        media.playsInline = true;
        media.muted = true;
        media.setAttribute('aria-labelledby', title.id);
        media.poster = figureVideo.poster;
        sources.set(media, {url: link.href, controller: null, blobUrl: null});
        media.addEventListener('error', () => retryWithVideoMime(media));
        media.src = sources.get(figureVideo)?.blobUrl || link.href;
      } else {
        media.alt = title.textContent;
        media.src = link.href;
      }

      opener = link;
      openingScroll = history.state?.semgeoScroll || [window.scrollX, window.scrollY];
      content.replaceChildren(media);
      preview.showModal();
      updateMotion();
    }

    function hide() {
      if (!preview.open) return;
      const media = content.firstElementChild;
      if (media instanceof HTMLVideoElement) {
        media.pause();
        clearSource(media);
        media.removeAttribute('src');
        media.load();
      }
      content.replaceChildren();
      preview.close();
      const link = opener;
      const [left, top] = openingScroll;
      opener = null;
      openingScroll = null;
      link.focus({preventScroll: true});
      // History traversal can refocus the document and scroll to its fragment
      // after popstate. Restore the opening view once that traversal finishes.
      requestAnimationFrame(() => {
        if (preview.open || (document.activeElement !== document.body && document.activeElement !== link)) return;
        link.focus({preventScroll: true});
        window.scrollTo({left, top, behavior: 'instant'});
      });
      updateMotion();
    }

    let returning = false;
    function closePreview() {
      if (returning) return;
      if (Number.isInteger(history.state?.semgeoMedia)) {
        returning = true;
        history.back();
      } else {
        hide();
      }
    }

    function restorePreview() {
      returning = false;
      const index = history.state?.semgeoMedia;
      if (Number.isInteger(index) && links[index]) show(links[index]);
      else hide();
    }

    links.forEach((link, index) => {
      link.setAttribute('aria-haspopup', 'dialog');
      link.addEventListener('click', event => {
        if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
        event.preventDefault();
        if (preview.open) return;
        // A same-URL entry makes browser Back dismiss the modal and Forward reopen it.
        try {
          history.pushState({...history.state, semgeoMedia: index, semgeoScroll: [window.scrollX, window.scrollY]}, '');
        } catch {
          // Some file viewers disable history operations. The Close button still works.
        }
        show(link);
      });
    });
    preview.querySelector('.preview-close').addEventListener('click', closePreview);
    preview.addEventListener('cancel', event => {
      event.preventDefault();
      closePreview();
    });
    preview.addEventListener('click', event => {
      if (event.target === preview) closePreview();
    });
    window.addEventListener('popstate', restorePreview);
    restorePreview();
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
  window.addEventListener('pagehide', event => {
    if (!event.persisted) videos.forEach(clearSource);
  });

  const query = new URLSearchParams(location.search);
  if ([...select.options].some(option => option.value === query.get('scene'))) select.value = query.get('scene');
  chooseScene(query.get('style'), false);
  document.querySelector('.scene-picker').hidden = false;
  document.querySelector('.style-picker').hidden = false;
  enablePreview();
  updateMotion();
})();
