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
  const referenceDescriptions = {
    2: 'Hand holding a reflective sphere',
    3: 'Watercolor leaves in many colors',
    5: 'Seated figure beside a vase in a green room',
    6: 'Woman framed by circular stained-glass patterns',
    7: 'Bird mosaic in earth tones',
    8: 'Pencil portrait of a woman',
    9: 'Angular portrait on an ochre ground',
    10: 'Rainy street with umbrellas and colorful lights',
    11: 'Portrait against multicolored square brushstrokes',
    12: 'Angular blue-gray figures',
    14: 'Swirling blue night sky with bright stars',
    16: 'Looping black ink lines',
    19: 'Cresting blue wave',
    24: 'Rocky landscape drawn in black ink',
    25: 'Portrait behind a web of dark lines',
    30: 'Boats in a harbor below a hillside town',
    31: 'Bright abstract shapes on a violet ground',
    32: 'Blue animal in a snowy landscape',
    35: 'Color blocks crossed by angular dark outlines',
    39: 'Long-faced figure in a green striped shirt',
    48: 'Sepia portrait of a seated man',
    50: 'Stone bridge painted in pink and violet',
    54: 'Abstract faces in pink, green and orange',
    59: 'Beach dotted with colorful umbrellas',
    63: 'Elongated brown figure on a pale textured ground',
    64: 'Faces and handwritten diagrams on a yellow ground',
    76: 'Interlocking pink, blue and gray shapes',
    77: 'Black figures and lettering on a brown ground',
    79: 'Fine hanging lines on a pale blue ground',
    87: 'Rounded pink and yellow shapes with curling green lines',
    89: 'Intertwined red and blue figures',
    94: 'Black crown on a tan ground',
    99: 'Loose figure studies in red and green',
    111: 'Red standing figure beside writing and a crown',
    114: 'Broad black and red brushstrokes',
    119: 'White outlined forms and orange spirals',
    120: 'Pen drawing of a man wearing glasses',
    121: 'Wall of irregular tan and gray stones',
    123: 'Frost crystals on pale blue',
    128: 'Illustrated playground with trees and benches',
    132: 'Wall of narrow multicolored bricks',
    133: 'Multicolored LEGO bricks',
    135: 'Snowy landscape with a gate and bare trees',
    137: 'A dark overhanging tree above blue water',
    138: 'A glowing cat-shaped bus at night',
    140: 'Pastel town scene in pink, blue and yellow',
  };

  const orbitMount = document.querySelector('#orbit-mount');
  const orbit = document.querySelector('#orbit-panel');
  const orbitOpen = document.querySelector('#orbit-open');
  const orbitVideo = document.querySelector('#orbit-video');
  const slider = document.querySelector('#orbit-range');
  const orbitStyles = document.querySelector('#orbit-styles');
  const orbitName = document.querySelector('#orbit-style-name');
  let pendingOrbitTime = null;
  const canvas = document.querySelector('#orbit-canvas');
  const context = canvas.getContext('2d');

  function drawOrbit() {
    if (!context || orbitVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    // The paired video stores both appearances of a camera pose in a single frame.
    const width = canvas.width;
    const height = canvas.height;
    const cut = Math.round(width * slider.valueAsNumber / 100);
    if (cut > 0) context.drawImage(orbitVideo, 0, 0, cut, height, 0, 0, cut, height);
    if (cut < width) context.drawImage(orbitVideo, width + cut, 0, width - cut, height, cut, 0, width - cut, height);
  }

  function updateSplit() {
    const percent = slider.valueAsNumber;
    orbit.style.setProperty('--split', percent + '%');
    slider.setAttribute('aria-valuetext', percent + '% original, ' + (100 - percent) + '% ' + orbitName.textContent);
    const labels = orbit.querySelector('.orbit-labels').children;
    labels[0].style.visibility = percent === 0 ? 'hidden' : '';
    labels[1].style.visibility = percent === 100 ? 'hidden' : '';
    drawOrbit();
  }

  // Present decoded frames on their own clock; use the display clock on older engines.
  const hasVideoClock = typeof orbitVideo.requestVideoFrameCallback === 'function';
  let pendingFrame = null;
  function paintOrbit() {
    pendingFrame = null;
    drawOrbit();
    if (!orbitVideo.paused && !orbitVideo.ended) {
      pendingFrame = hasVideoClock ? orbitVideo.requestVideoFrameCallback(paintOrbit) : requestAnimationFrame(paintOrbit);
    }
  }
  orbitVideo.addEventListener('playing', () => {
    if (pendingFrame === null) paintOrbit();
  });
  function stopOrbitPaint() {
    if (pendingFrame === null) return;
    if (hasVideoClock) orbitVideo.cancelVideoFrameCallback(pendingFrame);
    else cancelAnimationFrame(pendingFrame);
    pendingFrame = null;
  }
  orbitVideo.addEventListener('pause', () => {
    if (orbitVideo.paused) stopOrbitPaint();
  });
  orbitVideo.addEventListener('loadeddata', drawOrbit);
  orbitVideo.addEventListener('loadedmetadata', () => {
    if (pendingOrbitTime !== null && Number.isFinite(orbitVideo.duration)) {
      orbitVideo.currentTime = pendingOrbitTime % orbitVideo.duration;
      pendingOrbitTime = null;
    }
  });
  orbitVideo.addEventListener('seeked', drawOrbit);

  function selectOrbitStyle(input) {
    const label = input.labels[0];
    const name = label.querySelector('.orbit-name').textContent;
    const poster = label.querySelector('img').getAttribute('src');
    pendingOrbitTime ??= orbitVideo.currentTime;
    orbitVideo.pause();
    stopOrbitPaint();
    context?.clearRect(0, 0, canvas.width, canvas.height);
    orbit.style.setProperty('--orbit-poster', 'url("' + poster + '")');
    orbitName.textContent = name;
    orbit.querySelector('.orbit-stage').setAttribute('aria-label',
      'Kitchen loader with its original appearance on the left and ' + name + ' on the right');
    for (const figure of document.querySelectorAll('[data-orbit-credit]')) {
      figure.hidden = figure.dataset.orbitCredit !== input.value;
    }
    orbitVideo.src = input.dataset.video;
    orbitVideo.preload = pendingOrbitTime > 0 ? 'auto' : 'none';
    orbitVideo.load();
    updateSplit();
    updateOrbitMotion();
  }

  orbitStyles.addEventListener('change', event => selectOrbitStyle(event.target));
  orbitStyles.hidden = false;
  slider.hidden = false;
  orbit.querySelector('.orbit-handle').hidden = false;
  slider.addEventListener('input', updateSplit);

  function updateOrbitMotion() {
    const shown = preview?.open ? preview.contains(orbit) : visible.has(orbitMount);
    if (!context || !motion || !shown || document.hidden) {
      orbitVideo.pause();
    } else if (orbitVideo.paused) {
      orbitVideo.play().catch(() => {});
    }
  }

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
    updateOrbitMotion();
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
    const framed = scene.hasAttribute('data-framed');
    const version = framed ? '?v=framed1' : '';
    const videoDirectory = framed ? directory : videoRoot + '/' + scene.value;
    const clip = name => videoDirectory + '/' + name + '.mp4' + version;
    for (const video of videos) {
      const isOriginal = video === original;
      const name = isOriginal ? 'original' : style;
      const url = clip(name);
      clearSource(video);
      sources.set(video, {url, controller: null, blobUrl: null});
      video.pause();
      video.preload = 'none';
      video.poster = directory + '/' + name + '.jpg' + version;
      video.src = url;
      video.setAttribute('aria-label', (isOriginal ? 'Original ' : 'Stylized ') + scene.text + ' scene');
      video.load();
    }
    const referencePath = 'media/gallery/styles/' + style + '.jpg';
    reference.src = referencePath;
    reference.alt = referenceDescriptions[style] || 'Selected style reference';
    document.querySelector('#reference-link').href = referencePath;

    const originalGif = Boolean(scene.dataset.gifStyle);
    const styleGif = scene.dataset.gifStyle === style;
    setLink('original-link', originalGif ? 'media/' + scene.value + '-original.gif' + version : clip('original'), originalGif ? 'Open GIF' : 'Open clip');
    setLink('styled-link', styleGif ? 'media/' + scene.value + '-scene.gif' + version : clip(style), styleGif ? 'Open GIF' : 'Open clip');
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
      radio.setAttribute('aria-label', referenceDescriptions[id] || 'Style reference');
      const image = document.createElement('img');
      image.src = 'media/gallery/styles/' + id + '.jpg';
      image.alt = '';
      label.append(radio, image);
      choices.append(label);
    }
    picker.replaceChildren(choices);
    document.querySelector('#dataset').textContent = scene.dataset.dataset;
    showStyle(selected, updateUrl);
    return selected;
  }

  function enablePreview() {
    if (!preview || typeof preview.showModal !== 'function') return;

    orbitOpen.hidden = false;
    const links = [...document.querySelectorAll('#orbit-open, main a.media-box, .control-meta a, #original-link, #styled-link')];
    const title = preview.querySelector('#preview-title');
    const content = preview.querySelector('.preview-content');
    let opener = null;
    let openingScroll = null;

    function previewMedia(link) {
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

      return media;
    }

    function show(link) {
      if (preview.open) return;
      opener = link;
      openingScroll = history.state?.semgeoScroll || [window.scrollX, window.scrollY];
      if (link === orbitOpen) {
        title.textContent = 'Kitchen loader · ' + orbitName.textContent;
        orbitMount.style.height = orbitMount.offsetHeight + 'px';
        content.replaceChildren(orbit);
      } else {
        content.replaceChildren(previewMedia(link));
      }
      preview.showModal();
      updateMotion();
    }

    function hide() {
      if (!preview.open) return;
      const media = content.firstElementChild;
      if (media === orbit) {
        orbitMount.append(orbit);
        orbitMount.style.height = '';
      } else if (media instanceof HTMLVideoElement) {
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
    [...clips, comparison, orbitMount].forEach(element => observer.observe(element));
  } else {
    [...clips, comparison, orbitMount].forEach(element => visible.add(element));
  }
  preference.addEventListener('change', event => {
    motion = !event.matches;
    updateMotion();
  });
  document.addEventListener('visibilitychange', updateMotion);
  window.addEventListener('pagehide', event => {
    if (!event.persisted) videos.forEach(clearSource);
  });

  const initialUrl = new URL(location.href);
  const requestedScene = initialUrl.searchParams.get('scene');
  const requestedStyle = initialUrl.searchParams.get('style');
  const sceneAvailable = [...select.options].some(option => option.value === requestedScene);
  if (sceneAvailable) select.value = requestedScene;
  const selectedStyle = chooseScene(requestedScene && !sceneAvailable ? null : requestedStyle, false);
  const removedSelection = (requestedScene && !sceneAvailable) || (requestedStyle && requestedStyle !== selectedStyle);
  if (removedSelection && location.protocol !== 'file:') {
    initialUrl.searchParams.set('scene', select.value);
    initialUrl.searchParams.set('style', selectedStyle);
    history.replaceState(history.state, '', initialUrl);
  }
  document.querySelector('.scene-picker').hidden = false;
  document.querySelector('.style-picker').hidden = false;
  const initialOrbitStyle = orbitStyles.querySelector(':checked');
  if (initialOrbitStyle.dataset.video !== orbitVideo.getAttribute('src')) selectOrbitStyle(initialOrbitStyle);
  updateSplit();
  enablePreview();
  updateMotion();
})();
