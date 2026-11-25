(() => {
  const panels = [...document.querySelectorAll('.scene-panel')];
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const clips = [...document.querySelectorAll('img[data-animation]')];
  const toggle = document.querySelector('.motion');
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const visible = new Set();
  let enabled = !preference.matches;

  function updateAnimations() {
    for (const clip of clips) {
      const play = enabled && visible.has(clip) && !clip.closest('[hidden]') && !document.hidden;
      const source = play ? clip.dataset.animation : clip.dataset.poster;
      if (clip.getAttribute('src') !== source) clip.setAttribute('src', source);
    }
    toggle.textContent = enabled ? 'Pause motion' : 'Play motion';
  }

  function selectScene(index, focus = false) {
    tabs.forEach((tab, i) => {
      const active = i === index;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      panels[i].hidden = !active;
    });
    if (focus) tabs[index].focus();
    updateAnimations();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectScene(index));
    tab.addEventListener('keydown', event => {
      const destinations = {
        ArrowLeft: (index + tabs.length - 1) % tabs.length,
        ArrowRight: (index + 1) % tabs.length,
        Home: 0,
        End: tabs.length - 1
      };
      if (!Object.hasOwn(destinations, event.key)) return;
      event.preventDefault();
      selectScene(destinations[event.key], true);
    });
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      }
      updateAnimations();
    });
    clips.forEach(clip => observer.observe(clip));
  } else {
    clips.forEach(clip => visible.add(clip));
  }

  toggle.addEventListener('click', () => {
    enabled = !enabled;
    updateAnimations();
  });
  preference.addEventListener('change', event => {
    enabled = !event.matches;
    updateAnimations();
  });
  document.addEventListener('visibilitychange', updateAnimations);
  document.querySelector('.tabs').hidden = false;
  toggle.hidden = false;
  selectScene(0);
})();
