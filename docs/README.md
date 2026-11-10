# SemGeoSplat project website

The website is a static, dependency-free gallery, entirely contained in `docs/`.

**Public URL:** https://TeAmo-Lemon.github.io/SemGeoSplat/

## GitHub Pages

In the repository's **Settings → Pages**, select:

- Source: **Deploy from a branch**
- Branch: **main**
- Folder: **/docs**

The `.nojekyll` file keeps the site as plain HTML, CSS and JavaScript. All asset paths are relative, so the `/SemGeoSplat/` project-site prefix works without a build step. Application / research code can live outside `docs/` without being published to the website.

## Preview

From the repository root:

```sh
python3 docs/scripts/preview.py --port 8080
```

Open http://127.0.0.1:8080. The preview server supports HTTP byte ranges so Safari can stream MP4s. Use an HTTP server rather than opening `index.html` directly, because the gallery loads a JSON manifest.

## Media and provenance

The website displays **12 selected SemGeo results** across 10 scenes and 11 styles, in this order: `orchids-3`, `garden-35`, `counter-9`, `truck-133`, `trex-121`, `bonsai-2`, `flower-76`, `train-138`, `kitchen-120`, `horns-12`, `garden-121`, `orchids-87`. These were selected for clear subjects, coherent palettes, distinctive styles and varied scene content; representative frames from each animation were also reviewed.

The original imported media collection (174 results, 12 original scene animations and 51 style references) remains in `assets/` for future curation. Only the 12 entries in `data/results.json` appear on the website. The site's mapping explicitly identifies `semgeo` as the method. Other methods, participant responses, databases, admin pages and environment files are not included.

The source site's original GIFs had already been converted to H.264 MP4. The published MP4s are copied unchanged; they retain the source site's resolution, frame rate and duration. Looping MP4 provides the animated presentation with native playback controls. JPG posters are extracted from the videos for fast browsing. Style JPGs are copied unchanged.

```
docs/
  index.html              Page structure and metadata
  styles.css              Responsive presentation
  app.js                  Curated gallery, previews, and result viewer
  data/results.json       Ordered selection of 12 results
  assets/results/<scene>-<style>/semgeo.mp4
  assets/gt/<scene>.mp4
  assets/styles/<style>.jpg
  assets/posters/<scene>-<style>.jpg
  assets/posters/<scene>-original.jpg
```

The compact gallery uses four columns on desktop, three on tablets and two on phones. Desktop result thumbnails are at most 247 px wide; the featured result is about 422 px wide and the comparison viewer is capped at 760 px overall.

Videos in the grid load only on desktop hover or focus. The result viewer loads the selected clip; original animations have separate playback controls. Leaving the page pauses playback. Reduced-motion preferences disable automatic animation.

## Update the selection

Keep this a compact showcase. `data/results.json` is the ordered public selection; assets outside this list are not shown. The three homepage selections are set by `featuredIds` in `app.js` and must also appear in the manifest.

## Add a new result

1. Add the SemGeo MP4, original MP4 (for a new scene), and reference JPG to the matching asset directories.
2. Add `{ "id": "scene-style", "scene": "scene", "sceneLabel": "场景中文名", "styleId": "style" }` to `data/results.json`.
3. Add a display name to `names` in `app.js` for a new scene. Optional homepage selections are defined by `featuredIds`.
4. Generate the result poster with FFmpeg (replace the example ID):

```sh
ffmpeg -ss 0.1 -i docs/assets/results/truck-133/semgeo.mp4 -frames:v 1 -vf scale=960:-2 -q:v 3 docs/assets/posters/truck-133.jpg
```

For a new scene, generate `assets/posters/<scene>-original.jpg` from its original MP4 in the same way. Update the static metadata and introductory counts in `index.html` if the collection changes. The visible gallery counts are computed from the manifest.

## Content notes

Style numbers match the source collection. No artist names, paper details, affiliations, benchmarks or publication claims are inferred. Reference images remain associated with their original style IDs. No new license for third-party reference artwork is asserted by this gallery.
