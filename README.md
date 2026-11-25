# SemGeoSplat

Stylize a 3D Gaussian scene from one reference image. Semantic and geometric regions connect the scene to patches of the reference through fused Gromov–Wasserstein (FGW) transport. The same edit can be rendered from multiple cameras.

[Visual examples](docs/index.html) · [Kitchen LEGO brick orbit](docs/media/kitchen-orbit-stylized.gif) · [FGW texture placement on Flower](docs/media/flower-control.gif)

## Install

Use Linux with an NVIDIA GPU, a CUDA 12.4-compatible driver and Conda. Run commands from the repository root.

```bash
conda env create -f environment.yml
conda activate semgeo
```

The environment includes PyTorch and gsplat. Scene and reference features use DINOv3 from ModelScope. The first style fit also downloads the torchvision VGG16 weights. To use the optional fused SSIM implementation, run the following with a compatible CUDA compiler. The training code includes a PyTorch fallback.

```bash
python -m pip install --no-build-isolation 'git+https://github.com/rahul-goel/fused-ssim.git@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38'
```

## Stylize and render a scene

Point `SCENE` at a COLMAP reconstruction with `images/` and `sparse/0/`. For a downsampled image directory, pass the same `-i images_2` or `-i images_4` argument to both training commands. You can start at the second command if you already have a base Gaussian PLY registered to those cameras. Change `STYLE` to try your own image.

```bash
SCENE=/path/to/scene
STYLE=docs/media/garden-scene-reference.jpg
BASE=output/base
STYLED=output/styled

python train.py -s "$SCENE" -m "$BASE" --disable_viewer
python train_style.py -s "$SCENE" -m "$STYLED" \
  --start_ply "$BASE/point_cloud/iteration_30000/point_cloud.ply" \
  --style_image "$STYLE" --spatial_fgw_routing local --disable_viewer

python render.py -m "$STYLED" --iteration 1500 --skip_test --save-gif
python viewer.py -m "$STYLED" -i 1500 --host 127.0.0.1
```

The default style fit saves `point_cloud/iteration_1500/` in the style model. The GIF above uses the training cameras and is written under `train/ours_1500/`. To make a moving-camera comparison, use **Save Trajectory** in the viewer and render both models with the same path.

```bash
python render.py -m "$BASE" --iteration 30000 --camera-path trajectory.json --save-gif
python render.py -m "$STYLED" --iteration 1500 --camera-path trajectory.json --save-gif
```

## Choose a region and edit its appearance

Render the cluster colors and look up their IDs. The example below applies half the fitted appearance to region 6 and leaves the rest of the scene at its original appearance. Replace 6 with a fitted region in your scene. The export writes a new PLY and runs on CPU.

```bash
ITER="$STYLED/point_cloud/iteration_1500"
REFINE="$BASE/point_cloud/iteration_refine"

python render.py -m "$STYLED" --iteration 1500 --skip_test --render-regions --region-color-source clusters
python style_control.py list "$ITER"
python style_control.py export "$ITER" --source-ply "$REFINE/point_cloud.ply" \
  --strength 0 --region 6=0.5 --output output/edited.ply
```

For finer control, a recipe specifies how a region uses a particular patch of the reference. The [Flower recipe](configs/flower-placement.json) is one of the settings shown on the project page. It uses region 6, the marked patch of reference 77, a 50% share and a placement bias of +0.75. The region ID and 3D anchor belong to that reconstruction. When adapting the recipe, choose the ID from your region render and an anchor on the object in the coordinate system of your PLY.

| Recipe field | What it controls |
|---|---|
| `strength` | Appearance strength from 0 (original) to 1 (fully fitted). |
| `texture_mix.box` | A rectangle in the reference image, in normalized left, top, right, bottom coordinates. |
| `texture_mix.fraction` | The fraction of the region’s transport mass assigned to patches inside the rectangle. |
| `texture_mix.source_bias` | With the `centroid` profile, positive bias moves the selected texture toward the 3D anchor along the region graph. Negative bias moves it away. The patch share remains fixed. |

Pass `--style_control_file recipe.json --spatial_fgw_routing local` to `train_style.py` to use a recipe when fitting a new appearance. To change placement after fitting, use the saved region, PCA and reference-patch caches and two appearances fitted on the same Gaussians.

```bash
python -m scripts.solve_fgw_recipe \
  --source "$REFINE/point_cloud.ply" --regions "$ITER/regions.pt" \
  --pca "$REFINE/dino_features.pt" --style-primitives "$ITER/style_primitives.pt" \
  --style-image "$STYLE" --control recipe.json --region 6 \
  --contrast-weight 0.25 --output output/edit-plan.pt

python -m scripts.render_fgw_appearance \
  --source "$REFINE/point_cloud.ply" --regions "$ITER/regions.pt" \
  --transport output/edit-plan.pt --region 6 \
  --endpoint-a outside.ply --endpoint-b inside.ply \
  --cameras-json "$STYLED/cameras.json" --views 0 --output-dir output/preview \
  --export-ply output/placed.ply
```

Use the exact reference file that produced `style_primitives.pt`. To make `outside.ply` and `inside.ply`, fit two style models with the same base, reference, selected region and box. Set `fraction` to 0 for the first model and 1 for the second. Use their `point_cloud.ply` files as the two endpoints. For an original-to-stylized transition you can instead use the refined source as endpoint A and the existing stylized PLY as B. The solver runs on CPU. Rendering new views uses the GPU.

## Isolate the Kitchen loader

The Kitchen example uses a [calibrated 3D crop](configs/kitchen-loader.json), including the tabletop plane and the contacts of the tracks. The renderer selects the object once from the original Gaussian model and orbits it on white. The following command also exports a standalone object PLY.

```bash
python -m scripts.render_object_orbit \
  --source "$REFINE/point_cloud.ply" --recipe configs/kitchen-loader.json \
  --output output/kitchen-loader.gif --export-ply output/kitchen-loader.ply
```

Use a refined Kitchen reconstruction in the preset’s coordinate system. Add `--appearance /path/to/stylized/point_cloud.ply` to replay the same 72 views with a registered appearance. The script verifies positions, rotations, scales and opacities before applying the selection. The website pairs the original loader with the appearance fitted to the complete LEGO brick image (reference 133). The Kitchen scene was stylized before extracting the same object from both models. A reconstruction in a different coordinate frame requires its own crop.

## Project page and attribution

The scene selector covers twelve scenes. Each reference thumbnail selects a stylized result rendered on the original camera path.

Open [docs/index.html](docs/index.html) directly, or run `python -m http.server 8000` from the repository root and visit `http://localhost:8000/docs/`.

The implementation builds on the Graphdeco reconstruction code, gsplat rasterization and DINOv3 features. The gallery uses Mip-NeRF 360 and LLFF scenes. Upstream reconstruction evaluation is available through `metrics.py` and `full_eval.py`. The repository retains the Graphdeco license. [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) · [gsplat](https://github.com/nerfstudio-project/gsplat) · [DINOv3](https://github.com/facebookresearch/dinov3) · [License](LICENSE.md)
