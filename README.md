# SemGeoSplat

3D Gaussian scene stylization with semantic-geometric regions and fused Gromov-Wasserstein matching.

**[Project website and visual results](https://teamo-lemon.github.io/SemGeoSplat/)**

Starting from a trained Gaussian scene, SemGeoSplat refines the geometry, distills DINOv3 features, decomposes a reference image into style primitives, constructs 3D regions, matches regions to primitives, and optimizes the scene's appearance.

## Installation

Use Linux, Python 3.10, an NVIDIA GPU, and a CUDA 12.4-compatible driver. CUDA compilation also requires a C++ compiler. Run these commands from the repository root:

```bash
git clone --recurse-submodules https://github.com/TeAmo-Lemon/SemGeoSplat.git
cd SemGeoSplat
conda env create -f environment.yml
conda activate semgeo
pip install submodules/fused-ssim --no-build-isolation
```

For an existing Python 3.10 environment with the CUDA 12.4 toolkit:

```bash
pip install -r requirements.txt
git submodule update --init --recursive
pip install submodules/fused-ssim --no-build-isolation
```

The first stylization run downloads DINOv3 from ModelScope and VGG16 from torchvision unless their weights are already cached. Model weights and datasets are not included. The default DINO backbone is `facebook/dinov3-vith16plus-pretrain-lvd1689m`; select another compatible backbone with `--dino_model`.

## Prepare a scene

Use a COLMAP scene with `images/` and `sparse/0/`, or a NeRF synthetic scene with its transform JSON files. To prepare a photo collection, place the photos in `<scene>/input/` and run `python convert.py -s <scene>` with COLMAP installed. ImageMagick is needed only for `convert.py --resize`.

Train a base Gaussian scene, or use an existing compatible `point_cloud.ply`:

```bash
python train.py -s /path/to/scene -m output/base_scene --eval --disable_viewer
```

## Stylize

```bash
python train_style.py \
  -s /path/to/scene \
  -m output/stylized_scene \
  --start_ply output/base_scene/point_cloud/iteration_30000/point_cloud.ply \
  --style_image /path/to/style.jpg \
  --eval --disable_viewer
```

DINO feature extraction runs automatically. Refinement and distilled-feature caches are stored beside the input checkpoint in `point_cloud/iteration_refine/` and reused across styles, so the input model directory must be writable. Use `--force_refine` or `--force_dino` to rebuild these caches. Give each scene/style pair a separate output directory.

The final scene is saved as `output/stylized_scene/point_cloud/iteration_1500/point_cloud.ply` with region and style sidecars. `cfg_args` records the run configuration.

| Option | Default | Purpose |
| --- | --- | --- |
| `--refine_iterations` | `500` | Refinement steps |
| `--prune_percent` | `0.2` | Fraction removed per pruning round |
| `--dino_distill_iterations` | `3000` | Semantic distillation steps |
| `--dino_feature_dim` | `32` | Distilled feature dimension |
| `--style_primitive_k` | `8` | Number of style primitives |
| `--region_merge_max_cost` | `0.60` | Region merge threshold |
| `--fgw_alpha` | `0.5` | Balance between feature and structure costs |
| `--style_iterations` | `1500` | Appearance optimization steps |
| `--style_render_scale` | `0.5` | Training render scale |

Optional scene-specific overrides are listed in [`configs/scene_presets.json`](configs/scene_presets.json). They are not applied automatically: pass their keys as command-line options. For example, the counter preset adds `--style_primitive_k 6 --region_merge_max_cost 0.50 --fgw_alpha 0.7`. See `python train_style.py --help` for all options.

## View and export

```bash
python viewer.py -m output/stylized_scene --port 8081
python render.py -m output/stylized_scene --save-gif
```

The viewer supports RGB, semantic features, region visualization, and camera-path export. To render a saved trajectory without the original dataset:

```bash
python render.py -m output/stylized_scene \
  --camera-path /path/to/trajectory.json --save-gif
```

Dataset-view renders go into `train/ours_<iteration>/renders/` and `test/ours_<iteration>/renders/`. GIF export defaults to training views; add `--gif-test` for test views. Trajectory frames and animations go into `trajectory/<path-name>/ours_<iteration>/`. Use `--gif-max-size 640 --gif-colors 64` for smaller GIFs, and `--force` to rebuild existing renders. Omit `--disable_viewer` during training to enable the live viewer.

## Standard reconstruction metrics

The original 3DGS PSNR, SSIM and LPIPS evaluation is retained for reconstruction checks:

```bash
python render.py -m output/base_scene --skip_train --write-gt
python metrics.py -m output/base_scene
```

## Repository layout

```text
train.py                 Base Gaussian scene training
train_style.py           SemGeoSplat training entry point
render.py / viewer.py    Rendering, animation export and interactive viewing
arguments/              Command-line parameters
style_pipeline/         Refinement, features, regions, matching and style optimization
scene/                  Gaussian scene and dataset loading
gaussian_renderer/      gsplat rendering and live viewer integration
semgeo_utils/           Shared utilities and DINO feature extraction
configs/                Optional scene parameter presets
submodules/fused-ssim/   Optional accelerated SSIM implementation
docs/                   GitHub Pages website
```

## Acknowledgments and license

This implementation builds on [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [gsplat](https://github.com/nerfstudio-project/gsplat), [DINOv3](https://github.com/facebookresearch/dinov3), [POT](https://github.com/PythonOT/POT), [viser](https://github.com/nerfstudio-project/viser), [nerfview](https://github.com/nerfstudio-project/nerfview), and [fused-ssim](https://github.com/rahul-goel/fused-ssim).

See [`LICENSE.md`](LICENSE.md). Upstream copyright notices and third-party license terms are retained in the corresponding files and dependency repositories.
