"""Isolate a set of 3D Gaussians and render a white-background camera orbit."""

import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree

from gaussian_renderer import GaussianModel, render
from scene.cameras import MiniCam
from semgeo_utils.graphics_utils import getProjectionMatrix
from semgeo_utils.appearance_mix import regions_xyz_fingerprint, xyz_fingerprint


def box_mask(across, along, height, box):
    x0, x1, y0, y1, z0, z1 = box
    return (
        (across >= x0) & (across <= x1)
        & (along >= y0) & (along <= y1)
        & (height > z0) & (height < z1)
    )


def select_vertices(vertices, recipe):
    xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')])
    local = xyz @ np.asarray(recipe['world_to_local']).T - recipe['offset']
    x, y, z = local.T
    theta = math.radians(recipe['heading_degrees'])
    across = math.cos(theta) * x + math.sin(theta) * y
    along = -math.sin(theta) * x + math.cos(theta) * y
    a, b, c = recipe['support_plane']
    height = z - (a * x + b * y + c)

    mask = np.zeros(len(vertices), dtype=bool)
    for box in recipe['boxes']:
        mask |= box_mask(across, along, height, box)

    if recipe.get('dark_contact_boxes'):
        # Track and tabletop centers overlap at the contact surface. Use the
        # source color here so changes in stylized appearance preserve point IDs.
        dc = np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)])
        dark = (0.5 + 0.2820947918 * dc).max(axis=1) <= recipe['contact_max_rgb']
        for box in recipe['dark_contact_boxes']:
            mask |= dark & box_mask(across, along, height, box)

    scales = np.column_stack([np.exp(vertices[f'scale_{i}']) for i in range(3)])
    candidates = np.flatnonzero(mask & (scales.max(axis=1) < recipe['max_scale']))
    if not candidates.size:
        raise ValueError('No Gaussians intersect the object crop')
    return largest_component(candidates, local[candidates], recipe['voxel_size'])


def largest_component(indices, positions, voxel_size):
    grid = np.floor((positions - positions.min(axis=0)) / voxel_size).astype(int) + 1
    shape = tuple(int(size) + 2 for size in grid.max(axis=0))
    occupied = np.zeros(shape, dtype=bool)
    occupied[tuple(grid.T)] = True
    labels, _ = ndimage.label(
        ndimage.binary_dilation(occupied), structure=np.ones((3, 3, 3), dtype=bool)
    )
    point_labels = labels[tuple(grid.T)]
    return indices[point_labels == np.bincount(point_labels).argmax()]


def support_heights(background, foreground, tolerance):
    """Signed heights above the dominant plane fitted to nearby background centers."""
    rng = np.random.default_rng(0)
    sample = background[rng.choice(len(background), min(14000, len(background)), replace=False)]
    best = 0
    for _ in range(11):
        triples = rng.integers(len(sample), size=(48, 3))
        p, q, r = sample[triples].transpose(1, 0, 2)
        normals = np.cross(q - p, r - p)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-9
        if not valid.any():
            continue
        normals = normals[valid] / lengths[valid, None]
        offsets = (p[valid] * normals).sum(axis=1)
        counts = (np.abs(sample @ normals.T - offsets) < tolerance).sum(axis=0)
        winner = np.argmax(counts)
        if counts[winner] > best:
            normal, offset, best = normals[winner], offsets[winner], counts[winner]
    if best < 3:
        raise ValueError('The background points do not identify a support plane')
    for _ in range(4):
        inliers = background[np.abs(background @ normal - offset) < tolerance]
        if len(inliers) < 3:
            raise ValueError('The background has insufficient planar support')
        centered = inliers - np.median(inliers, axis=0)
        normal = np.linalg.eigh(centered.T @ centered)[1][:, 0]
        offset = np.median(inliers @ normal)
    heights = foreground @ normal - offset
    background_heights = background @ normal - offset
    if np.median(heights) < 0:
        heights = -heights
        background_heights = -background_heights
    slab = background_heights[np.abs(background_heights) < 2.5 * tolerance]
    return heights, np.quantile(slab, 0.97)


def select_supported_region(vertices, cache_path, region):
    """Keep the connected foreground above the support of a saved 3D region."""
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    labels = np.asarray(cache['cluster_labels'])
    if labels.shape != (len(vertices),) or regions_xyz_fingerprint(cache, cache_path) != xyz_fingerprint(vertices):
        raise ValueError('The saved region labels must match the ordered source Gaussians')
    if region < 0 or region >= cache['num_regions']:
        raise ValueError('Choose a region ID from the saved source partition')
    ids = np.flatnonzero(labels == region)
    if len(ids) < 3:
        raise ValueError('The selected region has too few Gaussians')

    xyz = np.column_stack([vertices[key] for key in ('x', 'y', 'z')])
    foreground = xyz[ids]
    center = np.median(foreground, axis=0)
    radius = np.quantile(np.linalg.norm(foreground - center, axis=1), 0.90)
    nearby = (labels >= 0) & (labels != region) & (np.linalg.norm(xyz - center, axis=1) < 1.65 * radius)
    background = xyz[nearby]
    distances = cKDTree(foreground).query(background, k=1)[0]
    background = background[distances < 0.5 * radius]
    if len(background) < 3:
        raise ValueError('No neighboring labeled region supplies a support surface')
    tolerance = 0.016 * radius
    heights, top = support_heights(background, foreground, tolerance)
    scales = np.maximum.reduce([np.exp(vertices[f'scale_{i}'][ids]) for i in range(3)])
    core_scales = scales[heights > 3 * tolerance]
    if not len(core_scales):
        raise ValueError('The selected region does not extend above its support')
    keep = (heights > top) & (scales < np.quantile(core_scales, 0.997))
    kept_positions = foreground[keep]
    if len(kept_positions) < 2:
        raise ValueError('Too few points remain above the support surface')
    spacing = np.median(cKDTree(kept_positions).query(kept_positions[::5], k=2)[0][:, 1])
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Cannot estimate the spacing between foreground Gaussians')
    return largest_component(ids[keep], kept_positions, 3 * spacing)


def orbit_view(azimuth, recipe, width, height):
    basis = np.asarray(recipe['world_to_local'])
    a, b, _ = recipe['support_plane']
    up = basis.T @ np.array([-a, -b, 1.0])
    up /= np.linalg.norm(up)
    horizontal = basis[0] - up * (basis[0] @ up)
    horizontal /= np.linalg.norm(horizontal)
    tangent = np.cross(up, horizontal)
    target = basis.T @ (np.asarray(recipe['target_local']) + recipe['offset'])

    elevation = math.radians(recipe['elevation'])
    radial = math.cos(azimuth) * horizontal + math.sin(azimuth) * tangent
    outward = math.cos(elevation) * radial + math.sin(elevation) * up
    eye = target + outward * recipe['distance']
    forward = -outward
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    w2c = np.eye(4)
    w2c[:3, :3] = np.stack([right, down, forward])
    w2c[:3, 3] = -w2c[:3, :3] @ eye

    fov_x = math.radians(recipe['horizontal_fov'])
    fov_y = 2 * math.atan(math.tan(fov_x / 2) * height / width)
    view = torch.as_tensor(w2c.T.copy(), device='cuda', dtype=torch.float32)
    projection = getProjectionMatrix(0.01, 100.0, fov_x, fov_y).T.to(view)
    return MiniCam(width, height, fov_y, fov_x, 0.01, 100.0, view, view @ projection)


def write_gif(paths, output, fps):
    # Build a shared palette and reserve pure white. Otherwise common encoders
    # dither even the untouched background from frame to frame.
    samples = []
    for path in paths:
        with Image.open(path) as image:
            samples.append(np.asarray(image.convert('RGB'))[::4, ::4])
    mosaic = Image.fromarray(np.concatenate(samples, axis=0))
    adaptive = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    palette = adaptive.getpalette()[:3 * 255] + [255, 255, 255]
    shared = Image.new('P', (1, 1))
    shared.putpalette(palette)
    frames = []
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            indexed = rgb.quantize(palette=shared, dither=Image.Dither.FLOYDSTEINBERG)
            white = Image.fromarray(np.all(np.asarray(rgb) == 255, axis=2))
            indexed.paste(255, mask=white)
            frames.append(indexed)
    # GIF delays are stored as integer centiseconds. Distribute the rounding.
    duration = [10 * (round(100 * (i + 1) / fps) - round(100 * i / fps)) for i in range(len(frames))]
    frames[0].save(
        output, format='GIF', save_all=True, append_images=frames[1:], duration=duration,
        loop=0, palette=bytes(palette), background=255, optimize=False, disposal=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original Gaussian PLY')
    parser.add_argument('--appearance', type=Path, help='Stylized PLY on the same geometry')
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--regions', type=Path, help='Saved region partition for a foreground object on a planar support')
    parser.add_argument('--region', type=int, help='Region to isolate and clean using neighboring background geometry')
    parser.add_argument('--output', type=Path, required=True, help='Destination GIF')
    parser.add_argument('--export-ply', type=Path, help='Save the isolated Gaussian model')
    parser.add_argument('--frames', type=int, default=72)
    parser.add_argument('--fps', type=int, default=18)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=368)
    args = parser.parse_args()
    if min(args.frames, args.fps, args.width, args.height) < 1 or args.fps > 100:
        parser.error('Frame count and dimensions must be positive, and FPS must be in [1, 100]')
    if (args.region is None) != (args.regions is None):
        parser.error('--regions and --region must be provided together')
    inputs = {path.resolve() for path in (args.source, args.appearance, args.recipe, args.regions) if path}
    if args.regions is not None:
        inputs.add(args.regions.with_name('point_cloud.ply').resolve())
    outputs = [path.resolve() for path in (args.output, args.export_ply) if path]
    if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
        parser.error('Use a different path for each output and each input')
    recipe = json.loads(args.recipe.read_text())
    original = PlyData.read(args.source, mmap='r')['vertex'].data
    chosen = (select_vertices(original, recipe) if args.regions is None
              else select_supported_region(original, args.regions, args.region))
    vertices = original
    if args.appearance:
        vertices = PlyData.read(args.appearance, mmap='r')['vertex'].data
        fields = ['x', 'y', 'z', 'opacity']
        fields.extend(f'scale_{i}' for i in range(3))
        fields.extend(f'rot_{i}' for i in range(4))
        if len(original) != len(vertices) or any(
            not np.array_equal(original[key], vertices[key]) for key in fields
        ):
            raise ValueError('Appearance checkpoint does not share the source geometry')
    rest = sum(key.startswith('f_rest_') for key in vertices.dtype.names)
    degree = math.isqrt(1 + rest // 3) - 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        ply = args.export_ply or folder / 'object.ply'
        ply.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertices[chosen], 'vertex')], text=False).write(ply)
        model = GaussianModel(degree)
        model.load_ply(str(ply))
        white = torch.ones(3, device='cuda')
        pipeline = SimpleNamespace(convert_SHs_python=False, antialiasing=False)
        frames = []
        with torch.inference_mode():
            for i in range(args.frames):
                azimuth = math.radians(recipe['initial_azimuth']) + 2 * math.pi * i / args.frames
                camera = orbit_view(azimuth, recipe, args.width * 2, args.height * 2)
                pixels = render(camera, model, pipeline, white, no_densify_hook=True)['render']
                rgb = pixels.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                frame = Image.fromarray(rgb).resize((args.width, args.height), Image.Resampling.LANCZOS)
                path = folder / f'{i:04}.png'
                frame.save(path)
                frames.append(path)
        write_gif(frames, args.output, args.fps)
    print(f'Saved {args.frames} views of {len(chosen):,} Gaussians to {args.output}')


if __name__ == '__main__':
    main()
