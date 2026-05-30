#!/usr/bin/env python3
"""Standalone bpy rendering subprocess for Blocksworld states.

This script is called as a subprocess from the main training pipeline.
It imports bpy (Blender Python) WITHOUT importing PyTorch, avoiding
the bpy+PyTorch crash.

Usage:
    python render_bpy_subprocess.py <states_pkl> <output_dir> [--samples N] [--width W] [--height H]
"""
import sys
import os
import pickle
import argparse
import random
import numpy as np

# Add ViPlan to path
VIPAN_ROOT = "/home/claudeuser/ViPlan"
sys.path.insert(0, VIPAN_ROOT)

# Import bpy and ViPlan rendering modules
import bpy
import viplan.rendering.blocksworld.blocks as blocks
import viplan.rendering.blocksworld.render_utils as render_utils
import viplan.rendering.blocksworld.utils as utils

# Block ID mapping
from viplan.planning.conversion import block_id, block_letter
COLOR_MAP = {
    block_id['R']: [1, 0, 0, 1],
    block_id['G']: [0, 0.8, 0, 1],
    block_id['B']: [0, 0, 0.8, 1],
    block_id['Y']: [1, 1, 0, 1],
    block_id['P']: [0.2, 0, 0.5, 1],
    block_id['O']: [1, 0.5, 0, 1],
}

RENDER_DATA = os.path.join(VIPAN_ROOT, "data", "blocksworld_rendering")


class SimpleArgs:
    """Minimal args object for render_scene."""
    pass


def make_render_args(width=224, height=224, n_samples=32, use_gpu=0):
    args = SimpleArgs()
    args.base_scene_blendfile = os.path.join(RENDER_DATA, "base_scene.blend")
    args.properties_json = os.path.join(RENDER_DATA, "properties.json")
    args.shape_dir = os.path.join(RENDER_DATA, "shapes")
    args.material_dir = os.path.join(RENDER_DATA, "materials")
    args.width = width
    args.height = height
    args.render_num_samples = n_samples
    args.render_min_bounces = 8
    args.render_max_bounces = 8
    args.render_tile_size = 256
    args.use_gpu = use_gpu
    args.key_light_jitter = 0.0
    args.fill_light_jitter = 0.0
    args.back_light_jitter = 0.0
    args.camera_jitter = 0.0
    args.seed = 0
    return args


def render_state_matrix(matrix, output_path, args, seed=0):
    """Render a single Blocksworld state matrix to an image file."""
    state = blocks.State(
        list(matrix),
        properties_json=args.properties_json,
        seed=seed,
    )

    render_dir = os.path.dirname(output_path)
    os.makedirs(render_dir, exist_ok=True)
    scene_json = os.path.join(render_dir, f"_scene_{seed}.json")

    random.seed(seed)
    np.random.seed(seed)

    render_utils.render_scene(
        args,
        output_image=output_path,
        output_scene=scene_json,
        objects=state.for_rendering(),
    )

    # Cleanup scene json
    if os.path.exists(scene_json):
        os.remove(scene_json)

    return os.path.exists(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("states_pkl", help="Pickle file with list of numpy matrices")
    parser.add_argument("output_dir", help="Output directory for rendered images")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--use-gpu", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    cli_args = parser.parse_args()

    with open(cli_args.states_pkl, "rb") as f:
        states = pickle.load(f)

    os.makedirs(cli_args.output_dir, exist_ok=True)
    args = make_render_args(cli_args.width, cli_args.height, cli_args.samples, cli_args.use_gpu)

    n = len(states)
    rendered = 0
    failed = 0

    for i in range(cli_args.start_idx, n):
        img_path = os.path.join(cli_args.output_dir, f"state_{i:05d}.png")
        if os.path.exists(img_path):
            rendered += 1
            continue

        try:
            ok = render_state_matrix(states[i], img_path, args, seed=i)
            if ok:
                rendered += 1
            else:
                failed += 1
                print(f"FAIL state {i}: render returned False")
        except Exception as e:
            failed += 1
            print(f"FAIL state {i}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"PROGRESS {i + 1}/{n} rendered={rendered} failed={failed}")

    print(f"DONE rendered={rendered} failed={failed} total={n}")

    # Write manifest
    manifest = {"total": n, "rendered": rendered, "failed": failed}
    with open(os.path.join(cli_args.output_dir, "_manifest.json"), "w") as f:
        import json
        json.dump(manifest, f)


if __name__ == "__main__":
    main()
