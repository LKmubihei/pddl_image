#!/usr/bin/env python3
"""
Direct bpy Blocksworld renderer (no .blend dependency).

Creates colored cubes on a ground plane with lighting, compatible with
bpy 5.x. Supports multi-view rendering via --views N.
"""
import sys
import os
import pickle
import argparse
import random
import json
import numpy as np

VIPAN_ROOT = "/home/claudeuser/ViPlan"
sys.path.insert(0, VIPAN_ROOT)

import bpy
from mathutils import Vector

# Block color map (matching ViPlan)
COLOR_MAP = {
    1: (1.0, 0.0, 0.0),    # R = Red
    2: (0.0, 0.8, 0.0),    # G = Green
    3: (0.0, 0.0, 0.8),    # B = Blue
    4: (1.0, 1.0, 0.0),    # Y = Yellow
    5: (0.8, 0.0, 0.5),    # P = Purple
    6: (1.0, 0.5, 0.0),    # O = Orange
}

# Multi-view camera configurations
CAMERA_VIEWS = [
    {"cam_loc": (0.0, -4.5, 5.0), "target": (0.0, 0.0, 1.2), "lens": 25},
    {"cam_loc": (1.2, -3.8, 5.5), "target": (0.0, 0.0, 1.2), "lens": 27},
    {"cam_loc": (-1.2, -3.8, 5.5), "target": (0.0, 0.0, 1.2), "lens": 27},
]

# Column marker colors
MARKER_COLORS = [
    (0.6, 0.6, 0.8),   # C1 light blue
    (0.8, 0.6, 0.6),   # C2 light red
    (0.6, 0.8, 0.6),   # C3 light green
    (0.8, 0.8, 0.6),   # C4 light yellow
]


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)


def setup_base_scene(width=224, height=224, samples=32):
    """Set up render settings, world, ground, lights (no camera)."""
    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.cycles.samples = samples
    scene.cycles.blur_glossy = 2.0
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    # World background
    world = bpy.data.worlds.get('World')
    if world is None:
        world = bpy.data.worlds.new('World')
        scene.world = world
    if world.use_nodes:
        bg = world.node_tree.nodes.get('Background')
        if bg:
            bg.inputs[0].default_value = (0.9, 0.9, 0.95, 1.0)
            bg.inputs[1].default_value = 1.0

    # Ground plane
    bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = 'Ground'
    mat = bpy.data.materials.new('GroundMat')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.8
    ground.data.materials.append(mat)

    # Key light
    bpy.ops.object.light_add(type='SUN', location=(5, -3, 10))
    key_light = bpy.context.active_object
    key_light.name = 'KeyLight'
    key_light.data.energy = 3.0
    key_light.rotation_euler = (0.8, 0.2, 0.5)

    # Fill light
    bpy.ops.object.light_add(type='SUN', location=(-3, 5, 8))
    fill_light = bpy.context.active_object
    fill_light.name = 'FillLight'
    fill_light.data.energy = 1.5
    fill_light.rotation_euler = (0.5, -0.3, -0.5)

    # Back light
    bpy.ops.object.light_add(type='SUN', location=(0, 3, -5))
    back_light = bpy.context.active_object
    back_light.name = 'BackLight'
    back_light.data.energy = 1.0
    back_light.rotation_euler = (2.5, 0, 0)


def create_block(name, x, z, size, color_rgb, rotation=0):
    """Create a single colored cube block."""
    bpy.ops.mesh.primitive_cube_add(size=size * 2, location=(x, 0, z))
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = (0, 0, rotation)

    mat = bpy.data.materials.new(f'Mat_{name}')
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Metallic'].default_value = 0.05
    obj.data.materials.append(mat)
    return obj


def render_state(matrix, output_prefix, width=224, height=224, samples=32,
                 seed=0, n_views=1):
    """Render a single Blocksworld state from multiple camera views.

    Output files: {output_prefix}_v0.png, {output_prefix}_v1.png, ...
    """
    random.seed(seed)
    np.random.seed(seed)

    clear_scene()
    setup_base_scene(width, height, samples)

    n_cols = len(matrix)
    max_h = len(matrix[0]) if n_cols > 0 else 1
    unit = 0.7

    # Fixed column positions
    total_span = 5.6
    col_xs = {ci: -total_span / 2 + ci * (total_span / max(n_cols - 1, 1))
              for ci in range(n_cols)}

    # Column markers
    for ci in range(n_cols):
        cx = col_xs[ci]
        bpy.ops.mesh.primitive_cube_add(size=1, location=(cx, 0, -0.02))
        marker = bpy.context.active_object
        marker.name = f'Marker_C{ci+1}'
        marker.scale = (0.55, 0.4, 0.02)
        mat = bpy.data.materials.new(f'MarkerMat_C{ci+1}')
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes['Principled BSDF']
        mc = MARKER_COLORS[ci % len(MARKER_COLORS)]
        bsdf.inputs['Base Color'].default_value = (*mc, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.9
        marker.data.materials.append(mat)

    # Place blocks
    for ci in range(n_cols):
        cx = col_xs[ci]
        for bi in range(max_h):
            bid = int(matrix[ci][bi])
            if bid == 0:
                continue
            color = COLOR_MAP.get(bid, (0.5, 0.5, 0.5))
            z = unit + bi * unit * 2
            rot = random.uniform(-0.05, 0.05)
            jitter_x = random.gauss(0, 0.02)
            create_block(f'Block_{ci}_{bi}', cx + jitter_x, z, unit, color, rot)

    # Create camera once, move for each view
    bpy.ops.object.camera_add(location=(0, -4.5, 5))
    camera = bpy.context.active_object
    camera.name = 'Camera'
    bpy.context.scene.camera = camera

    rendered = []
    for vi in range(n_views):
        view = CAMERA_VIEWS[vi % len(CAMERA_VIEWS)]
        cam_loc = Vector(view["cam_loc"])
        target = Vector(view["target"])
        camera.location = cam_loc
        direction = target - cam_loc
        quat = direction.to_track_quat('-Z', 'Y')
        camera.rotation_euler = quat.to_euler()
        camera.data.lens = view["lens"]

        out_path = f"{output_prefix}_v{vi}.png"
        bpy.context.scene.render.filepath = out_path
        bpy.ops.render.render(write_still=True)
        rendered.append(os.path.exists(out_path))

    return all(rendered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("states_pkl", help="Pickle file with list of numpy matrices")
    parser.add_argument("output_dir", help="Output directory for rendered images")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--views", type=int, default=1, help="Number of camera views per state")
    cli_args = parser.parse_args()

    with open(cli_args.states_pkl, "rb") as f:
        states = pickle.load(f)

    os.makedirs(cli_args.output_dir, exist_ok=True)
    n = len(states)
    n_views = cli_args.views
    rendered = 0
    failed = 0

    for i in range(n):
        prefix = os.path.join(cli_args.output_dir, f"state_{i:05d}")
        all_exist = all(
            os.path.exists(f"{prefix}_v{vi}.png") for vi in range(n_views)
        )
        if all_exist:
            rendered += 1
            continue
        try:
            ok = render_state(states[i], prefix, cli_args.width, cli_args.height,
                              cli_args.samples, seed=i, n_views=n_views)
            if ok:
                rendered += 1
            else:
                failed += 1
                print(f"FAIL {i}: render returned False")
        except Exception as e:
            failed += 1
            print(f"FAIL {i}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"PROGRESS {i + 1}/{n} rendered={rendered} failed={failed}")

    print(f"DONE rendered={rendered} failed={failed} total={n}")

    manifest = {"total": n, "rendered": rendered, "failed": failed, "n_views": n_views}
    with open(os.path.join(cli_args.output_dir, "_manifest.json"), "w") as f:
        json.dump(manifest, f)


if __name__ == "__main__":
    main()
