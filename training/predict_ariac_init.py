#!/usr/bin/env python3
"""Predict ARIAC PDDL ``(:init ...)`` from one image.

The trained ARIAC placement model predicts a stable placement assignment
``place(part) in location | other_part`` and the constrained decoder derives
``part_at/on/clear/robot_at/handempty`` init atoms.

Object presence is not learned in the current model.  Pass active parts with
``--parts`` or provide a PDDL file whose ``(:objects ...)`` section lists them.
If neither is provided, the script looks for a same-stem PDDL under
``data/ariac/pddl_y_valid``; otherwise it falls back to all parts from the
checkpoint.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paq.ariac_support import AriacPlacementSketch
from paq.model import PaQModel
from training.run_ariac_structured import (
    DINOv3ViTHPlus,
    append_spatial_coords,
    build_domain_info,
    build_feature_projector,
    parse_typed_objects,
)


DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "ariac_init_dinov3raw_d256_nodup_all154_20260601"
    / "k_154"
    / "placement"
    / "model.pt"
)
DEFAULT_DINOV3_WEIGHTS = ROOT / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"


def _extract_objects_section(pddl_path: Path) -> tuple[list[str], list[str]]:
    text = pddl_path.read_text()
    m = re.search(r"\(:objects(?P<body>.*?)\)\s*\(:init", text, flags=re.S | re.I)
    if not m:
        raise ValueError(f"Cannot find (:objects ...) section in {pddl_path}")
    return parse_typed_objects(m.group("body"))


def _parse_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def resolve_active_objects(args, checkpoint_parts, checkpoint_locations):
    parts = _parse_csv(args.parts)
    locations = _parse_csv(args.locations)

    pddl_path = args.objects_pddl
    if pddl_path is None:
        candidate = ROOT / "data" / "ariac" / "pddl_y_valid" / f"{args.image.stem}.pddl"
        if candidate.exists():
            pddl_path = candidate

    if parts is None and pddl_path is not None:
        parts, pddl_locations = _extract_objects_section(pddl_path)
        if locations is None:
            locations = pddl_locations

    if parts is None:
        parts = list(checkpoint_parts)
        print(
            "warning: no active object list was provided; using all checkpoint parts",
            file=sys.stderr,
        )
    if locations is None:
        locations = list(checkpoint_locations)

    unknown_parts = sorted(set(parts) - set(checkpoint_parts))
    unknown_locations = sorted(set(locations) - set(checkpoint_locations))
    if unknown_parts:
        raise ValueError(f"Unknown parts not in checkpoint: {unknown_parts}")
    if unknown_locations:
        raise ValueError(f"Unknown locations not in checkpoint: {unknown_locations}")
    return parts, locations, pddl_path


def extract_dinov3_feature(
    image_path: Path,
    weights_path: Path,
    d_slot: int,
    device: str,
    feature_seed: int,
    raw_patch_tokens: bool,
    dinov3_scales: list[int],
    dinov3_last_n_layers: int,
    dinov3_layer_fusion: str,
    dinov3_add_coords: bool,
):
    from torchvision import transforms

    if raw_patch_tokens:
        encoder = DINOv3ViTHPlus(
            weights_path,
            d_out=None,
            image_size=max(dinov3_scales),
            last_n_layers=dinov3_last_n_layers,
            layer_fusion=dinov3_layer_fusion,
        ).to(device).eval()
    else:
        # Older projected-feature checkpoints cached DINO after constructing a
        # random projection with the experiment seed.  Reconstruct it exactly.
        torch.manual_seed(feature_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(feature_seed)
        encoder = DINOv3ViTHPlus(
            weights_path,
            d_out=d_slot,
            image_size=max(dinov3_scales),
            last_n_layers=dinov3_last_n_layers,
            layer_fusion=dinov3_layer_fusion,
        ).to(device).eval()
    features_by_scale = []
    with torch.no_grad():
        for scale in dinov3_scales:
            transform = transforms.Compose([
                transforms.Resize((scale, scale)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            scale_features = encoder(image).cpu()
            if dinov3_add_coords:
                scale_features = append_spatial_coords(
                    scale_features,
                    scale=scale,
                    max_scale=max(dinov3_scales),
                )
            features_by_scale.append(scale_features)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(features_by_scale, dim=1)


def order_init_atoms(atoms: set[str], active_parts: list[str], locations: list[str]) -> list[str]:
    ordered = []
    for atom in ["(robot_at table)", "(handempty)"]:
        if atom in atoms:
            ordered.append(atom)

    for part in active_parts:
        atom = f"(clear {part})"
        if atom in atoms:
            ordered.append(atom)

    for part in active_parts:
        for other in active_parts:
            if part == other:
                continue
            atom = f"(on {part} {other})"
            if atom in atoms:
                ordered.append(atom)

    for part in active_parts:
        for loc in locations:
            atom = f"(part_at {part} {loc})"
            if atom in atoms:
                ordered.append(atom)

    remaining = sorted(atoms - set(ordered))
    return ordered + remaining


def format_init(atoms: list[str]) -> str:
    lines = ["(:init"]
    lines.extend(f"    {atom}" for atom in atoms)
    lines.append(")")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Input ARIAC image path")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dinov3-weights", type=Path, default=DEFAULT_DINOV3_WEIGHTS)
    parser.add_argument("--objects-pddl", type=Path, default=None,
                        help="Optional PDDL file; only (:objects ...) is read.")
    parser.add_argument("--parts", type=str, default=None,
                        help="Comma-separated active parts, e.g. blue_pump,red_pump")
    parser.add_argument("--locations", type=str, default=None,
                        help="Comma-separated active locations. Defaults to checkpoint locations.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-seed", type=int, default=42,
                        help="Seed used when the DINO projection was created during feature extraction.")
    parser.add_argument("--show-assignment", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("method") != "placement":
        raise ValueError(f"Expected a placement checkpoint, got method={ckpt.get('method')}")
    parts = list(ckpt["parts"])
    locations = list(ckpt["locations"])
    slot_init = ckpt["slot_init"]
    d_slot = int(slot_init.shape[1])
    metadata = ckpt.get("metadata", {})
    feature_source = ckpt.get("feature_source", metadata.get("feature_source", "dinov3"))
    input_feature_dim = int(ckpt.get("input_feature_dim", metadata.get("input_feature_dim", d_slot)))
    raw_patch_tokens = feature_source == "dinov3_raw"
    object_extractor_type = metadata.get("object_extractor_type", "slot_attention")
    object_query_relation_layers = int(metadata.get("object_query_relation_layers", 0))
    dense_global_bias = bool(metadata.get("dense_global_bias", False))
    support_head_type = metadata.get("support_head_type", "legacy")
    support_temperature = float(metadata.get("support_temperature", 1.0))
    support_geometry_type = metadata.get("support_geometry_type", "none")
    support_hidden_dim = metadata.get("support_hidden_dim", None)
    feature_projector = metadata.get("feature_projector", "linear")
    dinov3_base_dim = int(metadata.get("dinov3_base_dim", 1280))
    dinov3_scales = metadata.get("dinov3_scales", [224])
    dinov3_last_n_layers = int(metadata.get("dinov3_last_n_layers", 1))
    dinov3_layer_fusion = metadata.get("dinov3_layer_fusion", "last")
    dinov3_add_coords = bool(metadata.get("dinov3_add_coords", False))
    dinov3_peft = metadata.get("dinov3_peft", "none")
    dinov3_lora_rank = int(metadata.get("dinov3_lora_rank", 0))
    dinov3_lora_alpha = metadata.get("dinov3_lora_alpha", None)
    dinov3_lora_dropout = float(metadata.get("dinov3_lora_dropout", 0.0))
    dinov3_lora_last_blocks = int(metadata.get("dinov3_lora_last_blocks", 2))
    dinov3_lora_targets = metadata.get("dinov3_lora_targets", "qkv")
    state_dict_is_partial = bool(ckpt.get("state_dict_is_partial", False))
    online_dinov3 = feature_source == "dinov3_online"
    dinov3_scales = [int(x) for x in dinov3_scales]

    active_parts, active_locations, object_source = resolve_active_objects(
        args, parts, locations
    )
    domain_info = build_domain_info(parts, locations)
    sketch = AriacPlacementSketch.from_domain_info(domain_info)
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    visual_encoder = None
    if online_dinov3:
        visual_encoder = DINOv3ViTHPlus(
            args.dinov3_weights,
            d_out=d_slot,
            image_size=dinov3_scales[0],
            last_n_layers=dinov3_last_n_layers,
            layer_fusion=dinov3_layer_fusion,
            peft=dinov3_peft,
            lora_rank=dinov3_lora_rank,
            lora_alpha=dinov3_lora_alpha,
            lora_dropout=dinov3_lora_dropout,
            lora_last_blocks=dinov3_lora_last_blocks,
            lora_targets=dinov3_lora_targets,
        )

    model = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=len(domain_info.objects),
        d_slot=d_slot,
        n_slot_iters=3,
        use_real_encoder=online_dinov3,
        visual_encoder=visual_encoder,
        predict_slot_types=True,
        object_extractor_type=object_extractor_type,
        object_query_relation_layers=object_query_relation_layers,
        dense_global_bias=dense_global_bias,
        use_support_head=True,
        support_block_type="part",
        support_column_type="location",
        support_head_type=support_head_type,
        support_temperature=support_temperature,
        support_geometry_type=support_geometry_type,
        support_hidden_dim=support_hidden_dim,
        scoring_head_type="film",
    )
    if not online_dinov3 and input_feature_dim != d_slot:
        proj_args = SimpleNamespace(
            feature_projector=feature_projector,
            dinov3_last_n_layers=dinov3_last_n_layers,
            dinov3_base_dim=dinov3_base_dim,
        )
        model.feat_proj = build_feature_projector(input_feature_dim, d_slot, proj_args)
    model.load_state_dict(ckpt["model_state_dict"], strict=not state_dict_is_partial)
    model.to(args.device).eval()

    if online_dinov3:
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((dinov3_scales[0], dinov3_scales[0])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        features = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(args.device)
    else:
        features = extract_dinov3_feature(
            args.image,
            args.dinov3_weights,
            d_slot=d_slot,
            device=args.device,
            feature_seed=args.feature_seed,
            raw_patch_tokens=raw_patch_tokens,
            dinov3_scales=dinov3_scales,
            dinov3_last_n_layers=dinov3_last_n_layers,
            dinov3_layer_fusion=dinov3_layer_fusion,
            dinov3_add_coords=dinov3_add_coords,
        ).to(args.device)

    with torch.no_grad():
        out = model(
            features,
            object_type_ids=type_ids.to(args.device).unsqueeze(0),
            slot_init=slot_init.to(args.device).unsqueeze(0),
        )
    decoded = sketch.decode(out["support_scores"][0].cpu(), active_parts)
    atoms = order_init_atoms(decoded.atoms, active_parts, active_locations)

    if object_source is not None:
        print(f"; objects: {object_source}")
    if args.show_assignment:
        print("; placement:")
        for part in active_parts:
            print(f";   place({part}) = {decoded.assignment[part]}")
    print(format_init(atoms))


if __name__ == "__main__":
    main()
