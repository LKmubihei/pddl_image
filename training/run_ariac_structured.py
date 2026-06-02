#!/usr/bin/env python3
"""Few-shot ARIAC init-state grounding with placement-factor decoding.

This script uses only the truth atoms inside each ``(:init ...)`` section as
state labels.  ``(:goal ...)`` text and the VL conversation data are ignored.

Two PaQ variants are compared:
  - ``atom``: directly predicts canonical init atoms with BCE.
  - ``placement``: predicts ``place(part)`` scores and decodes them through an
    ARIAC constrained placement sketch, then derives init atoms.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from paq.ariac_support import AriacPlacementSketch
from paq.domain_compiler import DomainInfo, GroundAtom, ObjectInfo, PredicateSchema
from paq.model import PaQModel


DEFAULT_LOCATIONS = [
    "table",
    "pump_placement",
    "regulator_placement",
    "battery_placement",
    "buffer_placement",
]


@dataclass
class AriacSample:
    sample_id: str
    image_path: Path
    objects_text: str
    init_atoms: set[str]
    active_parts: list[str]
    active_locations: list[str]
    assignment: dict[str, str] | None
    valid_factor_state: bool
    invalid_reasons: list[str]


def _section_between(text: str, start_name: str, next_name: str) -> str:
    m = re.search(
        rf"\(:{re.escape(start_name)}(?P<body>.*?)\)\s*\(:{re.escape(next_name)}",
        text,
        flags=re.S | re.I,
    )
    if not m:
        raise ValueError(f"Cannot find :{start_name} before :{next_name}")
    return m.group("body")


def parse_typed_objects(objects_text: str) -> tuple[list[str], list[str]]:
    parts: list[str] = []
    locations: list[str] = []
    for names, typ in re.findall(r"([^()\-]+?)\s*-\s*(\w+)", objects_text):
        values = names.split()
        if typ == "part":
            parts.extend(values)
        elif typ == "location":
            locations.extend(values)
    return parts, locations


def parse_init_atoms(init_text: str) -> set[str]:
    atoms: set[str] = set()
    for raw in re.findall(r"\((?!not\b)([^()]+)\)", init_text):
        toks = raw.split()
        if toks:
            atoms.add("(" + " ".join(toks) + ")")
    return atoms


def parse_problem(pddl_path: Path, image_path: Path) -> AriacSample:
    text = pddl_path.read_text()
    objects_text = _section_between(text, "objects", "init")
    init_text = _section_between(text, "init", "goal")
    parts, locations = parse_typed_objects(objects_text)
    atoms = parse_init_atoms(init_text)
    assignment: dict[str, str] = {}
    reasons: list[str] = []

    for part in parts:
        supports: list[str] = []
        for atom in atoms:
            toks = atom.strip("()").split()
            if len(toks) == 3 and toks[0] == "on" and toks[1] == part:
                supports.append(toks[2])
            elif len(toks) == 3 and toks[0] == "part_at" and toks[1] == part:
                supports.append(toks[2])
        if len(supports) != 1:
            reasons.append(f"{part}: expected one support, got {supports}")
        else:
            assignment[part] = supports[0]

    active = set(parts)
    locs = set(locations)
    support_count = {p: 0 for p in parts}
    for part, support in assignment.items():
        if support == part or (support not in active and support not in locs):
            reasons.append(f"{part}: illegal support {support}")
        if support in support_count:
            support_count[support] += 1
            if support_count[support] > 1:
                reasons.append(f"{support}: supports multiple direct top parts")

    for part in parts:
        seen: set[str] = set()
        cur = part
        while cur in assignment:
            if cur in seen:
                reasons.append(f"{part}: placement cycle")
                break
            seen.add(cur)
            cur = assignment[cur]
        if cur not in locs and part in assignment:
            reasons.append(f"{part}: chain ends at non-location {cur}")

    clear_atoms = {
        atom.strip("()").split()[1]
        for atom in atoms
        if atom.startswith("(clear ")
    }
    for part in parts:
        expected_clear = not any(support == part for support in assignment.values())
        if (part in clear_atoms) != expected_clear:
            reasons.append(
                f"{part}: clear mismatch label={part in clear_atoms} "
                f"expected={expected_clear}"
            )

    return AriacSample(
        sample_id=pddl_path.stem,
        image_path=image_path,
        objects_text=objects_text,
        init_atoms=atoms,
        active_parts=parts,
        active_locations=locations,
        assignment=assignment if not reasons else None,
        valid_factor_state=not reasons,
        invalid_reasons=reasons,
    )


def load_samples(data_dir: Path, strict_valid: bool = True) -> list[AriacSample]:
    pddl_dir = data_dir / "pddl_y_valid"
    image_dir = data_dir / "real_pictures"
    image_by_stem = {p.stem: p for p in image_dir.glob("*") if p.is_file()}
    samples: list[AriacSample] = []
    for pddl_path in sorted(pddl_dir.glob("*.pddl")):
        image_path = image_by_stem.get(pddl_path.stem)
        if image_path is None:
            continue
        sample = parse_problem(pddl_path, image_path)
        if strict_valid and not sample.valid_factor_state:
            continue
        samples.append(sample)
    return samples


def build_domain_info(parts: list[str], locations: list[str]) -> DomainInfo:
    types = ["part", "location"]
    type_to_idx = {t: i for i, t in enumerate(types)}
    objects = [
        ObjectInfo(name=p, type_name="part", type_idx=type_to_idx["part"])
        for p in parts
    ] + [
        ObjectInfo(name=l, type_name="location", type_idx=type_to_idx["location"])
        for l in locations
    ]
    obj_name_to_idx = {o.name.lower(): i for i, o in enumerate(objects)}
    schemas = [
        PredicateSchema("clear", 1, ["part"], ["x"], ["precondition"], "part {0} is clear"),
        PredicateSchema("handempty", 0, [], [], ["precondition", "effect"], "robot hand is empty"),
        PredicateSchema("on", 2, ["part", "part"], ["top", "bottom"], ["precondition", "effect"], "part {0} is on part {1}"),
        PredicateSchema("part_at", 2, ["part", "location"], ["p", "l"], ["precondition", "effect"], "part {0} is at location {1}"),
        PredicateSchema("robot_at", 1, ["location"], ["l"], ["precondition", "effect"], "robot is at location {0}"),
    ]
    canonical: list[GroundAtom] = []
    for si, schema in enumerate(schemas):
        if schema.arity == 0:
            canonical.append(GroundAtom(schema.name, (), f"({schema.name})", si))
        elif schema.name == "clear":
            for p in parts:
                canonical.append(GroundAtom("clear", (p,), f"(clear {p})", si))
        elif schema.name == "on":
            for p in parts:
                for q in parts:
                    if p != q:
                        canonical.append(GroundAtom("on", (p, q), f"(on {p} {q})", si))
        elif schema.name == "part_at":
            for p in parts:
                for l in locations:
                    canonical.append(
                        GroundAtom("part_at", (p, l), f"(part_at {p} {l})", si)
                    )
        elif schema.name == "robot_at":
            for l in locations:
                canonical.append(GroundAtom("robot_at", (l,), f"(robot_at {l})", si))

    return DomainInfo(
        domain_name="ariac_init",
        types=types,
        type_to_idx=type_to_idx,
        objects=objects,
        obj_name_to_idx=obj_name_to_idx,
        predicate_schemas=schemas,
        canonical_atoms=canonical,
        action_semantics=[],
        obj_type_ids=[o.type_idx for o in objects],
        static_predicates=set(),
        n_canonical=len(canonical),
    )


def sample_labels(
    sample: AriacSample,
    sketch: AriacPlacementSketch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.zeros(len(sketch.canonical_atom_strings), dtype=torch.float32)
    for atom in sample.init_atoms:
        idx = sketch.atom_to_idx.get(atom)
        if idx is not None:
            labels[idx] = 1.0

    active_part_mask = torch.tensor(
        [1.0 if p in set(sample.active_parts) else 0.0 for p in sketch.parts],
        dtype=torch.float32,
    )
    active_atom_mask = sketch.active_atom_mask(sample.active_parts)
    return labels, active_part_mask, active_atom_mask


def duplicate_base_name(name: str) -> str:
    """Return the exchangeability class for duplicated ARIAC object instances."""
    return re.sub(r"_\d+$", "", name)


def duplicate_name_maps(active_parts: list[str]) -> list[dict[str, str]]:
    groups: dict[str, list[str]] = {}
    for part in active_parts:
        groups.setdefault(duplicate_base_name(part), []).append(part)

    grouped_perms: list[list[dict[str, str]]] = []
    for names in groups.values():
        ordered = list(names)
        if len(ordered) <= 1:
            grouped_perms.append([{ordered[0]: ordered[0]}])
            continue
        grouped_perms.append([
            dict(zip(ordered, permuted))
            for permuted in permutations(ordered)
        ])

    maps: list[dict[str, str]] = []
    for combo in product(*grouped_perms):
        merged: dict[str, str] = {}
        for item in combo:
            merged.update(item)
        maps.append(merged)
    return maps


def has_duplicate_active_parts(sample: AriacSample) -> bool:
    bases = [duplicate_base_name(part) for part in sample.active_parts]
    return len(bases) != len(set(bases))


def atom_dst_indices_for_name_map(
    canonical_atoms: list[str],
    atom_to_idx: dict[str, int],
    name_map: dict[str, str],
) -> torch.Tensor:
    dst = []
    for atom in canonical_atoms:
        toks = atom.strip("()").split()
        mapped = [name_map.get(tok, tok) for tok in toks]
        mapped_atom = "(" + " ".join(mapped) + ")"
        dst.append(atom_to_idx.get(mapped_atom, atom_to_idx[atom]))
    return torch.tensor(dst, dtype=torch.long)


def build_duplicate_label_variants(
    samples: list[AriacSample],
    labels: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    duplicate_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Build padded target variants under exchangeable duplicate renamings."""
    all_label_variants: list[list[torch.Tensor]] = []
    all_support_variants: list[list[torch.Tensor]] = []
    variant_counts: list[int] = []

    for i, sample in enumerate(samples):
        maps = (
            duplicate_name_maps(sample.active_parts)
            if duplicate_mode == "exchangeable"
            else [{p: p for p in sample.active_parts}]
        )
        label_vars: list[torch.Tensor] = []
        support_vars: list[torch.Tensor] = []
        for name_map in maps:
            dst_idx = atom_dst_indices_for_name_map(
                sketch.canonical_atom_strings,
                sketch.atom_to_idx,
                name_map,
            )
            label_var = torch.zeros_like(labels[i])
            label_var[dst_idx] = labels[i]
            label_vars.append(label_var)
            support_vars.append(
                sketch.labels_to_support_targets(label_var, active_part_masks[i])
            )
        all_label_variants.append(label_vars)
        all_support_variants.append(support_vars)
        variant_counts.append(len(label_vars))

    max_variants = max(variant_counts)
    label_variants = torch.zeros(
        labels.shape[0],
        max_variants,
        labels.shape[1],
        dtype=labels.dtype,
    )
    support_variants = torch.full(
        (labels.shape[0], max_variants, sketch.n_parts),
        -1,
        dtype=torch.long,
    )
    variant_mask = torch.zeros(labels.shape[0], max_variants, dtype=torch.bool)
    for i, (label_vars, support_vars) in enumerate(zip(all_label_variants, all_support_variants)):
        for vi, (label_var, support_var) in enumerate(zip(label_vars, support_vars)):
            label_variants[i, vi] = label_var
            support_variants[i, vi] = support_var
            variant_mask[i, vi] = True

    metadata = {
        "duplicate_mode": duplicate_mode,
        "max_duplicate_target_variants": int(max_variants),
        "samples_with_duplicate_variants": int(sum(v > 1 for v in variant_counts)),
    }
    return label_variants, support_variants, variant_mask, metadata


def raw_grid_features(
    image_path: Path,
    d_slot: int,
    grid_h: int = 8,
    grid_w: int = 20,
    patch_size: int = 16,
) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((grid_w * patch_size, grid_h * patch_size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    feats = []
    for yi in range(grid_h):
        for xi in range(grid_w):
            patch = arr[
                yi * patch_size:(yi + 1) * patch_size,
                xi * patch_size:(xi + 1) * patch_size,
                :,
            ]
            mean = patch.mean(axis=(0, 1))
            std = patch.std(axis=(0, 1))
            mn = patch.min(axis=(0, 1))
            mx = patch.max(axis=(0, 1))
            bright = float(mean.mean())
            sat = float(mx.max() - mn.min())
            x = (xi + 0.5) / grid_w
            y = (yi + 0.5) / grid_h
            base = np.array(
                [
                    *mean,
                    *std,
                    *mn,
                    *mx,
                    bright,
                    sat,
                    x,
                    y,
                    x * x,
                    y * y,
                    x * y,
                    math.sin(math.pi * x),
                    math.cos(math.pi * x),
                    math.sin(math.pi * y),
                    math.cos(math.pi * y),
                    math.sin(2 * math.pi * x),
                    math.cos(2 * math.pi * x),
                    math.sin(2 * math.pi * y),
                    math.cos(2 * math.pi * y),
                ],
                dtype=np.float32,
            )
            if base.shape[0] < d_slot:
                base = np.pad(base, (0, d_slot - base.shape[0]))
            elif base.shape[0] > d_slot:
                base = base[:d_slot]
            feats.append(base)
    return torch.tensor(np.stack(feats), dtype=torch.float32)


class LoRALinear(nn.Module):
    """LoRA wrapper for a frozen Linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scaling = float(alpha if alpha is not None else rank) / float(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.dropout(self.lora_a(x))) * self.scaling


def inject_dinov3_lora(
    backbone: nn.Module,
    rank: int,
    alpha: float | None,
    dropout: float,
    last_n_blocks: int,
    target_modules: str,
) -> int:
    """Inject LoRA into selected attention Linear layers of the last blocks."""
    blocks = list(getattr(backbone, "blocks", []))
    if not blocks:
        raise ValueError("DINOv3 backbone has no blocks attribute for LoRA injection")
    selected = blocks[-max(1, last_n_blocks):]
    targets = {x.strip() for x in target_modules.split(",") if x.strip()}
    if not targets:
        raise ValueError("At least one LoRA target module is required")

    injected = 0
    for block in selected:
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        if "qkv" in targets and isinstance(getattr(attn, "qkv", None), nn.Linear):
            attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
            injected += 1
        if "proj" in targets and isinstance(getattr(attn, "proj", None), nn.Linear):
            attn.proj = LoRALinear(attn.proj, rank=rank, alpha=alpha, dropout=dropout)
            injected += 1
    if injected == 0:
        raise ValueError(f"No LoRA modules injected for targets={sorted(targets)}")
    return injected


def spatial_coord_features(
    n_tokens: int,
    scale: int,
    max_scale: int,
    patch_size: int = 16,
) -> torch.Tensor:
    grid = int(round(n_tokens ** 0.5))
    expected = (scale // patch_size) ** 2
    if grid * grid != n_tokens or expected != n_tokens:
        raise ValueError(
            f"Cannot build square patch coords for n_tokens={n_tokens}, scale={scale}"
        )
    coords = []
    scale_norm = float(scale) / float(max_scale)
    for yi in range(grid):
        for xi in range(grid):
            x = (xi + 0.5) / grid
            y = (yi + 0.5) / grid
            coords.append([
                x,
                y,
                x * x,
                y * y,
                x * y,
                math.sin(math.pi * x),
                math.cos(math.pi * x),
                math.sin(math.pi * y),
                math.cos(math.pi * y),
                scale_norm,
            ])
    return torch.tensor(coords, dtype=torch.float32)


def append_spatial_coords(
    features: torch.Tensor,
    scale: int,
    max_scale: int,
) -> torch.Tensor:
    coords = spatial_coord_features(
        n_tokens=features.shape[1],
        scale=scale,
        max_scale=max_scale,
    )
    coords = coords.unsqueeze(0).expand(features.shape[0], -1, -1)
    return torch.cat([features, coords.to(features.dtype)], dim=-1)


class DenseLayerAttentionProjector(nn.Module):
    """Trainable layer-wise attention pooling for concat DINO dense tokens."""

    def __init__(
        self,
        input_dim: int,
        d_out: int,
        n_layers: int,
        base_dim: int = 1280,
    ):
        super().__init__()
        if n_layers <= 1:
            raise ValueError("DenseLayerAttentionProjector needs n_layers > 1")
        core_dim = n_layers * base_dim
        if input_dim < core_dim:
            raise ValueError(
                f"input_dim={input_dim} is smaller than n_layers*base_dim={core_dim}"
            )
        self.n_layers = n_layers
        self.base_dim = base_dim
        self.coord_dim = input_dim - core_dim
        self.layer_proj = nn.Linear(base_dim, d_out)
        self.layer_attn = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Linear(d_out, 1),
        )
        self.coord_proj = nn.Linear(self.coord_dim, d_out) if self.coord_dim > 0 else None
        self.out_norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core_dim = self.n_layers * self.base_dim
        core = x[..., :core_dim].reshape(
            x.shape[0],
            x.shape[1],
            self.n_layers,
            self.base_dim,
        )
        h = self.layer_proj(core)
        weights = torch.softmax(self.layer_attn(h).squeeze(-1), dim=-1)
        out = (h * weights.unsqueeze(-1)).sum(dim=2)
        if self.coord_proj is not None:
            out = out + self.coord_proj(x[..., core_dim:])
        return self.out_norm(out)


def build_feature_projector(
    input_dim: int,
    d_slot: int,
    args,
) -> nn.Module:
    if args.feature_projector == "linear":
        return nn.Linear(input_dim, d_slot)
    if args.feature_projector == "layer_attention":
        return DenseLayerAttentionProjector(
            input_dim=input_dim,
            d_out=d_slot,
            n_layers=args.dinov3_last_n_layers,
            base_dim=args.dinov3_base_dim,
        )
    raise ValueError(f"Unknown feature_projector: {args.feature_projector}")


class DINOv3ViTHPlus(nn.Module):
    """Local DINOv3 ViT-H+/16 feature extractor.

    The default path is frozen and used for offline feature caching.  For the
    PEFT ablation, LoRA can be injected into the last attention blocks and used
    in the online image path.
    """

    def __init__(
        self,
        ckpt_path: Path,
        d_out: int | None = 256,
        image_size: int = 224,
        last_n_layers: int = 1,
        layer_fusion: str = "last",
        peft: str = "none",
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        lora_dropout: float = 0.0,
        lora_last_blocks: int = 2,
        lora_targets: str = "qkv",
    ):
        super().__init__()
        if last_n_layers < 1:
            raise ValueError("last_n_layers must be >= 1")
        if layer_fusion not in {"last", "mean", "concat"}:
            raise ValueError(f"Unknown DINO layer fusion: {layer_fusion}")
        if peft not in {"none", "lora"}:
            raise ValueError(f"Unknown DINO PEFT mode: {peft}")

        sys.path.insert(0, str(ROOT / "dinov3"))
        from dinov3.models.vision_transformer import DinoVisionTransformer

        self.embed_dim = 1280
        self.last_n_layers = last_n_layers
        self.layer_fusion = layer_fusion
        self.peft = peft
        self.backbone = DinoVisionTransformer(
            img_size=image_size,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=self.embed_dim,
            depth=32,
            num_heads=20,
            ffn_ratio=6.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1e-5,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(sd, strict=True)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.freeze_backbone_forward = True
        self.lora_modules = 0
        if peft == "lora":
            if lora_rank <= 0:
                raise ValueError("--dinov3-lora-rank must be > 0 for PEFT LoRA")
            self.lora_modules = inject_dinov3_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                last_n_blocks=lora_last_blocks,
                target_modules=lora_targets,
            )
            self.freeze_backbone_forward = False
            self.backbone.train()

        proj_in = self.embed_dim
        if last_n_layers > 1 and layer_fusion == "concat":
            proj_in *= last_n_layers
        self.proj = nn.Linear(proj_in, d_out) if d_out is not None else None

    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.last_n_layers == 1:
            out = self.backbone.forward_features(x)
            return out["x_norm_patchtokens"] if isinstance(out, dict) else out[:, 1:, :]

        layers = self.backbone.get_intermediate_layers(
            x,
            n=self.last_n_layers,
            reshape=False,
            norm=True,
        )
        if self.layer_fusion == "last":
            return layers[-1]
        if self.layer_fusion == "mean":
            return torch.stack(list(layers), dim=0).mean(dim=0)
        return torch.cat(list(layers), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone_forward:
            with torch.no_grad():
                patches = self._extract_patch_tokens(x)
        else:
            patches = self._extract_patch_tokens(x)
        patches = patches.float()
        return self.proj(patches) if self.proj is not None else patches


def dinov3_features(
    samples: list[AriacSample],
    d_slot: int | None,
    weights_path: Path,
    device: str,
    batch_size: int,
    image_scales: list[int],
    last_n_layers: int,
    layer_fusion: str,
    add_spatial_coords: bool = False,
) -> torch.Tensor:
    from torchvision import transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    encoder = DINOv3ViTHPlus(
        weights_path,
        d_out=d_slot,
        image_size=max(image_scales),
        last_n_layers=last_n_layers,
        layer_fusion=layer_fusion,
    ).to(device)
    encoder.eval()
    scale_features: list[torch.Tensor] = []
    for scale in image_scales:
        transform = transforms.Compose([
            transforms.Resize((scale, scale)),
            transforms.ToTensor(),
            normalize,
        ])
        tensors = [
            transform(Image.open(s.image_path).convert("RGB"))
            for s in samples
        ]
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[i:i + batch_size]).to(device)
                chunks.append(encoder(batch).cpu())
                print(
                    f"    DINO scale={scale} encoded "
                    f"{min(i + batch_size, len(tensors))}/{len(tensors)}"
                )
        scale_feature = torch.cat(chunks, dim=0)
        if add_spatial_coords:
            scale_feature = append_spatial_coords(
                scale_feature,
                scale=scale,
                max_scale=max(image_scales),
            )
        print(f"    scale={scale} feature shape={tuple(scale_feature.shape)}")
        scale_features.append(scale_feature)
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(scale_features, dim=1)


def dinov3_image_tensors(
    samples: list[AriacSample],
    image_size: int,
) -> torch.Tensor:
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return torch.stack([
        transform(Image.open(s.image_path).convert("RGB"))
        for s in samples
    ])


def load_or_extract_features(
    samples: list[AriacSample],
    cache_path: Path,
    d_slot: int,
    feature_source: str,
    dinov3_weights: Path,
    device: str,
    dinov3_batch_size: int,
    dinov3_scales: list[int],
    dinov3_last_n_layers: int,
    dinov3_layer_fusion: str,
    dinov3_add_coords: bool,
    rebuild: bool = False,
) -> torch.Tensor:
    meta = {
        "sample_ids": [s.sample_id for s in samples],
        "feature_source": feature_source,
        "d_slot": d_slot,
    }
    if feature_source == "raw_grid":
        meta.update({"grid_h": 8, "grid_w": 20})
    elif feature_source in {"dinov3", "dinov3_raw"}:
        meta.update({
            "dinov3_weights": str(dinov3_weights),
            "image_transform": f"resize_{dinov3_scales}_imagenet",
            "dinov3_scales": list(dinov3_scales),
            "dinov3_last_n_layers": dinov3_last_n_layers,
            "dinov3_layer_fusion": dinov3_layer_fusion,
            "dinov3_add_coords": dinov3_add_coords,
            "raw_patch_tokens": feature_source == "dinov3_raw",
        })
    elif feature_source == "dinov3_online":
        if len(dinov3_scales) != 1:
            raise ValueError("dinov3_online currently supports exactly one image scale")
        meta.update({
            "dinov3_weights": str(dinov3_weights),
            "image_transform": f"resize_{dinov3_scales[0]}x{dinov3_scales[0]}_imagenet",
            "dinov3_scales": list(dinov3_scales),
            "online_dinov3": True,
        })
    else:
        raise ValueError(f"Unknown feature source: {feature_source}")

    if cache_path.exists() and not rebuild:
        cached = torch.load(cache_path, map_location="cpu")
        cached_meta = cached.get("metadata")
        expected_meta = meta
        if feature_source == "dinov3_raw" and isinstance(cached_meta, dict):
            cached_meta = dict(cached_meta)
            expected_meta = dict(meta)
            cached_meta.pop("d_slot", None)
            expected_meta.pop("d_slot", None)
        if cached_meta == expected_meta:
            print(f"  Loaded ARIAC feature cache: {cache_path}")
            return cached["features"]
        print("  ARIAC feature cache metadata mismatch; rebuilding.")

    if feature_source == "raw_grid":
        print(f"  Extracting raw-grid features for {len(samples)} images...")
        features = torch.stack([raw_grid_features(s.image_path, d_slot) for s in samples])
    elif feature_source in {"dinov3", "dinov3_raw"}:
        print(
            f"  Extracting DINOv3 features for {len(samples)} images "
            f"scales={dinov3_scales} layers={dinov3_last_n_layers}:{dinov3_layer_fusion}..."
        )
        features = dinov3_features(
            samples=samples,
            d_slot=None if feature_source == "dinov3_raw" else d_slot,
            weights_path=dinov3_weights,
            device=device,
            batch_size=dinov3_batch_size,
            image_scales=dinov3_scales,
            last_n_layers=dinov3_last_n_layers,
            layer_fusion=dinov3_layer_fusion,
            add_spatial_coords=dinov3_add_coords,
        )
    elif feature_source == "dinov3_online":
        print(f"  Loading image tensors for online DINOv3 at scale={dinov3_scales[0]}...")
        features = dinov3_image_tensors(samples, image_size=dinov3_scales[0])
    else:
        raise ValueError(f"Unknown feature source: {feature_source}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": meta, "features": features}, cache_path)
    print(f"  Saved feature cache: {cache_path} shape={tuple(features.shape)}")
    return features


def object_slot_init(parts: list[str], locations: list[str], d_slot: int) -> torch.Tensor:
    colors = ["blue", "green", "red", "yellow"]
    kinds = ["pump", "battery", "regulator", "sensor"]
    loc_xy = {
        "table": (0.25, 0.55),
        "pump_placement": (0.63, 0.68),
        "regulator_placement": (0.66, 0.36),
        "battery_placement": (0.78, 0.17),
        "buffer_placement": (0.87, 0.45),
    }
    vectors = []
    for name in parts + locations:
        v = np.zeros(d_slot, dtype=np.float32)
        off = 0
        if name in locations:
            v[off + len(colors) + len(kinds)] = 1.0
            x, y = loc_xy.get(name, (0.5, 0.5))
            if off + len(colors) + len(kinds) + 3 < d_slot:
                v[off + len(colors) + len(kinds) + 1] = x
                v[off + len(colors) + len(kinds) + 2] = y
                v[off + len(colors) + len(kinds) + 3] = 1.0
        else:
            for i, c in enumerate(colors):
                if name.startswith(c + "_"):
                    v[off + i] = 1.0
            for j, k in enumerate(kinds):
                if f"_{k}" in name:
                    v[off + len(colors) + j] = 1.0
            idx = off + len(colors) + len(kinds)
            if idx < d_slot:
                v[idx] = 0.0
            if name.endswith("_1") and idx + 1 < d_slot:
                v[idx + 1] = 1.0
        vectors.append(v)
    return torch.tensor(np.stack(vectors), dtype=torch.float32)


def masked_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    active_labels = labels[mask > 0]
    n_pos = active_labels.sum()
    n_neg = active_labels.numel() - n_pos
    pos_weight = torch.clamp(n_neg / torch.clamp(n_pos, min=1.0), min=1.0, max=50.0)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(labels > 0.5, pos_weight.to(labels.device), 1.0)
    loss = loss * weights * mask
    return loss.sum() / torch.clamp(mask.sum(), min=1.0)


def duplicate_invariant_bce_with_logits(
    logits: torch.Tensor,
    label_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    """BCE loss minimized over equivalent duplicate-object target renamings."""
    if label_variants.dim() != 3:
        raise ValueError(
            "label_variants must have shape (B, V, A), "
            f"got {tuple(label_variants.shape)}"
        )
    base_labels = label_variants[:, 0, :]
    active_labels = base_labels[atom_mask > 0]
    n_pos = active_labels.sum()
    n_neg = active_labels.numel() - n_pos
    pos_weight = torch.clamp(n_neg / torch.clamp(n_pos, min=1.0), min=1.0, max=50.0)

    logits_v = logits.unsqueeze(1).expand_as(label_variants)
    atom_mask_v = atom_mask.unsqueeze(1).expand_as(label_variants)
    loss = F.binary_cross_entropy_with_logits(
        logits_v,
        label_variants,
        reduction="none",
    )
    weights = torch.where(label_variants > 0.5, pos_weight.to(logits.device), 1.0)
    denom = torch.clamp(atom_mask.sum(dim=1, keepdim=True), min=1.0)
    per_variant = (loss * weights * atom_mask_v).sum(dim=2) / denom
    per_variant = per_variant.masked_fill(~variant_mask, float("inf"))
    return per_variant.min(dim=1).values.mean()


def duplicate_invariant_support_ce(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
) -> torch.Tensor:
    """Placement CE minimized over equivalent duplicate-object target renamings."""
    if support_target_variants.dim() != 3:
        raise ValueError(
            "support_target_variants must have shape (B, V, P), "
            f"got {tuple(support_target_variants.shape)}"
        )
    log_probs = F.log_softmax(support_scores, dim=-1)
    bsz, n_variants, n_parts = support_target_variants.shape
    log_probs = log_probs.unsqueeze(1).expand(bsz, n_variants, n_parts, -1)
    valid_targets = support_target_variants >= 0
    safe_targets = support_target_variants.clamp_min(0)
    picked = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    denom = torch.clamp(valid_targets.sum(dim=2), min=1)
    per_variant = -(picked * valid_targets).sum(dim=2) / denom
    per_variant = per_variant.masked_fill(~variant_mask, float("inf"))
    return per_variant.min(dim=1).values.mean()


_LEGAL_TARGET_CACHE: dict[tuple[int, ...], torch.Tensor] = {}
_COUNTERFACTUAL_TARGET_CACHE: dict[tuple[tuple[int, ...], tuple[int, ...]], torch.Tensor] = {}


def _legal_assignment_targets(
    active_part_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    device: torch.device,
) -> torch.Tensor:
    """Enumerate legal placement assignments as target rows.

    Each row has shape ``(n_parts,)`` with candidate indices for active parts
    and ``-1`` for inactive parts.
    """
    active_flags = tuple(
        1 if flag > 0 else 0
        for flag in active_part_mask.detach().cpu().tolist()
    )
    cached = _LEGAL_TARGET_CACHE.get(active_flags)
    if cached is not None:
        return cached.to(device)

    active = [p for p, flag in zip(sketch.parts, active_flags) if flag > 0]
    if not active:
        targets = torch.full(
            (1, sketch.n_parts),
            -1,
            dtype=torch.long,
        )
        _LEGAL_TARGET_CACHE[active_flags] = targets
        return targets.to(device)

    choices: list[list[tuple[str, int]]] = []
    for p in active:
        row = []
        for ci, cand in enumerate(sketch.place_candidates[p]):
            if cand in sketch.locations or cand in active:
                row.append((cand, ci))
        choices.append(row)

    rows: list[list[int]] = []
    for combo in product(*choices):
        assignment = {p: cand for p, (cand, _) in zip(active, combo)}
        if not sketch.is_valid_assignment(assignment, active):
            continue
        target = [-1] * sketch.n_parts
        for p, (_, ci) in zip(active, combo):
            target[sketch.part_index(p)] = ci
        rows.append(target)

    if not rows:
        raise RuntimeError(f"No legal assignment targets for active={active}")
    targets = torch.tensor(rows, dtype=torch.long)
    _LEGAL_TARGET_CACHE[active_flags] = targets
    return targets.to(device)


def _assignment_scores_from_targets(
    support_scores: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Score target rows by summing selected per-part support scores."""
    if targets.dim() != 2:
        raise ValueError(f"targets must be 2D, got {tuple(targets.shape)}")
    valid = targets >= 0
    safe_targets = targets.clamp_min(0)
    picked = support_scores.unsqueeze(0).expand(targets.shape[0], -1, -1)
    picked = picked.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (picked * valid.to(picked.dtype)).sum(dim=1)


def structured_legal_state_nll(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> torch.Tensor:
    """Exact NLL over enumerated legal placement assignments.

    This optimizes the same legal state space used by the constrained decoder:

        -log p(A_gold | image)
        = logsumexp_{A in Legal} Score(A)
          - logsumexp_{A in GoldVariants} Score(A)
    """
    losses = []
    for bi in range(support_scores.shape[0]):
        legal_targets = _legal_assignment_targets(
            active_part_masks[bi],
            sketch,
            support_scores.device,
        )
        legal_scores = _assignment_scores_from_targets(
            support_scores[bi],
            legal_targets,
        )

        valid_variant_ids = variant_mask[bi].nonzero(as_tuple=True)[0].tolist()
        gold_scores = []
        for vi in valid_variant_ids:
            gold = support_target_variants[bi, vi]
            active = gold >= 0
            if active.sum().item() == 0:
                continue
            matches = (legal_targets[:, active] == gold[active]).all(dim=1)
            if matches.any():
                gold_scores.append(legal_scores[matches])
        if not gold_scores:
            continue
        gold_scores_t = torch.cat(gold_scores)
        losses.append(torch.logsumexp(legal_scores, dim=0) - torch.logsumexp(gold_scores_t, dim=0))
    if not losses:
        return support_scores.sum() * 0.0
    return torch.stack(losses).mean()


def _target_is_legal(
    target: torch.Tensor,
    active_part_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> bool:
    active = [
        p for p, flag in zip(sketch.parts, active_part_mask.detach().cpu().tolist())
        if flag > 0
    ]
    assignment = {}
    target_cpu = target.detach().cpu()
    for p in active:
        pi = sketch.part_index(p)
        ci = int(target_cpu[pi].item())
        if ci < 0:
            return False
        assignment[p] = sketch.place_candidates[p][ci]
    return sketch.is_valid_assignment(assignment, active)


def _counterfactual_targets_for_gold(
    gold: torch.Tensor,
    active_part_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> torch.Tensor:
    """Generate legal-but-wrong PDDL counterfactual placement targets."""
    active_flags = tuple(
        1 if flag > 0 else 0
        for flag in active_part_mask.detach().cpu().tolist()
    )
    gold_cpu = gold.detach().cpu()
    gold_key = tuple(int(x) for x in gold_cpu.tolist())
    cache_key = (active_flags, gold_key)
    cached = _COUNTERFACTUAL_TARGET_CACHE.get(cache_key)
    if cached is not None:
        return cached.to(gold.device)

    active = [p for p, flag in zip(sketch.parts, active_flags) if flag > 0]
    rows: list[torch.Tensor] = []
    seen: set[tuple[int, ...]] = set()

    def add(row: torch.Tensor):
        key = tuple(int(x) for x in row.detach().cpu().tolist())
        if key == gold_key or key in seen:
            return
        if _target_is_legal(row, active_part_mask, sketch):
            seen.add(key)
            rows.append(row.clone())

    for p in active:
        pi = sketch.part_index(p)
        gold_ci = int(gold_cpu[pi].item())
        if gold_ci < 0:
            continue
        candidates = sketch.place_candidates[p]
        gold_name = candidates[gold_ci]

        # The two dominant observed errors: stack/region collapsed to table.
        if "table" in candidates and gold_name != "table":
            row = gold_cpu.clone()
            row[pi] = candidates.index("table")
            add(row)

        # Region confusions: true region -> other region/table.
        if gold_name in sketch.locations:
            for loc in sketch.locations:
                if loc == gold_name or loc not in candidates:
                    continue
                row = gold_cpu.clone()
                row[pi] = candidates.index(loc)
                add(row)

        # Support swap: true stack -> wrong active support part.
        if gold_name in sketch.parts:
            for other in active:
                if other == p or other == gold_name or other not in candidates:
                    continue
                row = gold_cpu.clone()
                row[pi] = candidates.index(other)
                add(row)

    if not rows:
        targets = torch.empty(0, sketch.n_parts, dtype=torch.long)
    else:
        targets = torch.stack(rows).to(torch.long)
    _COUNTERFACTUAL_TARGET_CACHE[cache_key] = targets
    return targets.to(gold.device)


def counterfactual_margin_loss(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    margin: float,
) -> torch.Tensor:
    """Rank gold legal states above PDDL-generated legal counterfactuals."""
    losses = []
    for bi in range(support_scores.shape[0]):
        valid_variant_ids = variant_mask[bi].nonzero(as_tuple=True)[0].tolist()
        for vi in valid_variant_ids:
            gold = support_target_variants[bi, vi]
            if (gold >= 0).sum().item() == 0:
                continue
            negatives = _counterfactual_targets_for_gold(
                gold,
                active_part_masks[bi],
                sketch,
            )
            if negatives.numel() == 0:
                continue
            gold_score = _assignment_scores_from_targets(
                support_scores[bi],
                gold.unsqueeze(0),
            ).squeeze(0)
            neg_scores = _assignment_scores_from_targets(
                support_scores[bi],
                negatives,
            )
            hardest = neg_scores.max()
            losses.append(torch.relu(margin + hardest - gold_score))
    if not losses:
        return support_scores.sum() * 0.0
    return torch.stack(losses).mean()


def dynamic_hard_negative_support_loss(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    margin: float,
    region_table_weight: float,
    stack_table_weight: float,
    wrong_support_weight: float,
) -> torch.Tensor:
    """Per-part margin against the current highest-scoring wrong candidate."""
    losses: list[torch.Tensor] = []
    for bi in range(support_scores.shape[0]):
        active = {
            p for p, flag in zip(sketch.parts, active_part_masks[bi].detach().cpu().tolist())
            if flag > 0.5
        }
        if not active:
            continue
        valid_variant_losses: list[torch.Tensor] = []
        for vi in variant_mask[bi].nonzero(as_tuple=True)[0].tolist():
            gold = support_target_variants[bi, vi]
            part_losses: list[torch.Tensor] = []
            for pi, part in enumerate(sketch.parts):
                if part not in active:
                    continue
                gold_ci = int(gold[pi].item())
                if gold_ci < 0:
                    continue
                candidates = sketch.place_candidates[part]
                valid_cands = [
                    ci for ci, cand in enumerate(candidates)
                    if ci != gold_ci and (cand in sketch.locations or cand in active)
                ]
                if not valid_cands:
                    continue
                valid_idx = torch.tensor(
                    valid_cands,
                    dtype=torch.long,
                    device=support_scores.device,
                )
                neg_scores = support_scores[bi, pi].index_select(0, valid_idx)
                hard_pos = int(torch.argmax(neg_scores).item())
                hard_ci = valid_cands[hard_pos]
                gold_name = candidates[gold_ci]
                hard_name = candidates[hard_ci]
                weight = 1.0
                if gold_name in sketch.locations and hard_name == "table":
                    weight = region_table_weight
                elif gold_name in sketch.parts and hard_name == "table":
                    weight = stack_table_weight
                elif gold_name in sketch.parts and hard_name in sketch.parts:
                    weight = wrong_support_weight
                part_losses.append(
                    torch.relu(
                        margin
                        + support_scores[bi, pi, hard_ci]
                        - support_scores[bi, pi, gold_ci]
                    )
                    * weight
                )
            if part_losses:
                valid_variant_losses.append(torch.stack(part_losses).mean())
        if valid_variant_losses:
            losses.append(torch.stack(valid_variant_losses).min())
    if not losses:
        return support_scores.sum() * 0.0
    return torch.stack(losses).mean()


def support_occupancy_loss(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> torch.Tensor:
    """Auxiliary occupied/clear consistency over part supports."""
    on_candidate = sketch._on_candidate_indices.to(support_scores.device)
    losses: list[torch.Tensor] = []
    for bi in range(support_scores.shape[0]):
        active_indices = torch.nonzero(
            active_part_masks[bi] > 0.5,
            as_tuple=False,
        ).flatten().tolist()
        if len(active_indices) <= 1:
            continue
        valid_variant_ids = variant_mask[bi].nonzero(as_tuple=True)[0].tolist()
        variant_losses: list[torch.Tensor] = []
        for vi in valid_variant_ids:
            gold = support_target_variants[bi, vi]
            support_losses: list[torch.Tensor] = []
            for support_pi in active_indices:
                support_logits: list[torch.Tensor] = []
                occupied = False
                for top_pi in active_indices:
                    if top_pi == support_pi:
                        continue
                    cand_idx = int(on_candidate[top_pi, support_pi].item())
                    if cand_idx < 0:
                        continue
                    support_logits.append(support_scores[bi, top_pi, cand_idx])
                    if int(gold[top_pi].item()) == cand_idx:
                        occupied = True
                if not support_logits:
                    continue
                occupied_logit = torch.logsumexp(torch.stack(support_logits), dim=0)
                label = occupied_logit.new_tensor(1.0 if occupied else 0.0)
                support_losses.append(
                    F.binary_cross_entropy_with_logits(occupied_logit, label)
                )
            if support_losses:
                variant_losses.append(torch.stack(support_losses).mean())
        if variant_losses:
            losses.append(torch.stack(variant_losses).min())
    if not losses:
        return support_scores.sum() * 0.0
    return torch.stack(losses).mean()


def hybrid_decode_vector(
    sketch: AriacPlacementSketch,
    placement_scores: torch.Tensor,
    atom_logits: torch.Tensor,
    active_parts: list[str],
    atom_weight: float,
) -> list[int]:
    """Decode by reranking legal placements with atom-branch evidence."""
    if atom_weight <= 0:
        return sketch.decode(placement_scores, active_parts).atom_vector

    active = [p for p in sketch.parts if p in set(active_parts)]
    if not active:
        return sketch.decode(placement_scores, active_parts).atom_vector

    scores = placement_scores.detach().cpu()
    atom_logits_cpu = atom_logits.detach().cpu()
    active_mask = sketch.active_atom_mask(active).to(atom_logits_cpu.dtype)
    part_to_idx = {p: i for i, p in enumerate(sketch.parts)}
    choices: list[list[tuple[str, int, float]]] = []
    for p in active:
        pi = part_to_idx[p]
        row = []
        for ci, cand in enumerate(sketch.place_candidates[p]):
            if cand in sketch.locations or cand in active:
                row.append((cand, ci, float(scores[pi, ci].item())))
        choices.append(row)

    best_vec: list[int] | None = None
    best_score = -float("inf")
    for combo in product(*choices):
        assignment = {p: cand for p, (cand, _, _) in zip(active, combo)}
        if not sketch.is_valid_assignment(assignment, active):
            continue
        placement_score = sum(val for _, _, val in combo)
        atoms = sketch.derive_atoms(assignment, active)
        vec = torch.tensor(
            sketch.atoms_to_vector(atoms),
            dtype=atom_logits_cpu.dtype,
        )
        atom_logprob = (
            vec * F.logsigmoid(atom_logits_cpu)
            + (1.0 - vec) * F.logsigmoid(-atom_logits_cpu)
        )
        atom_score = (atom_logprob * active_mask).sum()
        norm = torch.clamp(active_mask.sum(), min=1.0)
        total = placement_score + atom_weight * float((atom_score / norm).item())
        if total > best_score:
            best_score = total
            best_vec = [int(x) for x in vec.tolist()]

    if best_vec is None:
        raise RuntimeError(f"No legal hybrid ARIAC placement assignment for {active}")
    return best_vec


def _legal_assignment_features(
    support_scores: torch.Tensor,
    targets: torch.Tensor,
    active_part_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> torch.Tensor:
    """Small assignment-level features for legal-state reranking."""
    active = [
        p for p, flag in zip(sketch.parts, active_part_mask.detach().cpu().tolist())
        if flag > 0
    ]
    active_set = set(active)
    n_active = max(len(active), 1)
    rows: list[list[float]] = []
    scores_cpu = support_scores.detach().cpu()
    targets_cpu = targets.detach().cpu()
    for row in targets_cpu:
        table_count = 0
        region_count = 0
        stack_count = 0
        margins = []
        non_table_margins = []
        chosen_scores = []
        for part in active:
            pi = sketch.part_index(part)
            ci = int(row[pi].item())
            if ci < 0:
                continue
            cand = sketch.place_candidates[part][ci]
            if cand == sketch.table_location:
                table_count += 1
            elif cand in sketch.locations:
                region_count += 1
            elif cand in active_set:
                stack_count += 1

            valid = [
                cj for cj, name in enumerate(sketch.place_candidates[part])
                if name in sketch.locations or name in active_set
            ]
            chosen = float(scores_cpu[pi, ci].item())
            chosen_scores.append(chosen)
            alt_scores = [
                float(scores_cpu[pi, cj].item())
                for cj in valid
                if cj != ci
            ]
            if alt_scores:
                margins.append(chosen - max(alt_scores))
            if cand != sketch.table_location and sketch.table_location in sketch.place_candidates[part]:
                table_ci = sketch.place_candidates[part].index(sketch.table_location)
                non_table_margins.append(chosen - float(scores_cpu[pi, table_ci].item()))

        margin_mean = float(np.mean(margins)) if margins else 0.0
        margin_min = float(np.min(margins)) if margins else 0.0
        non_table_margin_mean = float(np.mean(non_table_margins)) if non_table_margins else 0.0
        chosen_mean = float(np.mean(chosen_scores)) if chosen_scores else 0.0
        rows.append(
            [
                table_count / n_active,
                region_count / n_active,
                stack_count / n_active,
                margin_mean,
                margin_min,
                non_table_margin_mean,
                chosen_mean,
            ]
        )
    return torch.tensor(rows, dtype=support_scores.dtype, device=support_scores.device)


def _topk_legal_assignment_targets(
    support_scores: torch.Tensor,
    active_part_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    top_k: int,
) -> torch.Tensor:
    legal_targets = _legal_assignment_targets(
        active_part_mask,
        sketch,
        support_scores.device,
    )
    if top_k <= 0 or top_k >= legal_targets.shape[0]:
        return legal_targets
    base_scores = _assignment_scores_from_targets(support_scores, legal_targets)
    keep = torch.topk(base_scores, k=top_k).indices
    return legal_targets.index_select(0, keep)


def _reranker_train_targets(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    top_k: int,
) -> torch.Tensor:
    rows: list[tuple[int, ...]] = []
    for row in _topk_legal_assignment_targets(
        support_scores,
        active_part_mask,
        sketch,
        top_k,
    ).detach().cpu().tolist():
        rows.append(tuple(int(x) for x in row))
    for vi in variant_mask.nonzero(as_tuple=True)[0].tolist():
        gold = support_target_variants[vi]
        if (gold >= 0).sum().item() == 0:
            continue
        rows.append(tuple(int(x) for x in gold.detach().cpu().tolist()))
    unique = list(dict.fromkeys(rows))
    return torch.tensor(unique, dtype=torch.long, device=support_scores.device)


def fit_legal_state_reranker(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    top_k: int,
    steps: int,
    lr: float,
    l2: float,
) -> torch.Tensor:
    """Fit a tiny linear energy over legal PDDL placement assignments."""
    device = support_scores.device
    weights = torch.zeros(7, dtype=support_scores.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weights], lr=lr, weight_decay=0.0)
    for _ in range(steps):
        losses = []
        optimizer.zero_grad()
        for bi in range(support_scores.shape[0]):
            legal_targets = _reranker_train_targets(
                support_scores[bi],
                support_target_variants[bi],
                variant_mask[bi],
                active_part_masks[bi],
                sketch,
                top_k,
            )
            base_scores = _assignment_scores_from_targets(
                support_scores[bi],
                legal_targets,
            )
            features = _legal_assignment_features(
                support_scores[bi],
                legal_targets,
                active_part_masks[bi],
                sketch,
            )
            scores = base_scores + features.matmul(weights)

            gold_scores = []
            for vi in variant_mask[bi].nonzero(as_tuple=True)[0].tolist():
                gold = support_target_variants[bi, vi]
                active = gold >= 0
                if active.sum().item() == 0:
                    continue
                matches = (legal_targets[:, active] == gold[active]).all(dim=1)
                if matches.any():
                    gold_scores.append(scores[matches])
            if gold_scores:
                losses.append(
                    torch.logsumexp(scores, dim=0)
                    - torch.logsumexp(torch.cat(gold_scores), dim=0)
                )
        if not losses:
            break
        loss = torch.stack(losses).mean() + l2 * weights.square().sum()
        loss.backward()
        optimizer.step()
    return weights.detach().cpu()


def legal_rerank_decode_vector(
    sketch: AriacPlacementSketch,
    placement_scores: torch.Tensor,
    active_part_mask: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
) -> list[int]:
    """Decode by adding a learned low-dimensional legal-state energy."""
    device = placement_scores.device
    legal_targets = _topk_legal_assignment_targets(
        placement_scores,
        active_part_mask,
        sketch,
        top_k,
    )
    base_scores = _assignment_scores_from_targets(placement_scores, legal_targets)
    features = _legal_assignment_features(
        placement_scores,
        legal_targets,
        active_part_mask,
        sketch,
    )
    total = base_scores + features.matmul(weights.to(device=device, dtype=placement_scores.dtype))
    best = int(torch.argmax(total).item())
    active = [
        p for p, flag in zip(sketch.parts, active_part_mask.detach().cpu().tolist())
        if flag > 0
    ]
    assignment: dict[str, str] = {}
    row = legal_targets[best].detach().cpu()
    for part in active:
        pi = sketch.part_index(part)
        ci = int(row[pi].item())
        assignment[part] = sketch.place_candidates[part][ci]
    return sketch.assignment_to_vector(assignment, active)


def placement_ranking_metrics(
    support_scores: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> dict[str, float]:
    """Compute per-part support ranking diagnostics for placement scores."""
    scores = support_scores.detach().cpu()
    targets = support_target_variants.detach().cpu()
    variants = variant_mask.detach().cpu()
    active = active_part_masks.detach().cpu()

    n_parts = 0
    top1 = 0
    top3 = 0
    top10 = 0
    ranks: list[int] = []
    missed_stack = 0
    location_region = 0
    wrong_support_part = 0
    false_stack = 0

    for bi in range(scores.shape[0]):
        valid_variant_ids = variants[bi].nonzero(as_tuple=True)[0].tolist()
        if not valid_variant_ids:
            continue
        for pi, is_active in enumerate(active[bi].tolist()):
            if is_active <= 0:
                continue
            golds = {
                int(targets[bi, vi, pi].item())
                for vi in valid_variant_ids
                if int(targets[bi, vi, pi].item()) >= 0
            }
            if not golds:
                continue
            n_parts += 1
            order = torch.argsort(scores[bi, pi], descending=True).tolist()
            rank = min(order.index(gold) + 1 for gold in golds)
            ranks.append(rank)
            top1_cand = int(order[0])
            if rank <= 1:
                top1 += 1
            if rank <= 3:
                top3 += 1
            if rank <= 10:
                top10 += 1
            if top1_cand in golds:
                continue

            part = sketch.parts[pi]
            pred_name = sketch.place_candidates[part][top1_cand]
            gold_names = [sketch.place_candidates[part][gold] for gold in golds]
            pred_is_part = pred_name in sketch.parts
            gold_has_part = any(name in sketch.parts for name in gold_names)
            gold_has_location = any(name in sketch.locations for name in gold_names)
            if gold_has_part and not pred_is_part:
                missed_stack += 1
            elif gold_has_location and pred_name in sketch.locations:
                location_region += 1
            elif gold_has_part and pred_is_part:
                wrong_support_part += 1
            elif gold_has_location and pred_is_part:
                false_stack += 1

    denom = max(n_parts, 1)
    return {
        "placement_parts": float(n_parts),
        "placement_part_top1": float(top1 / denom),
        "placement_part_top3": float(top3 / denom),
        "placement_part_top10": float(top10 / denom),
        "placement_mean_gold_rank": float(np.mean(ranks)) if ranks else 0.0,
        "placement_max_gold_rank": float(max(ranks)) if ranks else 0.0,
        "missed_stack_top1": float(missed_stack),
        "location_region_top1": float(location_region),
        "wrong_support_part_top1": float(wrong_support_part),
        "false_stack_top1": float(false_stack),
    }


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    active_part_masks: torch.Tensor,
    label_variants: torch.Tensor | None = None,
    variant_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    preds = preds.float().cpu()
    labels = labels.float().cpu()
    masks = masks.cpu()
    preds = preds * masks
    labels = labels * masks
    if label_variants is None:
        label_variants = labels.unsqueeze(1)
        variant_mask = torch.ones(labels.shape[0], 1, dtype=torch.bool)
    else:
        label_variants = label_variants.float().cpu() * masks.unsqueeze(1)
        if variant_mask is None:
            variant_mask = torch.ones(
                label_variants.shape[:2],
                dtype=torch.bool,
            )
        else:
            variant_mask = variant_mask.cpu()

    pred_v = preds.unsqueeze(1)
    mask_v = masks.unsqueeze(1)
    tp_v = ((pred_v == 1) & (label_variants == 1) & (mask_v == 1)).sum(dim=2)
    fp_v = ((pred_v == 1) & (label_variants == 0) & (mask_v == 1)).sum(dim=2)
    fn_v = ((pred_v == 0) & (label_variants == 1) & (mask_v == 1)).sum(dim=2)
    mismatch_v = fp_v + fn_v
    mismatch_v = mismatch_v.masked_fill(~variant_mask, 10**9)
    best_variant = mismatch_v.argmin(dim=1)
    row_idx = torch.arange(preds.shape[0])
    best_labels = label_variants[row_idx, best_variant]

    tp = ((preds == 1) & (best_labels == 1) & (masks == 1)).sum().item()
    fp = ((preds == 1) & (best_labels == 0) & (masks == 1)).sum().item()
    fn = ((preds == 0) & (best_labels == 1) & (masks == 1)).sum().item()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    exact = (mismatch_v.min(dim=1).values == 0).float().mean().item()
    legal = np.mean([
        float(is_legal_prediction(preds[i], sketch, active_part_masks[i]))
        for i in range(preds.shape[0])
    ])
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "exact_match": float(exact),
        "legal": float(legal),
        "pred_pos_rate": float(preds[masks > 0].mean().item()),
        "label_pos_rate": float(labels[masks > 0].mean().item()),
    }


def is_legal_prediction(
    pred: torch.Tensor,
    sketch: AriacPlacementSketch,
    active_part_mask: torch.Tensor,
) -> bool:
    active = [p for p, flag in zip(sketch.parts, active_part_mask.tolist()) if flag > 0]
    targets = sketch.labels_to_support_targets(pred, active_part_mask)
    if any(targets[sketch.part_index(p)].item() < 0 for p in active):
        return False
    assignment: dict[str, str] = {}
    for p in active:
        pi = sketch.part_index(p)
        ci = int(targets[pi].item())
        assignment[p] = sketch.place_candidates[p][ci]
    if not sketch.is_valid_assignment(assignment, active):
        return False
    derived = torch.tensor(sketch.assignment_to_vector(assignment, active))
    mask = sketch.active_atom_mask(active).long()
    return bool(((derived.long() * mask) == (pred.long() * mask)).all().item())


def snapshot_state_dict(
    model: nn.Module,
    trainable_only: bool = False,
) -> dict[str, torch.Tensor]:
    """Clone model state, optionally keeping only trainable parameters.

    The online DINOv3 PEFT path keeps a frozen 3GB backbone in the module.  For
    that path we only need the trainable LoRA/projection/head parameters; the
    frozen base is reconstructed from --dinov3-weights when loading.
    """
    if not trainable_only:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    trainable_names = {
        name for name, param in model.named_parameters() if param.requires_grad
    }
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k in trainable_names
    }


def train_one(
    method: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    label_variants: torch.Tensor,
    active_part_masks: torch.Tensor,
    active_atom_masks: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_masks: torch.Tensor,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    domain_info: DomainInfo,
    sketch: AriacPlacementSketch,
    type_ids: torch.Tensor,
    slot_init: torch.Tensor,
    args,
) -> dict:
    device = torch.device(args.device)
    use_placement = method == "placement"
    online_dinov3 = args.feature_source == "dinov3_online"
    cache_features_on_device = args.features_on_device and not online_dinov3
    if cache_features_on_device:
        labels = labels.to(device)
        label_variants = label_variants.to(device)
        active_part_masks = active_part_masks.to(device)
        active_atom_masks = active_atom_masks.to(device)
        support_target_variants = support_target_variants.to(device)
        variant_masks = variant_masks.to(device)
    visual_encoder = None
    if online_dinov3:
        visual_encoder = DINOv3ViTHPlus(
            args.dinov3_weights,
            d_out=args.d_slot,
            image_size=args.dinov3_scales_list[0],
            last_n_layers=args.dinov3_last_n_layers,
            layer_fusion=args.dinov3_layer_fusion,
            peft=args.dinov3_peft,
            lora_rank=args.dinov3_lora_rank,
            lora_alpha=args.dinov3_lora_alpha,
            lora_dropout=args.dinov3_lora_dropout,
            lora_last_blocks=args.dinov3_lora_last_blocks,
            lora_targets=args.dinov3_lora_targets,
        )
    model = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=len(domain_info.objects),
        d_slot=args.d_slot,
        n_slot_iters=args.n_slot_iters,
        use_real_encoder=online_dinov3,
        visual_encoder=visual_encoder,
        predict_slot_types=True,
        object_extractor_type=args.object_extractor_type,
        object_query_relation_layers=args.object_query_relation_layers,
        object_query_local_refine=args.object_query_local_refine,
        object_query_local_top_k=args.object_query_local_top_k,
        object_query_local_radius=args.object_query_local_radius,
        dense_global_bias=args.dense_global_bias,
        use_support_head=use_placement,
        support_block_type="part",
        support_column_type="location",
        support_head_type=args.support_head_type,
        support_temperature=args.support_temperature,
        support_geometry_type=args.support_geometry_type,
        support_location_prior_weight=args.support_location_prior_weight,
        support_location_prior_sigma=args.support_location_prior_sigma,
        support_patch_evidence_type=args.support_patch_evidence_type,
        support_patch_location_scale_init=args.support_patch_location_scale_init,
        support_patch_table_scale_init=args.support_patch_table_scale_init,
        support_patch_contact_scale_init=args.support_patch_contact_scale_init,
        support_patch_location_sigma=args.support_patch_location_sigma,
        support_patch_temperature=args.support_patch_temperature,
        support_patch_contact_top_k=args.support_patch_contact_top_k,
        support_patch_contact_sigma_x=args.support_patch_contact_sigma_x,
        support_patch_contact_sigma_y=args.support_patch_contact_sigma_y,
        support_patch_contact_gap=args.support_patch_contact_gap,
        support_hidden_dim=args.support_hidden_dim,
        scoring_head_type=args.scoring_head_type,
    ).to(device)
    input_feature_dim = args.d_slot if online_dinov3 else int(features.shape[-1])
    if not online_dinov3 and input_feature_dim != args.d_slot:
        model.feat_proj = build_feature_projector(
            input_dim=input_feature_dim,
            d_slot=args.d_slot,
            args=args,
        ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    type_ids_dev = type_ids.to(device)
    slot_init_dev = slot_init.to(device)

    train_features = (
        features[train_idx].to(device)
        if cache_features_on_device
        else features[train_idx]
    )
    ds = TensorDataset(
        train_features,
        labels[train_idx],
        label_variants[train_idx],
        active_part_masks[train_idx],
        active_atom_masks[train_idx],
        support_target_variants[train_idx],
        variant_masks[train_idx],
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    best_state = None
    best_train_loss = float("inf")
    hist = []
    partial_state_dict = online_dinov3

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        nb = 0
        for feats, labs, lab_vars, act_parts, act_atoms, sup_tgt_vars, var_mask in loader:
            feats = feats.to(device)
            labs = labs.to(device)
            lab_vars = lab_vars.to(device)
            act_parts = act_parts.to(device)
            act_atoms = act_atoms.to(device)
            sup_tgt_vars = sup_tgt_vars.to(device)
            var_mask = var_mask.to(device)
            batch = feats.shape[0]
            optimizer.zero_grad()
            out = model(
                feats,
                object_type_ids=type_ids_dev.unsqueeze(0).expand(batch, -1),
                slot_init=slot_init_dev.unsqueeze(0).expand(batch, -1, -1),
            )
            if use_placement:
                if args.placement_loss == "ce":
                    loss = duplicate_invariant_support_ce(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                    )
                elif args.placement_loss == "structured":
                    loss = structured_legal_state_nll(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                        act_parts,
                        sketch,
                    )
                elif args.placement_loss == "ce_structured":
                    loss = duplicate_invariant_support_ce(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                    )
                    loss = loss + args.structured_loss_weight * structured_legal_state_nll(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                        act_parts,
                        sketch,
                    )
                else:
                    raise ValueError(f"Unknown placement_loss={args.placement_loss}")
                if args.counterfactual_margin_weight > 0:
                    loss = loss + args.counterfactual_margin_weight * counterfactual_margin_loss(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                        act_parts,
                        sketch,
                        margin=args.counterfactual_margin,
                    )
                if args.dynamic_hard_negative_weight > 0:
                    loss = loss + args.dynamic_hard_negative_weight * dynamic_hard_negative_support_loss(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                        act_parts,
                        sketch,
                        margin=args.dynamic_hard_negative_margin,
                        region_table_weight=args.dynamic_region_table_weight,
                        stack_table_weight=args.dynamic_stack_table_weight,
                        wrong_support_weight=args.dynamic_wrong_support_weight,
                    )
                if args.occupancy_loss_weight > 0:
                    loss = loss + args.occupancy_loss_weight * support_occupancy_loss(
                        out["support_scores"],
                        sup_tgt_vars,
                        var_mask,
                        act_parts,
                        sketch,
                    )
                if args.aux_atom_weight > 0:
                    loss = loss + args.aux_atom_weight * duplicate_invariant_bce_with_logits(
                        out["canonical_scores"], lab_vars, var_mask, act_atoms
                    )
            else:
                loss = duplicate_invariant_bce_with_logits(
                    out["canonical_scores"], lab_vars, var_mask, act_atoms
                )
            if args.type_weight > 0:
                loss = loss + args.type_weight * model.compute_type_loss(
                    type_ids_dev, forward_output=out
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            nb += 1
        train_loss = total / max(nb, 1)
        hist.append({"epoch": epoch, "train_loss": train_loss})
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_state = snapshot_state_dict(
                model,
                trainable_only=partial_state_dict,
            )
        if epoch == 1 or epoch % args.print_every == 0:
            print(f"    [{method} epoch {epoch:03d}] loss={train_loss:.4f}")

    assert best_state is not None
    if cache_features_on_device:
        del ds, loader, train_features
        torch.cuda.empty_cache()
    model.load_state_dict(best_state, strict=not partial_state_dict)
    model.to(device).eval()

    def forward_indices(
        indices: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        pred_chunks = []
        score_chunks = []
        support_chunks = []
        eval_features = (
            features[indices].to(device)
            if cache_features_on_device
            else features[indices]
        )
        eval_ds = TensorDataset(eval_features, active_part_masks[indices])
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for feats, act_parts in eval_loader:
                feats = feats.to(device)
                batch = feats.shape[0]
                out = model(
                    feats,
                    object_type_ids=type_ids_dev.unsqueeze(0).expand(batch, -1),
                    slot_init=slot_init_dev.unsqueeze(0).expand(batch, -1, -1),
                )
                if use_placement:
                    support_cpu = out["support_scores"].cpu()
                    if args.hybrid_atom_decode_weight > 0:
                        vec_rows = []
                        atom_cpu = out["canonical_scores"].cpu()
                        act_cpu = act_parts.cpu()
                        for row_i in range(support_cpu.shape[0]):
                            active_names = [
                                p for p, flag in zip(sketch.parts, act_cpu[row_i].tolist())
                                if flag > 0
                            ]
                            vec_rows.append(
                                hybrid_decode_vector(
                                    sketch,
                                    support_cpu[row_i],
                                    atom_cpu[row_i],
                                    active_names,
                                    atom_weight=args.hybrid_atom_decode_weight,
                                )
                            )
                        vecs = torch.tensor(vec_rows, dtype=torch.float32)
                    else:
                        vecs, _ = sketch.decode_batch(
                            support_cpu,
                            act_parts.cpu(),
                            device="cpu",
                        )
                    pred_chunks.append(vecs)
                    support_chunks.append(support_cpu)
                    score_chunks.append(out["canonical_scores"].cpu())
                else:
                    score_chunks.append(out["canonical_scores"].cpu())
        scores = torch.cat(score_chunks, dim=0)
        if use_placement:
            preds = torch.cat(pred_chunks, dim=0)
            support_scores = torch.cat(support_chunks, dim=0)
        else:
            preds = scores
            support_scores = None
        return preds, scores, support_scores

    train_pred_or_logits, train_scores, train_support_scores = forward_indices(train_idx)
    test_pred_or_logits, test_scores, test_support_scores = forward_indices(test_idx)

    threshold = None
    legal_reranker_weights = None
    if use_placement:
        if args.legal_reranker == "linear":
            assert train_support_scores is not None and test_support_scores is not None
            legal_reranker_weights = fit_legal_state_reranker(
                train_support_scores,
                support_target_variants[train_idx].cpu(),
                variant_masks[train_idx].cpu(),
                active_part_masks[train_idx].cpu(),
                sketch,
                top_k=args.legal_reranker_top_k,
                steps=args.legal_reranker_steps,
                lr=args.legal_reranker_lr,
                l2=args.legal_reranker_l2,
            )

            def decode_with_reranker(
                support: torch.Tensor,
                active_masks: torch.Tensor,
            ) -> torch.Tensor:
                rows = []
                for bi in range(support.shape[0]):
                    rows.append(
                        legal_rerank_decode_vector(
                            sketch,
                            support[bi],
                            active_masks[bi],
                            legal_reranker_weights,
                            top_k=args.legal_reranker_top_k,
                        )
                    )
                return torch.tensor(rows, dtype=torch.float32)

            train_preds = decode_with_reranker(
                train_support_scores,
                active_part_masks[train_idx].cpu(),
            )
            test_preds = decode_with_reranker(
                test_support_scores,
                active_part_masks[test_idx].cpu(),
            )
        else:
            train_preds = train_pred_or_logits
            test_preds = test_pred_or_logits
    else:
        threshold_grid = [float(x) for x in np.linspace(0.05, 0.95, 19)]
        best = (-1.0, 0.5)
        train_labs = labels[train_idx]
        train_masks = active_atom_masks[train_idx]
        train_label_variants = label_variants[train_idx]
        train_variant_masks = variant_masks[train_idx]
        for thr in threshold_grid:
            cand = (torch.sigmoid(train_scores) >= thr).float()
            score = compute_metrics(
                cand,
                train_labs,
                train_masks,
                sketch,
                active_part_masks[train_idx],
                train_label_variants,
                train_variant_masks,
            )["exact_match"]
            if score > best[0]:
                best = (score, thr)
        threshold = best[1]
        train_preds = (torch.sigmoid(train_scores) >= threshold).float()
        test_preds = (torch.sigmoid(test_scores) >= threshold).float()

    train_metrics = compute_metrics(
        train_preds,
        labels[train_idx],
        active_atom_masks[train_idx],
        sketch,
        active_part_masks[train_idx],
        label_variants[train_idx],
        variant_masks[train_idx],
    )
    test_metrics = compute_metrics(
        test_preds,
        labels[test_idx],
        active_atom_masks[test_idx],
        sketch,
        active_part_masks[test_idx],
        label_variants[test_idx],
        variant_masks[test_idx],
    )
    train_placement_metrics = None
    test_placement_metrics = None
    if use_placement:
        assert train_support_scores is not None and test_support_scores is not None
        train_placement_metrics = placement_ranking_metrics(
            train_support_scores,
            support_target_variants[train_idx],
            variant_masks[train_idx],
            active_part_masks[train_idx],
            sketch,
        )
        test_placement_metrics = placement_ranking_metrics(
            test_support_scores,
            support_target_variants[test_idx],
            variant_masks[test_idx],
            active_part_masks[test_idx],
            sketch,
        )

    return {
        "method": method,
        "best_train_loss": best_train_loss,
        "threshold": threshold,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_placement_metrics": train_placement_metrics,
        "test_placement_metrics": test_placement_metrics,
        "history": hist,
        "state_dict": best_state,
        "state_dict_is_partial": partial_state_dict,
        "input_feature_dim": input_feature_dim,
        "legal_reranker_weights": (
            None if legal_reranker_weights is None else legal_reranker_weights.tolist()
        ),
    }


def write_report(exp_dir: Path, rows: list[dict], metadata: dict):
    csv_path = exp_dir / "results.csv"
    fields = [
        "method", "k", "n_train", "n_test", "threshold",
        "test_exact_match", "test_f1", "test_precision", "test_recall",
        "test_legal", "train_exact_match", "train_f1",
        "placement_part_top1", "placement_part_top3", "placement_part_top10",
        "placement_mean_gold_rank", "placement_max_gold_rank",
        "missed_stack_top1", "location_region_top1", "wrong_support_part_top1",
        "false_stack_top1",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})

    md_path = exp_dir / "report.md"
    with md_path.open("w") as f:
        f.write("# ARIAC Init-State Placement Decoder\n\n")
        f.write("Only true atoms from `(:init ...)` are used as labels. Goals and VL text are ignored.\n\n")
        f.write("## Metadata\n\n")
        for k, v in metadata.items():
            f.write(f"- `{k}`: `{v}`\n")
        f.write("\n## Results\n\n")
        eval_name = metadata.get("eval_name", "test")
        f.write(
            f"| method | K | {eval_name} EM | {eval_name} F1 | P | R | legal | "
            "pl top1 | pl top3 | pl top10 | miss stack | loc err | threshold |\n"
        )
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            thr = row["threshold"]
            thr_s = "" if thr is None else f"{thr:.2f}"
            pl_top1 = "" if row.get("placement_part_top1") is None else f"{row['placement_part_top1']:.4f}"
            pl_top3 = "" if row.get("placement_part_top3") is None else f"{row['placement_part_top3']:.4f}"
            pl_top10 = "" if row.get("placement_part_top10") is None else f"{row['placement_part_top10']:.4f}"
            miss_stack = "" if row.get("missed_stack_top1") is None else f"{row['missed_stack_top1']:.0f}"
            loc_err = "" if row.get("location_region_top1") is None else f"{row['location_region_top1']:.0f}"
            f.write(
                f"| {row['method']} | {row['k']} | "
                f"{row['test_exact_match']:.4f} | {row['test_f1']:.4f} | "
                f"{row['test_precision']:.4f} | {row['test_recall']:.4f} | "
                f"{row['test_legal']:.4f} | "
                f"{pl_top1} | {pl_top3} | {pl_top10} | "
                f"{miss_stack} | {loc_err} | "
                f"{thr_s} |\n"
            )


def parse_k_values(raw: str) -> list[int]:
    values = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        values.append(-1 if x.lower() == "all" else int(x))
    return values


def parse_int_values(raw: str) -> list[int]:
    values = []
    for x in raw.split(","):
        x = x.strip()
        if x:
            values.append(int(x))
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "ariac")
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--k-values", type=str, default="50,100,150")
    parser.add_argument("--methods", type=str, default="atom,placement")
    parser.add_argument(
        "--test-size",
        type=int,
        default=None,
        help="Number of held-out test samples. Defaults to one fifth of the usable data.",
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Use all samples for training and report fit on the same samples.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-slot", type=int, default=64)
    parser.add_argument("--n-slot-iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--aux-atom-weight", type=float, default=0.2)
    parser.add_argument("--type-weight", type=float, default=0.05)
    parser.add_argument("--scoring-head-type", type=str, default="film", choices=["film", "legacy"])
    parser.add_argument(
        "--placement-loss",
        type=str,
        default="ce",
        choices=["ce", "structured", "ce_structured"],
        help=(
            "Training objective for placement models. `structured` optimizes "
            "exact NLL over legal PDDL placement assignments."
        ),
    )
    parser.add_argument("--structured-loss-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-margin-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-margin", type=float, default=1.0)
    parser.add_argument(
        "--dynamic-hard-negative-weight",
        type=float,
        default=0.0,
        help="Weight for current top-wrong support-candidate margin loss.",
    )
    parser.add_argument("--dynamic-hard-negative-margin", type=float, default=1.0)
    parser.add_argument("--dynamic-region-table-weight", type=float, default=2.0)
    parser.add_argument("--dynamic-stack-table-weight", type=float, default=2.0)
    parser.add_argument("--dynamic-wrong-support-weight", type=float, default=1.5)
    parser.add_argument(
        "--occupancy-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary occupied/clear consistency weight for part supports.",
    )
    parser.add_argument(
        "--hybrid-atom-decode-weight",
        type=float,
        default=0.0,
        help="Rerank legal placement assignments with atom-branch log likelihood.",
    )
    parser.add_argument(
        "--legal-reranker",
        type=str,
        default="none",
        choices=["none", "linear"],
        help="Train a tiny post-hoc linear energy over legal PDDL assignments.",
    )
    parser.add_argument("--legal-reranker-steps", type=int, default=200)
    parser.add_argument("--legal-reranker-top-k", type=int, default=25)
    parser.add_argument("--legal-reranker-lr", type=float, default=0.05)
    parser.add_argument("--legal-reranker-l2", type=float, default=0.05)
    parser.add_argument(
        "--object-extractor-type",
        type=str,
        default="slot_attention",
        choices=["slot_attention", "object_queries"],
    )
    parser.add_argument("--object-query-relation-layers", type=int, default=0)
    parser.add_argument("--object-query-local-refine", action="store_true")
    parser.add_argument("--object-query-local-top-k", type=int, default=4)
    parser.add_argument("--object-query-local-radius", type=int, default=2)
    parser.add_argument("--dense-global-bias", action="store_true")
    parser.add_argument(
        "--support-head-type",
        type=str,
        default="legacy",
        choices=[
            "legacy",
            "pair",
            "two_stage",
            "typed_two_stage",
            "calibrated_two_stage",
        ],
    )
    parser.add_argument("--support-temperature", type=float, default=1.0)
    parser.add_argument(
        "--support-geometry-type",
        type=str,
        default="none",
        choices=["none", "attention"],
        help="Optional geometry features for support scoring.",
    )
    parser.add_argument("--support-location-prior-weight", type=float, default=0.0)
    parser.add_argument("--support-location-prior-sigma", type=float, default=0.2)
    parser.add_argument(
        "--support-patch-evidence-type",
        type=str,
        default="none",
        choices=["none", "location", "location_table", "location_table_contact"],
    )
    parser.add_argument("--support-patch-location-scale-init", type=float, default=0.5)
    parser.add_argument("--support-patch-table-scale-init", type=float, default=0.5)
    parser.add_argument("--support-patch-contact-scale-init", type=float, default=0.5)
    parser.add_argument("--support-patch-location-sigma", type=float, default=0.18)
    parser.add_argument("--support-patch-temperature", type=float, default=1.0)
    parser.add_argument("--support-patch-contact-top-k", type=int, default=16)
    parser.add_argument("--support-patch-contact-sigma-x", type=float, default=0.12)
    parser.add_argument("--support-patch-contact-sigma-y", type=float, default=0.12)
    parser.add_argument("--support-patch-contact-gap", type=float, default=0.06)
    parser.add_argument("--support-hidden-dim", type=int, default=None)
    parser.add_argument(
        "--feature-projector",
        type=str,
        default="linear",
        choices=["linear", "layer_attention"],
    )
    parser.add_argument("--dinov3-base-dim", type=int, default=1280)
    parser.add_argument(
        "--feature-source",
        type=str,
        default="raw_grid",
        choices=["raw_grid", "dinov3", "dinov3_raw", "dinov3_online"],
    )
    parser.add_argument(
        "--dinov3-weights",
        type=Path,
        default=ROOT / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    )
    parser.add_argument("--dinov3-batch-size", type=int, default=1)
    parser.add_argument(
        "--dinov3-scales",
        type=str,
        default="224",
        help="Comma-separated square image sizes for DINOv3 feature extraction, e.g. 224,448.",
    )
    parser.add_argument("--dinov3-last-n-layers", type=int, default=1)
    parser.add_argument(
        "--dinov3-layer-fusion",
        type=str,
        default="last",
        choices=["last", "mean", "concat"],
    )
    parser.add_argument("--dinov3-add-coords", action="store_true")
    parser.add_argument(
        "--dinov3-peft",
        type=str,
        default="none",
        choices=["none", "lora"],
        help="PEFT mode for feature_source=dinov3_online.",
    )
    parser.add_argument("--dinov3-lora-rank", type=int, default=8)
    parser.add_argument("--dinov3-lora-alpha", type=float, default=None)
    parser.add_argument("--dinov3-lora-dropout", type=float, default=0.0)
    parser.add_argument("--dinov3-lora-last-blocks", type=int, default=2)
    parser.add_argument("--dinov3-lora-targets", type=str, default="qkv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Seed for train/test split. Defaults to --seed.",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=None,
        help="Seed for model initialization/training randomness. Defaults to --seed.",
    )
    parser.add_argument("--print-every", type=int, default=40)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument(
        "--features-on-device",
        action="store_true",
        help=(
            "Keep cached features and target tensors on the training device. "
            "Useful for small cached-feature experiments when GPU memory is ample."
        ),
    )
    parser.add_argument("--keep-invalid", action="store_true")
    parser.add_argument(
        "--exclude-duplicate-parts",
        action="store_true",
        help="Drop samples whose active objects contain exchangeable duplicates such as red_pump/red_pump_1.",
    )
    parser.add_argument(
        "--duplicate-mode",
        type=str,
        default="exchangeable",
        choices=["exchangeable", "strict_names"],
        help=(
            "Treat same-base duplicate instances like red_pump/red_pump_1 as "
            "exchangeable during loss/evaluation, or require exact object names."
        ),
    )
    args = parser.parse_args()
    args.dinov3_scales_list = parse_int_values(args.dinov3_scales)
    for scale in args.dinov3_scales_list:
        if scale <= 0 or scale % 16 != 0:
            raise ValueError(f"DINOv3 scale must be a positive multiple of 16, got {scale}")
    if args.feature_source == "dinov3_online" and len(args.dinov3_scales_list) != 1:
        raise ValueError("dinov3_online currently supports exactly one --dinov3-scales value")
    if args.feature_source != "dinov3_online" and args.dinov3_peft != "none":
        raise ValueError("--dinov3-peft requires --feature-source dinov3_online")
    if args.dinov3_add_coords and args.feature_source == "dinov3_online":
        raise ValueError("--dinov3-add-coords is only supported for cached DINO features")
    if args.feature_projector == "layer_attention":
        if args.feature_source not in {"dinov3", "dinov3_raw"}:
            raise ValueError("--feature-projector layer_attention requires cached DINO features")
        if args.dinov3_layer_fusion != "concat" or args.dinov3_last_n_layers <= 1:
            raise ValueError(
                "--feature-projector layer_attention requires "
                "--dinov3-layer-fusion concat and --dinov3-last-n-layers > 1"
            )
    if args.object_query_relation_layers < 0:
        raise ValueError("--object-query-relation-layers must be non-negative")
    if args.object_query_local_top_k <= 0:
        raise ValueError("--object-query-local-top-k must be positive")
    if args.object_query_local_radius < 0:
        raise ValueError("--object-query-local-radius must be non-negative")
    if args.object_query_local_refine and args.object_extractor_type != "object_queries":
        raise ValueError("--object-query-local-refine requires --object-extractor-type object_queries")
    if args.support_temperature <= 0:
        raise ValueError("--support-temperature must be positive")
    if args.support_location_prior_weight < 0:
        raise ValueError("--support-location-prior-weight must be non-negative")
    if args.support_location_prior_sigma <= 0:
        raise ValueError("--support-location-prior-sigma must be positive")
    if args.support_location_prior_weight > 0 and args.support_geometry_type != "attention":
        raise ValueError("--support-location-prior-weight requires --support-geometry-type attention")
    if args.support_patch_evidence_type != "none" and args.object_extractor_type != "object_queries":
        raise ValueError("--support-patch-evidence-type requires object_queries")
    if args.support_patch_location_sigma <= 0:
        raise ValueError("--support-patch-location-sigma must be positive")
    if args.support_patch_temperature <= 0:
        raise ValueError("--support-patch-temperature must be positive")
    if args.support_patch_contact_top_k <= 0:
        raise ValueError("--support-patch-contact-top-k must be positive")
    if args.support_patch_contact_sigma_x <= 0 or args.support_patch_contact_sigma_y <= 0:
        raise ValueError("support patch contact sigmas must be positive")
    if args.structured_loss_weight < 0:
        raise ValueError("--structured-loss-weight must be non-negative")
    if args.counterfactual_margin_weight < 0:
        raise ValueError("--counterfactual-margin-weight must be non-negative")
    if args.counterfactual_margin <= 0:
        raise ValueError("--counterfactual-margin must be positive")
    if args.dynamic_hard_negative_weight < 0:
        raise ValueError("--dynamic-hard-negative-weight must be non-negative")
    if args.dynamic_hard_negative_margin <= 0:
        raise ValueError("--dynamic-hard-negative-margin must be positive")
    for name in (
        "dynamic_region_table_weight",
        "dynamic_stack_table_weight",
        "dynamic_wrong_support_weight",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.occupancy_loss_weight < 0:
        raise ValueError("--occupancy-loss-weight must be non-negative")
    if args.hybrid_atom_decode_weight < 0:
        raise ValueError("--hybrid-atom-decode-weight must be non-negative")
    if args.legal_reranker != "none" and args.hybrid_atom_decode_weight > 0:
        raise ValueError("--legal-reranker cannot be combined with --hybrid-atom-decode-weight")
    if args.legal_reranker_steps <= 0:
        raise ValueError("--legal-reranker-steps must be positive")
    if args.legal_reranker_top_k <= 0:
        raise ValueError("--legal-reranker-top-k must be positive")
    if args.legal_reranker_lr <= 0:
        raise ValueError("--legal-reranker-lr must be positive")
    if args.legal_reranker_l2 < 0:
        raise ValueError("--legal-reranker-l2 must be non-negative")
    if args.test_size is not None and args.test_size <= 0:
        raise ValueError("--test-size must be positive")

    args.split_seed_value = args.seed if args.split_seed is None else args.split_seed
    args.init_seed_value = args.seed if args.init_seed is None else args.init_seed

    torch.manual_seed(args.init_seed_value)
    np.random.seed(args.init_seed_value)

    exp_name = args.exp_name or f"ariac_init_structured_{int(time.time())}"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("ARIAC init-state grounding")
    print(f"  data: {args.data_dir}")
    print(f"  output: {exp_dir}")
    print(f"  device: {args.device}")
    print("=" * 72)

    samples = load_samples(args.data_dir, strict_valid=not args.keep_invalid)
    if not samples:
        raise RuntimeError("No ARIAC samples loaded")
    excluded_duplicate_ids: list[str] = []
    if args.exclude_duplicate_parts:
        kept_samples = []
        for sample in samples:
            if has_duplicate_active_parts(sample):
                excluded_duplicate_ids.append(sample.sample_id)
            else:
                kept_samples.append(sample)
        samples = kept_samples
        if not samples:
            raise RuntimeError("No ARIAC samples left after excluding duplicate parts")
        print(
            f"  excluded duplicate-part samples={len(excluded_duplicate_ids)} "
            f"remaining={len(samples)}"
        )
        with (exp_dir / "excluded_duplicate_samples.txt").open("w") as f:
            for sample_id in excluded_duplicate_ids:
                f.write(sample_id + "\n")
    all_parts = sorted({p for s in samples for p in s.active_parts})
    all_locations = [
        loc for loc in DEFAULT_LOCATIONS
        if any(loc in s.active_locations for s in samples)
    ]
    for loc in sorted({l for s in samples for l in s.active_locations}):
        if loc not in all_locations:
            all_locations.append(loc)

    domain_info = build_domain_info(all_parts, all_locations)
    sketch = AriacPlacementSketch.from_domain_info(domain_info)
    print(f"  samples={len(samples)} parts={len(all_parts)} locations={len(all_locations)}")
    print(f"  canonical init atoms={domain_info.n_canonical}")

    labels, active_part_masks, active_atom_masks = [], [], []
    for sample in samples:
        y, part_mask, atom_mask = sample_labels(sample, sketch)
        labels.append(y)
        active_part_masks.append(part_mask)
        active_atom_masks.append(atom_mask)
    labels_t = torch.stack(labels)
    active_part_masks_t = torch.stack(active_part_masks)
    active_atom_masks_t = torch.stack(active_atom_masks)
    label_variants_t, support_target_variants_t, variant_masks_t, duplicate_meta = (
        build_duplicate_label_variants(
            samples=samples,
            labels=labels_t,
            active_part_masks=active_part_masks_t,
            sketch=sketch,
            duplicate_mode=args.duplicate_mode,
        )
    )
    print(
        f"  active label pos rate="
        f"{(labels_t * active_atom_masks_t).sum().item() / active_atom_masks_t.sum().item():.4f}"
    )
    print(
        f"  duplicate mode={args.duplicate_mode} "
        f"variant_samples={duplicate_meta['samples_with_duplicate_variants']} "
        f"max_variants={duplicate_meta['max_duplicate_target_variants']}"
    )

    scale_tag = "-".join(str(x) for x in args.dinov3_scales_list)
    layer_tag = f"l{args.dinov3_last_n_layers}_{args.dinov3_layer_fusion}"
    coord_tag = "_coords" if args.dinov3_add_coords else ""
    feature_cache = exp_dir / (
        f"ariac_{args.feature_source}_s{scale_tag}_{layer_tag}{coord_tag}_features.pt"
    )
    features = load_or_extract_features(
        samples,
        feature_cache,
        d_slot=args.d_slot,
        feature_source=args.feature_source,
        dinov3_weights=args.dinov3_weights,
        device=args.device,
        dinov3_batch_size=args.dinov3_batch_size,
        dinov3_scales=args.dinov3_scales_list,
        dinov3_last_n_layers=args.dinov3_last_n_layers,
        dinov3_layer_fusion=args.dinov3_layer_fusion,
        dinov3_add_coords=args.dinov3_add_coords,
        rebuild=args.rebuild_feature_cache,
    )

    n = len(samples)
    rng = np.random.default_rng(args.split_seed_value)
    perm = rng.permutation(n)
    if args.train_all:
        test_idx = np.arange(n)
        train_pool = np.arange(n)
    else:
        n_test = args.test_size if args.test_size is not None else max(1, n // 5)
        if n_test >= n:
            raise ValueError(
                f"--test-size must be smaller than the number of samples ({n}), got {n_test}"
            )
        test_idx = perm[:n_test]
        train_pool = perm[n_test:]
    k_values = parse_k_values(args.k_values)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    slot_init = object_slot_init(all_parts, all_locations, args.d_slot)

    rows: list[dict] = []
    artifacts = {"metadata": {}, "runs": []}
    metadata = {
        "n_samples": len(samples),
        "n_test": len(test_idx),
        "n_train_pool": len(train_pool),
        "n_parts": len(all_parts),
        "n_locations": len(all_locations),
        "n_canonical_atoms": domain_info.n_canonical,
        "feature_source": args.feature_source,
        "input_feature_dim": args.d_slot if args.feature_source == "dinov3_online" else int(features.shape[-1]),
        "features_on_device": args.features_on_device,
        "object_extractor_type": args.object_extractor_type,
        "object_query_relation_layers": args.object_query_relation_layers,
        "object_query_local_refine": args.object_query_local_refine,
        "object_query_local_top_k": args.object_query_local_top_k,
        "object_query_local_radius": args.object_query_local_radius,
        "dense_global_bias": args.dense_global_bias,
        "support_head_type": args.support_head_type,
        "support_temperature": args.support_temperature,
        "support_geometry_type": args.support_geometry_type,
        "support_location_prior_weight": args.support_location_prior_weight,
        "support_location_prior_sigma": args.support_location_prior_sigma,
        "support_patch_evidence_type": args.support_patch_evidence_type,
        "support_patch_location_scale_init": args.support_patch_location_scale_init,
        "support_patch_table_scale_init": args.support_patch_table_scale_init,
        "support_patch_contact_scale_init": args.support_patch_contact_scale_init,
        "support_patch_location_sigma": args.support_patch_location_sigma,
        "support_patch_temperature": args.support_patch_temperature,
        "support_patch_contact_top_k": args.support_patch_contact_top_k,
        "support_patch_contact_sigma_x": args.support_patch_contact_sigma_x,
        "support_patch_contact_sigma_y": args.support_patch_contact_sigma_y,
        "support_patch_contact_gap": args.support_patch_contact_gap,
        "support_hidden_dim": args.support_hidden_dim,
        "placement_loss": args.placement_loss,
        "structured_loss_weight": args.structured_loss_weight,
        "counterfactual_margin_weight": args.counterfactual_margin_weight,
        "counterfactual_margin": args.counterfactual_margin,
        "dynamic_hard_negative_weight": args.dynamic_hard_negative_weight,
        "dynamic_hard_negative_margin": args.dynamic_hard_negative_margin,
        "dynamic_region_table_weight": args.dynamic_region_table_weight,
        "dynamic_stack_table_weight": args.dynamic_stack_table_weight,
        "dynamic_wrong_support_weight": args.dynamic_wrong_support_weight,
        "occupancy_loss_weight": args.occupancy_loss_weight,
        "hybrid_atom_decode_weight": args.hybrid_atom_decode_weight,
        "legal_reranker": args.legal_reranker,
        "legal_reranker_steps": args.legal_reranker_steps,
        "legal_reranker_top_k": args.legal_reranker_top_k,
        "legal_reranker_lr": args.legal_reranker_lr,
        "legal_reranker_l2": args.legal_reranker_l2,
        "feature_projector": args.feature_projector,
        "dinov3_base_dim": args.dinov3_base_dim,
        "dinov3_scales": args.dinov3_scales_list,
        "dinov3_last_n_layers": args.dinov3_last_n_layers,
        "dinov3_layer_fusion": args.dinov3_layer_fusion,
        "dinov3_add_coords": args.dinov3_add_coords,
        "dinov3_peft": args.dinov3_peft,
        "dinov3_lora_rank": args.dinov3_lora_rank if args.dinov3_peft == "lora" else 0,
        "dinov3_lora_alpha": args.dinov3_lora_alpha if args.dinov3_peft == "lora" else None,
        "dinov3_lora_dropout": args.dinov3_lora_dropout if args.dinov3_peft == "lora" else 0.0,
        "dinov3_lora_last_blocks": args.dinov3_lora_last_blocks if args.dinov3_peft == "lora" else 0,
        "dinov3_lora_targets": args.dinov3_lora_targets if args.dinov3_peft == "lora" else "",
        "d_slot": args.d_slot,
        "strict_valid_factor_states": not args.keep_invalid,
        "exclude_duplicate_parts": args.exclude_duplicate_parts,
        "excluded_duplicate_part_samples": len(excluded_duplicate_ids),
        "train_all": args.train_all,
        "eval_name": "train-fit" if args.train_all else "test",
        "seed": args.seed,
        "split_seed": args.split_seed_value,
        "init_seed": args.init_seed_value,
        **duplicate_meta,
    }

    for k in k_values:
        if k == -1:
            k = len(train_pool)
        if k > len(train_pool):
            print(f"  Skipping K={k}: train pool only has {len(train_pool)} samples")
            continue
        train_idx = train_pool[:k]
        for method in methods:
            if method not in {"atom", "placement"}:
                raise ValueError(f"Unknown method: {method}")
            print(f"\n[run] method={method} K={k} train={len(train_idx)} test={len(test_idx)}")
            result = train_one(
                method=method,
                features=features,
                labels=labels_t,
                label_variants=label_variants_t,
                active_part_masks=active_part_masks_t,
                active_atom_masks=active_atom_masks_t,
                support_target_variants=support_target_variants_t,
                variant_masks=variant_masks_t,
                train_idx=train_idx,
                test_idx=test_idx,
                domain_info=domain_info,
                sketch=sketch,
                type_ids=type_ids,
                slot_init=slot_init,
                args=args,
            )
            k_dir = exp_dir / f"k_{k}" / method
            k_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": result["state_dict"],
                    "method": method,
                    "k": k,
                    "domain_summary": domain_info.summary(),
                    "parts": all_parts,
                    "locations": all_locations,
                    "canonical_atoms": domain_info.canonical_atom_strings,
                    "slot_init": slot_init,
                    "metadata": metadata,
                    "input_feature_dim": result["input_feature_dim"],
                    "feature_source": args.feature_source,
                    "d_slot": args.d_slot,
                    "state_dict_is_partial": result.get("state_dict_is_partial", False),
                    "legal_reranker_weights": result.get("legal_reranker_weights"),
                },
                k_dir / "model.pt",
            )
            with (k_dir / "metrics.json").open("w") as f:
                json.dump(
                    {kk: vv for kk, vv in result.items() if kk != "state_dict"},
                    f,
                    indent=2,
                )
            row = {
                "method": method,
                "k": k,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "threshold": result["threshold"],
                "test_exact_match": result["test_metrics"]["exact_match"],
                "test_f1": result["test_metrics"]["f1"],
                "test_precision": result["test_metrics"]["precision"],
                "test_recall": result["test_metrics"]["recall"],
                "test_legal": result["test_metrics"]["legal"],
                "train_exact_match": result["train_metrics"]["exact_match"],
                "train_f1": result["train_metrics"]["f1"],
            }
            if result.get("test_placement_metrics") is not None:
                row.update(result["test_placement_metrics"])
            rows.append(row)
            artifacts["runs"].append({k: v for k, v in row.items()})
            print(
                f"  -> test EM={row['test_exact_match']:.4f} "
                f"F1={row['test_f1']:.4f} legal={row['test_legal']:.4f}"
            )
            if result.get("test_placement_metrics") is not None:
                pm = result["test_placement_metrics"]
                print(
                    "     placement "
                    f"top1={pm['placement_part_top1']:.4f} "
                    f"top3={pm['placement_part_top3']:.4f} "
                    f"top10={pm['placement_part_top10']:.4f} "
                    f"missed_stack={pm['missed_stack_top1']:.0f} "
                    f"location_region={pm['location_region_top1']:.0f}"
                )

    artifacts["metadata"] = metadata
    with (exp_dir / "results.json").open("w") as f:
        json.dump(artifacts, f, indent=2)
    write_report(exp_dir, rows, metadata)
    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
