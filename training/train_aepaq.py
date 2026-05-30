#!/usr/bin/env python3
"""AE-PaQ: Action-Equivariant Visual Predicate Grounding
=========================================================

Core question:
    Does action structure improve visual grounding BEYOND what state labels provide?

Four experimental conditions (SAME number of labeled frames):

  1. Static Grounding     — L_seed only, no action structure
  2. Random Pairs         — L_seed + consistency on random state pairs
  3. Adjacent Transition  — L_seed + action equivariance (Γ_a consistency)
  4. Full AE-PaQ          — L_seed + equivariance + counterfactual discrimination

The key comparison: conditions 3/4 vs 1/2 tests whether the PDDL transition
structure Γ_a improves grounding quality, beyond what additional image pairs
can provide.

Three structural constraints:
  C1: Pointwise Correctness     G(I) ≈ S
  C2: Action Equivariance       G(I_{t+1}) ≈ Γ_a(G(I_t))
  C3: Counterfactual Discrim.   E(G(I_t), a_true, G(I_{t+1})) < E(..., a_false, ...)

Predicate bottleneck: two images NEVER fused at visual level.
Each passes through the SAME single-image PaQ grounder Gθ independently.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

# Force unbuffered output regardless of redirection target
sys.stdout.reconfigure(write_through=True)
sys.stderr.reconfigure(write_through=True)
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ---- Paths ----
PDDL_ROOT = Path("/home/claudeuser/RL4VLA/PDDL")
VIPAN_ROOT = Path("/home/claudeuser/ViPlan")
DINOV3_REPO = Path("/home/claudeuser/facebookresearch/dinov3")
DINOV3_WEIGHTS = "/home/claudeuser/RL4VLA/PDDL/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"
RENDER_SCRIPT = PDDL_ROOT / "render_bpy_direct.py"
BWS_DOMAIN = VIPAN_ROOT / "data" / "planning" / "blocksworld" / "domain.pddl"

sys.path.insert(0, str(PDDL_ROOT))
sys.path.insert(0, str(DINOV3_REPO))

from paq.domain_compiler import PDDLDomainCompiler
from paq.model import PaQModel
from paq.losses import (
    PredicateStateLoss,
    PredicateContrastiveLoss,
    ActionEquivarianceLoss,
    CounterfactualDiscriminabilityLoss,
    TransitionEnergyScorer,
)
from training.data.ae_dataset import (
    StateDataset,
    TransitionDataset,
    collate_state_batch,
    collate_trans_batch,
)

# Blocksworld config
BLOCKS = ["Y", "P", "R", "O"]
COLUMNS = ["C1", "C2", "C3", "C4"]
STATIC_PREDS = {"rightof", "leftof"}
N_VIEWS = 3
N_AUGS = 3
BLOCK_ID = {"R": 1, "G": 2, "B": 3, "Y": 4, "P": 5, "O": 6}

from itertools import product as iter_product, permutations


# =========================================================================
# State representation
# =========================================================================

class BlocksworldState:
    def __init__(self, blocks, columns, on, inColumn, clear, rightOf, leftOf):
        self.blocks = blocks
        self.columns = columns
        self.on = on
        self.inColumn = inColumn
        self.clear = clear
        self.rightOf = rightOf
        self.leftOf = leftOf

    def to_numpy(self):
        n_cols = len(self.columns)
        matrix = np.zeros((n_cols, 5), dtype=int)
        for ci, col in enumerate(self.columns):
            col_blocks = [b for b in self.blocks if self.inColumn.get((b, col), False)]
            sorted_blocks = self._sort_column(col_blocks)
            for bi, block in enumerate(sorted_blocks):
                matrix[ci, bi] = BLOCK_ID[block.upper()]
        return matrix

    def _sort_column(self, blocks):
        if len(blocks) <= 1:
            return blocks
        below_map = {}
        for b1 in blocks:
            for b2 in blocks:
                if self.on.get((b1, b2), False):
                    below_map[b2] = b1
        tops = set(below_map.values())
        bottoms = [b for b in blocks if b not in tops]
        if not bottoms:
            return blocks
        result = [bottoms[0]]
        while result[-1] in below_map:
            result.append(below_map[result[-1]])
        return result

    def get_predicates(self):
        preds = set()
        for (b1, b2), v in self.on.items():
            if v: preds.add(f"(on {b1} {b2})")
        for (b, c), v in self.inColumn.items():
            if v: preds.add(f"(inColumn {b} {c})")
        for b, v in self.clear.items():
            if v: preds.add(f"(clear {b})")
        for (c1, c2), v in self.rightOf.items():
            if v: preds.add(f"(rightOf {c1} {c2})")
        for (c1, c2), v in self.leftOf.items():
            if v: preds.add(f"(leftOf {c1} {c2})")
        return preds


def enumerate_all_states(blocks, columns):
    n_blocks = len(blocks)
    n_cols = len(columns)
    states = []
    for assignment in iter_product(range(n_cols), repeat=n_blocks):
        col_contents = {c: [] for c in columns}
        for bi, ci in enumerate(assignment):
            col_contents[columns[ci]].append(blocks[bi])
        col_orderings = {}
        for ci, col in enumerate(columns):
            if len(col_contents[col]) <= 1:
                col_orderings[ci] = [col_contents[col]]
            else:
                col_orderings[ci] = list(permutations(col_contents[col]))
        for combo in iter_product(*[col_orderings[ci] for ci in range(n_cols)]):
            on, inColumn, clear = {}, {}, {}
            for ci, col in enumerate(columns):
                stack = combo[ci]
                for si, block in enumerate(stack):
                    inColumn[(block, col)] = True
                    if si > 0:
                        on[(block, stack[si - 1])] = True
                    if si == len(stack) - 1:
                        clear[block] = True
            for b in blocks:
                clear.setdefault(b, False)
            for b1 in blocks:
                for b2 in blocks:
                    on.setdefault((b1, b2), False)
                for c in columns:
                    inColumn.setdefault((b1, c), False)
            rightOf, leftOf = {}, {}
            for i, c1 in enumerate(columns):
                for j, c2 in enumerate(columns):
                    if i > j:
                        rightOf[(c1, c2)] = True
                        leftOf[(c2, c1)] = True
                    else:
                        rightOf[(c1, c2)] = False
                        leftOf[(c2, c1)] = False
            states.append(BlocksworldState(
                blocks=blocks, columns=columns,
                on=on, inColumn=inColumn, clear=clear,
                rightOf=rightOf, leftOf=leftOf,
            ))
    return states


def state_to_labels(state, canonical_preds):
    true_preds = state.get_predicates()
    labels = torch.zeros(len(canonical_preds), dtype=torch.float32)
    for i, gp in enumerate(canonical_preds):
        if gp in true_preds:
            labels[i] = 1.0
    return labels


# =========================================================================
# Rendering & feature extraction (reuse from train_viplan_dinov3)
# =========================================================================

def render_states(states, out_dir, n_views=N_VIEWS):
    """Render states via Blender subprocess.

    Uses render_bpy_direct.py which accepts: states_pkl output_dir --views N
    """
    import subprocess
    import pickle
    import tempfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.png"))
    if len(existing) >= len(states) * n_views:
        return

    # Write all state matrices to a temp pickle file
    matrices = [s.to_numpy() for s in states]
    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    pickle.dump(matrices, tmp)
    tmp.close()

    try:
        cmd = [
            sys.executable, str(RENDER_SCRIPT),
            tmp.name, str(out_dir),
            "--views", str(n_views),
        ]
        print(f"    Rendering {len(states)} states ({n_views} views each)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            print(f"    Render stderr: {result.stderr[:500]}")
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                if "PROGRESS" in line or "DONE" in line or "FAIL" in line:
                    print(f"    {line}")
    finally:
        os.unlink(tmp.name)


def extract_features(images_dir, encoder, n_states, n_views, n_augs, device="cuda"):
    """Extract DINOv3 features from rendered images with augmentation.

    Supports two filename formats:
      - state_XXXX_vY.png  (multi-view)
      - state_XXXX_view_Y.png  (multi-view, old format)
      - state_XXXX.png  (single-view)

    For each state × view, generates n_augs augmented versions.
    Total features: n_states × n_views × n_augs
    """
    from torchvision import transforms
    from PIL import Image
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    base_tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])
    aug_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(), normalize,
    ])

    images_dir = Path(images_dir)
    encoder.eval()

    all_features = []
    with torch.no_grad():
        for si in range(n_states):
            for vi in range(n_views):
                # Try multiple filename formats
                candidates = [
                    images_dir / f"state_{si:05d}_v{vi}.png",
                    images_dir / f"state_{si:04d}_v{vi}.png",
                    images_dir / f"state_{si:05d}_view_{vi}.png",
                    images_dir / f"state_{si:04d}_view_{vi}.png",
                ]
                if vi == 0:
                    candidates.append(images_dir / f"state_{si:05d}.png")
                    candidates.append(images_dir / f"state_{si:04d}.png")

                img_path = None
                for c in candidates:
                    if c.exists():
                        img_path = c
                        break

                if img_path is None:
                    # Try glob fallback
                    matches = list(images_dir.glob(f"state_{si:05d}*")) or \
                              list(images_dir.glob(f"state_{si:04d}*"))
                    if matches:
                        img_path = matches[0]
                    else:
                        continue

                img = Image.open(img_path).convert("RGB")

                # Base feature
                t = base_tf(img).unsqueeze(0).to(device)
                feat = encoder(t).cpu().squeeze(0)
                all_features.append(feat)

                # Augmented features
                for _ in range(n_augs - 1):
                    t = aug_tf(img).unsqueeze(0).to(device)
                    feat = encoder(t).cpu().squeeze(0)
                    all_features.append(feat)

    result = torch.stack(all_features)
    print(f"  Extracted {result.shape[0]} features from {images_dir} "
          f"({result.shape[0]}/{n_states * n_views * n_augs} expected)")
    return result


# =========================================================================
# DINOv3 encoder
# =========================================================================

class DINOv3ViTHPlus(nn.Module):
    def __init__(self, ckpt_path, d_out=256):
        super().__init__()
        from dinov3.models.vision_transformer import DinoVisionTransformer
        self.backbone = DinoVisionTransformer(
            img_size=224, patch_size=16, in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1280, depth=32, num_heads=20,
            ffn_ratio=6.0, qkv_bias=True, drop_path_rate=0.0,
            layerscale_init=1e-5, norm_layer="layernormbf16",
            ffn_layer="swiglu", ffn_bias=True, proj_bias=True,
            n_storage_tokens=4, mask_k_bias=True,
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(sd, strict=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(1280, d_out)

    @torch.no_grad()
    def forward(self, x):
        out = self.backbone.forward_features(x)
        patches = out["x_norm_patchtokens"] if isinstance(out, dict) else out[:, 1:, :]
        return self.proj(patches.float())


# =========================================================================
# Transition data generation
# =========================================================================

def generate_transitions(states, blocks, columns, include_action: bool = False):
    """Generate valid (state_t, state_t1) pairs from moveBlock actions."""
    transitions = []
    for state in states:
        for block in blocks:
            if not state.clear.get(block, False):
                continue
            cur_col = None
            for c in columns:
                if state.inColumn.get((block, c), False):
                    cur_col = c
                    break
            for target_col in columns:
                if target_col == cur_col:
                    continue
                new_on = dict(state.on)
                new_inColumn = dict(state.inColumn)
                new_clear = dict(state.clear)
                for b2 in blocks:
                    if new_on.get((block, b2), False):
                        new_on[(block, b2)] = False
                        new_clear[b2] = True
                if cur_col:
                    new_inColumn[(block, cur_col)] = False
                top_block = None
                col_blocks = [b for b in blocks
                              if new_inColumn.get((b, target_col), False) and b != block]
                for b in col_blocks:
                    is_top = True
                    for b2 in col_blocks:
                        if b2 != b and new_on.get((b2, b), False):
                            is_top = False
                            break
                    if is_top:
                        top_block = b
                        break
                if top_block is not None:
                    new_on[(block, top_block)] = True
                    new_clear[top_block] = False
                new_inColumn[(block, target_col)] = True
                new_clear[block] = True
                new_state = BlocksworldState(
                    blocks=blocks, columns=columns,
                    on=new_on, inColumn=new_inColumn, clear=new_clear,
                    rightOf=dict(state.rightOf), leftOf=dict(state.leftOf),
                )
                if include_action:
                    transitions.append((state, new_state, f"moveBlock({block}, {target_col})"))
                else:
                    transitions.append((state, new_state))
    return transitions


def generate_random_pairs(states, n_pairs, rng=None):
    """Generate random (state_t, state_t1) pairs — no action structure."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(states)
    pairs = []
    for _ in range(n_pairs):
        i, j = rng.integers(0, n, size=2)
        while i == j:
            j = rng.integers(0, n)
        pairs.append((states[i], states[j]))
    return pairs


def _build_state_index(states):
    """Build a dict from predicate-frozenset -> state index."""
    idx = {}
    for i, s in enumerate(states):
        key = frozenset(s.get_predicates())
        if key not in idx:
            idx[key] = i
    return idx


def _compute_pos_weight(labels: torch.Tensor, max_weight: float = 20.0) -> torch.Tensor:
    """Per-atom positive weights for imbalanced closed-world predicate labels."""
    labels = labels.float()
    valid = labels >= 0
    pos = ((labels == 1) & valid).sum(dim=0).float()
    neg = ((labels == 0) & valid).sum(dim=0).float()
    pos_weight = neg / pos.clamp_min(1.0)
    pos_weight[pos == 0] = 1.0
    return pos_weight.clamp(min=0.25, max=max_weight)


def _build_object_slot_init(domain_info, d_slot: int, device: str | torch.device) -> torch.Tensor:
    """Stable object-slot initialization keyed by object identity.

    The scoring head assumes a deterministic object-slot order. Random slot
    initialization makes that binding unstable, so each slot gets a small,
    fixed identity/type/color cue before attention updates it from image tokens.
    """
    init = torch.zeros(domain_info.n_objects, d_slot, device=device)
    block_colors = {
        "Y": (1.0, 1.0, 0.0),
        "P": (0.8, 0.0, 0.5),
        "R": (1.0, 0.0, 0.0),
        "O": (1.0, 0.5, 0.0),
        "G": (0.0, 0.8, 0.0),
        "B": (0.0, 0.0, 1.0),
    }
    columns = [o.name for o in domain_info.objects if o.type_name == "column"]
    col_to_pos = {
        c: (i / max(len(columns) - 1, 1)) * 2.0 - 1.0
        for i, c in enumerate(columns)
    }

    for i, obj in enumerate(domain_info.objects):
        if obj.type_name == "block":
            rgb = block_colors.get(obj.name.upper(), (0.5, 0.5, 0.5))
            init[i, 0:3] = torch.tensor(rgb, device=device)
            init[i, 3] = 1.0
        elif obj.type_name == "column":
            init[i, 0] = col_to_pos.get(obj.name, 0.0)
            init[i, 4] = 1.0
        identity_dim = 8 + i
        if identity_dim < d_slot:
            init[i, identity_dim] = 1.0
    return init


def _parse_k_values(k_values: str | list[int] | tuple[int, ...]) -> list[int]:
    """Parse a comma-separated K sweep or normalize an integer list."""
    if isinstance(k_values, (list, tuple)):
        parsed = [int(k) for k in k_values]
    else:
        parsed = [int(x.strip()) for x in str(k_values).split(",") if x.strip()]
    parsed = [k for k in parsed if k > 0]
    if not parsed:
        raise ValueError("k_values must contain at least one positive integer")
    return parsed


def _parse_conditions(conditions: str | list[str] | tuple[str, ...] | None) -> list[str] | None:
    """Parse a comma-separated condition list."""
    if conditions is None:
        return None
    valid = {"static", "random_pairs", "adjacent", "full"}
    if isinstance(conditions, (list, tuple)):
        parsed = [str(c).strip() for c in conditions if str(c).strip()]
    else:
        parsed = [c.strip() for c in str(conditions).split(",") if c.strip()]
    unknown = [c for c in parsed if c not in valid]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}. Valid choices: {sorted(valid)}")
    return parsed or None


def _build_fewshot_state_dataset(
    all_state_features,
    all_state_labels,
    type_ids,
    k: int,
    fewshot_unit: str = "image",
    seed: int = 42,
    feat_per_state: int | None = None,
    feature_state_ids=None,
):
    """Sample a few-shot training dataset and return metadata.

    Args:
        fewshot_unit:
            - "image": sample K labeled visual samples/features directly.
            - "state": sample K symbolic states and keep all features per state.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    features = all_state_features
    labels = all_state_labels
    if isinstance(features, np.ndarray):
        n_items = features.shape[0]
    else:
        n_items = len(features)

    rng = np.random.default_rng(seed)
    covered_state_ids = None

    if fewshot_unit == "image":
        sample_size = min(k, n_items)
        selected_indices = np.sort(rng.choice(n_items, size=sample_size, replace=False))
        if feature_state_ids is not None:
            feature_state_ids = np.asarray(feature_state_ids)
            covered_state_ids = np.unique(feature_state_ids[selected_indices])
        elif feat_per_state:
            covered_state_ids = np.unique(selected_indices // int(feat_per_state))
    elif fewshot_unit == "state":
        if feature_state_ids is not None:
            feature_state_ids = np.asarray(feature_state_ids)
            unique_states = np.unique(feature_state_ids)
            sample_size = min(k, len(unique_states))
            chosen_states = np.sort(rng.choice(unique_states, size=sample_size, replace=False))
            mask = np.isin(feature_state_ids, chosen_states)
            selected_indices = np.flatnonzero(mask)
            covered_state_ids = chosen_states
        else:
            if not feat_per_state:
                raise ValueError(
                    "feat_per_state is required when feature_state_ids is not provided "
                    "for fewshot_unit='state'"
                )
            n_states = n_items // int(feat_per_state)
            sample_size = min(k, n_states)
            chosen_states = np.sort(rng.choice(n_states, size=sample_size, replace=False))
            blocks = [
                np.arange(si * int(feat_per_state), (si + 1) * int(feat_per_state))
                for si in chosen_states
            ]
            selected_indices = np.concatenate(blocks) if blocks else np.array([], dtype=np.int64)
            covered_state_ids = chosen_states
    else:
        raise ValueError(f"Unknown fewshot_unit: {fewshot_unit}")

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    indexer = (
        torch.as_tensor(selected_indices, dtype=torch.long)
        if isinstance(features, torch.Tensor) else selected_indices
    )
    selected_features = features[indexer]
    selected_labels = labels[indexer]
    dataset = StateDataset(selected_features, selected_labels, type_ids)

    if feature_state_ids is not None:
        feature_state_ids = np.asarray(feature_state_ids)
        n_labeled_states = int(len(np.unique(feature_state_ids[selected_indices])))
    elif covered_state_ids is not None:
        n_labeled_states = int(len(np.unique(covered_state_ids)))
    elif feat_per_state:
        n_labeled_states = int(np.ceil(len(selected_indices) / int(feat_per_state)))
    else:
        n_labeled_states = None

    metadata = {
        "k_requested": int(k),
        "fewshot_unit": fewshot_unit,
        "n_labeled_samples": int(len(selected_indices)),
        "n_labeled_states": n_labeled_states,
        "feat_per_state": int(feat_per_state) if feat_per_state else None,
    }
    return dataset, metadata, selected_indices


def _canonical_state_atoms(state, canonical_set: set[str]) -> set[str]:
    return {a for a in state.get_predicates() if a in canonical_set}


def _masks_from_atom_sets(
    atoms_t: set[str],
    atoms_t1: set[str],
    atom_to_idx: dict[str, int],
    n_canon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    add = torch.zeros(n_canon)
    delete = torch.zeros(n_canon)
    for atom in atoms_t1 - atoms_t:
        idx = atom_to_idx.get(atom)
        if idx is not None:
            add[idx] = 1.0
    for atom in atoms_t - atoms_t1:
        idx = atom_to_idx.get(atom)
        if idx is not None:
            delete[idx] = 1.0
    frame = 1.0 - torch.clamp(add + delete, max=1.0)
    return add, delete, frame


def _label_from_atom_set(
    atoms: set[str],
    atom_to_idx: dict[str, int],
    n_canon: int,
) -> torch.Tensor:
    label = torch.zeros(n_canon)
    for atom in atoms:
        idx = atom_to_idx.get(atom)
        if idx is not None:
            label[idx] = 1.0
    return label


def _build_transition_masks(
    action_idx: int,
    action_masks: dict[str, torch.Tensor],
    mask_source: str,
    atoms_t: set[str] | None = None,
    atoms_t1: set[str] | None = None,
    atom_to_idx: dict[str, int] | None = None,
    n_canon: int | None = None,
    state_t=None,
    action_name: str | None = None,
    canonical_set: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pre/add/del/frame masks from either state diff or PDDL semantics.

    Semantics:
        state_diff:
            C-style supervision from the observed S_t -> S_t+1 difference.
        pddl:
            B-style weak supervision from the action's static compiled PDDL
            add/delete declarations only. It intentionally does not inspect
            S_t, so conditional effects remain incomplete.
        pddl_conservative:
            Same static add/delete masks as pddl, but disables frame masks.
        pddl_sim:
            Dynamic simulator supervision retained for diagnostics/backward
            compatibility. In deterministic Blocksworld this is equivalent to
            state_diff and therefore is not a valid B-vs-C comparison.
    """
    pre_mask = action_masks["precondition_mask"][action_idx].clone()

    if mask_source == "state_diff":
        if atoms_t is None or atoms_t1 is None or atom_to_idx is None or n_canon is None:
            raise ValueError("state_diff masks require atoms_t, atoms_t1, atom_to_idx, and n_canon")
        add_mask, del_mask, frame_mask = _masks_from_atom_sets(
            atoms_t, atoms_t1, atom_to_idx, n_canon,
        )
        return pre_mask, add_mask, del_mask, frame_mask

    if mask_source in ("pddl", "pddl_conservative"):
        add_mask = action_masks["add_mask"][action_idx].clone()
        del_mask = action_masks["del_mask"][action_idx].clone()
        if mask_source == "pddl_conservative":
            frame_mask = torch.zeros_like(add_mask)
        else:
            frame_mask = action_masks["frame_mask"][action_idx].clone()
        return pre_mask, add_mask, del_mask, frame_mask

    if mask_source == "pddl_sim":
        # Dynamically simulate the action to capture conditional effects
        if state_t is not None and action_name is not None and canonical_set is not None:
            sim_t1 = _simulate_move_atoms(state_t, action_name, canonical_set)
            if atoms_t is None:
                atoms_t = _canonical_state_atoms(state_t, canonical_set)
            if atom_to_idx is None or n_canon is None:
                raise ValueError("pddl_sim requires atom_to_idx and n_canon")
            add_mask, del_mask, frame_mask = _masks_from_atom_sets(
                atoms_t, sim_t1, atom_to_idx, n_canon,
            )
        else:
            # Fallback for non-Blocksworld callers that have no simulator hook.
            add_mask = action_masks["add_mask"][action_idx].clone()
            del_mask = action_masks["del_mask"][action_idx].clone()
            frame_mask = action_masks["frame_mask"][action_idx].clone()
        return pre_mask, add_mask, del_mask, frame_mask

    raise ValueError(f"Unknown transition mask source: {mask_source}")


def _build_negative_action_masks(
    neg_action_idx: int,
    state_t,
    action_masks: dict[str, torch.Tensor],
    domain_info,
    mask_source: str,
    atoms_t: set[str],
    canonical_set: set[str],
    atom_to_idx: dict[str, int],
    n_canon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build masks for counterfactual negative actions without mixing B/C modes."""
    neg_pre = action_masks["precondition_mask"][neg_action_idx].clone()
    neg_action_name = domain_info.action_semantics[neg_action_idx].action_name

    if mask_source == "state_diff":
        neg_atoms_t1 = _simulate_move_atoms(
            state_t, neg_action_name, canonical_set,
        )
        neg_add, neg_del, neg_frame = _masks_from_atom_sets(
            atoms_t, neg_atoms_t1, atom_to_idx, n_canon,
        )
        return neg_pre, neg_add, neg_del, neg_frame

    if mask_source in ("pddl", "pddl_conservative"):
        neg_add = action_masks["add_mask"][neg_action_idx].clone()
        neg_del = action_masks["del_mask"][neg_action_idx].clone()
        neg_frame = action_masks["frame_mask"][neg_action_idx].clone()
        if mask_source == "pddl_conservative":
            neg_frame = torch.zeros_like(neg_frame)
        return neg_pre, neg_add, neg_del, neg_frame

    if mask_source == "pddl_sim":
        # Dynamically simulate to capture conditional effects
        neg_atoms_t1 = _simulate_move_atoms(
            state_t, neg_action_name, canonical_set,
        )
        neg_add, neg_del, neg_frame = _masks_from_atom_sets(
            atoms_t, neg_atoms_t1, atom_to_idx, n_canon,
        )
        return neg_pre, neg_add, neg_del, neg_frame

    raise ValueError(f"Unknown transition mask source: {mask_source}")


def _parse_move_action(action_name: str) -> tuple[str, str] | None:
    match = re.search(r"moveblock\(([^,]+),\s*([^)]+)\)", action_name, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().upper(), match.group(2).strip().upper()


def _simulate_move_atoms(state, action_name: str, canonical_set: set[str]) -> set[str]:
    parsed = _parse_move_action(action_name)
    if parsed is None:
        return _canonical_state_atoms(state, canonical_set)
    block, target_col = parsed
    if not state.clear.get(block, False):
        return _canonical_state_atoms(state, canonical_set)

    cur_col = None
    for c in state.columns:
        if state.inColumn.get((block, c), False):
            cur_col = c
            break
    if target_col == cur_col:
        return _canonical_state_atoms(state, canonical_set)

    new_on = dict(state.on)
    new_in_column = dict(state.inColumn)
    new_clear = dict(state.clear)

    for b2 in state.blocks:
        if new_on.get((block, b2), False):
            new_on[(block, b2)] = False
            new_clear[b2] = True

    if cur_col:
        new_in_column[(block, cur_col)] = False

    top_block = None
    col_blocks = [
        b for b in state.blocks
        if new_in_column.get((b, target_col), False) and b != block
    ]
    for b in col_blocks:
        is_top = True
        for b2 in col_blocks:
            if b2 != b and new_on.get((b2, b), False):
                is_top = False
                break
        if is_top:
            top_block = b
            break

    if top_block is not None:
        new_on[(block, top_block)] = True
        new_clear[top_block] = False

    new_in_column[(block, target_col)] = True
    new_clear[block] = True

    next_state = BlocksworldState(
        blocks=state.blocks,
        columns=state.columns,
        on=new_on,
        inColumn=new_in_column,
        clear=new_clear,
        rightOf=dict(state.rightOf),
        leftOf=dict(state.leftOf),
    )
    return _canonical_state_atoms(next_state, canonical_set)


def build_transition_features(
    transitions, states, all_features, n_views, n_augs,
    canonical_preds, domain_info, n_negatives=3,
    mask_source="state_diff",
):
    """Build transition dataset features with action masks."""
    if mask_source not in {"state_diff", "pddl", "pddl_conservative", "pddl_sim"}:
        raise ValueError(f"Unknown transition mask source: {mask_source}")
    state_to_idx = _build_state_index(states)
    feat_per_state = n_views * n_augs
    action_masks = domain_info.get_action_masks_tensor(device="cpu")
    atom_to_idx = {a.str_repr: i for i, a in enumerate(domain_info.canonical_atoms)}
    canonical_set = set(canonical_preds)

    n_canon = len(canonical_preds)

    features_t, features_t1 = [], []
    feature_idx_t, feature_idx_t1 = [], []
    action_indices = []
    pre_masks, add_masks, del_masks, frame_masks = [], [], [], []
    neg_pre_l, neg_add_l, neg_del_l = [], [], []
    state_labels_t, state_labels_t1 = [], []

    for transition in transitions:
        if len(transition) == 2:
            state_t, state_t1 = transition
            action_hint = None
        elif len(transition) >= 3:
            state_t, state_t1, action_hint = transition[:3]
        else:
            continue

        t_key = frozenset(state_t.get_predicates())
        t1_key = frozenset(state_t1.get_predicates())
        t_idx = state_to_idx.get(t_key)
        t1_idx = state_to_idx.get(t1_key)
        if t_idx is None or t1_idx is None:
            continue

        aug_t = t_idx * feat_per_state + np.random.randint(feat_per_state)
        aug_t1 = t1_idx * feat_per_state + np.random.randint(feat_per_state)
        feature_idx_t.append(aug_t)
        feature_idx_t1.append(aug_t1)
        features_t.append(all_features[aug_t])
        features_t1.append(all_features[aug_t1])

        # Match action
        t_preds = _canonical_state_atoms(state_t, canonical_set)
        t1_preds = _canonical_state_atoms(state_t1, canonical_set)
        state_labels_t.append(_label_from_atom_set(t_preds, atom_to_idx, n_canon))
        state_labels_t1.append(_label_from_atom_set(t1_preds, atom_to_idx, n_canon))
        added = t1_preds - t_preds
        deleted = t_preds - t1_preds

        if action_hint is not None:
            matched_action = _match_action_hint(action_hint, domain_info)
        else:
            matched_action = _match_action(
                added, deleted, canonical_preds, atom_to_idx, domain_info, n_canon,
            )
        action_name = domain_info.action_semantics[matched_action].action_name
        pre_m, add_m, del_m, frame_m = _build_transition_masks(
            action_idx=matched_action,
            action_masks=action_masks,
            mask_source=mask_source,
            atoms_t=t_preds,
            atoms_t1=t1_preds,
            atom_to_idx=atom_to_idx,
            n_canon=n_canon,
            state_t=state_t,
            action_name=action_name,
            canonical_set=canonical_set,
        )
        action_indices.append(matched_action)
        pre_masks.append(pre_m)
        add_masks.append(add_m)
        del_masks.append(del_m)
        frame_masks.append(frame_m)

        # Negative actions
        n_act = len(domain_info.action_semantics)
        candidates = [a for a in range(n_act) if a != matched_action]
        n_neg = min(n_negatives, len(candidates))
        neg_indices = np.random.choice(candidates, size=n_neg, replace=False) if candidates else [0]

        np_pre = torch.zeros(n_negatives, n_canon)
        np_add = torch.zeros(n_negatives, n_canon)
        np_del = torch.zeros(n_negatives, n_canon)
        for k, ni in enumerate(neg_indices[:n_negatives]):
            neg_pre, neg_add, neg_del, _ = _build_negative_action_masks(
                neg_action_idx=ni,
                state_t=state_t,
                action_masks=action_masks,
                domain_info=domain_info,
                mask_source=mask_source,
                atoms_t=t_preds,
                canonical_set=canonical_set,
                atom_to_idx=atom_to_idx,
                n_canon=n_canon,
            )
            np_pre[k] = neg_pre
            np_add[k] = neg_add
            np_del[k] = neg_del
        neg_pre_l.append(np_pre)
        neg_add_l.append(np_add)
        neg_del_l.append(np_del)

    pre_stack = torch.stack(pre_masks)
    add_stack = torch.stack(add_masks)
    del_stack = torch.stack(del_masks)
    frame_stack = torch.stack(frame_masks)
    print(
        f"  Transition masks ({mask_source}): "
        f"pre={pre_stack.sum(dim=1).float().mean().item():.2f} "
        f"add={add_stack.sum(dim=1).float().mean().item():.2f} "
        f"del={del_stack.sum(dim=1).float().mean().item():.2f} "
        f"frame={frame_stack.sum(dim=1).float().mean().item():.2f}"
    )

    return {
        "features_t": torch.stack(features_t),
        "features_t1": torch.stack(features_t1),
        "feature_idx_t": torch.tensor(feature_idx_t, dtype=torch.long),
        "feature_idx_t1": torch.tensor(feature_idx_t1, dtype=torch.long),
        "action_idx": torch.tensor(action_indices, dtype=torch.long),
        "pre_mask": pre_stack,
        "add_mask": add_stack,
        "del_mask": del_stack,
        "frame_mask": frame_stack,
        "neg_pre_masks": torch.stack(neg_pre_l),
        "neg_add_masks": torch.stack(neg_add_l),
        "neg_del_masks": torch.stack(neg_del_l),
        "state_label_t": torch.stack(state_labels_t),
        "state_label_t1": torch.stack(state_labels_t1),
    }


def _match_action(added, deleted, canonical_preds, atom_to_idx, domain_info, n_canon):
    """Find the action whose effects match the observed add/del."""
    for atom in added:
        parts = atom.strip("()").split()
        if len(parts) == 3 and parts[0].lower() == "incolumn":
            block, col = parts[1].upper(), parts[2]
            for ai, asem in enumerate(domain_info.action_semantics):
                parsed = _parse_move_action(asem.action_name)
                if parsed == (block, col):
                    return ai

    for ai, asem in enumerate(domain_info.action_semantics):
        eff_add_strs = {canonical_preds[j] for j in range(n_canon) if asem.add_mask[j] > 0}
        eff_del_strs = {canonical_preds[j] for j in range(n_canon) if asem.del_mask[j] > 0}
        if added == eff_add_strs and deleted == eff_del_strs:
            return ai

    best_score, best_ai = -1, 0
    for ai, asem in enumerate(domain_info.action_semantics):
        mc = 0
        for atom_str in added:
            if atom_str in atom_to_idx and asem.add_mask[atom_to_idx[atom_str]] > 0:
                mc += 1
        for atom_str in deleted:
            if atom_str in atom_to_idx and asem.del_mask[atom_to_idx[atom_str]] > 0:
                mc += 1
        if mc > best_score:
            best_score, best_ai = mc, ai
    return best_ai


def _match_action_hint(action_hint, domain_info):
    """Map an explicit generated action label to a grounded action index."""
    parsed_hint = _parse_move_action(str(action_hint))
    if parsed_hint is not None:
        for ai, asem in enumerate(domain_info.action_semantics):
            if _parse_move_action(asem.action_name) == parsed_hint:
                return ai
    hint = str(action_hint)
    for ai, asem in enumerate(domain_info.action_semantics):
        if asem.action_name == hint:
            return ai
    raise ValueError(f"Could not match transition action hint: {action_hint}")


# =========================================================================
# Training — single condition
# =========================================================================

def train_condition(
    condition: str,
    domain_info,
    state_train_ds: StateDataset,
    state_val_ds: StateDataset,
    state_test_ds: StateDataset,
    trans_ds: TransitionDataset | None,
    device: str = "cuda",
    n_epochs: int = 100,
    lr: float = 1e-4,
    batch_size: int = 32,
    d_slot: int = 256,
    n_object_slots: int = 16,
    n_slot_iters: int = 3,
    w_seed: float = 1.0,
    w_type: float = 0.3,
    w_contrast: float = 0.1,
    w_equiv: float = 1.0,
    w_cf: float = 0.3,
    w_transition_seed: float = 0.0,
    w_transition_type: float = 0.0,
    w_transition_contrast: float = 0.0,
    contrast_temperature: float = 0.5,
    eval_every: int = 5,
    use_real_encoder: bool = True,
    visual_encoder=None,
    dinov3_kwargs: dict | None = None,
    transition_mask_source: str = "state_diff",
    direct_object_tokens: bool = False,
    transition_warmup_epochs: int = 0,
    pos_weight_max: float = 20.0,
    scoring_head_type: str = "film",
    train_seed: int | None = None,
) -> dict:
    """Train one experimental condition.

    Conditions:
        'static'      — C1 only (pointwise correctness)
        'random_pairs' — C1 + frame consistency on random pairs (no action structure)
        'adjacent'     — C1 + C2 (pointwise + action equivariance)
        'full'         — C1 + C2 + C3 (pointwise + equivariance + counterfactual)
    """
    if train_seed is not None:
        torch.manual_seed(train_seed)
        np.random.seed(train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_seed)

    # Match n_object_slots to actual domain objects
    effective_n_slots = domain_info.n_objects
    model_kwargs = dict(
        n_object_slots=effective_n_slots, d_slot=d_slot,
        n_slot_iters=n_slot_iters,
        use_real_encoder=False,  # Always False: features are pre-extracted
        predict_slot_types=True,
        direct_object_tokens=direct_object_tokens,
        scoring_head_type=scoring_head_type,
    )
    if visual_encoder is not None:
        model_kwargs["visual_encoder"] = visual_encoder
    # Note: dinov3_kwargs ignored — features are pre-extracted, no encoder needed

    model = PaQModel.from_domain_info(domain_info, **model_kwargs).to(device)
    with torch.no_grad():
        model.object_slot_init.copy_(
            _build_object_slot_init(domain_info, d_slot=d_slot, device=device)
        )

    # Loss functions
    pos_weight = _compute_pos_weight(
        state_train_ds.state_labels, max_weight=pos_weight_max,
    ).to(device)
    loss_seed = PredicateStateLoss(pos_weight=pos_weight)
    loss_contrast = PredicateContrastiveLoss(temperature=contrast_temperature)
    loss_equiv = ActionEquivarianceLoss(
        w_pre=0.5, w_eff=1.0, w_frame=0.5,
    )
    loss_cf = CounterfactualDiscriminabilityLoss(margin=1.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    state_loader = DataLoader(
        state_train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_state_batch, num_workers=0,
    )
    trans_loader = DataLoader(
        trans_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_trans_batch, num_workers=0,
    ) if trans_ds and len(trans_ds) > 0 else None

    type_ids_tensor = torch.tensor(domain_info.obj_type_ids, dtype=torch.long, device=device)

    use_equiv = condition in ("adjacent", "full", "random_pairs")
    use_cf = condition == "full"

    history = []
    best_val_f1 = 0.0
    best_threshold = 0.5
    best_state = None

    condition_desc = {
        "static": "C1: Pointwise Correctness only",
        "random_pairs": "C1 + random pair consistency (no action structure)",
        "adjacent": "C1 + C2: Pointwise + Action Equivariance",
        "full": "C1 + C2 + C3: Pointwise + Equivariance + Counterfactual",
    }

    print(f"\n{'='*60}")
    print(f"Condition: {condition}")
    print(f"  {condition_desc[condition]}")
    print(f"  State samples: {len(state_train_ds)}")
    print(f"  Transition samples: {len(trans_ds) if trans_ds else 0}")
    print(f"  Model params: {model.count_parameters()}")
    print(
        f"  Seed pos_weight: mean={pos_weight.mean().item():.2f} "
        f"min={pos_weight.min().item():.2f} max={pos_weight.max().item():.2f}"
    )
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = defaultdict(float)
        n_steps = 0

        # ---- State batch: always active (C1) ----
        for batch in state_loader:
            feats = batch["features"].to(device)
            labels = batch["state_labels"].to(device)
            obj_types = type_ids_tensor.unsqueeze(0).expand(feats.shape[0], -1)

            optimizer.zero_grad()
            out = model(feats, object_type_ids=obj_types)
            scores = out["canonical_scores"]

            L_seed = loss_seed(scores, labels)
            L_contrast = loss_contrast(out["predicate_slots"], out["predicate_queries"])
            loss = w_seed * L_seed + w_contrast * L_contrast

            if "type_logits" in out:
                L_type = model.compute_type_loss(obj_types, forward_output=out)
                loss = loss + w_type * L_type
                epoch_losses["type"] += L_type.item()

            epoch_losses["seed"] += L_seed.item()
            epoch_losses["contrast"] += L_contrast.item()

            loss.backward()
            optimizer.step()
            n_steps += 1

        # ---- Transition batch: active for adjacent, full, random_pairs ----
        transition_active = (epoch + 1) > transition_warmup_epochs
        if trans_loader is not None and use_equiv and transition_active:
            for batch in trans_loader:
                feats_t = batch["features_t"].to(device)
                feats_t1 = batch["features_t1"].to(device)
                obj_types = type_ids_tensor.unsqueeze(0).expand(feats_t.shape[0], -1)

                optimizer.zero_grad()

                # CRITICAL: predicate bottleneck — same model, no visual fusion
                out_t = model(feats_t, object_type_ids=obj_types)
                out_t1 = model(feats_t1, object_type_ids=obj_types)

                scores_t = out_t["canonical_scores"]
                scores_t1 = out_t1["canonical_scores"]
                probs_t = torch.sigmoid(scores_t)
                probs_t1 = torch.sigmoid(scores_t1)

                pre = batch["pre_mask"].to(device)
                add = batch["add_mask"].to(device)
                dl = batch["del_mask"].to(device)
                frame = batch["frame_mask"].to(device)

                loss = torch.tensor(0.0, device=device)

                if w_transition_seed > 0 and "state_label_t" in batch and "state_label_t1" in batch:
                    labels_t = batch["state_label_t"].to(device)
                    labels_t1 = batch["state_label_t1"].to(device)
                    L_trans_seed = 0.5 * (
                        loss_seed(scores_t, labels_t) +
                        loss_seed(scores_t1, labels_t1)
                    )
                    loss = loss + w_transition_seed * L_trans_seed
                    epoch_losses["seed_trans"] += L_trans_seed.item()

                if w_transition_type > 0 and "type_logits" in out_t and "type_logits" in out_t1:
                    L_trans_type = 0.5 * (
                        model.compute_type_loss(obj_types, forward_output=out_t) +
                        model.compute_type_loss(obj_types, forward_output=out_t1)
                    )
                    loss = loss + w_transition_type * L_trans_type
                    epoch_losses["type_trans"] += L_trans_type.item()

                if w_transition_contrast > 0:
                    L_trans_contrast = 0.5 * (
                        loss_contrast(out_t["predicate_slots"], out_t["predicate_queries"]) +
                        loss_contrast(out_t1["predicate_slots"], out_t1["predicate_queries"])
                    )
                    loss = loss + w_transition_contrast * L_trans_contrast
                    epoch_losses["contrast_trans"] += L_trans_contrast.item()

                if condition == "random_pairs":
                    # Random pairs: only frame consistency, no action-specific masks
                    # This tests whether just "seeing more image pairs" helps
                    # without action structure
                    L_frame = torch.tensor(0.0, device=device)
                    if frame.sum() > 0:
                        diff = (probs_t1 - probs_t).abs()
                        # For random pairs, just penalize overall stability
                        # (weaker signal than action-conditioned)
                        L_frame = diff.mean()
                    loss = loss + w_equiv * L_frame
                    epoch_losses["frame"] += L_frame.item()

                elif condition in ("adjacent", "full"):
                    # C2: Action equivariance — Γ_a consistency
                    equiv_result = loss_equiv(
                        scores_t, scores_t1, pre, add, dl, frame,
                    )
                    L_equiv = equiv_result["total"]
                    loss = loss + w_equiv * L_equiv
                    epoch_losses["equiv"] += L_equiv.item()
                    epoch_losses["pre"] += equiv_result["L_pre"].item()
                    epoch_losses["eff"] += equiv_result["L_eff"].item()
                    epoch_losses["frame"] += equiv_result["L_frame"].item()

                    # C3: Counterfactual discrimination (full only)
                    if use_cf and "neg_pre_masks" in batch:
                        neg_pre = batch["neg_pre_masks"].to(device)
                        neg_add = batch["neg_add_masks"].to(device)
                        neg_del = batch["neg_del_masks"].to(device)
                        cf_result = loss_cf(
                            probs_t, probs_t1,
                            pre, add, dl,
                            neg_pre, neg_add, neg_del,
                        )
                        L_cf = cf_result["total"]
                        loss = loss + w_cf * L_cf
                        epoch_losses["cf"] += L_cf.item()
                        epoch_losses["cf_viol"] += cf_result["violation_rate"].item()

                loss.backward()
                optimizer.step()
                n_steps += 1

        scheduler.step()

        # Average losses
        for k in list(epoch_losses.keys()):
            epoch_losses[k] /= max(n_steps, 1)

        # ---- Evaluation ----
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            metrics = evaluate(model, state_val_ds, device, domain_info, threshold=None)

            history.append({
                "epoch": epoch + 1,
                **{f"loss_{k}": v for k, v in epoch_losses.items()},
                "val_f1": metrics["f1"],
                "val_type_acc": metrics["type_acc"],
            })

            loss_str = " ".join(f"{k}={v:.3f}" for k, v in epoch_losses.items())
            print(
                f"  [{condition}] Epoch {epoch+1:3d}/{n_epochs} | "
                f"{loss_str} | "
                f"F1={metrics['f1']:.3f} "
                f"MacroF1={metrics['macro_f1']:.3f} "
                f"TypeAcc={metrics['type_acc']:.3f} "
                f"Pred+={metrics['pred_pos_rate']:.3f} "
                f"Thr={metrics['threshold']:.2f}"
            )

            if metrics["f1"] > best_val_f1:
                best_val_f1 = metrics["f1"]
                best_threshold = metrics["threshold"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            loss_str = " ".join(f"{k}={v:.3f}" for k, v in epoch_losses.items())
            print(f"  [{condition}] Epoch {epoch+1:3d}/{n_epochs} | {loss_str}")

    # Final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_metrics = evaluate(model, state_test_ds, device, domain_info, threshold=best_threshold)

    return {
        "condition": condition,
        "description": condition_desc[condition],
        "transition_mask_source": transition_mask_source,
        "test": test_metrics,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "history": history,
        "model_state": best_state,
    }


# =========================================================================
# Evaluation
# =========================================================================

@torch.no_grad()
def evaluate(model, dataset, device, domain_info, use_predicted_types=False, threshold=0.5):
    """Evaluate model on a state dataset.

    Note: Scoring always uses oracle type_ids for consistent canonical dimensions.
    The use_predicted_types flag controls whether type accuracy is measured.
    Full predicted-type scoring requires scoring_head changes (separate task).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_state_batch)
    type_ids_tensor = torch.tensor(domain_info.obj_type_ids, dtype=torch.long, device=device)
    all_probs, all_labels = [], []
    all_type_preds, all_type_labels = [], []

    for batch in loader:
        feats = batch["features"].to(device)
        labels = batch["state_labels"]
        obj_types = type_ids_tensor.unsqueeze(0).expand(feats.shape[0], -1)

        # Always use oracle types for scoring (canonical dimensions must match)
        out = model(feats, object_type_ids=obj_types)

        probs = torch.sigmoid(out["canonical_scores"]).cpu()
        all_probs.append(probs)
        all_labels.append(labels.long())

        if "predicted_type_ids" in out:
            all_type_preds.append(out["predicted_type_ids"].cpu())
            all_type_labels.append(obj_types.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    def _score_at(thr: float):
        preds = (all_probs >= thr).long()
        tp = (preds * all_labels).sum().float()
        fp = (preds * (1 - all_labels)).sum().float()
        fn = ((1 - preds) * all_labels).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return preds, precision, recall, f1

    if threshold is None:
        best = None
        for thr in torch.linspace(0.05, 0.95, steps=19).tolist():
            preds_i, precision_i, recall_i, f1_i = _score_at(float(thr))
            if best is None or f1_i > best[3]:
                best = (float(thr), preds_i, precision_i, recall_i, f1_i)
        threshold, all_preds, precision, recall, f1 = best
    else:
        threshold = float(threshold)
        all_preds, precision, recall, f1 = _score_at(threshold)

    exact_match = (all_preds == all_labels).all(dim=1).float().mean()

    per_tp = (all_preds * all_labels).sum(dim=0).float()
    per_fp = (all_preds * (1 - all_labels)).sum(dim=0).float()
    per_fn = ((1 - all_preds) * all_labels).sum(dim=0).float()
    per_prec = per_tp / (per_tp + per_fp + 1e-8)
    per_rec = per_tp / (per_tp + per_fn + 1e-8)
    per_f1 = 2 * per_prec * per_rec / (per_prec + per_rec + 1e-8)
    macro_f1 = per_f1.mean()

    type_acc = 0.0
    if all_type_preds:
        type_preds = torch.cat(all_type_preds)
        type_labels = torch.cat(all_type_labels)
        type_acc = (type_preds == type_labels).float().mean().item()

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "macro_f1": macro_f1.item(),
        "exact_match": exact_match.item(),
        "type_acc": type_acc,
        "threshold": threshold,
        "pred_pos_rate": all_preds.float().mean().item(),
        "label_pos_rate": all_labels.float().mean().item(),
        "avg_prob": all_probs.mean().item(),
    }


# =========================================================================
# The Key Experiment: 4 conditions, same labeled frames
# =========================================================================

def run_structural_experiment(
    domain_info,
    state_train_ds, state_val_ds, state_test_ds,
    trans_adjacent_ds, trans_random_ds,
    device="cuda",
    n_epochs=100,
    batch_size=32,
    lr=1e-4,
    d_slot=256,
    n_object_slots=16,
    dinov3_kwargs=None,
    exp_dir=None,
    conditions=None,
    transition_mask_source="state_diff",
    direct_object_tokens: bool = False,
    w_seed: float = 1.0,
    w_type: float = 0.3,
    w_contrast: float = 0.1,
    w_equiv: float = 1.0,
    w_cf: float = 0.3,
    w_transition_seed: float = 0.0,
    w_transition_type: float = 0.0,
    w_transition_contrast: float = 0.0,
    contrast_temperature: float = 0.5,
    transition_warmup_epochs: int = 0,
    pos_weight_max: float = 20.0,
    scoring_head_type: str = "film",
    train_seed: int | None = None,
):
    """Run the 4-condition structural experiment.

    All conditions use the SAME state_train_ds (same labeled frames).
    The ONLY difference is what transition data and constraints are used:

      1. Static:       no transition data
      2. Random Pairs: transition data from random pairs (no action structure)
      3. Adjacent:     transition data from adjacent actions (equivariance)
      4. Full:         adjacent + counterfactual discrimination

    This directly answers: does action structure improve grounding beyond labels?
    """
    if conditions is None:
        conditions = ["static", "random_pairs", "adjacent", "full"]
    else:
        conditions = _parse_conditions(conditions)
    trans_map = {
        "static": None,
        "random_pairs": trans_random_ds,
        "adjacent": trans_adjacent_ds,
        "full": trans_adjacent_ds,
    }

    all_results = {}

    for cond in conditions:
        result = train_condition(
            condition=cond,
            domain_info=domain_info,
            state_train_ds=state_train_ds,
            state_val_ds=state_val_ds,
            state_test_ds=state_test_ds,
            trans_ds=trans_map[cond],
            device=device,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            d_slot=d_slot,
            n_object_slots=n_object_slots,
            w_seed=w_seed,
            w_type=w_type,
            w_contrast=w_contrast,
            w_equiv=w_equiv,
            w_cf=w_cf,
            w_transition_seed=w_transition_seed,
            w_transition_type=w_transition_type,
            w_transition_contrast=w_transition_contrast,
            contrast_temperature=contrast_temperature,
            transition_warmup_epochs=transition_warmup_epochs,
            pos_weight_max=pos_weight_max,
            scoring_head_type=scoring_head_type,
            use_real_encoder=True if dinov3_kwargs else False,
            dinov3_kwargs=dinov3_kwargs,
            transition_mask_source=transition_mask_source,
            direct_object_tokens=direct_object_tokens,
            train_seed=train_seed,
        )
        all_results[cond] = result

        # Save model
        if exp_dir and result["model_state"]:
            exp_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result["model_state"], exp_dir / f"model_{cond}.pt")

    # Print comparison table
    print_comparison_table(all_results)

    # Save results
    if exp_dir:
        save_results = {}
        for cond, res in all_results.items():
            save_results[cond] = {
                "condition": cond,
                "description": res["description"],
                "transition_mask_source": res.get("transition_mask_source", transition_mask_source),
                "test": res["test"],
                "best_val_f1": res["best_val_f1"],
                "best_threshold": res.get("best_threshold", 0.5),
                "transition_mask_source": res.get("transition_mask_source", transition_mask_source),
            }
        with open(exp_dir / "structural_results.json", "w") as f:
            json.dump(save_results, f, indent=2)

    return all_results


def print_comparison_table(results: dict):
    """Print the key comparison table."""
    print()
    print("=" * 90)
    print("STRUCTURAL EXPERIMENT: Does action structure improve grounding?")
    print("  All conditions use the SAME labeled frames.")
    print("  Difference is ONLY in transition constraints.")
    print("=" * 90)
    print()
    print(f"{'Condition':<20} {'F1':>8} {'EM':>8} {'TypeAcc':>8} {'P':>8} {'R':>8}")
    print("-" * 60)
    ordered = [c for c in ["static", "random_pairs", "adjacent", "full"] if c in results]
    for cond in ordered:
        r = results[cond]
        t = r["test"]
        print(f"{cond:<20} {t['f1']:>8.3f} {t['exact_match']:>8.3f} "
              f"{t['type_acc']:>8.3f} {t['precision']:>8.3f} {t['recall']:>8.3f}")

    # Delta analysis
    if "static" in results:
        print()
        base_f1 = results["static"]["test"]["f1"]
        print(f"Improvement over Static Grounding:")
        for cond in ["random_pairs", "adjacent", "full"]:
            if cond not in results:
                continue
            f1 = results[cond]["test"]["f1"]
            delta = f1 - base_f1
            print(f"  {cond:<20} ΔF1 = {delta:+.3f}  ({delta/max(abs(base_f1),1e-6)*100:+.1f}%)")


# =========================================================================
# Few-shot sweep: same structural experiment at different K
# =========================================================================

def run_fewshot_structural(
    domain_info,
    all_state_features, all_state_labels, type_ids,
    state_val_ds, state_test_ds,
    trans_adjacent_ds, trans_random_ds,
    k_values=None,
    fewshot_unit="image",
    feat_per_state=None,
    feature_state_ids=None,
    device="cuda",
    n_epochs=60,
    lr=1e-4,
    batch_size=32,
    d_slot=256,
    dinov3_kwargs=None,
    exp_dir=None,
    conditions=None,
    transition_mask_source="state_diff",
):
    """Run the 4-condition experiment at different few-shot K values.

    By default, K means labeled visual samples/features. Pass
    fewshot_unit="state" to reproduce the older state-level setting where each
    selected symbolic state contributes all views/augmentations.
    """
    if k_values is None:
        k_values = [5, 10, 30, 50]
    k_values = _parse_k_values(k_values)

    all_k_results = {}

    for k in k_values:
        print(f"\n{'#'*60}")
        if fewshot_unit == "image":
            print(f"# Few-shot K={k} labeled visual samples/images")
        else:
            print(f"# Few-shot K_state={k} symbolic states")
        print(f"{'#'*60}")

        state_train_ds, metadata, _ = _build_fewshot_state_dataset(
            all_state_features,
            all_state_labels,
            type_ids,
            k=k,
            fewshot_unit=fewshot_unit,
            seed=42,
            feat_per_state=feat_per_state,
            feature_state_ids=feature_state_ids,
        )
        metadata["transition_mask_source"] = transition_mask_source
        metadata["conditions"] = _parse_conditions(conditions) or ["static", "random_pairs", "adjacent", "full"]
        print(
            "  Labeled state supervision: "
            f"{metadata['n_labeled_samples']} visual samples, "
            f"{metadata['n_labeled_states']} symbolic states covered"
        )

        k_results = run_structural_experiment(
            domain_info=domain_info,
            state_train_ds=state_train_ds,
            state_val_ds=state_val_ds,
            state_test_ds=state_test_ds,
            trans_adjacent_ds=trans_adjacent_ds,
            trans_random_ds=trans_random_ds,
            device=device,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            d_slot=d_slot,
            dinov3_kwargs=dinov3_kwargs,
            exp_dir=exp_dir / f"k_{k}" if exp_dir else None,
            conditions=conditions,
            transition_mask_source=transition_mask_source,
        )
        k_summary = {"_metadata": metadata}
        for cond, res in k_results.items():
            k_summary[cond] = {
                "test_f1": res["test"]["f1"],
                "test_em": res["test"]["exact_match"],
                "test_type_acc": res["test"]["type_acc"],
                "test": res["test"],
                "best_val_f1": res["best_val_f1"],
                "best_threshold": res.get("best_threshold", 0.5),
            }
        all_k_results[k] = k_summary

    # Summary table
    print()
    print("=" * 80)
    print("FEW-SHOT STRUCTURAL EXPERIMENT SUMMARY")
    print("=" * 80)
    header = f"{'K':>5} {'Unit':>8} {'Samples':>8} {'States':>8}"
    ordered_conditions = _parse_conditions(conditions) or ["static", "random_pairs", "adjacent", "full"]
    for cond in ordered_conditions:
        header += f" {cond:>14}"
    print(header)
    print("-" * len(header))

    for k in k_values:
        meta = all_k_results[k]["_metadata"]
        row = (
            f"{k:>5} {meta['fewshot_unit']:>8} "
            f"{meta['n_labeled_samples']:>8} "
            f"{str(meta['n_labeled_states']):>8}"
        )
        for cond in ordered_conditions:
            r = all_k_results[k][cond]
            row += f" {r['test_f1']:>14.3f}"
        print(row)

    if exp_dir:
        save_results = {str(k): v for k, v in all_k_results.items()}
        with open(exp_dir / "fewshot_structural.json", "w") as f:
            json.dump(save_results, f, indent=2)

    return all_k_results


# =========================================================================
# Main entry point
# =========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AE-PaQ: Action-Equivariant Visual Predicate Grounding")
    parser.add_argument("--mode", choices=["structural", "fewshot", "single"], default="structural",
                        help="Experiment mode: structural=4-condition, fewshot=K sweep, single=one condition")
    parser.add_argument("--condition", choices=["static", "random_pairs", "adjacent", "full"],
                        default="full", help="Condition for single mode")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-slot", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--mock", action="store_true", help="Use mock encoder for quick testing")
    parser.add_argument(
        "--fewshot-unit",
        choices=["image", "state"],
        default="image",
        help=(
            "Unit for --mode fewshot: image=K labeled visual samples/features; "
            "state=K symbolic states, each expanded to all views/augmentations"
        ),
    )
    parser.add_argument(
        "--k-values",
        default="5,10,30,50",
        help="Comma-separated K sweep for --mode fewshot, e.g. 5,10,30,50",
    )
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated conditions to run: static,random_pairs,adjacent,full",
    )
    parser.add_argument(
        "--transition-mask-source",
        type=str,
        default="state_diff",
        choices=["state_diff", "pddl", "pddl_conservative", "pddl_sim"],
        help=(
            "How to build add/del/frame masks: state_diff=C observed diff, "
            "pddl=static PDDL declaration masks, pddl_sim=dynamic simulator."
        ),
    )
    args = parser.parse_args()
    fewshot_k_values = _parse_k_values(args.k_values)
    selected_conditions = _parse_conditions(args.conditions)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exp_name = args.exp_name or f"aepaq_{int(time.time())}"
    exp_dir = PDDL_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"AE-PaQ Experiment: {exp_name}")
    print(f"Mode: {args.mode}, Device: {args.device}, Mock: {args.mock}")
    if args.mode == "fewshot":
        print(f"Few-shot: unit={args.fewshot_unit}, k_values={fewshot_k_values}")
    if selected_conditions:
        print(f"Conditions: {selected_conditions}")
    print(f"Transition mask source: {args.transition_mask_source}")

    # ---- Step 1: Compile PDDL domain ----
    print("\n[1/6] Compiling PDDL domain...")
    compiler = PDDLDomainCompiler(str(BWS_DOMAIN))
    domain_info = compiler.compile(
        objects={"block": BLOCKS, "column": COLUMNS},
        static_predicates=STATIC_PREDS,
    )
    print(f"  {domain_info.n_canonical} canonical atoms, {len(domain_info.action_semantics)} actions")

    with open(exp_dir / "domain_info.json", "w") as f:
        json.dump(domain_info.summary(), f, indent=2, default=str)

    # ---- Step 2: Enumerate states ----
    print("\n[2/6] Enumerating states...")
    states = enumerate_all_states(BLOCKS, COLUMNS)
    canonical_preds = domain_info.canonical_atom_strings
    print(f"  {len(states)} states")

    # ---- Step 3: Features ----
    print("\n[3/6] Preparing features...")
    feat_cache = exp_dir / "features.pt"
    dinov3_kwargs = None

    if args.mock:
        # Mock encoder: random features
        print("  Using mock encoder")
        n_patches = 16
        feat_per_state = N_VIEWS * N_AUGS
        all_features = torch.randn(len(states) * feat_per_state, n_patches, args.d_slot)
    else:
        if feat_cache.exists():
            print(f"  Loading cached features from {feat_cache}")
            all_features = torch.load(feat_cache, map_location="cpu")
        else:
            # Find existing rendered images (prefer multi-view)
            images_dir = exp_dir / "images"
            existing_dirs = sorted(PDDL_ROOT.glob("experiments/viplan_dinov3_*/images"))
            reuse_dir = None
            for d in existing_dirs:
                n_imgs = len(list(d.glob("*.png")))
                if n_imgs >= len(states) * N_VIEWS:
                    reuse_dir = d
                    break
            if reuse_dir:
                images_dir = reuse_dir
                print(f"  Reusing images from {reuse_dir}")
            else:
                print(f"  Rendering {len(states)} states...")
                render_states(states, images_dir)

            print(f"  Loading DINOv3 ViT-H+/16 encoder...")
            dinov3_enc = DINOv3ViTHPlus(DINOV3_WEIGHTS, d_out=args.d_slot).to(args.device)

            print(f"  Extracting DINOv3 features (n_views={N_VIEWS}, n_augs={N_AUGS})...")
            all_features = extract_features(
                images_dir, dinov3_enc,
                n_states=len(states), n_views=N_VIEWS, n_augs=N_AUGS,
                device=args.device,
            )
            torch.save(all_features, feat_cache)
            print(f"  Cached features to {feat_cache}")
        dinov3_kwargs = {
            "use_dinov3": True, "dinov3_source": "local",
            "dinov3_repo_dir": str(DINOV3_REPO),
            "dinov3_weights_path": DINOV3_WEIGHTS,
        }

    # ---- Step 4: Build datasets ----
    print("\n[4/6] Building datasets...")
    n_states = len(states)
    feat_per_state = N_VIEWS * N_AUGS
    all_labels = torch.stack([state_to_labels(s, canonical_preds) for s in states])
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    expanded_labels = all_labels.repeat_interleave(feat_per_state, dim=0)

    state_indices = np.random.permutation(n_states)
    n_train = int(0.7 * n_states)
    n_val = int(0.15 * n_states)

    train_idx = state_indices[:n_train]
    val_idx = state_indices[n_train:n_train + n_val]
    test_idx = state_indices[n_train + n_val:]

    def get_features(idx_array):
        feat_indices = np.concatenate([
            np.arange(si * feat_per_state, (si + 1) * feat_per_state) for si in idx_array
        ])
        return all_features[feat_indices], expanded_labels[feat_indices]

    train_feats, train_labels = get_features(train_idx)
    val_feats, val_labels = get_features(val_idx)
    test_feats, test_labels = get_features(test_idx)

    state_train_ds = StateDataset(train_feats, train_labels, type_ids)
    state_val_ds = StateDataset(val_feats, val_labels, type_ids)
    state_test_ds = StateDataset(test_feats, test_labels, type_ids)
    print(f"  State: {len(state_train_ds)} train / {len(state_val_ds)} val / {len(state_test_ds)} test")

    # ---- Build transition datasets ----
    print("  Building transitions...")
    train_states = [states[i] for i in train_idx]

    # Adjacent transitions (action-conditioned)
    adj_transitions = generate_transitions(train_states, BLOCKS, COLUMNS, include_action=True)
    print(f"  Adjacent transitions: {len(adj_transitions)}")

    if adj_transitions:
        adj_data = build_transition_features(
            adj_transitions, states, all_features, N_VIEWS, N_AUGS,
            canonical_preds, domain_info, n_negatives=3,
            mask_source=args.transition_mask_source,
        )
        trans_adjacent_ds = TransitionDataset(
            features_t=adj_data["features_t"],
            features_t1=adj_data["features_t1"],
            action_idx=adj_data["action_idx"],
            pre_masks=adj_data["pre_mask"],
            add_masks=adj_data["add_mask"],
            del_masks=adj_data["del_mask"],
            frame_masks=adj_data["frame_mask"],
            neg_pre_masks=adj_data["neg_pre_masks"],
            neg_add_masks=adj_data["neg_add_masks"],
            neg_del_masks=adj_data["neg_del_masks"],
            object_type_ids=type_ids,
            state_labels_t=adj_data.get("state_label_t"),
            state_labels_t1=adj_data.get("state_label_t1"),
        )
    else:
        trans_adjacent_ds = None

    # Random pair transitions (no action structure, same count)
    rng = np.random.default_rng(42)
    random_pairs = generate_random_pairs(train_states, len(adj_transitions), rng)
    if random_pairs:
        rand_data = build_transition_features(
            random_pairs, states, all_features, N_VIEWS, N_AUGS,
            canonical_preds, domain_info, n_negatives=3,
            mask_source=args.transition_mask_source,
        )
        trans_random_ds = TransitionDataset(
            features_t=rand_data["features_t"],
            features_t1=rand_data["features_t1"],
            action_idx=rand_data["action_idx"],
            pre_masks=rand_data["pre_mask"],
            add_masks=rand_data["add_mask"],
            del_masks=rand_data["del_mask"],
            frame_masks=rand_data["frame_mask"],
            neg_pre_masks=rand_data["neg_pre_masks"],
            neg_add_masks=rand_data["neg_add_masks"],
            neg_del_masks=rand_data["neg_del_masks"],
            object_type_ids=type_ids,
            state_labels_t=rand_data.get("state_label_t"),
            state_labels_t1=rand_data.get("state_label_t1"),
        )
    else:
        trans_random_ds = None

    # ---- Step 5: Run experiment ----
    print("\n[5/6] Running experiment...")

    common_kwargs = dict(
        domain_info=domain_info,
        state_train_ds=state_train_ds,
        state_val_ds=state_val_ds,
        state_test_ds=state_test_ds,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        d_slot=args.d_slot,
        device=args.device,
        dinov3_kwargs=dinov3_kwargs,
    )

    if args.mode == "structural":
        results = run_structural_experiment(
            trans_adjacent_ds=trans_adjacent_ds,
            trans_random_ds=trans_random_ds,
            exp_dir=exp_dir,
            conditions=selected_conditions,
            transition_mask_source=args.transition_mask_source,
            **common_kwargs,
        )

    elif args.mode == "fewshot":
        train_feature_state_ids = np.repeat(train_idx, feat_per_state)
        results = run_fewshot_structural(
            domain_info=domain_info,
            all_state_features=train_feats,
            all_state_labels=train_labels,
            type_ids=type_ids,
            state_val_ds=state_val_ds,
            state_test_ds=state_test_ds,
            trans_adjacent_ds=trans_adjacent_ds,
            trans_random_ds=trans_random_ds,
            k_values=fewshot_k_values,
            fewshot_unit=args.fewshot_unit,
            feat_per_state=feat_per_state,
            feature_state_ids=train_feature_state_ids,
            device=args.device,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            d_slot=args.d_slot,
            dinov3_kwargs=dinov3_kwargs,
            exp_dir=exp_dir,
            conditions=selected_conditions,
            transition_mask_source=args.transition_mask_source,
        )

    elif args.mode == "single":
        result = train_condition(
            condition=args.condition,
            trans_ds=trans_adjacent_ds if args.condition in ("adjacent", "full") else trans_random_ds,
            transition_mask_source=args.transition_mask_source,
            **common_kwargs,
        )
        if result["model_state"]:
            torch.save(result["model_state"], exp_dir / f"model_{args.condition}.pt")

    # ---- Step 6: Save report ----
    print("\n[6/6] Saving report...")
    report = {
        "exp_name": exp_name,
        "mode": args.mode,
        "mock_encoder": args.mock,
        "domain": domain_info.summary(),
        "data": {
            "n_states": len(states),
            "n_adjacent_transitions": len(adj_transitions),
            "n_random_pairs": len(random_pairs) if random_pairs else 0,
            "train_states": len(train_idx),
            "val_states": len(val_idx),
            "test_states": len(test_idx),
        },
        "transition_mask_source": args.transition_mask_source,
        "conditions": selected_conditions,
    }
    if args.mode == "fewshot":
        report["fewshot"] = {
            "unit": args.fewshot_unit,
            "k_values": fewshot_k_values,
            "meaning": (
                "K labeled visual samples/features"
                if args.fewshot_unit == "image"
                else "K symbolic states expanded to all views/augmentations"
            ),
            "feat_per_state": feat_per_state,
        }
    with open(exp_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nExperiment saved to {exp_dir}")


if __name__ == "__main__":
    main()
