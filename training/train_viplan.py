#!/usr/bin/env python3
"""
ViPlan Blocksworld → PaQ Training Pipeline
============================================
1. Enumerate valid Blocksworld states from ViPlan metadata
2. Render images with Blender (ViPlan's renderer)
3. Extract patch-level features with DINOv3 ViT-H+/16
4. Train PaQ model with canonical predicate alignment

Architecture (object-centric):
    Image → DINOv3 (frozen) → Patch Tokens (B, N_patches, D)
                                        ↓
                         Object Slot Attention → Object Slots (B, N_obj, D)
                                        ↓
               Predicate Queries → Predicate Slot Attention → Predicate Slots
                                        ↓
                         Type-Aware Scoring Head → Canonical Scores (B, N_canonical)

Fixes vs v1:
    - Train/val/test split (70/15/15)
    - Self-relations filtered from canonical ground predicates
    - Patch tokens → slot attention (real object-centric, not CLS broadcast)
    - pos_weight in BCEWithLogitsLoss for class imbalance
    - DINOv3 ViT-H+/16 backbone
    - Dead code removed
    - use_real_encoder=True
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import subprocess
import time
from itertools import product as iter_product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Paths
VIPAN_ROOT = Path("/home/pc/PDDL/ViPlan")
PDDL_ROOT = Path("/home/pc/PDDL")
BLENDER = str(VIPAN_ROOT / "blender-3.0.0-linux-x64" / "blender")
RENDER_SCRIPT = str(VIPAN_ROOT / "viplan" / "rendering" / "blocksworld" / "render.py")
RENDER_DATA = str(VIPAN_ROOT / "data" / "blocksworld_rendering")
DOMAIN_PDDL = str(VIPAN_ROOT / "data" / "planning" / "blocksworld" / "domain.pddl")
PROBLEMS_DIR = str(VIPAN_ROOT / "data" / "planning" / "blocksworld" / "problems")

sys.path.insert(0, str(PDDL_ROOT))
sys.path.insert(0, str(VIPAN_ROOT))

# Block ID mapping (from ViPlan)
BLOCK_ID = {'R': 1, 'G': 2, 'B': 3, 'Y': 4, 'P': 5, 'O': 6}
BLOCK_LETTER = {v: k for k, v in BLOCK_ID.items()}
COLOR_MAP = {
    1: [1, 0, 0, 1],    # R = Red
    2: [0, 0.8, 0, 1],  # G = Green
    3: [0, 0, 0.8, 1],  # B = Blue
    4: [1, 1, 0, 1],    # Y = Yellow
    5: [0.2, 0, 0.5, 1],# P = Purple
    6: [1, 0.5, 0, 1],  # O = Orange
}

# Blocksworld domain definition
BLOCKSWORLD_TYPES = ["block", "column"]
BLOCKSWORLD_PREDICATES = {
    "on":        {"arity": 2, "param_types": ["block", "block"]},
    "inColumn":  {"arity": 2, "param_types": ["block", "column"]},
    "clear":     {"arity": 1, "param_types": ["block"]},
    "rightOf":   {"arity": 2, "param_types": ["column", "column"]},
    "leftOf":    {"arity": 2, "param_types": ["column", "column"]},
}
STATIC_PREDS = {"rightOf", "leftOf"}


# =========================================================================
# State representation and enumeration
# =========================================================================

class BlocksworldState:
    """Represents a Blocksworld state as a numpy matrix + predicate dict."""

    def __init__(self, blocks: list[str], columns: list[str],
                 on: dict, inColumn: dict, clear: dict,
                 rightOf: dict, leftOf: dict):
        self.blocks = blocks
        self.columns = columns
        self.on = on
        self.inColumn = inColumn
        self.clear = clear
        self.rightOf = rightOf
        self.leftOf = leftOf

    def to_numpy(self) -> np.ndarray:
        """Convert to (n_cols, max_height) matrix for rendering."""
        n_cols = len(self.columns)
        max_h = max(
            sum(1 for b in self.blocks if self.inColumn.get((b, c), False))
            for c in self.columns
        ) if self.blocks else 1
        max_h = max(max_h, 1)

        matrix = np.zeros((max(5, n_cols), max(5, max_h)), dtype=int)

        for ci, col in enumerate(self.columns):
            col_blocks = [b for b in self.blocks if self.inColumn.get((b, col), False)]
            sorted_blocks = self._sort_column(col_blocks)
            for bi, block in enumerate(sorted_blocks):
                matrix[ci, bi] = BLOCK_ID[block.upper()]

        return matrix

    def _sort_column(self, blocks: list[str]) -> list[str]:
        """Sort blocks in a column from bottom to top."""
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

    def get_predicates(self) -> set[str]:
        """Get all true ground predicates as strings."""
        preds = set()
        for (b1, b2), v in self.on.items():
            if v:
                preds.add(f"(on {b1} {b2})")
        for (b, c), v in self.inColumn.items():
            if v:
                preds.add(f"(inColumn {b} {c})")
        for b, v in self.clear.items():
            if v:
                preds.add(f"(clear {b})")
        for (c1, c2), v in self.rightOf.items():
            if v:
                preds.add(f"(rightOf {c1} {c2})")
        for (c1, c2), v in self.leftOf.items():
            if v:
                preds.add(f"(leftOf {c1} {c2})")
        return preds


def enumerate_all_states(blocks: list[str], columns: list[str]) -> list[BlocksworldState]:
    """Enumerate all valid Blocksworld states for given blocks and columns."""
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
                from itertools import permutations
                col_orderings[ci] = list(permutations(col_contents[col]))

        for combo in iter_product(*[col_orderings[ci] for ci in range(n_cols)]):
            on = {}
            inColumn = {}
            clear = {}
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

            rightOf = {}
            leftOf = {}
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


# =========================================================================
# Rendering
# =========================================================================

def render_state_blender(np_matrix: np.ndarray, output_path: str,
                         seed: int = 42, n_samples: int = 32):
    """Render a Blocksworld state using ViPlan's Blender renderer."""
    from viplan.rendering.blocksworld.blocks import State as RenderState

    state = RenderState(
        list(np_matrix),
        properties_json=os.path.join(RENDER_DATA, "properties.json"),
        seed=seed,
    )

    render_dir = os.path.dirname(output_path)
    os.makedirs(render_dir, exist_ok=True)
    pkl_path = os.path.join(render_dir, "_render_state.pkl")

    with open(pkl_path, 'wb') as f:
        pickle.dump(state, f)

    cmd = [
        BLENDER, "-noaudio", "--background",
        "--python", RENDER_SCRIPT, "--",
        "--output-dir", render_dir,
        "--render-num-samples", str(n_samples),
        "--width", "224", "--height", "224",
        "--render-state", pkl_path,
        "--base-scene-blendfile", os.path.join(RENDER_DATA, "base_scene.blend"),
        "--properties-json", os.path.join(RENDER_DATA, "properties.json"),
        "--shape-dir", os.path.join(RENDER_DATA, "shapes"),
        "--material-dir", os.path.join(RENDER_DATA, "materials"),
        "--use-gpu", "1",
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=120)
    rendered = os.path.join(render_dir, "render.png")

    if os.path.exists(rendered):
        os.rename(rendered, output_path)
        for cleanup in [
            os.path.join(render_dir, "scene.json"),
            pkl_path,
        ]:
            if os.path.exists(cleanup):
                os.remove(cleanup)
        return True
    return False


def render_all_states(states: list[BlocksworldState], output_dir: str,
                      max_samples: int = 0, n_blender_samples: int = 32):
    """Render states and return list of (image_path, state) tuples."""
    os.makedirs(output_dir, exist_ok=True)
    rendered = []

    n = len(states) if max_samples == 0 else min(max_samples, len(states))
    print(f"  Rendering {n}/{len(states)} states...")

    for i, state in enumerate(states[:n]):
        img_path = os.path.join(output_dir, f"state_{i:05d}.png")
        if os.path.exists(img_path):
            rendered.append((img_path, state))
            continue

        np_mat = state.to_numpy()
        ok = render_state_blender(np_mat, img_path, seed=i, n_samples=n_blender_samples)
        if ok:
            rendered.append((img_path, state))
        else:
            print(f"  WARNING: render failed for state {i}")

        if (i + 1) % 10 == 0:
            print(f"    Rendered {i + 1}/{n}...")

    return rendered


# =========================================================================
# Canonical ground predicates for Blocksworld
# =========================================================================

def build_canonical_ground_preds(blocks: list[str], columns: list[str]) -> list[str]:
    """Build canonical ordered list of dynamic ground predicates.

    Order: pred types sorted by name, groundings in product order.
    Static predicates (rightOf, leftOf) are excluded.
    Self-relations (e.g. on Y Y, on R R) are excluded — always false.
    """
    dynamic_pred_defs = sorted(
        [(k, v) for k, v in BLOCKSWORLD_PREDICATES.items() if k not in STATIC_PREDS],
        key=lambda x: x[0],
    )

    type_to_objs = {"block": blocks, "column": columns}
    canonical = []

    for pname, pinfo in dynamic_pred_defs:
        arity = pinfo["arity"]
        param_types = pinfo["param_types"]

        if arity == 0:
            canonical.append(f"({pname})")
        else:
            valid_names = [type_to_objs[pt] for pt in param_types]
            for combo in iter_product(*valid_names):
                # Filter self-relations for binary predicates with same type
                if arity == 2 and param_types[0] == param_types[1]:
                    if combo[0] == combo[1]:
                        continue
                args = " ".join(combo)
                canonical.append(f"({pname} {args})")

    return canonical


def state_to_labels(state: BlocksworldState, canonical_preds: list[str]) -> torch.Tensor:
    """Convert state to label tensor in canonical order."""
    true_preds = state.get_predicates()
    labels = torch.zeros(len(canonical_preds), dtype=torch.float32)
    for i, gp in enumerate(canonical_preds):
        if gp in true_preds:
            labels[i] = 1.0
    return labels


# =========================================================================
# Feature extraction with DINOv3
# =========================================================================

def extract_dinov3_patch_features(
    image_paths: list[str],
    batch_size: int = 8,
    dinov3_repo_dir: str | None = None,
) -> torch.Tensor:
    """Extract DINOv3 ViT-H+/16 patch-level features for all images.

    Returns: (N_images, N_patches, D) tensor.
    """
    from paq.visual_encoder import DINOv3VisualEncoder

    print("  Loading DINOv3 ViT-H+/16...")
    encoder = DINOv3VisualEncoder(
        model_name="dinov3_vith16plus",
        d_out=256,
        source="local" if dinov3_repo_dir else "github",
        repo_dir=dinov3_repo_dir,
    )
    encoder.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.to(device)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    all_features = []
    from PIL import Image

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(transform(img))

        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            # Get raw patch features (before projection)
            patch_feats = encoder._extract_patch_features(batch_tensor)

        all_features.append(patch_feats.cpu())

        if (i + batch_size) % 40 == 0 or (i + batch_size) >= len(image_paths):
            print(f"    Extracted {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    return torch.cat(all_features, dim=0)


# =========================================================================
# Main training pipeline
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-render", type=int, default=200,
                        help="Max number of states to render (0=all)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--blender-samples", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--dinov3-repo-dir", type=str, default=None,
                        help="Path to local DINOv3 repo checkout")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir or str(
        PDDL_ROOT / "experiments" / f"viplan_bw_{int(time.time())}"
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  ViPlan Blocksworld → PaQ Training Pipeline (v2)")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    # ---- Step 1: Load ViPlan metadata ----
    print("\n[Phase 1] Loading ViPlan Blocksworld problems...")

    with open(os.path.join(PROBLEMS_DIR, "simple", "metadata.json")) as f:
        metadata = json.load(f)

    first_key = list(metadata.keys())[0]
    blocks = metadata[first_key]["blocks"]
    columns = ["C1", "C2", "C3", "C4"]

    print(f"  Blocks: {blocks}")
    print(f"  Columns: {columns}")

    # ---- Step 2: Build canonical ground predicates ----
    print("\n[Phase 2] Building canonical ground predicates (no self-relations)...")
    canonical_preds = build_canonical_ground_preds(blocks, columns)
    print(f"  Canonical ground predicates: {len(canonical_preds)}")
    for i, p in enumerate(canonical_preds):
        print(f"    [{i}] {p}")

    # Build pred info for model
    dynamic_pred_names = sorted(
        [k for k in BLOCKSWORLD_PREDICATES if k not in STATIC_PREDS]
    )
    pred_arities = {k: v["arity"] for k, v in BLOCKSWORLD_PREDICATES.items()
                    if k not in STATIC_PREDS}
    pred_param_types = {k: v["param_types"] for k, v in BLOCKSWORLD_PREDICATES.items()
                        if k not in STATIC_PREDS}

    print(f"  Dynamic predicates: {dynamic_pred_names}")

    # ---- Step 3: Enumerate states ----
    print("\n[Phase 3] Enumerating Blocksworld states...")
    all_states = enumerate_all_states(blocks, columns)
    print(f"  Total unique states: {len(all_states)}")

    # ---- Step 4: Render images ----
    print(f"\n[Phase 4] Rendering images (max {args.max_render}, samples={args.blender_samples})...")
    img_dir = os.path.join(output_dir, "images")
    rendered = render_all_states(
        all_states, img_dir,
        max_samples=args.max_render,
        n_blender_samples=args.blender_samples,
    )
    print(f"  Rendered {len(rendered)} images")

    # ---- Step 5: Extract DINOv3 patch features ----
    print("\n[Phase 5] Extracting DINOv3 ViT-H+/16 patch features...")
    image_paths = [r[0] for r in rendered]
    patch_features = extract_dinov3_patch_features(
        image_paths, batch_size=8, dinov3_repo_dir=args.dinov3_repo_dir,
    )
    print(f"  Patch features shape: {patch_features.shape}")  # (N, N_patches, D_backbone)

    # ---- Step 6: Build labels ----
    print("\n[Phase 6] Building label tensors...")
    label_list = [state_to_labels(r[1], canonical_preds) for r in rendered]
    labels = torch.stack(label_list)
    print(f"  Labels shape: {labels.shape}")
    print(f"  Positive ratio: {labels.mean():.3f}")
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    print(f"  Positive samples: {int(n_pos)}, Negative samples: {int(n_neg)}")

    # ---- Step 7: Train/val/test split ----
    print("\n[Phase 7] Splitting data...")
    n_total = len(rendered)
    n_test = max(1, int(n_total * args.test_ratio))
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val - n_test

    # Create indices and shuffle deterministically
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_feats = patch_features[train_idx]
    train_labels = labels[train_idx]
    val_feats = patch_features[val_idx]
    val_labels = labels[val_idx]
    test_feats = patch_features[test_idx]
    test_labels = labels[test_idx]

    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")

    # ---- Step 8: Build PaQ model with real DINOv3 encoder ----
    print("\n[Phase 8] Building PaQ model (DINOv3 + slot attention)...")

    d_slot = 256
    n_pred_types = len(dynamic_pred_names)  # 3: clear, inColumn, on
    n_obj_slots = len(blocks) + len(columns)  # 7 = 3 blocks + 4 columns

    # Object type IDs: blocks get type 0, columns get type 1
    obj_type_ids = torch.tensor(
        [0] * len(blocks) + [1] * len(columns),
        dtype=torch.long,
    )  # [0, 0, 0, 1, 1, 1, 1]

    from paq.visual_encoder import DINOv3VisualEncoder

    # Build DINOv3 encoder (frozen backbone + learnable proj)
    dinov3_encoder = DINOv3VisualEncoder(
        model_name="dinov3_vith16plus",
        d_out=d_slot,
        source="local" if args.dinov3_repo_dir else "github",
        repo_dir=args.dinov3_repo_dir,
    )

    from paq.model import PaQModel

    model = PaQModel(
        predicate_names=dynamic_pred_names,
        predicate_arities=pred_arities,
        predicate_param_types=pred_param_types,
        type_names=["block", "column"],
        n_object_slots=n_obj_slots,
        d_slot=d_slot,
        n_slot_iters=3,
        use_real_encoder=True,
        tau_unknown=0.3,
        visual_encoder=dinov3_encoder,
    ).to(device)

    params = model.count_parameters()
    print(f"  Parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    # ---- Step 9: Train with pos_weight BCE ----
    print(f"\n[Phase 9] Training for {args.epochs} epochs...")

    # pos_weight for class imbalance: n_neg / n_pos
    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"  pos_weight: {pos_weight_val:.2f} (neg/pos ratio)")

    train_dataset = TensorDataset(train_feats, train_labels)
    val_dataset = TensorDataset(val_feats, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        tp, fp, fn = 0, 0, 0

        for batch in train_loader:
            feats, labs = batch
            feats = feats.to(device)
            labs = labs.to(device)

            optimizer.zero_grad()
            out = model(feats, obj_type_ids.unsqueeze(0).expand(feats.shape[0], -1).to(device))
            scores = out["canonical_scores"]  # (B, N_canonical) logits

            loss = nn.functional.binary_cross_entropy_with_logits(
                scores, labs, pos_weight=pos_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                preds = (scores > 0).float()
                n_correct += (preds == labs).sum().item()
                n_total += labs.numel()
                tp += ((preds == 1) & (labs == 1)).sum().item()
                fp += ((preds == 1) & (labs == 0)).sum().item()
                fn += ((preds == 0) & (labs == 1)).sum().item()

        scheduler.step()

        train_acc = n_correct / max(n_total, 1)
        train_prec = tp / max(tp + fp, 1)
        train_rec = tp / max(tp + fn, 1)
        train_f1 = 2 * train_prec * train_rec / max(train_prec + train_rec, 1e-8)
        avg_loss = epoch_loss / max(len(train_loader), 1)

        # --- Validation ---
        model.eval()
        val_tp, val_fp, val_fn, val_total = 0, 0, 0, 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch in val_loader:
                feats, labs = batch
                feats = feats.to(device)
                labs = labs.to(device)

                out = model(feats, obj_type_ids.unsqueeze(0).expand(feats.shape[0], -1).to(device))
                scores = out["canonical_scores"]

                val_loss_sum += nn.functional.binary_cross_entropy_with_logits(
                    scores, labs, pos_weight=pos_weight,
                ).item()

                preds = (scores > 0).float()
                val_total += labs.numel()
                val_tp += ((preds == 1) & (labs == 1)).sum().item()
                val_fp += ((preds == 1) & (labs == 0)).sum().item()
                val_fn += ((preds == 0) & (labs == 1)).sum().item()

        val_prec = val_tp / max(val_tp + val_fp, 1)
        val_rec = val_tp / max(val_tp + val_fn, 1)
        val_f1 = 2 * val_prec * val_rec / max(val_prec + val_rec, 1e-8)
        val_acc = (val_tp + (val_total - val_tp - val_fp - val_fn) + (val_total - val_tp - val_fn - (val_total - val_tp - val_fp - val_fn))) / max(val_total, 1)
        # Simplified accuracy
        val_correct = val_total - (val_fp + val_fn)
        val_acc = val_correct / max(val_total, 1)
        val_avg_loss = val_loss_sum / max(len(val_loader), 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:4d}] "
                  f"train_loss={avg_loss:.4f} train_F1={train_f1:.3f} | "
                  f"val_loss={val_avg_loss:.4f} val_F1={val_f1:.3f} val_P={val_prec:.3f} val_R={val_rec:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss, "train_f1": train_f1,
            "val_loss": val_avg_loss, "val_f1": val_f1,
            "val_precision": val_prec, "val_recall": val_rec,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_f1": val_f1,
                "train_f1": train_f1,
                "canonical_preds": canonical_preds,
                "blocks": blocks,
                "columns": columns,
                "obj_type_ids": obj_type_ids,
                "pos_weight": pos_weight_val,
            }, os.path.join(output_dir, "best_model.pt"))

    # ---- Step 10: Final evaluation on test set ----
    print("\n[Phase 10] Evaluating on test set...")

    checkpoint = torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loader = DataLoader(
        TensorDataset(test_feats, test_labels), batch_size=8, shuffle=False,
    )

    test_tp, test_fp, test_fn, test_total = 0, 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            feats, labs = batch
            feats = feats.to(device)
            labs = labs.to(device)

            out = model(feats, obj_type_ids.unsqueeze(0).expand(feats.shape[0], -1).to(device))
            scores = out["canonical_scores"]

            preds = (scores > 0).float()
            test_total += labs.numel()
            test_tp += ((preds == 1) & (labs == 1)).sum().item()
            test_fp += ((preds == 1) & (labs == 0)).sum().item()
            test_fn += ((preds == 0) & (labs == 1)).sum().item()

    test_prec = test_tp / max(test_tp + test_fp, 1)
    test_rec = test_tp / max(test_tp + test_fn, 1)
    test_f1 = 2 * test_prec * test_rec / max(test_prec + test_rec, 1e-8)

    # Majority baseline: predict all negative → F1 = 0
    # Or per-predicate majority baseline
    all_labels = torch.cat([train_labels, val_labels], dim=0)
    majority_pred = (all_labels.mean(dim=0) >= 0.5).float()
    # Apply majority prediction to test
    maj_tp = (majority_pred.unsqueeze(0).expand(test_labels.shape[0], -1) * test_labels).sum().item()
    maj_fp = (majority_pred.unsqueeze(0).expand(test_labels.shape[0], -1) * (1 - test_labels)).sum().item()
    maj_fn = ((1 - majority_pred.unsqueeze(0).expand(test_labels.shape[0], -1)) * test_labels).sum().item()
    maj_prec = maj_tp / max(maj_tp + maj_fp, 1)
    maj_rec = maj_tp / max(maj_tp + maj_fn, 1)
    maj_f1 = 2 * maj_prec * maj_rec / max(maj_prec + maj_rec, 1e-8)

    print(f"\n  === Test Results ===")
    print(f"  Model:     F1={test_f1:.4f}  P={test_prec:.3f}  R={test_rec:.3f}")
    print(f"  Majority:  F1={maj_f1:.4f}  P={maj_prec:.3f}  R={maj_rec:.3f}")
    print(f"  Model {'BEATS' if test_f1 > maj_f1 else 'DOES NOT BEAT'} majority baseline")

    # ---- Final report ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  Training complete in {elapsed / 60:.1f} min")
    print(f"  Best val F1: {best_val_f1:.4f}")
    print(f"  Test F1: {test_f1:.4f}")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)

    config = {
        "blocks": blocks,
        "columns": columns,
        "canonical_preds": canonical_preds,
        "n_states": len(rendered),
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "epochs": args.epochs,
        "best_val_f1": best_val_f1,
        "test_f1": test_f1,
        "majority_f1": maj_f1,
        "pos_weight": pos_weight_val,
        "elapsed_min": round(elapsed / 60, 1),
        "backbone": "dinov3_vith16plus",
        "n_obj_slots": n_obj_slots,
    }

    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return {"best_val_f1": best_val_f1, "test_f1": test_f1, "n_states": len(rendered)}


if __name__ == "__main__":
    main()
