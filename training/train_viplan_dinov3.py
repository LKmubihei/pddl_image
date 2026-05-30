#!/usr/bin/env python3
"""
ViPlan Blocksworld + DINOv3 ViT-H+/16 -> PaQ v3 Training Pipeline
==================================================================

Domain-conditioned pipeline using PDDLDomainCompiler:
  1. Parse PDDL domain file -> types, predicates, action semantics
  2. Enumerate Blocksworld states (4 blocks: Y, P, R, O)
  3. Render via subprocess (bpy) with 3 camera views per state
  4. Extract DINOv3 features with image augmentation
  5. Train PaQ model with schema queries + type classifier + action loss
  6. Evaluate on test set
"""
from __future__ import annotations
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from itertools import product as iter_product, permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
from paq.losses import PredicateStateLoss, ActionSemanticsLoss, PredicateContrastiveLoss

# Block ID mapping
BLOCK_ID = {'R': 1, 'G': 2, 'B': 3, 'Y': 4, 'P': 5, 'O': 6}
BLOCK_LETTER = {v: k for k, v in BLOCK_ID.items()}

# 4 blocks for expanded dataset
BLOCKS = ['Y', 'P', 'R', 'O']
COLUMNS = ['C1', 'C2', 'C3', 'C4']
STATIC_PREDS = {"rightof", "leftof"}
N_VIEWS = 3
N_AUGS = 3  # base + 2 augmented per image


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
        max_h = max(
            sum(1 for b in self.blocks if self.inColumn.get((b, c), False))
            for c in self.columns
        ) if self.blocks else 1
        max_h = max(max_h, 1)
        matrix = np.zeros((n_cols, max(5, max_h)), dtype=int)
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
# Direct Predicate Predictor baseline
# =========================================================================

class DirectPredicateModel(nn.Module):
    """Predict all canonical predicates directly from patch features."""
    def __init__(self, d_in=256, n_patches=196, n_predicates=32, n_heads=8):
        super().__init__()
        self.d_in = d_in
        self.n_predicates = n_predicates
        self.pred_queries = nn.Parameter(torch.randn(1, n_predicates, d_in))
        nn.init.xavier_uniform_(self.pred_queries)
        self.cross_attn = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d_in)
        self.norm_kv = nn.LayerNorm(d_in)
        self.scorer = nn.Sequential(
            nn.Linear(d_in * 2, d_in), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_in, 1),
        )
        self.global_scorer = nn.Sequential(
            nn.Linear(d_in, d_in), nn.GELU(), nn.Linear(d_in, n_predicates),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, patch_features):
        B = patch_features.shape[0]
        q = self.norm_q(self.pred_queries.expand(B, -1, -1))
        kv = self.norm_kv(patch_features)
        pred_feats, _ = self.cross_attn(q, kv, kv)
        global_feat = patch_features.mean(dim=1)
        global_exp = global_feat.unsqueeze(1).expand_as(pred_feats)
        per_pred_scores = self.scorer(
            torch.cat([pred_feats, global_exp], dim=-1)
        ).squeeze(-1)
        global_scores = self.global_scorer(global_feat)
        g = torch.sigmoid(self.gate)
        return g * per_pred_scores + (1 - g) * global_scores


# =========================================================================
# Action transition data generation
# =========================================================================

def generate_transitions(states, blocks, columns):
    """Generate valid (state_t_idx, state_t1_idx, delta_vector) pairs."""
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

                transitions.append((state, new_state))
    return transitions


# =========================================================================
# Main
# =========================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Try to reuse an existing experiment with fully rendered images
    existing_rendered = None
    for edir in sorted(PDDL_ROOT.glob("experiments/viplan_dinov3_*/images")):
        n_png = len(list(edir.glob("*.png")))
        if n_png >= 2520:  # 840 states × 3 views
            existing_rendered = str(edir)
            print(f"  Found existing rendered images: {existing_rendered} ({n_png} PNGs)")
            break

    output_dir = str(PDDL_ROOT / "experiments" / f"viplan_dinov3_{int(time.time())}")
    os.makedirs(output_dir, exist_ok=True)
    img_dir = existing_rendered if existing_rendered else os.path.join(output_dir, "images")

    n_gpus = torch.cuda.device_count()
    dev_direct = torch.device("cuda:0")
    dev_paq = torch.device("cuda:1") if n_gpus > 1 else torch.device("cuda:0")
    print("=" * 70)
    print("  PaQ v3: 4 Blocks + Multi-View + Augmentation (Parallel GPU)")
    print(f"  GPUs: {n_gpus}")
    print(f"  Output: {output_dir}")
    print(f"  Blocks: {BLOCKS} ({len(BLOCKS)} blocks)")
    print(f"  Views: {N_VIEWS}, Augmentations: {N_AUGS}")
    print(f"  PDDL domain: {BWS_DOMAIN}")
    print("=" * 70)

    # ==================================================================
    # Phase 1: PDDL Domain Compilation
    # ==================================================================
    print("\n[Phase 1] Compiling PDDL domain...")

    compiler = PDDLDomainCompiler(str(BWS_DOMAIN))
    domain_info = compiler.compile(
        objects={"block": BLOCKS, "column": COLUMNS},
        static_predicates=STATIC_PREDS,
    )

    print(f"  Domain: {domain_info.domain_name}")
    print(f"  Types: {domain_info.types}")
    print(f"  Objects: {domain_info.n_objects} ({domain_info.objects})")
    print(f"  Dynamic predicates: {domain_info.n_predicates}")
    for s in domain_info.predicate_schemas:
        print(f"    {s.schema_str}  roles={s.action_roles}  gloss=\"{s.gloss}\"")
    print(f"  Canonical atoms: {domain_info.n_canonical}")
    for i, a in enumerate(domain_info.canonical_atoms):
        print(f"    [{i:2d}] {a}")

    with open(os.path.join(output_dir, "domain_info.json"), "w") as f:
        json.dump(domain_info.summary(), f, indent=2, default=str)

    canonical_preds = domain_info.canonical_atom_strings

    # ==================================================================
    # Phase 2: Enumerate states (4 blocks)
    # ==================================================================
    print("\n[Phase 2] Enumerating Blocksworld states...")
    all_states = enumerate_all_states(BLOCKS, COLUMNS)
    print(f"  Total unique states: {len(all_states)}")

    # ==================================================================
    # Phase 3: Render with multi-view
    # ==================================================================
    print("\n[Phase 3] Rendering via bpy subprocess (multi-view)...")

    matrices = [s.to_numpy() for s in all_states]
    pkl_path = os.path.join(output_dir, "_state_matrices.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(matrices, f)

    # Check if all views already rendered
    all_rendered = all(
        all(os.path.exists(os.path.join(img_dir, f"state_{i:05d}_v{vi}.png"))
            for vi in range(N_VIEWS))
        for i in range(len(all_states))
    )

    if not all_rendered:
        render_cmd = [
            sys.executable, str(RENDER_SCRIPT),
            pkl_path, img_dir,
            "--samples", "32", "--width", "224", "--height", "224",
            "--views", str(N_VIEWS),
        ]
        print(f"  CMD: {' '.join(render_cmd)}")
        try:
            proc = subprocess.Popen(
                render_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            for line in proc.stdout:
                line = line.strip()
                if line:
                    print(f"  [bpy] {line}")
            proc.wait(timeout=600)
            print(f"  Render subprocess exited with code {proc.returncode}")
        except Exception as e:
            print(f"  bpy subprocess failed: {e}")
    else:
        print("  All multi-view images already rendered.")

    # Collect multi-view images and map to state indices
    print("\n  Collecting multi-view images...")
    from PIL import Image as PILImage

    # Build mapping: (state_idx, view_idx) -> image_path
    image_map = {}  # state_idx -> list of view image paths
    valid_state_indices = []

    for i in range(len(all_states)):
        views = []
        skip = False
        for vi in range(N_VIEWS):
            img_path = os.path.join(img_dir, f"state_{i:05d}_v{vi}.png")
            if not os.path.exists(img_path):
                skip = True
                break
            # Quality check
            img_arr = np.array(PILImage.open(img_path))[:, :, :3]
            r, g, b = img_arr[:,:,0].astype(int), img_arr[:,:,1].astype(int), img_arr[:,:,2].astype(int)
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            saturation = (max_c - min_c).astype(float) / 255.0
            pct_colored = (saturation > 0.1).sum() / (224 * 224)
            if pct_colored < 0.05:
                skip = True
                break
            views.append(img_path)
        if not skip:
            valid_state_indices.append(i)
            image_map[i] = views

    valid_states = [all_states[i] for i in valid_state_indices]
    print(f"  Valid states: {len(valid_states)}/{len(all_states)}")
    print(f"  Total images (views): {sum(len(v) for v in image_map.values())}")

    # ==================================================================
    # Phase 4: Extract DINOv3 features with augmentation
    # ==================================================================
    print("\n[Phase 4] Loading DINOv3 ViT-H+/16...")
    encoder = DINOv3ViTHPlus(DINOV3_WEIGHTS, d_out=256).to(dev_direct)
    encoder.eval()

    from torchvision import transforms
    from PIL import Image

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    aug_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Extract features: for each image, N_AUGS augmented versions
    # Each augmented feature gets the same label as its source state
    print(f"  Extracting features ({N_AUGS} augmentations per image)...")
    all_features = []       # [N_total, 196, 256]
    all_state_indices = []  # maps each feature to its state index

    torch.manual_seed(42)  # deterministic augmentation
    total_images = sum(len(v) for v in image_map.values())
    processed = 0

    for state_idx, view_paths in image_map.items():
        for img_path in view_paths:
            img = Image.open(img_path).convert("RGB")
            for aug_i in range(N_AUGS):
                if aug_i == 0:
                    tensor = base_transform(img)
                else:
                    tensor = aug_transform(img)
                all_features.append(tensor)
                all_state_indices.append(state_idx)
            processed += 1
        if processed % 200 == 0 or processed == total_images:
            print(f"    Transformed {processed}/{total_images} images")

    # Batch encode
    print(f"  Encoding {len(all_features)} feature tensors...")
    batch_size = 16
    encoded_features = []
    for i in range(0, len(all_features), batch_size):
        batch = torch.stack(all_features[i:i + batch_size]).to(dev_direct)
        with torch.no_grad():
            feats = encoder(batch)
        encoded_features.append(feats.cpu())

    patch_features = torch.cat(encoded_features, dim=0)
    print(f"  Features shape: {patch_features.shape}")

    # ==================================================================
    # Phase 5: Build labels (map features to state labels)
    # ==================================================================
    print("\n[Phase 5] Building labels...")
    state_labels = {}
    for i, state in enumerate(all_states):
        state_labels[i] = state_to_labels(state, canonical_preds)

    labels = torch.stack([state_labels[si] for si in all_state_indices])
    n_pos = labels.sum().item()
    n_neg = labels.numel() - n_pos
    print(f"  Labels: {labels.shape}, pos_ratio={labels.mean():.3f}")
    print(f"  Unique states in features: {len(set(all_state_indices))}")

    # ==================================================================
    # Phase 5b: Build grounded action semantics from compiler
    # ==================================================================
    n_canonical = len(canonical_preds)
    print(f"\n[Phase 5b] Building grounded action semantics from compiler...")
    print(f"  Compiler generated {len(domain_info.action_semantics)} grounded actions")

    # For each state, find applicable grounded actions with proper
    # precondition_mask and effect_delta from the compiler
    state_applicable_actions = {}  # state_idx -> list of (pre_mask, eff_delta, s_t1_labels)

    for si in valid_state_indices:
        labels_t = state_labels[si]  # (n_canonical,)

        for action_sem in domain_info.action_semantics:
            pre_mask = torch.tensor(action_sem.precondition_mask, dtype=torch.float)
            eff_delta = torch.tensor(action_sem.effect_delta, dtype=torch.float)

            # Skip actions with no preconditions or no effects
            if pre_mask.sum() == 0 and eff_delta.sum() == 0:
                continue

            # Check preconditions: all precondition atoms must be True
            if pre_mask.sum() > 0:
                pre_atoms_mask = pre_mask > 0
                if not (labels_t[pre_atoms_mask] == 1.0).all():
                    continue

            # Compute s_t1 by applying effects
            s_t1_labels = labels_t.clone()
            add_mask = eff_delta > 0
            del_mask = eff_delta < 0
            s_t1_labels[add_mask] = 1.0
            s_t1_labels[del_mask] = 0.0

            # Skip no-op transitions (e.g. moveblock(Y,C1) when Y already in C1)
            if (s_t1_labels == labels_t).all():
                continue

            state_applicable_actions.setdefault(si, []).append(
                (pre_mask, eff_delta, s_t1_labels)
            )

    total_applicable = sum(len(v) for v in state_applicable_actions.values())
    states_with_actions = len(state_applicable_actions)
    print(f"  {states_with_actions}/{len(valid_state_indices)} states have applicable actions")
    print(f"  Total applicable (state, action) pairs: {total_applicable}")

    # ==================================================================
    # Phase 6: Split by state (prevent data leakage)
    # ==================================================================
    print("\n[Phase 6] Train/val/test split (by state)...")

    n_states = len(valid_state_indices)
    n_test_states = max(1, int(n_states * 0.15))
    n_val_states = max(1, int(n_states * 0.15))
    n_train_states = n_states - n_val_states - n_test_states

    perm = torch.randperm(n_states)
    train_state_indices = [valid_state_indices[i] for i in perm[:n_train_states]]
    val_state_indices = [valid_state_indices[i] for i in perm[n_train_states:n_train_states+n_val_states]]
    test_state_indices = [valid_state_indices[i] for i in perm[n_train_states+n_val_states:]]

    train_state_set = set(train_state_indices)
    val_state_set = set(val_state_indices)
    test_state_set = set(test_state_indices)

    # Map features to splits
    train_mask = torch.tensor([si in train_state_set for si in all_state_indices])
    val_mask = torch.tensor([si in val_state_set for si in all_state_indices])
    test_mask = torch.tensor([si in test_state_set for si in all_state_indices])

    train_feats = patch_features[train_mask]
    train_labels = labels[train_mask]
    val_feats = patch_features[val_mask]
    val_labels = labels[val_mask]
    test_feats = patch_features[test_mask]
    test_labels = labels[test_mask]

    # Track state index per training feature (for action loss alignment)
    train_feat_sidx = torch.tensor(
        [si for si, m in zip(all_state_indices, train_mask) if m],
        dtype=torch.long,
    )
    val_feat_sidx = torch.tensor(
        [si for si, m in zip(all_state_indices, val_mask) if m],
        dtype=torch.long,
    )

    print(f"  States: train={n_train_states}, val={n_val_states}, test={n_test_states}")
    print(f"  Features: train={len(train_feats)}, val={len(val_feats)}, test={len(test_feats)}")

    # ==================================================================
    # Phase 7: Build models with domain conditioning (multi-GPU)
    # ==================================================================
    print("\n[Phase 7] Building models (multi-GPU parallel)...")

    print(f"  DirectPred -> {dev_direct}, PaQ -> {dev_paq}")
    print(f"  DirectPred -> {dev_direct}, PaQ -> {dev_paq}")

    d_slot = 256
    n_obj_slots = len(BLOCKS) + len(COLUMNS)  # 8
    obj_type_ids = torch.tensor(
        [domain_info.type_to_idx["block"]] * len(BLOCKS) +
        [domain_info.type_to_idx["column"]] * len(COLUMNS),
        dtype=torch.long,
    )

    # --- Model A: PaQ v3 ---
    model_paq = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=n_obj_slots,
        d_slot=d_slot,
        n_slot_iters=3,
        use_real_encoder=False,
        predict_slot_types=True,
    ).to(dev_paq)

    # Color-grounded slot initialization for 4 blocks + 4 columns
    color_init = torch.zeros(n_obj_slots, d_slot)
    block_rgb = [
        torch.tensor([1.0, 1.0, 0.0]),   # Y = Yellow
        torch.tensor([0.8, 0.0, 0.5]),    # P = Purple
        torch.tensor([1.0, 0.0, 0.0]),    # R = Red
        torch.tensor([1.0, 0.5, 0.0]),    # O = Orange
    ]
    color_proj = nn.Linear(3, d_slot)
    torch.nn.init.xavier_uniform_(color_proj.weight)
    with torch.no_grad():
        for i, rgb in enumerate(block_rgb):
            color_init[i] = color_proj(rgb.unsqueeze(0)).squeeze(0)
        for j in range(len(COLUMNS)):
            pos = torch.tensor([(j / max(len(COLUMNS)-1, 1)) * 2 - 1])
            color_init[len(BLOCKS) + j] = color_proj(
                torch.tensor([pos.item(), 0.0, 0.0]).unsqueeze(0)
            ).squeeze(0)
    color_init_paq = color_init.to(dev_paq)

    # --- Model B: Direct Predicate Predictor (baseline) ---
    n_canonical = len(canonical_preds)
    model_direct = DirectPredicateModel(
        d_in=d_slot, n_patches=196,
        n_predicates=n_canonical, n_heads=8,
    ).to(dev_direct)

    for name, m, d in [("PaQ+ColorInit+v3", model_paq, dev_paq),
                        ("DirectPred", model_direct, dev_direct)]:
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        print(f"  {name} ({d}): {trainable:,} trainable / {total:,} total params")

    # ==================================================================
    # Phase 8: Loss functions
    # ==================================================================
    print("\n[Phase 8] Setting up loss functions...")

    pos_weight_val = n_neg / max(n_pos, 1)
    pos_weight_direct = torch.tensor([pos_weight_val], device=dev_direct)
    pos_weight_paq = torch.tensor([pos_weight_val], device=dev_paq)
    print(f"  pos_weight: {pos_weight_val:.2f}")

    state_loss_fn = PredicateStateLoss()
    contrastive_loss_fn = PredicateContrastiveLoss(temperature=0.1)
    action_loss_fn = ActionSemanticsLoss()

    w_state = 1.0
    w_type = 0.3
    w_contrast = 0.1
    w_action = 0.2
    print(f"  Loss weights: state={w_state}, type={w_type}, contrast={w_contrast}, action={w_action}")

    # ==================================================================
    # Phase 9: Train both models in PARALLEL on separate GPUs
    # ==================================================================
    n_epochs = 100
    print(f"\n[Phase 9] Parallel training ({n_epochs} epochs, 2 GPUs)...")

    # Separate data loaders for each GPU (DataLoader is not thread-safe)
    train_loader_direct = DataLoader(
        TensorDataset(train_feats, train_labels, train_feat_sidx),
        batch_size=16, shuffle=True, drop_last=True,
    )
    train_loader_paq = DataLoader(
        TensorDataset(train_feats, train_labels, train_feat_sidx),
        batch_size=16, shuffle=True, drop_last=True,
    )
    val_loader_direct = DataLoader(
        TensorDataset(val_feats, val_labels, val_feat_sidx),
        batch_size=16, shuffle=False,
    )
    val_loader_paq = DataLoader(
        TensorDataset(val_feats, val_labels, val_feat_sidx),
        batch_size=16, shuffle=False,
    )
    obj_type_ids_paq = obj_type_ids.to(dev_paq)

    # Thread-safe printing
    _print_lock = threading.Lock()

    def train_model(model, model_name, is_direct, dev, train_loader, val_loader,
                    pos_weight_dev, obj_type_ids_dev=None, color_init_dev=None):
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        best_vf1 = 0.0
        hist = []
        best_state = None

        for epoch in range(1, n_epochs + 1):
            model.train()
            eloss, tp, fp, fn = 0.0, 0, 0, 0
            type_correct, type_total = 0, 0

            for batch in train_loader:
                feats, labs, feat_sidx = batch[0].to(dev), batch[1].to(dev), batch[2]
                B = feats.shape[0]
                optimizer.zero_grad()

                total_loss = torch.tensor(0.0, device=dev)

                if is_direct:
                    scores = model(feats)
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        scores, labs, pos_weight=pos_weight_dev
                    )
                    total_loss = total_loss + loss
                else:
                    out = model(
                        feats,
                        object_type_ids=obj_type_ids_dev.unsqueeze(0).expand(B, -1),
                        slot_init=color_init_dev.unsqueeze(0).expand(B, -1, -1),
                    )
                    scores = out["canonical_scores"]

                    loss_state = state_loss_fn(scores, labs)
                    total_loss = total_loss + w_state * loss_state

                    loss_type = model.compute_type_loss(
                        obj_type_ids_dev, forward_output=out,
                    )
                    total_loss = total_loss + w_type * loss_type

                    loss_contrast = contrastive_loss_fn(
                        out["predicate_slots"], out["predicate_queries"]
                    )
                    total_loss = total_loss + w_contrast * loss_contrast

                    if state_applicable_actions and epoch > 20:
                        batch_pre_masks = []
                        for k in range(B):
                            si = feat_sidx[k].item()
                            actions = state_applicable_actions.get(si)
                            if actions:
                                idx = torch.randint(len(actions), (1,)).item()
                                pre_m = actions[idx][0]  # only precondition_mask
                                batch_pre_masks.append(pre_m)
                            else:
                                batch_pre_masks.append(torch.zeros(n_canonical))
                        action_info = {
                            "precondition_mask": torch.stack(batch_pre_masks).to(dev),
                        }
                        loss_action = action_loss_fn(scores, action_info)
                        total_loss = total_loss + w_action * loss_action

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                eloss += total_loss.item()
                with torch.no_grad():
                    preds = (scores > 0).float()
                    tp += ((preds == 1) & (labs == 1)).sum().item()
                    fp += ((preds == 1) & (labs == 0)).sum().item()
                    fn += ((preds == 0) & (labs == 1)).sum().item()

                    if not is_direct and "predicted_type_ids" in out:
                        pred_types = out["predicted_type_ids"]
                        true_types = obj_type_ids_dev.unsqueeze(0).expand(B, -1)
                        type_correct += (pred_types == true_types).sum().item()
                        type_total += pred_types.numel()

            scheduler.step()
            t_prec = tp / max(tp + fp, 1)
            t_rec = tp / max(tp + fn, 1)
            t_f1 = 2 * t_prec * t_rec / max(t_prec + t_rec, 1e-8)
            t_type_acc = type_correct / max(type_total, 1) if type_total > 0 else 0.0

            # Validate
            model.eval()
            v_tp, v_fp, v_fn = 0, 0, 0
            v_type_correct, v_type_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    feats, labs = batch[0].to(dev), batch[1].to(dev)
                    B = feats.shape[0]
                    if is_direct:
                        scores = model(feats)
                    else:
                        out = model(
                            feats,
                            object_type_ids=obj_type_ids_dev.unsqueeze(0).expand(B, -1),
                            slot_init=color_init_dev.unsqueeze(0).expand(B, -1, -1),
                        )
                        scores = out["canonical_scores"]
                        if "predicted_type_ids" in out:
                            pred_types = out["predicted_type_ids"]
                            true_types = obj_type_ids_dev.unsqueeze(0).expand(B, -1)
                            v_type_correct += (pred_types == true_types).sum().item()
                            v_type_total += pred_types.numel()

                    preds = (scores > 0).float()
                    v_tp += ((preds == 1) & (labs == 1)).sum().item()
                    v_fp += ((preds == 1) & (labs == 0)).sum().item()
                    v_fn += ((preds == 0) & (labs == 1)).sum().item()

            v_prec = v_tp / max(v_tp + v_fp, 1)
            v_rec = v_tp / max(v_tp + v_fn, 1)
            v_f1 = 2 * v_prec * v_rec / max(v_prec + v_rec, 1e-8)
            v_type_acc = v_type_correct / max(v_type_total, 1) if v_type_total > 0 else 0.0

            if epoch % 20 == 0 or epoch == 1:
                type_str = f" type_acc={t_type_acc:.3f}/{v_type_acc:.3f}" if not is_direct else ""
                with _print_lock:
                    print(f"  [{model_name} {epoch:3d}] loss={eloss/len(train_loader):.4f} "
                          f"train_F1={t_f1:.3f} val_F1={v_f1:.3f}{type_str}")

            hist.append({
                "epoch": epoch, "train_f1": t_f1, "val_f1": v_f1,
                "val_prec": v_prec, "val_rec": v_rec,
                "train_type_acc": t_type_acc, "val_type_acc": v_type_acc,
            })

            if v_f1 > best_vf1:
                best_vf1 = v_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if epoch % 20 != 0:
                    with _print_lock:
                        print(f"  [{model_name}] New best val_F1: {v_f1:.4f} (epoch {epoch})")

        return hist, best_vf1, best_state

    # Launch parallel training
    parallel_results = {}

    def run_direct():
        parallel_results['direct'] = train_model(
            model_direct, "DirectPred", is_direct=True, dev=dev_direct,
            train_loader=train_loader_direct, val_loader=val_loader_direct,
            pos_weight_dev=pos_weight_direct,
        )

    def run_paq():
        parallel_results['paq'] = train_model(
            model_paq, "PaQ+ColorInit+v3", is_direct=False, dev=dev_paq,
            train_loader=train_loader_paq, val_loader=val_loader_paq,
            pos_weight_dev=pos_weight_paq,
            obj_type_ids_dev=obj_type_ids_paq, color_init_dev=color_init_paq,
        )

    t0 = time.time()
    t_direct = threading.Thread(target=run_direct)
    t_paq = threading.Thread(target=run_paq)
    t_direct.start()
    t_paq.start()
    t_direct.join()
    t_paq.join()
    elapsed = time.time() - t0
    print(f"\n  Parallel training completed in {elapsed:.1f}s")

    direct_hist, direct_best_vf1, direct_best_state = parallel_results['direct']
    paq_hist, paq_best_vf1, paq_best_state = parallel_results['paq']

    # ==================================================================
    # Phase 10: Evaluate both on test set
    # ==================================================================
    print("\n[Phase 10] Test set evaluation...")

    test_loader_direct = DataLoader(
        TensorDataset(test_feats, test_labels),
        batch_size=16, shuffle=False,
    )
    test_loader_paq = DataLoader(
        TensorDataset(test_feats, test_labels),
        batch_size=16, shuffle=False,
    )

    def evaluate_model(model, best_state, model_name, is_direct, dev,
                       obj_type_ids_dev=None, color_init_dev=None):
        model.load_state_dict(best_state)
        model.to(dev).eval()
        t_tp, t_fp, t_fn = 0, 0, 0
        type_correct, type_total = 0, 0
        test_loader = test_loader_direct if is_direct else test_loader_paq
        with torch.no_grad():
            for feats, labs in test_loader:  # test_loader still 2-tensor
                feats, labs = feats.to(dev), labs.to(dev)
                B = feats.shape[0]
                if is_direct:
                    scores = model(feats)
                else:
                    out = model(
                        feats,
                        object_type_ids=obj_type_ids_dev.unsqueeze(0).expand(B, -1),
                        slot_init=color_init_dev.unsqueeze(0).expand(B, -1, -1),
                    )
                    scores = out["canonical_scores"]
                    if "predicted_type_ids" in out:
                        pred_types = out["predicted_type_ids"]
                        true_types = obj_type_ids_dev.unsqueeze(0).expand(B, -1)
                        type_correct += (pred_types == true_types).sum().item()
                        type_total += pred_types.numel()

                preds = (scores > 0).float()
                t_tp += ((preds == 1) & (labs == 1)).sum().item()
                t_fp += ((preds == 1) & (labs == 0)).sum().item()
                t_fn += ((preds == 0) & (labs == 1)).sum().item()

        prec = t_tp / max(t_tp + t_fp, 1)
        rec = t_tp / max(t_tp + t_fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        type_acc = type_correct / max(type_total, 1) if type_total > 0 else -1.0
        return f1, prec, rec, type_acc

    direct_f1, direct_prec, direct_rec, _ = evaluate_model(
        model_direct, direct_best_state, "DirectPred", is_direct=True, dev=dev_direct)
    paq_f1, paq_prec, paq_rec, paq_type_acc = evaluate_model(
        model_paq, paq_best_state, "PaQ+ColorInit+v3", is_direct=False, dev=dev_paq,
        obj_type_ids_dev=obj_type_ids_paq, color_init_dev=color_init_paq)

    # Majority baseline
    all_train_val_labels = torch.cat([train_labels, val_labels], dim=0)
    maj_pred = (all_train_val_labels.mean(dim=0) >= 0.5).float()
    m_tp = (maj_pred.unsqueeze(0) * test_labels).sum().item()
    m_fp = (maj_pred.unsqueeze(0) * (1 - test_labels)).sum().item()
    m_fn = ((1 - maj_pred.unsqueeze(0)) * test_labels).sum().item()
    m_prec = m_tp / max(m_tp + m_fp, 1)
    m_rec = m_tp / max(m_tp + m_fn, 1)
    maj_f1 = 2 * m_prec * m_rec / max(m_prec + m_rec, 1e-8)

    print(f"\n  === Test Results ===")
    print(f"  DirectPred:     F1={direct_f1:.4f}  P={direct_prec:.3f}  R={direct_rec:.3f}")
    print(f"  PaQ+ColorInit+v3: F1={paq_f1:.4f}  P={paq_prec:.3f}  R={paq_rec:.3f}  type_acc={paq_type_acc:.3f}")
    print(f"  Majority:       F1={maj_f1:.4f}  P={m_prec:.3f}  R={m_rec:.3f}")

    # Save results
    report = {
        "pipeline": "v3_domain_conditioned",
        "backbone": "dinov3_vith16plus",
        "domain_file": str(BWS_DOMAIN),
        "n_blocks": len(BLOCKS),
        "n_views": N_VIEWS,
        "n_augs": N_AUGS,
        "n_states": len(valid_states),
        "canonical_preds": canonical_preds,
        "domain_summary": domain_info.summary(),
        "split": {
            "train_states": n_train_states,
            "val_states": n_val_states,
            "test_states": n_test_states,
            "train_features": len(train_feats),
            "val_features": len(val_feats),
            "test_features": len(test_feats),
        },
        "loss_weights": {
            "state": w_state, "type": w_type,
            "contrast": w_contrast, "action": w_action,
        },
        "direct_pred": {
            "best_val_f1": direct_best_vf1, "test_f1": direct_f1,
            "test_precision": direct_prec, "test_recall": direct_rec,
        },
        "paq_v3": {
            "best_val_f1": paq_best_vf1, "test_f1": paq_f1,
            "test_precision": paq_prec, "test_recall": paq_rec,
            "test_type_accuracy": paq_type_acc,
            "predict_slot_types": True,
            "schema_conditioned_queries": True,
            "action_semantics_loss": True,
        },
        "majority_f1": maj_f1,
        "pos_weight": pos_weight_val,
    }
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(output_dir, "direct_history.json"), "w") as f:
        json.dump(direct_hist, f, indent=2)
    with open(os.path.join(output_dir, "paq_history.json"), "w") as f:
        json.dump(paq_hist, f, indent=2)

    torch.save(
        {"model_state_dict": direct_best_state, "val_f1": direct_best_vf1},
        os.path.join(output_dir, "best_direct_model.pt"),
    )
    torch.save(
        {
            "model_state_dict": paq_best_state,
            "val_f1": paq_best_vf1,
            "obj_type_ids": obj_type_ids,
            "canonical_preds": canonical_preds,
            "domain_info": domain_info.summary(),
            "color_init": color_init_paq.cpu(),
        },
        os.path.join(output_dir, "best_paq_model.pt"),
    )

    print(f"\n  Results saved to: {output_dir}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
