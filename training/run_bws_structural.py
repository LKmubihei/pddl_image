#!/usr/bin/env python3
"""Blocksworld structural experiment with synthetic features.

This runner is a structural sanity check, not a visual grounding result.  It
uses direct synthetic object tokens and oracle object types.  State-diff masks
are oracle transition supervision.

Tests: does action structure (C2/C3) improve grounding beyond state labels (C1)?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paq.domain_compiler import PDDLDomainCompiler
from training.train_aepaq import (
    enumerate_all_states,
    state_to_labels,
    _parse_k_values,
    _parse_conditions,
    _build_fewshot_state_dataset,
    run_structural_experiment,
)
from training.data.ae_dataset import StateDataset, TransitionDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BWS_DOMAIN = Path("/home/claudeuser/ViPlan") / "data" / "planning" / "blocksworld" / "domain.pddl"
BWS_PROBLEM = (
    Path("/home/claudeuser/ViPlan")
    / "data" / "planning" / "blocksworld" / "problems" / "simple" / "simple_problem_0.pddl"
)
STATIC_PREDS = {"rightof", "leftof"}


def _problem_objects(problem_path: Path) -> tuple[dict[str, list[str]], list[str], list[str]]:
    from pddl_parser import PDDLProblemParser

    problem = PDDLProblemParser(str(problem_path))
    blocks = list(problem.objects.get("block", []))
    columns = list(problem.objects.get("column", []))
    if not blocks or not columns:
        raise ValueError(
            f"Problem {problem_path} must define block and column objects; "
            f"got {problem.objects}"
        )
    return problem.objects, blocks, columns


def _label_duplicate_metadata(
    labels: torch.Tensor,
    pool_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict:
    label_keys = [tuple(int(x) for x in row.tolist()) for row in labels.long()]
    pool_keys = {label_keys[int(i)] for i in pool_idx}
    val_keys = [label_keys[int(i)] for i in val_idx]
    test_keys = [label_keys[int(i)] for i in test_idx]
    val_dups = sum(1 for key in val_keys if key in pool_keys)
    test_dups = sum(1 for key in test_keys if key in pool_keys)
    return {
        "num_unique_labels": len(set(label_keys)),
        "duplicate_rate_pool_val": val_dups / max(len(val_keys), 1),
        "duplicate_rate_pool_test": test_dups / max(len(test_keys), 1),
    }


# ---------------------------------------------------------------------------
# Synthetic features (adapted from run_strips_structural)
# ---------------------------------------------------------------------------
def _build_synthetic_features(
    domain_info,
    state_labels: torch.Tensor,
    views_per_state: int,
    d_slot: int,
    noise: float,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    n_states = state_labels.shape[0]
    n_objects = domain_info.n_objects

    base = rng.normal(0.0, 0.18, size=(n_objects, d_slot)).astype(np.float32)
    type_centers = rng.normal(0.0, 0.70, size=(domain_info.n_types, d_slot)).astype(np.float32)
    for oi, obj in enumerate(domain_info.objects):
        base[oi] += type_centers[obj.type_idx]
        identity_dim = 8 + oi
        if identity_dim < d_slot:
            base[oi, identity_dim] += 1.5

    schema_vecs = rng.normal(0.0, 0.75, size=(len(domain_info.predicate_schemas), d_slot)).astype(np.float32)
    role_vecs = rng.normal(0.0, 0.25, size=(3, d_slot)).astype(np.float32)
    partner_vecs = rng.normal(0.0, 0.35, size=(n_objects, d_slot)).astype(np.float32)

    obj_name_to_idx = {obj.name.lower(): i for i, obj in enumerate(domain_info.objects)}
    atom_to_objects: list[list[int]] = []
    for atom in domain_info.canonical_atoms:
        obj_ids = [obj_name_to_idx[arg.lower()] for arg in atom.arguments if arg.lower() in obj_name_to_idx]
        atom_to_objects.append(obj_ids or list(range(n_objects)))

    features = []
    labels_np = state_labels.numpy()
    for si in range(n_states):
        true_atoms = np.flatnonzero(labels_np[si] > 0.5)
        state_feat = base.copy()
        for ai in true_atoms:
            atom = domain_info.canonical_atoms[ai]
            obj_ids = atom_to_objects[ai]
            schema_vec = schema_vecs[atom.predicate_idx]
            if not atom.arguments:
                state_feat += schema_vec
            elif len(obj_ids) == 1:
                state_feat[obj_ids[0]] += schema_vec + role_vecs[0]
            elif len(obj_ids) == 2:
                left, right = obj_ids
                state_feat[left] += schema_vec + role_vecs[0] + partner_vecs[right]
                state_feat[right] += schema_vec + role_vecs[1] + partner_vecs[left]
            else:
                for pos, oi in enumerate(obj_ids):
                    state_feat[oi] += schema_vec + role_vecs[min(pos, len(role_vecs) - 1)]
        for _ in range(views_per_state):
            aug = state_feat + rng.normal(0.0, noise, size=state_feat.shape).astype(np.float32)
            features.append(aug)
    return torch.from_numpy(np.stack(features)).float()


# ---------------------------------------------------------------------------
# Blocksworld transitions
# ---------------------------------------------------------------------------
def _enumerate_transitions(blocks, columns):
    """Generate all valid moveBlock transitions as (state_idx_t, state_idx_t1, action_desc)."""
    states = enumerate_all_states(blocks, columns)
    state_to_idx = {}
    for i, s in enumerate(states):
        key = frozenset(s.get_predicates())
        state_to_idx[key] = i

    transitions = []
    for i, s in enumerate(states):
        for b in blocks:
            if not s.clear.get(b, False):
                continue
            for c in columns:
                if s.inColumn.get((b, c), False):
                    continue  # already there
                # Apply moveBlock(b, c)
                s2 = _apply_move(s, b, c, blocks, columns)
                key2 = frozenset(s2.get_predicates())
                j = state_to_idx.get(key2)
                if j is not None and j != i:
                    transitions.append((i, j, f"moveBlock({b},{c})"))
    return states, transitions


def _apply_move(state, block, target_col, blocks, columns):
    """Apply moveBlock(block, target_col) to a state, return new state."""
    from training.train_aepaq import BlocksworldState
    on = dict(state.on)
    inCol = dict(state.inColumn)
    cl = dict(state.clear)

    # Remove block from its current column
    old_col = None
    for c in columns:
        if inCol.get((block, c), False):
            old_col = c
            inCol[(block, c)] = False
            break

    # Remove on-relation from block below
    for b2 in blocks:
        if on.get((block, b2), False):
            on[(block, b2)] = False
            cl[b2] = True  # block below becomes clear

    # Find landing spot in target column
    blocks_in_col = [b for b in blocks if inCol.get((b, target_col), False)]
    top_block = None
    for b in blocks_in_col:
        is_top = True
        for b2 in blocks_in_col:
            if on.get((b2, b), False):
                is_top = False
                break
        if is_top and b != block:
            top_block = b
            break

    # Place block
    inCol[(block, target_col)] = True
    cl[block] = True
    if top_block is not None:
        on[(block, top_block)] = True
        cl[top_block] = False

    return BlocksworldState(
        blocks=blocks, columns=columns,
        on=on, inColumn=inCol, clear=cl,
        rightOf=state.rightOf, leftOf=state.leftOf,
    )


def _build_transition_dataset(
    states, transitions, state_labels, all_features, train_state_ids,
    views_per_state, type_ids, n_negatives, seed, domain_info,
):
    """Build TransitionDataset from Blocksworld transitions with state_diff masks."""
    rng = np.random.default_rng(seed)
    n_canon = domain_info.n_canonical

    features_t, features_t1 = [], []
    action_indices = []  # unused but required
    pre_masks, add_masks, del_masks, frame_masks = [], [], [], []
    neg_pre_l, neg_add_l, neg_del_l = [], [], []

    # Group transitions by source state for negative sampling
    source_groups = {}
    for (t_idx, t1_idx, desc) in transitions:
        if t_idx not in train_state_ids or t1_idx not in train_state_ids:
            continue
        source_groups.setdefault(t_idx, []).append((t1_idx, desc))

    all_trans = [(t, t1, d) for t, t1, d in transitions
                 if t in train_state_ids and t1 in train_state_ids]

    for (t_idx, t1_idx, desc) in all_trans:
        feat_t_idx = t_idx * views_per_state + int(rng.integers(views_per_state))
        feat_t1_idx = t1_idx * views_per_state + int(rng.integers(views_per_state))
        features_t.append(all_features[feat_t_idx])
        features_t1.append(all_features[feat_t1_idx])

        # state_diff masks
        s_t = state_labels[t_idx]
        s_t1 = state_labels[t1_idx]
        add = ((s_t1 > 0.5) & (s_t <= 0.5)).float()
        delete = ((s_t > 0.5) & (s_t1 <= 0.5)).float()
        frame = 1.0 - torch.clamp(add + delete, max=1.0)
        pre = torch.zeros(n_canon)  # not used with state_diff

        action_indices.append(0)
        pre_masks.append(pre)
        add_masks.append(add)
        del_masks.append(delete)
        frame_masks.append(frame)

        # Negative masks: random other transitions from same source
        others = [(t1o, do) for (t1o, do) in source_groups.get(t_idx, []) if t1o != t1_idx]
        neg_pre = torch.zeros(n_negatives, n_canon)
        neg_add = torch.zeros(n_negatives, n_canon)
        neg_del = torch.zeros(n_negatives, n_canon)
        if others:
            chosen = rng.choice(len(others), size=min(n_negatives, len(others)), replace=False)
            for k, ci in enumerate(chosen[:n_negatives]):
                t1o, _ = others[ci]
                s_t1o = state_labels[t1o]
                na = ((s_t1o > 0.5) & (s_t <= 0.5)).float()
                nd = ((s_t > 0.5) & (s_t1o <= 0.5)).float()
                neg_add[k] = na
                neg_del[k] = nd
        neg_pre_l.append(neg_pre)
        neg_add_l.append(neg_add)
        neg_del_l.append(neg_del)

    if not features_t:
        raise RuntimeError("No train-split transitions produced")

    add_stack = torch.stack(add_masks)
    del_stack = torch.stack(del_masks)
    frame_stack = torch.stack(frame_masks)
    print(
        f"  Transitions: {len(features_t)} "
        f"add={add_stack.sum(1).float().mean():.2f} "
        f"del={del_stack.sum(1).float().mean():.2f} "
        f"frame={frame_stack.sum(1).float().mean():.2f}"
    )

    return TransitionDataset(
        features_t=torch.stack(features_t),
        features_t1=torch.stack(features_t1),
        action_idx=torch.tensor(action_indices, dtype=torch.long),
        pre_masks=torch.stack(pre_masks),
        add_masks=add_stack,
        del_masks=del_stack,
        frame_masks=frame_stack,
        neg_pre_masks=torch.stack(neg_pre_l),
        neg_add_masks=torch.stack(neg_add_l),
        neg_del_masks=torch.stack(neg_del_l),
        object_type_ids=type_ids,
    )


def _feature_split(all_features, all_labels, state_ids, views_per_state):
    idx = np.concatenate([np.arange(si * views_per_state, (si + 1) * views_per_state) for si in state_ids])
    return all_features[idx], all_labels[idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-slot", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-equiv", type=float, default=0.15)
    parser.add_argument("--w-cf", type=float, default=0.05)
    parser.add_argument("--w-contrast", type=float, default=0.0)
    parser.add_argument("--k-values", default="50,100,200")
    parser.add_argument("--conditions", default="static,adjacent,full")
    parser.add_argument("--views-per-state", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--transition-warmup-epochs", type=int, default=20)
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--problem", default=str(BWS_PROBLEM))
    args = parser.parse_args()
    if args.train_seed is None:
        args.train_seed = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_values = _parse_k_values(args.k_values)
    conditions = _parse_conditions(args.conditions) or ["static", "adjacent", "full"]
    exp_name = args.exp_name or f"bws_structural_{int(time.time())}"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Blocksworld structural experiment: {exp_name}")
    print(f"  device={args.device} warmup={args.transition_warmup_epochs}")
    print(f"  k_values={k_values} conditions={conditions}")
    print(f"  w_equiv={args.w_equiv} w_cf={args.w_cf} w_contrast={args.w_contrast}")

    # Compile domain from the same problem objects used for state enumeration.
    problem_path = Path(args.problem)
    problem_objects, blocks, columns = _problem_objects(problem_path)
    compiler = PDDLDomainCompiler(str(BWS_DOMAIN))
    domain_info = compiler.compile(
        objects=problem_objects,
        static_predicates=STATIC_PREDS,
    )
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    compiled_blocks = [
        obj.name for obj in domain_info.objects
        if domain_info.types[obj.type_idx] == "block"
    ]
    compiled_columns = [
        obj.name for obj in domain_info.objects
        if domain_info.types[obj.type_idx] == "column"
    ]
    if compiled_blocks != blocks or compiled_columns != columns:
        raise RuntimeError(
            "Blocksworld object mismatch: problem/state enumeration and "
            f"compiled domain disagree. problem blocks={blocks}, "
            f"compiled blocks={compiled_blocks}, problem columns={columns}, "
            f"compiled columns={compiled_columns}"
        )
    print(
        f"  domain={domain_info.domain_name} objects={domain_info.n_objects} "
        f"atoms={domain_info.n_canonical} actions={len(domain_info.action_semantics)}"
    )
    print(f"  problem={problem_path}")
    print(f"  blocks={blocks} columns={columns}")

    # Enumerate ALL Blocksworld states and transitions
    print("  Enumerating states and transitions...")
    states, transitions = _enumerate_transitions(blocks, columns)
    n_states = len(states)
    print(f"  reachable states={n_states} transitions={len(transitions)}")

    # Build labels
    canonical_preds = []
    for atom in domain_info.canonical_atoms:
        canonical_preds.append(str(atom))
    state_labels = torch.stack([state_to_labels(s, canonical_preds) for s in states])
    print(f"  label_pos_rate={state_labels.float().mean():.4f}")

    # Synthetic features
    all_features = _build_synthetic_features(
        domain_info, state_labels,
        views_per_state=args.views_per_state,
        d_slot=args.d_slot,
        noise=args.noise,
        seed=args.seed,
    )
    expanded_labels = state_labels.repeat_interleave(args.views_per_state, dim=0)

    # Split: 15% val, 15% test, rest pool
    state_indices = np.random.default_rng(args.seed).permutation(n_states)
    n_val = max(1, int(0.15 * n_states))
    n_test = max(1, int(0.15 * n_states))
    val_idx = state_indices[:n_val]
    test_idx = state_indices[n_val:n_val + n_test]
    pool_idx = state_indices[n_val + n_test:]
    pool_set = set(int(x) for x in pool_idx.tolist())
    split_label_meta = _label_duplicate_metadata(state_labels, pool_idx, val_idx, test_idx)
    print(
        f"  unique_labels={split_label_meta['num_unique_labels']} "
        f"pool/test duplicate_rate={split_label_meta['duplicate_rate_pool_test']:.4f}"
    )

    val_feats, val_labels = _feature_split(all_features, expanded_labels, val_idx, args.views_per_state)
    test_feats, test_labels = _feature_split(all_features, expanded_labels, test_idx, args.views_per_state)

    state_val_ds = StateDataset(val_feats, val_labels, type_ids)
    state_test_ds = StateDataset(test_feats, test_labels, type_ids)

    # Transition dataset (state_diff masks — required for conditional effects)
    trans_ds = _build_transition_dataset(
        states=states,
        transitions=transitions,
        state_labels=state_labels,
        all_features=all_features,
        train_state_ids=pool_set,
        views_per_state=args.views_per_state,
        type_ids=type_ids,
        n_negatives=3,
        seed=args.seed,
        domain_info=domain_info,
    )

    all_results = {}
    for k in k_values:
        print(f"\n{'#' * 70}")
        print(f"# Blocksworld few-shot K={k}")
        print(f"{'#' * 70}")

        pool_feats, pool_labels = _feature_split(all_features, expanded_labels, pool_idx, args.views_per_state)
        pool_state_ids = np.repeat(pool_idx, args.views_per_state)

        state_train_ds, metadata, _ = _build_fewshot_state_dataset(
            pool_feats, pool_labels, type_ids,
            k=k, fewshot_unit="image", seed=args.seed,
            feat_per_state=args.views_per_state,
            feature_state_ids=pool_state_ids,
        )
        metadata["conditions"] = conditions
        metadata["train_seed"] = args.train_seed
        metadata["transition_warmup_epochs"] = args.transition_warmup_epochs
        metadata.update({
            "feature_source": "synthetic_object_token",
            "direct_object_tokens": True,
            "transition_mask_source": "state_diff",
            "transition_supervision": "oracle_state_diff",
            "object_type_source": "oracle",
            "domain": str(BWS_DOMAIN),
            "problem": str(problem_path),
            "num_objects": domain_info.n_objects,
            "num_canonical_atoms": domain_info.n_canonical,
            "blocks": blocks,
            "columns": columns,
            "threshold_source": "validation",
            **split_label_meta,
        })
        print(f"  Train: {metadata['n_labeled_samples']} samples, {metadata['n_labeled_states']} states")

        k_results = run_structural_experiment(
            domain_info=domain_info,
            state_train_ds=state_train_ds,
            state_val_ds=state_val_ds,
            state_test_ds=state_test_ds,
            trans_adjacent_ds=trans_ds,
            trans_random_ds=None,
            device=args.device,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            d_slot=args.d_slot,
            dinov3_kwargs=None,
            exp_dir=exp_dir / f"k_{k}",
            conditions=conditions,
            transition_mask_source="state_diff",
            direct_object_tokens=True,
            w_contrast=args.w_contrast,
            w_equiv=args.w_equiv,
            w_cf=args.w_cf,
            train_seed=args.train_seed,
            transition_warmup_epochs=args.transition_warmup_epochs,
        )
        all_results[str(k)] = {"_metadata": metadata, **k_results}

    # Save
    save_data = {}
    for k, k_res in all_results.items():
        save_data[k] = {
            "_metadata": k_res["_metadata"],
            **{
                cond: {
                    "test": res["test"],
                    "best_val_f1": res["best_val_f1"],
                    "best_threshold": res.get("best_threshold", 0.5),
                }
                for cond, res in k_res.items()
                if cond != "_metadata"
            },
        }
    with open(exp_dir / "bws_structural_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
