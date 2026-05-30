#!/usr/bin/env python3
"""Few-shot structural experiment with REAL DINOv3 features.

Core experiment: fix transition data, vary K labeled visual samples by default.
Use --fewshot-unit state to reproduce the old state-level setting.
Same 4 conditions per K: static, random_pairs, adjacent, full.

Uses cached DINOv3 features from experiments/dinov3_features_cached.pt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path("/home/claudeuser/RL4VLA/PDDL")))

from paq.domain_compiler import PDDLDomainCompiler
from training.train_aepaq import (
    enumerate_all_states, state_to_labels, generate_transitions,
    generate_random_pairs, _build_state_index, build_transition_features,
    StateDataset, TransitionDataset, _parse_k_values, _parse_conditions,
    _build_fewshot_state_dataset,
    run_structural_experiment,
)

BLOCKS = ["Y", "P", "R", "O"]
COLUMNS = ["C1", "C2", "C3", "C4"]
STATIC_PREDS = {"rightof", "leftof"}
N_VIEWS = 3
N_AUGS = 3

PDDL_ROOT = Path("/home/claudeuser/RL4VLA/PDDL")
BWS_DOMAIN = Path("/home/claudeuser/ViPlan") / "data" / "planning" / "blocksworld" / "domain.pddl"

DINOV3_KWARGS = {
    "use_dinov3": True,
    "dinov3_source": "local",
    "dinov3_repo_dir": str(Path("/home/claudeuser/facebookresearch/dinov3")),
    "dinov3_weights_path": str(PDDL_ROOT / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"),
}

TRANSITION_CACHE_VERSION = 3
DEFAULT_TRANSITION_CACHE = PDDL_ROOT / "experiments" / "fewshot_transition_cache.pt"


def _transition_cache_metadata(
    all_features, n_states, feat_per_state, pool_idx, canonical_preds, domain_info,
    transition_mask_source, need_random_pairs,
):
    """Metadata that must match for a transition cache to be reused."""
    return {
        "version": TRANSITION_CACHE_VERSION,
        "feature_shape": list(all_features.shape),
        "n_states": int(n_states),
        "feat_per_state": int(feat_per_state),
        "n_views": int(N_VIEWS),
        "n_augs": int(N_AUGS),
        "blocks": list(BLOCKS),
        "columns": list(COLUMNS),
        "static_preds": sorted(STATIC_PREDS),
        "pool_idx": [int(x) for x in pool_idx.tolist()],
        "canonical_preds": list(canonical_preds),
        "action_names": [a.action_name for a in domain_info.action_semantics],
        "n_negatives": 3,
        "split_seed": 42,
        "transition_mask_source": transition_mask_source,
        "need_random_pairs": bool(need_random_pairs),
    }


def _compact_transition_data(data):
    """Keep masks and feature indices, not duplicated feature tensors."""
    keys = [
        "feature_idx_t", "feature_idx_t1",
        "action_idx", "pre_mask", "add_mask", "del_mask", "frame_mask",
        "neg_pre_masks", "neg_add_masks", "neg_del_masks",
        "state_label_t", "state_label_t1",
    ]
    compact = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        compact[key] = value.cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return compact


def _transition_dataset_from_compact(data, all_features, type_ids):
    if data is None:
        return None
    idx_t = data["feature_idx_t"].long()
    idx_t1 = data["feature_idx_t1"].long()
    return TransitionDataset(
        features_t=all_features[idx_t],
        features_t1=all_features[idx_t1],
        action_idx=data["action_idx"],
        pre_masks=data["pre_mask"],
        add_masks=data["add_mask"],
        del_masks=data["del_mask"],
        frame_masks=data["frame_mask"],
        neg_pre_masks=data["neg_pre_masks"],
        neg_add_masks=data["neg_add_masks"],
        neg_del_masks=data["neg_del_masks"],
        object_type_ids=type_ids,
        state_labels_t=data.get("state_label_t"),
        state_labels_t1=data.get("state_label_t1"),
    )


def _load_cached_features() -> tuple[torch.Tensor, Path] | tuple[None, None]:
    """Load DINOv3 features from the canonical cache or an AE-PaQ run cache."""
    cache_path = PDDL_ROOT / "experiments" / "dinov3_features_cached.pt"
    candidates = [cache_path]
    candidates.extend(
        sorted(
            PDDL_ROOT.glob("experiments/aepaq_*/features.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    for path in candidates:
        if not path.exists():
            continue
        cached = torch.load(path, map_location="cpu")
        features = cached["features"] if isinstance(cached, dict) and "features" in cached else cached
        if isinstance(features, torch.Tensor):
            print(f"  Loaded features from {path}: {tuple(features.shape)}")
            return features, path
        print(f"  Ignoring unsupported feature cache payload: {path}")
    return None, None


def _transition_supervision_label(mask_source: str) -> str:
    if mask_source == "state_diff":
        return "oracle_state_diff"
    if mask_source in {"pddl", "pddl_conservative"}:
        return "static_pddl_weak"
    if mask_source == "pddl_sim":
        return "diagnostic_pddl_sim"
    return mask_source


def _subsample_dataset(dataset, max_samples: int | None, seed: int = 42):
    if dataset is None or max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False).tolist()
    return Subset(dataset, indices)


def _try_load_transition_cache(cache_path, metadata):
    if not cache_path.exists():
        return None
    try:
        cached = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"  Transition cache exists but failed to load: {exc}")
        return None
    if cached.get("metadata") != metadata:
        print("  Transition cache metadata mismatch; rebuilding.")
        return None
    return cached


def _remove_lock(lock_path):
    try:
        for child in lock_path.iterdir():
            child.unlink()
        lock_path.rmdir()
    except FileNotFoundError:
        pass


def _load_or_build_transition_cache(
    cache_path,
    metadata,
    pool_states,
    states,
    all_features,
    canonical_preds,
    domain_info,
    transition_mask_source,
    need_random_pairs=True,
    rebuild=False,
    use_cache=True,
):
    """Load/build transition data with a simple directory lock for parallel runs."""
    cache_path = Path(cache_path)
    lock_path = Path(str(cache_path) + ".lock")

    if use_cache and not rebuild:
        cached = _try_load_transition_cache(cache_path, metadata)
        if cached is not None:
            print(f"  Loaded transition cache: {cache_path}")
            return cached

    have_lock = False
    if use_cache:
        wait_started = time.time()
        warned = False
        while not have_lock:
            try:
                lock_path.mkdir(parents=True, exist_ok=False)
                (lock_path / "pid").write_text(str(os.getpid()))
                have_lock = True
            except FileExistsError:
                if not warned:
                    print(f"  Waiting for transition cache lock: {lock_path}")
                    warned = True
                if not rebuild:
                    cached = _try_load_transition_cache(cache_path, metadata)
                    if cached is not None:
                        print(f"  Loaded transition cache: {cache_path}")
                        return cached
                if time.time() - wait_started > 6 * 3600:
                    raise TimeoutError(f"Timed out waiting for transition cache lock: {lock_path}")
                if time.time() - lock_path.stat().st_mtime > 6 * 3600:
                    print("  Removing stale transition cache lock.")
                    _remove_lock(lock_path)
                    continue
                time.sleep(5)

        if not rebuild:
            cached = _try_load_transition_cache(cache_path, metadata)
            if cached is not None:
                _remove_lock(lock_path)
                print(f"  Loaded transition cache: {cache_path}")
                return cached

    try:
        print("  Building transition cache...")
        pool_states = list(pool_states)
        adj_transitions = generate_transitions(pool_states, BLOCKS, COLUMNS, include_action=True)
        print(f"  Adjacent transitions: {len(adj_transitions)}")

        rng = np.random.default_rng(42)
        random_pairs = generate_random_pairs(pool_states, len(adj_transitions), rng) if need_random_pairs else []
        print(f"  Random pairs: {len(random_pairs)}")

        adj_data = None
        rand_data = None
        if adj_transitions:
            adj_data = _compact_transition_data(build_transition_features(
                adj_transitions, states, all_features, N_VIEWS, N_AUGS,
                canonical_preds, domain_info, n_negatives=3,
                mask_source=transition_mask_source,
            ))
        if random_pairs:
            rand_data = _compact_transition_data(build_transition_features(
                random_pairs, states, all_features, N_VIEWS, N_AUGS,
                canonical_preds, domain_info, n_negatives=3,
                mask_source=transition_mask_source,
            ))

        cached = {
            "metadata": metadata,
            "n_adjacent_transitions": len(adj_transitions),
            "n_random_pairs": len(random_pairs),
            "adjacent": adj_data,
            "random_pairs": rand_data,
        }
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = Path(str(cache_path) + f".tmp.{os.getpid()}")
            torch.save(cached, tmp_path)
            tmp_path.replace(cache_path)
            print(f"  Saved transition cache: {cache_path}")
        return cached
    finally:
        if have_lock:
            _remove_lock(lock_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-slot", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--w-equiv", type=float, default=1.0)
    parser.add_argument("--w-cf", type=float, default=0.3)
    parser.add_argument("--w-contrast", type=float, default=0.1)
    parser.add_argument("--pos-weight-max", type=float, default=20.0)
    parser.add_argument("--scoring-head-type", choices=["film", "legacy"], default="film")
    parser.add_argument("--transition-warmup-epochs", type=int, default=20)
    parser.add_argument(
        "--max-transition-samples",
        type=int,
        default=0,
        help="Optional deterministic transition subsample for quick ablations. 0 uses all transitions.",
    )
    parser.add_argument("--fewshot-unit", choices=["image", "state"], default="image")
    parser.add_argument("--k-values", default="20,50,100,200")
    parser.add_argument("--transition-cache", default=str(DEFAULT_TRANSITION_CACHE))
    parser.add_argument("--rebuild-transition-cache", action="store_true")
    parser.add_argument("--no-transition-cache", action="store_true")
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
    parser.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated conditions to run: static,random_pairs,adjacent,full",
    )
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()
    k_values = _parse_k_values(args.k_values)
    selected_conditions = _parse_conditions(args.conditions) or ["static", "random_pairs", "adjacent", "full"]

    torch.manual_seed(42)
    np.random.seed(42)

    exp_name = args.exp_name or f"aepaq_fewshot_{int(time.time())}"
    exp_dir = PDDL_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load cached DINOv3 features ----
    print("Loading cached DINOv3 features...")
    all_features, feature_cache_path = _load_cached_features()
    if all_features is None:
        print(f"ERROR: Cached features not found under {PDDL_ROOT / 'experiments'}")
        print("Run feature extraction first, or run training/train_aepaq.py once to create features.pt.")
        return

    # ---- Compile PDDL domain ----
    print("\nCompiling PDDL domain...")
    compiler = PDDLDomainCompiler(str(BWS_DOMAIN))
    domain_info = compiler.compile(
        objects={"block": BLOCKS, "column": COLUMNS},
        static_predicates=STATIC_PREDS,
    )
    canonical_preds = domain_info.canonical_atom_strings
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    print(f"  {domain_info.n_canonical} atoms, {len(domain_info.action_semantics)} actions")

    # ---- Enumerate states & labels ----
    print("\nEnumerating states...")
    states = enumerate_all_states(BLOCKS, COLUMNS)
    n_states = len(states)
    feat_per_state = N_VIEWS * N_AUGS  # 9
    all_labels = torch.stack([state_to_labels(s, canonical_preds) for s in states])
    expanded_labels = all_labels.repeat_interleave(feat_per_state, dim=0)
    print(f"  {n_states} states, feat_per_state={feat_per_state}")

    assert all_features.shape[0] == n_states * feat_per_state, \
        f"Feature count mismatch: {all_features.shape[0]} vs {n_states * feat_per_state}"

    # ---- Split by state (not by feature) ----
    state_indices = np.random.permutation(n_states)
    n_val = int(0.15 * n_states)
    n_test = int(0.15 * n_states)
    val_idx = state_indices[:n_val]
    test_idx = state_indices[n_val:n_val + n_test]
    pool_idx = state_indices[n_val + n_test:]  # remaining ~70% for few-shot sampling

    def get_features(idx_array):
        feat_indices = np.concatenate([
            np.arange(si * feat_per_state, (si + 1) * feat_per_state) for si in idx_array
        ])
        return all_features[feat_indices], expanded_labels[feat_indices]

    val_feats, val_labels = get_features(val_idx)
    test_feats, test_labels = get_features(test_idx)
    pool_feats, pool_labels = get_features(pool_idx)
    pool_feature_state_ids = np.repeat(pool_idx, feat_per_state)

    state_val_ds = StateDataset(val_feats, val_labels, type_ids)
    state_test_ds = StateDataset(test_feats, test_labels, type_ids)
    print(f"  Val: {len(state_val_ds)}, Test: {len(state_test_ds)}, Pool: {len(pool_feats)}")

    # ---- Load/build transition data (from ALL pool states) ----
    print("\nPreparing transition data...")
    pool_states = [states[i] for i in pool_idx]
    need_random_pairs = "random_pairs" in selected_conditions
    transition_metadata = _transition_cache_metadata(
        all_features, n_states, feat_per_state, pool_idx, canonical_preds, domain_info,
        args.transition_mask_source, need_random_pairs,
    )
    transition_cache = _load_or_build_transition_cache(
        cache_path=Path(args.transition_cache),
        metadata=transition_metadata,
        pool_states=pool_states,
        states=states,
        all_features=all_features,
        canonical_preds=canonical_preds,
        domain_info=domain_info,
        transition_mask_source=args.transition_mask_source,
        need_random_pairs=need_random_pairs,
        rebuild=args.rebuild_transition_cache,
        use_cache=not args.no_transition_cache,
    )
    adj_transitions_count = int(transition_cache["n_adjacent_transitions"])
    random_pairs_count = int(transition_cache["n_random_pairs"])
    trans_adjacent_ds = _transition_dataset_from_compact(
        transition_cache["adjacent"], all_features, type_ids,
    )
    trans_random_ds = _transition_dataset_from_compact(
        transition_cache["random_pairs"], all_features, type_ids,
    )
    trans_adjacent_ds = _subsample_dataset(
        trans_adjacent_ds, args.max_transition_samples, seed=42,
    )
    trans_random_ds = _subsample_dataset(
        trans_random_ds, args.max_transition_samples, seed=43,
    )
    print(f"  Adjacent transitions: {adj_transitions_count}")
    print(f"  Random pairs: {random_pairs_count}")
    if args.max_transition_samples > 0:
        print(
            "  Transition samples used per epoch: "
            f"adjacent={len(trans_adjacent_ds) if trans_adjacent_ds else 0} "
            f"random={len(trans_random_ds) if trans_random_ds else 0}"
        )

    # ---- Run few-shot experiments ----
    all_results = {}

    for k in k_values:
        print(f"\n{'#'*70}")
        if args.fewshot_unit == "image":
            print(f"# FEW-SHOT K={k} labeled visual samples/images")
        else:
            print(f"# FEW-SHOT K_state={k} symbolic states")
        print(f"{'#'*70}")

        state_train_ds, metadata, _ = _build_fewshot_state_dataset(
            pool_feats,
            pool_labels,
            type_ids,
            k=k,
            fewshot_unit=args.fewshot_unit,
            seed=42,
            feat_per_state=feat_per_state,
            feature_state_ids=pool_feature_state_ids,
        )
        metadata["transition_mask_source"] = args.transition_mask_source
        metadata["conditions"] = selected_conditions
        metadata["transition_scope"] = "all_pool"
        metadata["transition_warmup_epochs"] = args.transition_warmup_epochs
        metadata["w_equiv"] = args.w_equiv
        metadata["w_cf"] = args.w_cf
        metadata["w_contrast"] = args.w_contrast
        metadata["pos_weight_max"] = args.pos_weight_max
        metadata["scoring_head_type"] = args.scoring_head_type
        metadata["max_transition_samples"] = args.max_transition_samples
        metadata.update({
            "feature_source": "cached_dinov3",
            "feature_cache_path": str(feature_cache_path),
            "feature_shape": list(all_features.shape),
            "direct_object_tokens": False,
            "transition_supervision": _transition_supervision_label(args.transition_mask_source),
            "object_type_source": "oracle",
            "domain": str(BWS_DOMAIN),
            "problem": None,
            "num_objects": domain_info.n_objects,
            "num_canonical_atoms": domain_info.n_canonical,
            "train_seed": 42,
            "split_seed": 42,
            "threshold_source": "validation",
            "checkpoint_source": None,
        })
        print(
            f"  Train: {metadata['n_labeled_samples']} visual samples "
            f"covering {metadata['n_labeled_states']} states"
        )
        print(f"  Transitions: {len(trans_adjacent_ds) if trans_adjacent_ds else 0}")
        print(f"  Conditions: {selected_conditions}")
        print(f"  Transition mask source: {args.transition_mask_source}")

        k_results = run_structural_experiment(
            domain_info=domain_info,
            state_train_ds=state_train_ds,
            state_val_ds=state_val_ds,
            state_test_ds=state_test_ds,
            trans_adjacent_ds=trans_adjacent_ds,
            trans_random_ds=trans_random_ds,
            device=args.device,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            d_slot=args.d_slot,
            dinov3_kwargs=DINOV3_KWARGS,
            exp_dir=exp_dir / f"k_{k}",
            conditions=selected_conditions,
            transition_mask_source=args.transition_mask_source,
            w_contrast=args.w_contrast,
            w_equiv=args.w_equiv,
            w_cf=args.w_cf,
            transition_warmup_epochs=args.transition_warmup_epochs,
            pos_weight_max=args.pos_weight_max,
            scoring_head_type=args.scoring_head_type,
        )
        all_results[k] = {
            "_metadata": metadata,
            **k_results,
        }

    # ---- Final summary ----
    print()
    print("=" * 90)
    print("FEW-SHOT STRUCTURAL EXPERIMENT (Real DINOv3 Features)")
    print(f"  {n_states} total states, {len(pool_idx)} pool for sampling")
    print(f"  {adj_transitions_count} transitions, {args.n_epochs} epochs per condition")
    print(f"  Few-shot unit: {args.fewshot_unit}")
    print(f"  Transition mask source: {args.transition_mask_source}")
    print(f"  Transition warmup epochs: {args.transition_warmup_epochs}")
    print(f"  Conditions: {selected_conditions}")
    print("=" * 90)

    # Table: K × Condition → F1
    print()
    header = f"{'K':>6} {'Unit':>8} {'Samples':>8} {'States':>8}"
    for cond in selected_conditions:
        header += f" | {cond:>14}"
    if "static" in selected_conditions:
        header += f" | {'Δ(adj-st)':>10} {'Δ(full-st)':>10}"
    print(header)
    print("-" * len(header))

    for k in k_values:
        meta = all_results[k]["_metadata"]
        row = (
            f"{k:>6} {meta['fewshot_unit']:>8} "
            f"{meta['n_labeled_samples']:>8} "
            f"{str(meta['n_labeled_states']):>8}"
        )
        for cond in selected_conditions:
            f1 = all_results[k][cond]["test"]["f1"]
            row += f" | {f1:>14.3f}"
        if "static" in selected_conditions:
            base_f1 = all_results[k]["static"]["test"]["f1"]
            adj_delta = (
                all_results[k]["adjacent"]["test"]["f1"] - base_f1
                if "adjacent" in all_results[k] else float("nan")
            )
            full_delta = (
                all_results[k]["full"]["test"]["f1"] - base_f1
                if "full" in all_results[k] else float("nan")
            )
            row += f" | {adj_delta:>+10.3f} {full_delta:>+10.3f}"
        print(row)

    # Save
    save_data = {}
    for k, k_res in all_results.items():
        save_data[str(k)] = {
            "_metadata": k_res["_metadata"],
            **{
                cond: {
                    "test": res["test"],
                    "best_val_f1": res["best_val_f1"],
                    "best_threshold": res.get("best_threshold", 0.5),
                    "transition_mask_source": res.get("transition_mask_source", args.transition_mask_source),
                }
                for cond, res in k_res.items()
                if cond != "_metadata"
            }
        }
    with open(exp_dir / "fewshot_structural_results.json", "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
