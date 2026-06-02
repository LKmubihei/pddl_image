#!/usr/bin/env python3
"""Few-shot structural experiment on a STRIPS-style domain.

This runner uses the TV screw assembly PDDL domain in ``solver/domain.pddl``.
Its action effects are unconditional, so static PDDL masks are complete and
``--transition-mask-source pddl`` is a real B-style weak-supervision setting.
The visual input is synthetic object-token features generated from predicate
states; no rendering or DINO cache is required.
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
from pddl_parser import PDDLProblemParser
from training.data.ae_dataset import StateDataset, TransitionDataset
from training.data.transition_sampler import TransitionSampler
from training.train_aepaq import (
    _build_fewshot_state_dataset,
    _parse_conditions,
    _parse_k_values,
    run_structural_experiment,
)


DOMAIN_PATH = ROOT / "solver" / "domain.pddl"
PROBLEM_PATH = ROOT / "solver" / "p_real.pddl"
STATIC_PREDS = {"screw-for-hole", "requires-predecessor"}
VALID_MASK_SOURCES = {"pddl"}


def _transition_supervision_label(mask_source: str) -> str:
    if mask_source == "pddl":
        return "static_pddl_weak"
    return mask_source


def _predicate_name(atom: str) -> str:
    return atom.strip("()").split()[0]


def _initial_dynamic_state(problem: PDDLProblemParser) -> set[str]:
    return {
        atom for atom in problem.init_state
        if _predicate_name(atom) not in STATIC_PREDS
    }


def _collect_sampler_states(sampler: TransitionSampler) -> tuple[torch.Tensor, dict[bytes, int]]:
    vectors: list[np.ndarray] = []
    index: dict[bytes, int] = {}

    def add(vec: np.ndarray):
        key = vec.astype(np.float32).tobytes()
        if key not in index:
            index[key] = len(vectors)
            vectors.append(vec.astype(np.float32, copy=True))

    add(sampler.state_to_vector(sampler.initial_state))
    for tr in sampler.transitions:
        add(tr["state_t_vec"])
        add(tr["state_t1_vec"])

    return torch.from_numpy(np.stack(vectors)).float(), index


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
    n_atoms = domain_info.n_canonical

    base = rng.normal(0.0, 0.18, size=(n_objects, d_slot)).astype(np.float32)
    type_centers = rng.normal(
        0.0, 0.70, size=(domain_info.n_types, d_slot),
    ).astype(np.float32)
    for oi, obj in enumerate(domain_info.objects):
        base[oi] += type_centers[obj.type_idx]
        identity_dim = 8 + oi
        if identity_dim < d_slot:
            base[oi, identity_dim] += 1.5

    schema_vecs = rng.normal(
        0.0, 0.75, size=(len(domain_info.predicate_schemas), d_slot),
    ).astype(np.float32)
    role_vecs = rng.normal(0.0, 0.25, size=(3, d_slot)).astype(np.float32)
    partner_vecs = rng.normal(0.0, 0.35, size=(n_objects, d_slot)).astype(np.float32)
    obj_name_to_idx = {
        obj.name.lower(): i for i, obj in enumerate(domain_info.objects)
    }
    atom_to_objects: list[list[int]] = []
    for atom in domain_info.canonical_atoms:
        obj_ids = [
            obj_name_to_idx[arg.lower()]
            for arg in atom.arguments
            if arg.lower() in obj_name_to_idx
        ]
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
            aug = state_feat + rng.normal(
                0.0, noise, size=state_feat.shape,
            ).astype(np.float32)
            features.append(aug)
    return torch.from_numpy(np.stack(features)).float()


def _diff_masks(
    state_t: torch.Tensor,
    state_t1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    add = ((state_t1 > 0.5) & (state_t <= 0.5)).float()
    delete = ((state_t > 0.5) & (state_t1 <= 0.5)).float()
    frame = 1.0 - torch.clamp(add + delete, max=1.0)
    return add, delete, frame


def _masks_for_transition(
    sampler: TransitionSampler,
    action_masks: dict[str, torch.Tensor],
    action_idx: int,
    state_t: torch.Tensor,
    state_t1: torch.Tensor,
    mask_source: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_source != "pddl":
        raise ValueError(f"Only PDDL transition masks are supported; got {mask_source}")
    pre = action_masks["precondition_mask"][action_idx].clone()
    add = action_masks["add_mask"][action_idx].clone()
    delete = action_masks["del_mask"][action_idx].clone()
    frame = action_masks["frame_mask"][action_idx].clone()
    return pre, add, delete, frame


def _negative_masks(
    sampler: TransitionSampler,
    action_masks: dict[str, torch.Tensor],
    neg_action_idx: int,
    state_t: torch.Tensor,
    mask_source: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_source != "pddl":
        raise ValueError(f"Only PDDL transition masks are supported; got {mask_source}")
    pre = action_masks["precondition_mask"][neg_action_idx].clone()
    return (
        pre,
        action_masks["add_mask"][neg_action_idx].clone(),
        action_masks["del_mask"][neg_action_idx].clone(),
    )


def _transition_dataset_from_sampler(
    sampler: TransitionSampler,
    domain_info,
    state_labels: torch.Tensor,
    state_index: dict[bytes, int],
    all_features: torch.Tensor,
    train_state_ids: set[int],
    views_per_state: int,
    type_ids: torch.Tensor,
    mask_source: str,
    n_negatives: int,
    seed: int,
    include_state_labels: bool = False,
) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    action_masks = domain_info.get_action_masks_tensor(device="cpu")
    n_canon = domain_info.n_canonical

    features_t, features_t1 = [], []
    action_indices = []
    pre_masks, add_masks, del_masks, frame_masks = [], [], [], []
    neg_pre_l, neg_add_l, neg_del_l = [], [], []
    state_labels_t_l, state_labels_t1_l = [], []

    for tr in sampler.transitions:
        t_idx = state_index.get(tr["state_t_vec"].astype(np.float32).tobytes())
        t1_idx = state_index.get(tr["state_t1_vec"].astype(np.float32).tobytes())
        if t_idx is None or t1_idx is None:
            continue
        if t_idx not in train_state_ids or t1_idx not in train_state_ids:
            continue

        feat_t_idx = t_idx * views_per_state + int(rng.integers(views_per_state))
        feat_t1_idx = t1_idx * views_per_state + int(rng.integers(views_per_state))
        features_t.append(all_features[feat_t_idx])
        features_t1.append(all_features[feat_t1_idx])

        action_idx = int(tr["action_idx"])
        state_t = state_labels[t_idx]
        state_t1 = state_labels[t1_idx]
        if include_state_labels:
            state_labels_t_l.append(state_t)
            state_labels_t1_l.append(state_t1)
        pre, add, delete, frame = _masks_for_transition(
            sampler, action_masks, action_idx, state_t, state_t1, mask_source,
        )
        action_indices.append(action_idx)
        pre_masks.append(pre)
        add_masks.append(add)
        del_masks.append(delete)
        frame_masks.append(frame)

        candidates = [a for a in range(sampler.n_actions) if a != action_idx]
        chosen = rng.choice(
            candidates, size=min(n_negatives, len(candidates)), replace=False,
        )
        neg_pre = torch.zeros(n_negatives, n_canon)
        neg_add = torch.zeros(n_negatives, n_canon)
        neg_del = torch.zeros(n_negatives, n_canon)
        for k, neg_idx in enumerate(chosen[:n_negatives]):
            np_m, na_m, nd_m = _negative_masks(
                sampler, action_masks, int(neg_idx), state_t, mask_source,
            )
            neg_pre[k] = np_m
            neg_add[k] = na_m
            neg_del[k] = nd_m
        neg_pre_l.append(neg_pre)
        neg_add_l.append(neg_add)
        neg_del_l.append(neg_del)

    if not features_t:
        raise RuntimeError("No train-split transitions were produced.")

    add_stack = torch.stack(add_masks)
    del_stack = torch.stack(del_masks)
    frame_stack = torch.stack(frame_masks)
    print(
        f"  Adjacent masks ({mask_source}): "
        f"add={add_stack.sum(1).float().mean().item():.2f} "
        f"del={del_stack.sum(1).float().mean().item():.2f} "
        f"frame={frame_stack.sum(1).float().mean().item():.2f}"
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
        state_labels_t=torch.stack(state_labels_t_l) if include_state_labels else None,
        state_labels_t1=torch.stack(state_labels_t1_l) if include_state_labels else None,
    )


def _random_pair_dataset(
    all_features: torch.Tensor,
    state_labels: torch.Tensor,
    train_state_ids: np.ndarray,
    n_pairs: int,
    views_per_state: int,
    type_ids: torch.Tensor,
    n_canon: int,
    n_negatives: int,
    seed: int,
    include_state_labels: bool = False,
) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    features_t, features_t1 = [], []
    labels_t, labels_t1 = [], []
    for _ in range(n_pairs):
        t_idx, t1_idx = rng.choice(train_state_ids, size=2, replace=False)
        features_t.append(all_features[t_idx * views_per_state + int(rng.integers(views_per_state))])
        features_t1.append(all_features[t1_idx * views_per_state + int(rng.integers(views_per_state))])
        if include_state_labels:
            labels_t.append(state_labels[t_idx])
            labels_t1.append(state_labels[t1_idx])

    zeros = torch.zeros(n_pairs, n_canon)
    return TransitionDataset(
        features_t=torch.stack(features_t),
        features_t1=torch.stack(features_t1),
        action_idx=torch.zeros(n_pairs, dtype=torch.long),
        pre_masks=zeros,
        add_masks=zeros.clone(),
        del_masks=zeros.clone(),
        frame_masks=torch.ones(n_pairs, n_canon),
        neg_pre_masks=torch.zeros(n_pairs, n_negatives, n_canon),
        neg_add_masks=torch.zeros(n_pairs, n_negatives, n_canon),
        neg_del_masks=torch.zeros(n_pairs, n_negatives, n_canon),
        object_type_ids=type_ids,
        state_labels_t=torch.stack(labels_t) if include_state_labels else None,
        state_labels_t1=torch.stack(labels_t1) if include_state_labels else None,
    )


def _feature_split(
    all_features: torch.Tensor,
    all_labels_expanded: torch.Tensor,
    state_ids: np.ndarray,
    views_per_state: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_indices = np.concatenate([
        np.arange(si * views_per_state, (si + 1) * views_per_state)
        for si in state_ids
    ])
    return all_features[feat_indices], all_labels_expanded[feat_indices]


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
    parser.add_argument("--w-contrast", type=float, default=0.1)
    parser.add_argument("--w-transition-seed", type=float, default=0.0)
    parser.add_argument("--w-transition-type", type=float, default=0.0)
    parser.add_argument("--w-transition-contrast", type=float, default=0.0)
    parser.add_argument("--contrast-temperature", type=float, default=0.5)
    parser.add_argument(
        "--transition-warmup-epochs",
        type=int,
        default=0,
        help="Run C1/state batches only for this many epochs before transition losses.",
    )
    parser.add_argument(
        "--transition-state-labels",
        action="store_true",
        help="Attach true S_t/S_t1 labels to transition batches. This is extra supervision.",
    )
    parser.add_argument("--k-values", default="20,50,100")
    parser.add_argument("--conditions", default="static,random_pairs,adjacent,full")
    parser.add_argument("--transition-mask-source", default="pddl", choices=sorted(VALID_MASK_SOURCES))
    parser.add_argument("--max-states", type=int, default=800)
    parser.add_argument("--views-per-state", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Model/training RNG seed reused for each condition. Defaults to --seed.",
    )
    parser.add_argument("--exp-name", default=None)
    args = parser.parse_args()
    if args.train_seed is None:
        args.train_seed = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    k_values = _parse_k_values(args.k_values)
    conditions = _parse_conditions(args.conditions) or ["static", "random_pairs", "adjacent", "full"]
    exp_name = args.exp_name or f"strips_structural_{int(time.time())}"
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"STRIPS structural experiment: {exp_name}")
    print(f"  device={args.device} mask_source={args.transition_mask_source}")
    print(
        f"  k_values={k_values} conditions={conditions} lr={args.lr} "
        f"w_equiv={args.w_equiv} w_cf={args.w_cf} "
        f"w_contrast={args.w_contrast} temp={args.contrast_temperature} "
        f"transition_warmup={args.transition_warmup_epochs}"
    )
    if args.transition_state_labels:
        print(
            "  transition_state_labels=ON "
            f"w_transition_seed={args.w_transition_seed}"
        )

    problem = PDDLProblemParser(str(PROBLEM_PATH))
    compiler = PDDLDomainCompiler(str(DOMAIN_PATH))
    domain_info = compiler.compile(
        objects=problem.objects,
        static_predicates=STATIC_PREDS,
    )
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    print(
        f"  domain={domain_info.domain_name} objects={domain_info.n_objects} "
        f"atoms={domain_info.n_canonical} actions={len(domain_info.action_semantics)}"
    )

    sampler = TransitionSampler(
        domain_info=domain_info,
        initial_state=_initial_dynamic_state(problem),
        max_states=args.max_states,
        seed=args.seed,
    )
    state_labels, state_index = _collect_sampler_states(sampler)
    print(f"  reachable states={len(state_labels)} transitions={sampler.n_transitions}")

    all_features = _build_synthetic_features(
        domain_info, state_labels,
        views_per_state=args.views_per_state,
        d_slot=args.d_slot,
        noise=args.noise,
        seed=args.seed,
    )
    expanded_labels = state_labels.repeat_interleave(args.views_per_state, dim=0)
    feature_state_ids = np.repeat(np.arange(len(state_labels)), args.views_per_state)

    state_indices = np.random.default_rng(args.seed).permutation(len(state_labels))
    n_val = max(1, int(0.15 * len(state_indices)))
    n_test = max(1, int(0.15 * len(state_indices)))
    val_idx = state_indices[:n_val]
    test_idx = state_indices[n_val:n_val + n_test]
    pool_idx = state_indices[n_val + n_test:]
    pool_set = set(int(x) for x in pool_idx.tolist())

    val_feats, val_labels = _feature_split(
        all_features, expanded_labels, val_idx, args.views_per_state,
    )
    test_feats, test_labels = _feature_split(
        all_features, expanded_labels, test_idx, args.views_per_state,
    )
    pool_feats, pool_labels = _feature_split(
        all_features, expanded_labels, pool_idx, args.views_per_state,
    )
    pool_feature_state_ids = feature_state_ids[np.concatenate([
        np.arange(si * args.views_per_state, (si + 1) * args.views_per_state)
        for si in pool_idx
    ])]

    state_val_ds = StateDataset(val_feats, val_labels, type_ids)
    state_test_ds = StateDataset(test_feats, test_labels, type_ids)
    trans_adjacent_ds = _transition_dataset_from_sampler(
        sampler=sampler,
        domain_info=domain_info,
        state_labels=state_labels,
        state_index=state_index,
        all_features=all_features,
        train_state_ids=pool_set,
        views_per_state=args.views_per_state,
        type_ids=type_ids,
        mask_source=args.transition_mask_source,
        n_negatives=3,
        seed=args.seed,
        include_state_labels=args.transition_state_labels,
    )
    trans_random_ds = _random_pair_dataset(
        all_features=all_features,
        state_labels=state_labels,
        train_state_ids=pool_idx,
        n_pairs=len(trans_adjacent_ds),
        views_per_state=args.views_per_state,
        type_ids=type_ids,
        n_canon=domain_info.n_canonical,
        n_negatives=3,
        seed=args.seed,
        include_state_labels=args.transition_state_labels,
    ) if "random_pairs" in conditions else None

    all_results = {}
    for k in k_values:
        print(f"\n{'#' * 70}")
        print(f"# STRIPS few-shot K={k} labeled synthetic visual samples")
        print(f"{'#' * 70}")
        state_train_ds, metadata, _ = _build_fewshot_state_dataset(
            pool_feats,
            pool_labels,
            type_ids,
            k=k,
            fewshot_unit="image",
            seed=args.seed,
            feat_per_state=args.views_per_state,
            feature_state_ids=pool_feature_state_ids,
        )
        metadata["transition_mask_source"] = args.transition_mask_source
        metadata["conditions"] = conditions
        metadata["train_seed"] = args.train_seed
        metadata["transition_state_labels"] = args.transition_state_labels
        metadata["w_transition_seed"] = args.w_transition_seed
        metadata["transition_warmup_epochs"] = args.transition_warmup_epochs
        metadata.update({
            "feature_source": "synthetic_object_token",
            "feature_cache_path": None,
            "direct_object_tokens": True,
            "transition_supervision": _transition_supervision_label(args.transition_mask_source),
            "object_type_source": "oracle",
            "domain": str(DOMAIN_PATH),
            "problem": str(PROBLEM_PATH),
            "num_objects": domain_info.n_objects,
            "num_canonical_atoms": domain_info.n_canonical,
            "split_seed": args.seed,
            "threshold_source": "validation",
            "checkpoint_source": None,
        })
        print(
            f"  Train: {metadata['n_labeled_samples']} samples "
            f"covering {metadata['n_labeled_states']} states"
        )

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
            dinov3_kwargs=None,
            exp_dir=exp_dir / f"k_{k}",
            conditions=conditions,
            transition_mask_source=args.transition_mask_source,
            direct_object_tokens=True,
            w_contrast=args.w_contrast,
            w_equiv=args.w_equiv,
            w_cf=args.w_cf,
            w_transition_seed=args.w_transition_seed,
            w_transition_type=args.w_transition_type,
            w_transition_contrast=args.w_transition_contrast,
            contrast_temperature=args.contrast_temperature,
            transition_warmup_epochs=args.transition_warmup_epochs,
            train_seed=args.train_seed,
        )
        all_results[str(k)] = {"_metadata": metadata, **k_results}

    save_data = {}
    for k, k_res in all_results.items():
        save_data[k] = {
            "_metadata": k_res["_metadata"],
            **{
                cond: {
                    "test": res["test"],
                    "best_val_f1": res["best_val_f1"],
                    "best_threshold": res.get("best_threshold", 0.5),
                    "transition_mask_source": res.get(
                        "transition_mask_source", args.transition_mask_source,
                    ),
                }
                for cond, res in k_res.items()
                if cond != "_metadata"
            },
        }
    with open(exp_dir / "strips_structural_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    with open(exp_dir / "domain_info.json", "w") as f:
        json.dump(domain_info.summary(), f, indent=2, default=str)

    print(f"\nResults saved to {exp_dir}")


if __name__ == "__main__":
    main()
