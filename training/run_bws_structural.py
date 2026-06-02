#!/usr/bin/env python3
"""Blocksworld structural experiment with synthetic features.

This runner is a structural sanity check, not a visual grounding result.  It
uses direct synthetic object tokens and oracle object types.  Transition pairs
are stable-state options compiled from primitive PDDL actions.

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
_LOCAL_BWS_DIR = ROOT / "data" / "planning" / "blocksworld"
_LEGACY_BWS_DIR = Path("/home/claudeuser/ViPlan") / "data" / "planning" / "blocksworld"
_DEFAULT_BWS_DOMAIN = _LOCAL_BWS_DIR / "domain.pddl"
BWS_DOMAIN = (
    _DEFAULT_BWS_DOMAIN
    if _DEFAULT_BWS_DOMAIN.exists()
    else _LEGACY_BWS_DIR / "domain.pddl"
)
BWS_PROBLEM = (
    _LOCAL_BWS_DIR / "problem.pddl"
    if (_LOCAL_BWS_DIR / "problem.pddl").exists()
    else _LEGACY_BWS_DIR / "problems" / "simple" / "simple_problem_0.pddl"
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
# Blocksworld option transitions
# ---------------------------------------------------------------------------
def _parse_grounded_action_args(action_name: str) -> tuple[str, list[str]]:
    name, _, rest = action_name.partition("(")
    args = rest.rstrip(")")
    return name.strip().lower(), [a.strip() for a in args.split(",") if a.strip()]


def _action_index(domain_info, name: str, args: list[str]) -> int:
    wanted = (name.lower(), [str(a) for a in args])
    for idx, action in enumerate(domain_info.action_semantics):
        if _parse_grounded_action_args(action.action_name) == wanted:
            return idx
    raise KeyError(f"Could not find grounded action {name}({', '.join(args)})")


def _support_assignment(state, blocks: list[str], columns: list[str]) -> dict[str, str]:
    assignment = {}
    for b in blocks:
        below = [x for x in blocks if state.on.get((b, x), False)]
        if below:
            assignment[b] = below[0]
            continue
        cols = [c for c in columns if state.inColumn.get((b, c), False)]
        if len(cols) != 1:
            raise ValueError(f"State has no unique column support for {b}")
        assignment[b] = cols[0]
    return assignment


def _chain_end_column(block: str, assignment: dict[str, str], columns: set[str]) -> str:
    cur = block
    seen = set()
    while True:
        if cur in seen:
            raise ValueError(f"Cycle in support assignment at {block}")
        seen.add(cur)
        support = assignment[cur]
        if support in columns:
            return support
        cur = support


def _atoms_from_assignment(
    assignment: dict[str, str],
    blocks: list[str],
    columns: list[str],
) -> set[str]:
    column_set = set(columns)
    atoms = set()
    for b in blocks:
        support = assignment[b]
        if support in blocks:
            atoms.add(f"(on {b} {support})")
        end_col = _chain_end_column(b, assignment, column_set)
        atoms.add(f"(inColumn {b} {end_col})")
    for b in blocks:
        if not any(assignment[x] == b for x in blocks):
            atoms.add(f"(clear {b})")
    return atoms


def _label_from_atoms(atoms: set[str], atom_to_idx: dict[str, int], n_canon: int) -> torch.Tensor:
    label = torch.zeros(n_canon, dtype=torch.long)
    for atom in atoms:
        idx = atom_to_idx.get(atom)
        if idx is not None:
            label[idx] = 1
    return label


def _option_pre_mask(
    moving: str,
    source_support: str,
    target_support: str,
    source_col: str,
    target_col: str | None,
    blocks: list[str],
    atom_to_idx: dict[str, int],
    n_canon: int,
) -> torch.Tensor:
    atoms = {f"(clear {moving})"}
    if source_support in blocks:
        atoms.add(f"(on {moving} {source_support})")
    else:
        atoms.add(f"(inColumn {moving} {source_support})")
    atoms.add(f"(inColumn {moving} {source_col})")
    if target_support in blocks:
        atoms.add(f"(clear {target_support})")
        if target_col is not None:
            atoms.add(f"(inColumn {target_support} {target_col})")
    return _label_from_atoms(atoms, atom_to_idx, n_canon).float()


def _primitive_plan_for_option(
    moving: str,
    source_support: str,
    target_support: str,
    source_col: str,
    blocks: list[str],
    domain_info,
) -> tuple[list[int], list[str], str]:
    if source_support in blocks:
        first_name = "unstack"
        first_args = [moving, source_support]
        source_kind = "block"
    else:
        first_name = "pickup"
        first_args = [moving, source_col]
        source_kind = "column"

    if target_support in blocks:
        second_name = "stack"
        second_args = [moving, target_support]
        target_kind = "block"
    else:
        second_name = "putdown"
        second_args = [moving, target_support]
        target_kind = "column"

    first_idx = _action_index(domain_info, first_name, first_args)
    second_idx = _action_index(domain_info, second_name, second_args)
    plan = [domain_info.action_semantics[first_idx].action_name, domain_info.action_semantics[second_idx].action_name]
    return [first_idx, second_idx], plan, f"{source_kind}_to_{target_kind}"


def _enumerate_option_transitions(states, state_labels: torch.Tensor, domain_info, blocks, columns):
    """Compile stable-state options from primitive PDDL actions.

    Each transition corresponds to a two-action primitive plan:
    pickup/unstack followed by putdown/stack.  Intermediate primitive states are
    latent; the supervision masks are the final stable-state option effects.
    """
    labels = state_labels.long()
    label_to_idx = {tuple(row.tolist()): i for i, row in enumerate(labels)}
    atom_to_idx = {a: i for i, a in enumerate(domain_info.canonical_atom_strings)}
    n_canon = domain_info.n_canonical

    transitions = []
    by_option: dict[str, int] = {}
    columns_set = set(columns)
    for state_idx, state in enumerate(states):
        assignment = _support_assignment(state, blocks, columns)
        clear_blocks = [b for b in blocks if state.clear.get(b, False)]

        for moving in clear_blocks:
            source_support = assignment[moving]
            source_col = _chain_end_column(moving, assignment, columns_set)
            candidate_supports = [c for c in columns] + [
                b for b in blocks
                if b != moving and state.clear.get(b, False)
            ]
            for target_support in candidate_supports:
                if target_support == source_support:
                    continue
                if target_support in blocks:
                    target_col = _chain_end_column(target_support, assignment, columns_set)
                else:
                    target_col = target_support

                next_assignment = dict(assignment)
                next_assignment[moving] = target_support
                next_atoms = _atoms_from_assignment(next_assignment, blocks, columns)
                next_label = _label_from_atoms(next_atoms, atom_to_idx, n_canon)
                next_idx = label_to_idx.get(tuple(next_label.tolist()))
                if next_idx is None or next_idx == state_idx:
                    continue

                pre = _option_pre_mask(
                    moving=moving,
                    source_support=source_support,
                    target_support=target_support,
                    source_col=source_col,
                    target_col=target_col,
                    blocks=blocks,
                    atom_to_idx=atom_to_idx,
                    n_canon=n_canon,
                )
                add = ((next_label > 0) & (labels[state_idx] <= 0)).float()
                delete = ((labels[state_idx] > 0) & (next_label <= 0)).float()
                frame = 1.0 - torch.clamp(add + delete, max=1.0)
                primitive_indices, primitive_plan, option_name = _primitive_plan_for_option(
                    moving=moving,
                    source_support=source_support,
                    target_support=target_support,
                    source_col=source_col,
                    blocks=blocks,
                    domain_info=domain_info,
                )
                option = {
                    "name": option_name,
                    "moving": moving,
                    "source_support": source_support,
                    "target_support": target_support,
                    "primitive_action_indices": primitive_indices,
                    "primitive_plan": primitive_plan,
                    "pre_mask": pre,
                    "add_mask": add,
                    "del_mask": delete,
                    "frame_mask": frame,
                }
                transitions.append((state_idx, next_idx, option))
                by_option[option_name] = by_option.get(option_name, 0) + 1

    return transitions, by_option


def _build_transition_dataset(
    states, transitions, state_labels, all_features, train_state_ids,
    views_per_state, type_ids, n_negatives, seed, domain_info,
):
    """Build TransitionDataset from option-level PDDL transitions."""
    rng = np.random.default_rng(seed)
    n_canon = domain_info.n_canonical

    features_t, features_t1 = [], []
    action_indices = []
    pre_masks, add_masks, del_masks, frame_masks = [], [], [], []
    neg_pre_l, neg_add_l, neg_del_l = [], [], []

    all_trans = [(t, t1, option) for t, t1, option in transitions
                 if t in train_state_ids and t1 in train_state_ids]
    option_pool = [option for _, _, option in all_trans if isinstance(option, dict)]

    for (t_idx, t1_idx, option) in all_trans:
        feat_t_idx = t_idx * views_per_state + int(rng.integers(views_per_state))
        feat_t1_idx = t1_idx * views_per_state + int(rng.integers(views_per_state))
        features_t.append(all_features[feat_t_idx])
        features_t1.append(all_features[feat_t1_idx])

        if not isinstance(option, dict):
            pre = torch.zeros(n_canon)
            add = torch.zeros(n_canon)
            delete = torch.zeros(n_canon)
            frame = torch.ones(n_canon)
            actual_action_idx = 0
        else:
            actual_action_idx = int(option["primitive_action_indices"][0])
            pre = option["pre_mask"].clone()
            add = option["add_mask"].clone()
            delete = option["del_mask"].clone()
            frame = option["frame_mask"].clone()

        action_indices.append(actual_action_idx)
        pre_masks.append(pre)
        add_masks.append(add)
        del_masks.append(delete)
        frame_masks.append(frame)

        neg_pre = torch.zeros(n_negatives, n_canon)
        neg_add = torch.zeros(n_negatives, n_canon)
        neg_del = torch.zeros(n_negatives, n_canon)
        if isinstance(option, dict) and option_pool:
            candidates = [o for o in option_pool if o is not option]
            if candidates:
                chosen = rng.choice(len(candidates), size=min(n_negatives, len(candidates)), replace=False)
                for k, choice_idx in enumerate(chosen[:n_negatives]):
                    neg_option = candidates[int(choice_idx)]
                    neg_pre[k] = neg_option["pre_mask"].clone()
                    neg_add[k] = neg_option["add_mask"].clone()
                    neg_del[k] = neg_option["del_mask"].clone()
        neg_pre_l.append(neg_pre)
        neg_add_l.append(neg_add)
        neg_del_l.append(neg_del)

    if not features_t:
        raise RuntimeError("No train-split transitions produced")

    add_stack = torch.stack(add_masks)
    del_stack = torch.stack(del_masks)
    frame_stack = torch.stack(frame_masks)
    print(
        f"  Transitions (pddl option): {len(features_t)} "
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


def _select_transition_state_ids(
    transition_source: str,
    seed_state_ids: set[int],
    pool_idx: np.ndarray,
    budget_states: int,
    seed: int,
) -> set[int]:
    """Select which symbolic states may appear as transition endpoints."""
    pool_states = [int(x) for x in pool_idx.tolist()]
    if transition_source == "seed_only":
        return set(seed_state_ids)
    if transition_source == "all_pool":
        return set(pool_states)
    if transition_source != "pool_subset":
        raise ValueError(f"Unknown transition_source: {transition_source}")

    if budget_states <= 0:
        return set(pool_states)
    selected = set(seed_state_ids)
    remaining = [s for s in pool_states if s not in selected]
    n_extra = max(0, min(int(budget_states) - len(selected), len(remaining)))
    if n_extra > 0:
        rng = np.random.default_rng(seed)
        selected.update(int(x) for x in rng.choice(remaining, size=n_extra, replace=False))
    return selected


def _filter_transition_pairs(
    transitions,
    transition_state_ids: set[int],
    budget_pairs: int,
    seed: int,
):
    """Filter transitions by endpoint set and optional pair budget."""
    eligible = [
        (t, t1, d) for (t, t1, d) in transitions
        if int(t) in transition_state_ids and int(t1) in transition_state_ids
    ]
    if budget_pairs and budget_pairs > 0 and len(eligible) > budget_pairs:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(eligible), size=int(budget_pairs), replace=False)
        eligible = [eligible[int(i)] for i in np.sort(chosen)]
    return eligible


def _sample_random_transition_pairs(
    transition_state_ids: set[int],
    n_pairs: int,
    seed: int,
):
    """Sample random directed state pairs from the same endpoint budget."""
    states = sorted(int(s) for s in transition_state_ids)
    if len(states) < 2 or n_pairs <= 0:
        return []
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(int(n_pairs)):
        t, t1 = rng.choice(states, size=2, replace=False)
        pairs.append((int(t), int(t1), f"randomPair({i})"))
    return pairs


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
    parser.add_argument(
        "--transition-source",
        choices=["seed_only", "pool_subset", "all_pool"],
        default="all_pool",
        help=(
            "Which states can appear as transition endpoints: seed_only=only "
            "K labeled states, pool_subset=K states plus sampled pool states, "
            "all_pool=current semi-supervised/oracle-upper-bound setting."
        ),
    )
    parser.add_argument(
        "--transition-budget-states",
        type=int,
        default=0,
        help="Total transition endpoint state budget for pool_subset. 0 uses all available pool states.",
    )
    parser.add_argument(
        "--transition-budget-pairs",
        type=int,
        default=0,
        help="Optional transition pair budget after endpoint filtering. 0 uses all eligible pairs.",
    )
    parser.add_argument("--exp-name", default=None)
    parser.add_argument("--domain", default=str(BWS_DOMAIN))
    parser.add_argument("--problem", default=str(BWS_PROBLEM))
    parser.add_argument(
        "--use-support-head",
        action="store_true",
        help="Train PaQ with a Blocksworld support(block) structural head.",
    )
    parser.add_argument(
        "--decode-support",
        action="store_true",
        help="Evaluate final states with the constrained support decoder.",
    )
    parser.add_argument(
        "--w-support",
        type=float,
        default=1.0,
        help="Weight for support cross-entropy when --use-support-head is set.",
    )
    args = parser.parse_args()
    if args.train_seed is None:
        args.train_seed = args.seed
    support_head_enabled = args.use_support_head or args.decode_support

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
    print(
        f"  transition_source={args.transition_source} "
        f"budget_states={args.transition_budget_states} "
        f"budget_pairs={args.transition_budget_pairs} "
        "mask_source=pddl"
    )
    if support_head_enabled:
        print(
            "  support_head=ON "
            f"decode_support={args.decode_support or support_head_enabled} "
            f"w_support={args.w_support}"
        )

    # Compile domain from the same problem objects used for state enumeration.
    domain_path = Path(args.domain)
    problem_path = Path(args.problem)
    problem_objects, blocks, columns = _problem_objects(problem_path)
    compiler = PDDLDomainCompiler(str(domain_path))
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
    print(f"  domain_path={domain_path}")
    print(f"  problem={problem_path}")
    print(f"  blocks={blocks} columns={columns}")

    # Enumerate all legal Blocksworld states.
    print("  Enumerating legal states...")
    states = enumerate_all_states(blocks, columns)
    n_states = len(states)

    # Build labels
    canonical_preds = []
    for atom in domain_info.canonical_atoms:
        canonical_preds.append(str(atom))
    state_labels = torch.stack([state_to_labels(s, canonical_preds) for s in states])
    print(f"  label_pos_rate={state_labels.float().mean():.4f}")

    # Compile stable-state options from the original primitive PDDL domain.
    transitions, transition_action_counts = _enumerate_option_transitions(
        states, state_labels, domain_info, blocks, columns,
    )
    print(
        f"  legal states={n_states} pddl_option_transitions={len(transitions)} "
        f"by_option={transition_action_counts}"
    )

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

    all_results = {}
    for k in k_values:
        print(f"\n{'#' * 70}")
        print(f"# Blocksworld few-shot K={k}")
        print(f"{'#' * 70}")

        pool_feats, pool_labels = _feature_split(all_features, expanded_labels, pool_idx, args.views_per_state)
        pool_state_ids = np.repeat(pool_idx, args.views_per_state)

        state_train_ds, metadata, selected_indices = _build_fewshot_state_dataset(
            pool_feats, pool_labels, type_ids,
            k=k, fewshot_unit="image", seed=args.seed,
            feat_per_state=args.views_per_state,
            feature_state_ids=pool_state_ids,
        )
        seed_state_ids = set(int(x) for x in np.unique(pool_state_ids[selected_indices]))
        transition_state_ids = _select_transition_state_ids(
            transition_source=args.transition_source,
            seed_state_ids=seed_state_ids,
            pool_idx=pool_idx,
            budget_states=args.transition_budget_states,
            seed=args.seed + int(k),
        )
        selected_transitions = _filter_transition_pairs(
            transitions=transitions,
            transition_state_ids=transition_state_ids,
            budget_pairs=args.transition_budget_pairs,
            seed=args.seed + int(k) * 1009,
        )
        if selected_transitions:
            trans_ds = _build_transition_dataset(
                states=states,
                transitions=selected_transitions,
                state_labels=state_labels,
                all_features=all_features,
                train_state_ids=transition_state_ids,
                views_per_state=args.views_per_state,
                type_ids=type_ids,
                n_negatives=3,
                seed=args.seed,
                domain_info=domain_info,
            )
        else:
            trans_ds = None
            print("  Transitions: 0 (transition losses inactive)")
        random_transitions = _sample_random_transition_pairs(
            transition_state_ids=transition_state_ids,
            n_pairs=len(selected_transitions),
            seed=args.seed + int(k) * 2027,
        )
        if random_transitions and "random_pairs" in conditions:
            trans_random_ds = _build_transition_dataset(
                states=states,
                transitions=random_transitions,
                state_labels=state_labels,
                all_features=all_features,
                train_state_ids=transition_state_ids,
                views_per_state=args.views_per_state,
                type_ids=type_ids,
                n_negatives=3,
                seed=args.seed,
                domain_info=domain_info,
            )
        else:
            trans_random_ds = None

        metadata["conditions"] = conditions
        metadata["train_seed"] = args.train_seed
        metadata["transition_warmup_epochs"] = args.transition_warmup_epochs
        metadata.update({
            "feature_source": "synthetic_object_token",
            "feature_noise": args.noise,
            "d_slot": args.d_slot,
            "seed": args.seed,
            "direct_object_tokens": True,
            "transition_mask_source": "pddl",
            "transition_supervision": "pddl_option_final_effects",
            "uses_gt_state_diff_for_transition": False,
            "transition_generator": "original_domain_stable_state_options",
            "option_transition_counts": transition_action_counts,
            "transition_source": args.transition_source,
            "transition_budget_states_requested": args.transition_budget_states,
            "transition_budget_pairs_requested": args.transition_budget_pairs,
            "transition_endpoint_budget": len(transition_state_ids),
            "transition_pair_budget": len(selected_transitions),
            "random_pair_budget": len(random_transitions),
            "state_label_budget": k,
            "state_label_unique_states": len(seed_state_ids),
            "object_type_source": "oracle",
            "domain": str(domain_path),
            "problem": str(problem_path),
            "num_objects": domain_info.n_objects,
            "num_canonical_atoms": domain_info.n_canonical,
            "blocks": blocks,
            "columns": columns,
            "threshold_source": "validation",
            "use_support_head": support_head_enabled,
            "decode_support": args.decode_support or support_head_enabled,
            "w_support": args.w_support,
            **split_label_meta,
        })
        print(f"  Train: {metadata['n_labeled_samples']} samples, {metadata['n_labeled_states']} states")

        k_results = run_structural_experiment(
            domain_info=domain_info,
            state_train_ds=state_train_ds,
            state_val_ds=state_val_ds,
            state_test_ds=state_test_ds,
            trans_adjacent_ds=trans_ds,
            trans_random_ds=trans_random_ds,
            device=args.device,
            n_epochs=args.n_epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            d_slot=args.d_slot,
            dinov3_kwargs=None,
            exp_dir=exp_dir / f"k_{k}",
            conditions=conditions,
            transition_mask_source="pddl",
            direct_object_tokens=True,
            w_contrast=args.w_contrast,
            w_equiv=args.w_equiv,
            w_cf=args.w_cf,
            train_seed=args.train_seed,
            transition_warmup_epochs=args.transition_warmup_epochs,
            use_support_head=support_head_enabled,
            decode_support=args.decode_support or support_head_enabled,
            w_support=args.w_support,
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
