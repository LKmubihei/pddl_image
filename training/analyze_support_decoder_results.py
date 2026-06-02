#!/usr/bin/env python3
"""Analyze PaQ atom-threshold vs support-decoder experiment outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paq.blocksworld_support import BlocksworldSupportSketch
from paq.domain_compiler import PDDLDomainCompiler
from paq.model import PaQModel
from training.data.ae_dataset import StateDataset, collate_state_batch
from training.run_bws_structural import _build_synthetic_features, _feature_split
from training.train_aepaq import enumerate_all_states, state_to_labels


STATIC_PREDS = {"rightof", "leftof"}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_support_checkpoint(path: Path) -> bool:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return any(k.startswith("support_head.") for k in state)


def _infer_d_slot(path: Path) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return int(state["object_slot_init"].shape[1])


def _build_test_dataset(metadata: dict, d_slot: int) -> tuple:
    blocks = metadata["blocks"]
    columns = metadata["columns"]
    views_per_state = int(metadata["feat_per_state"])
    seed = int(metadata.get("split_seed", metadata.get("train_seed", 42)))
    noise = float(metadata.get("feature_noise", 0.08))

    compiler = PDDLDomainCompiler(metadata["domain"])
    domain_info = compiler.compile(
        objects={"block": blocks, "column": columns},
        static_predicates=STATIC_PREDS,
    )
    states = enumerate_all_states(blocks, columns)
    labels = torch.stack([
        state_to_labels(s, domain_info.canonical_atom_strings) for s in states
    ])
    features = _build_synthetic_features(
        domain_info,
        labels,
        views_per_state=views_per_state,
        d_slot=d_slot,
        noise=noise,
        seed=seed,
    )
    expanded_labels = labels.repeat_interleave(views_per_state, dim=0)
    state_indices = np.random.default_rng(seed).permutation(len(states))
    n_val = max(1, int(0.15 * len(states)))
    n_test = max(1, int(0.15 * len(states)))
    test_idx = state_indices[n_val:n_val + n_test]
    test_feats, test_labels = _feature_split(
        features, expanded_labels, test_idx, views_per_state
    )
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    return domain_info, StateDataset(test_feats, test_labels, type_ids)


def _legal_vector(vec: torch.Tensor, sketch: BlocksworldSupportSketch) -> bool:
    atoms = {
        atom for atom, flag in zip(sketch.canonical_atom_strings, vec.tolist())
        if int(flag) == 1
    }
    assignment = {}
    for b in sketch.blocks:
        on_supports = [s for s in sketch.blocks if f"(on {b} {s})" in atoms]
        if len(on_supports) > 1:
            return False
        if len(on_supports) == 1:
            assignment[b] = on_supports[0]
            continue
        cols = [c for c in sketch.columns if f"(inColumn {b} {c})" in atoms]
        if len(cols) != 1:
            return False
        assignment[b] = cols[0]

    if not sketch.is_valid_assignment(assignment):
        return False
    return atoms == sketch.derive_atoms(assignment)


@torch.no_grad()
def _evaluate_checkpoint(
    checkpoint: Path,
    result_metrics: dict,
    metadata: dict,
    device: str,
) -> dict:
    d_slot = _infer_d_slot(checkpoint)
    use_support_head = _is_support_checkpoint(checkpoint)
    domain_info, dataset = _build_test_dataset(metadata, d_slot)
    sketch = BlocksworldSupportSketch.from_domain_info(domain_info)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    model = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=domain_info.n_objects,
        d_slot=d_slot,
        use_real_encoder=False,
        predict_slot_types=True,
        direct_object_tokens=True,
        scoring_head_type="film",
        use_support_head=use_support_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_state_batch)
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long, device=device)
    pred_vectors, labels = [], []
    for batch in loader:
        feats = batch["features"].to(device)
        obj_types = type_ids.unsqueeze(0).expand(feats.shape[0], -1)
        out = model(feats, object_type_ids=obj_types)
        if use_support_head:
            preds, _ = sketch.decode_batch(out["support_scores"], device="cpu")
        else:
            thr = float(result_metrics["threshold"])
            preds = (torch.sigmoid(out["canonical_scores"]).cpu() >= thr).long()
        pred_vectors.append(preds.cpu())
        labels.append(batch["state_labels"].long())

    pred = torch.cat(pred_vectors)
    gold = torch.cat(labels)
    legal = torch.tensor([_legal_vector(row, sketch) for row in pred], dtype=torch.float32)
    exact = (pred == gold).all(dim=1).float()
    illegal_exact = ((1.0 - legal) * exact).sum().item()

    return {
        "legal_state_rate": legal.mean().item(),
        "illegal_state_rate": 1.0 - legal.mean().item(),
        "checked_samples": int(pred.shape[0]),
        "recomputed_exact_match": exact.mean().item(),
        "illegal_exact_count": int(illegal_exact),
    }


def _collect_rows(exp_dirs: list[Path], device: str) -> list[dict]:
    rows = []
    for exp_dir in exp_dirs:
        summary_path = exp_dir / "bws_structural_results.json"
        data = _load_json(summary_path)
        for k, k_data in data.items():
            metadata = k_data["_metadata"]
            for condition, res in k_data.items():
                if condition == "_metadata":
                    continue
                ckpt = exp_dir / f"k_{k}" / f"model_{condition}.pt"
                diagnostics = _evaluate_checkpoint(
                    ckpt,
                    res["test"],
                    metadata,
                    device=device,
                )
                test = res["test"]
                if condition == "static":
                    pair_source = "none"
                    effective_endpoint_budget = 0
                    effective_pair_budget = 0
                    effective_mask_source = "none"
                    uses_gt_state_diff = False
                elif condition == "random_pairs":
                    pair_source = "random"
                    effective_endpoint_budget = int(metadata.get("transition_endpoint_budget", 0))
                    effective_pair_budget = int(metadata.get("random_pair_budget", 0))
                    effective_mask_source = metadata.get("transition_mask_source", "unknown")
                    uses_gt_state_diff = bool(metadata.get("uses_gt_state_diff_for_transition", False))
                else:
                    pair_source = "adjacent"
                    effective_endpoint_budget = int(metadata.get("transition_endpoint_budget", 0))
                    effective_pair_budget = int(metadata.get("transition_pair_budget", 0))
                    effective_mask_source = metadata.get("transition_mask_source", "unknown")
                    uses_gt_state_diff = bool(metadata.get("uses_gt_state_diff_for_transition", False))
                if effective_pair_budget == 0:
                    pair_source = "none"
                    effective_endpoint_budget = 0
                    effective_mask_source = "none"
                    uses_gt_state_diff = False

                rows.append({
                    "experiment": exp_dir.name,
                    "K": int(k),
                    "state_label_budget": int(metadata.get("state_label_budget", k)),
                    "state_label_unique_states": int(metadata.get("state_label_unique_states", metadata.get("n_labeled_states", 0))),
                    "condition": condition,
                    "decoder": "support" if metadata.get("decode_support") else "atom_threshold",
                    "use_support_head": bool(metadata.get("use_support_head")),
                    "transition_source": metadata.get("transition_source", "legacy"),
                    "pair_source": pair_source,
                    "transition_mask_source": effective_mask_source,
                    "transition_supervision": (
                        metadata.get("transition_supervision", "unknown")
                        if effective_pair_budget > 0 else "none"
                    ),
                    "uses_gt_state_diff_for_transition": uses_gt_state_diff,
                    "transition_budget_states_requested": int(metadata.get("transition_budget_states_requested", 0)),
                    "transition_budget_pairs_requested": int(metadata.get("transition_budget_pairs_requested", 0)),
                    "transition_endpoint_budget": effective_endpoint_budget,
                    "transition_pair_budget": effective_pair_budget,
                    "available_transition_endpoint_budget": int(metadata.get("transition_endpoint_budget", 0)),
                    "available_transition_pair_budget": int(metadata.get("transition_pair_budget", 0)),
                    "random_pair_budget": int(metadata.get("random_pair_budget", 0)),
                    "f1": float(test["f1"]),
                    "exact_match": float(test["exact_match"]),
                    "precision": float(test["precision"]),
                    "recall": float(test["recall"]),
                    "pred_pos_rate": float(test["pred_pos_rate"]),
                    "legal_state_rate": diagnostics["legal_state_rate"],
                    "illegal_state_rate": diagnostics["illegal_state_rate"],
                    "checked_samples": diagnostics["checked_samples"],
                    "recomputed_exact_match": diagnostics["recomputed_exact_match"],
                    "best_val_f1": float(res["best_val_f1"]),
                    "threshold": float(test["threshold"]),
                })
    return sorted(
        rows,
        key=lambda r: (
            r["K"],
            r["condition"],
            r["transition_source"],
            r["transition_endpoint_budget"],
            r["transition_pair_budget"],
            r["decoder"],
        ),
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict], path: Path) -> None:
    def fmt(x):
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    headers = [
        "K", "U_end", "U_pair", "pair", "mask", "oracle_diff",
        "condition", "decoder", "F1", "EM", "legal", "precision", "recall",
    ]
    lines = [
        "# PaQ Support-Head + PDDL Decoder Comparison",
        "",
        "All rows use the same Blocksworld objects, synthetic object-token features, split seed 42, and test set.",
        "`U_end` and `U_pair` are effective transition endpoint/pair budgets used by that condition; static rows are reported as U=0.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        vals = [
            r["K"],
            r["transition_endpoint_budget"],
            r["transition_pair_budget"],
            r["pair_source"],
            r["transition_mask_source"],
            r["uses_gt_state_diff_for_transition"],
            r["condition"],
            r["decoder"],
            r["f1"],
            r["exact_match"],
            r["legal_state_rate"],
            r["precision"],
            r["recall"],
        ]
        lines.append("| " + " | ".join(fmt(v) for v in vals) + " |")

    lines.extend([
        "",
        "Key readout:",
    ])
    for k in sorted({r["K"] for r in rows}):
        group_keys = sorted({
            (
                r["condition"],
                r["transition_source"],
                r["transition_endpoint_budget"],
                r["transition_pair_budget"],
                r["pair_source"],
                r["transition_mask_source"],
            )
            for r in rows if r["K"] == k
        })
        for cond, source, u_end, u_pair, pair_source, mask_source in group_keys:
            pair = [
                r for r in rows
                if r["K"] == k
                and r["condition"] == cond
                and r["transition_source"] == source
                and r["transition_endpoint_budget"] == u_end
                and r["transition_pair_budget"] == u_pair
                and r["pair_source"] == pair_source
                and r["transition_mask_source"] == mask_source
            ]
            atom = next((r for r in pair if r["decoder"] == "atom_threshold"), None)
            sup = next((r for r in pair if r["decoder"] == "support"), None)
            if atom and sup:
                lines.append(
                    f"- K={k} {cond} U_end={u_end} U_pair={u_pair}: support decoder changes EM "
                    f"{atom['exact_match']:.4f} -> {sup['exact_match']:.4f} "
                    f"(delta {sup['exact_match'] - atom['exact_match']:+.4f}) and "
                    f"legal-state rate {atom['legal_state_rate']:.4f} -> {sup['legal_state_rate']:.4f}."
                )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dirs", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "experiments" / "support_decoder_comparison_20260531")
    args = parser.parse_args()

    rows = _collect_rows(args.exp_dirs, args.device)
    _write_csv(rows, args.out_dir / "comparison.csv")
    _write_markdown(rows, args.out_dir / "comparison.md")
    print(f"Wrote {args.out_dir / 'comparison.csv'}")
    print(f"Wrote {args.out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
