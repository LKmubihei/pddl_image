#!/usr/bin/env python3
"""Pure evaluation diagnostic for ARIAC workspace calibration reranking.

This script does not train. It loads one checkpoint, computes support scores
once, and compares:

  A. normal legal decoder
  B. workspace rerank with object-query attention centers
  C. workspace rerank with available part bbox centers, falling back to
     attention centers when a bbox assignment is missing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from paq.ariac_support import AriacPlacementSketch
from paq.model import PaQModel
from training import run_ariac_structured as base


def _metadata_get(meta: dict, name: str, default):
    return meta[name] if name in meta else default


def _build_model(checkpoint: dict, domain_info, input_feature_dim: int) -> PaQModel:
    meta = checkpoint.get("metadata", {})
    d_slot = int(checkpoint.get("d_slot", meta.get("d_slot", 256)))
    model = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=len(domain_info.objects),
        d_slot=d_slot,
        n_slot_iters=int(_metadata_get(meta, "n_slot_iters", 3)),
        use_real_encoder=False,
        visual_encoder=None,
        predict_slot_types=True,
        object_extractor_type=_metadata_get(meta, "object_extractor_type", "object_queries"),
        object_query_relation_layers=int(_metadata_get(meta, "object_query_relation_layers", 1)),
        object_query_local_refine=bool(_metadata_get(meta, "object_query_local_refine", False)),
        object_query_local_top_k=int(_metadata_get(meta, "object_query_local_top_k", 4)),
        object_query_local_radius=int(_metadata_get(meta, "object_query_local_radius", 2)),
        dense_global_bias=bool(_metadata_get(meta, "dense_global_bias", False)),
        use_support_head=True,
        support_block_type="part",
        support_column_type="location",
        support_head_type=_metadata_get(meta, "support_head_type", "two_stage"),
        support_temperature=float(_metadata_get(meta, "support_temperature", 1.5)),
        support_geometry_type=_metadata_get(meta, "support_geometry_type", "none"),
        support_location_prior_weight=float(_metadata_get(meta, "support_location_prior_weight", 0.0)),
        support_location_prior_sigma=float(_metadata_get(meta, "support_location_prior_sigma", 0.2)),
        support_patch_evidence_type=_metadata_get(meta, "support_patch_evidence_type", "none"),
        support_patch_location_scale_init=float(_metadata_get(meta, "support_patch_location_scale_init", 0.5)),
        support_patch_table_scale_init=float(_metadata_get(meta, "support_patch_table_scale_init", 0.5)),
        support_patch_contact_scale_init=float(_metadata_get(meta, "support_patch_contact_scale_init", 0.5)),
        support_patch_location_sigma=float(_metadata_get(meta, "support_patch_location_sigma", 0.18)),
        support_patch_temperature=float(_metadata_get(meta, "support_patch_temperature", 1.0)),
        support_patch_contact_top_k=int(_metadata_get(meta, "support_patch_contact_top_k", 16)),
        support_patch_contact_sigma_x=float(_metadata_get(meta, "support_patch_contact_sigma_x", 0.12)),
        support_patch_contact_sigma_y=float(_metadata_get(meta, "support_patch_contact_sigma_y", 0.12)),
        support_patch_contact_gap=float(_metadata_get(meta, "support_patch_contact_gap", 0.06)),
        support_hidden_dim=_metadata_get(meta, "support_hidden_dim", 512),
        scoring_head_type=_metadata_get(meta, "scoring_head_type", "film"),
    )
    if input_feature_dim != d_slot:
        fp_args = argparse.Namespace(
            feature_projector=_metadata_get(meta, "feature_projector", "linear"),
        )
        model.feat_proj = base.build_feature_projector(input_feature_dim, d_slot, fp_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _active_parts(sketch: AriacPlacementSketch, active_mask: torch.Tensor) -> list[str]:
    return [p for p, flag in zip(sketch.parts, active_mask.tolist()) if flag > 0]


def _assignment_from_target_row(
    sketch: AriacPlacementSketch,
    target_row: torch.Tensor,
    active_parts: list[str],
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for part in active_parts:
        pi = sketch.part_index(part)
        ci = int(target_row[pi].item())
        assignment[part] = sketch.place_candidates[part][ci]
    return assignment


def _workspace_decode_assignment(
    sketch: AriacPlacementSketch,
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    part_centers: torch.Tensor,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
    top_k: int,
    weight: float,
    table_residual_penalty: float,
    outside_penalty: float,
) -> tuple[dict[str, str], int, float, float]:
    legal_targets = base._topk_legal_assignment_targets(
        support_scores,
        active_mask,
        sketch,
        top_k,
    )
    model_scores = base._assignment_scores_from_targets(support_scores, legal_targets)
    active = _active_parts(sketch, active_mask.detach().cpu())
    geo_scores: list[float] = []
    for row in legal_targets.detach().cpu():
        assignment = _assignment_from_target_row(sketch, row, active)
        geo_scores.append(
            base._workspace_assignment_score(
                sketch,
                assignment,
                active,
                part_centers.detach().cpu(),
                workspace_boxes,
                table_residual_penalty=table_residual_penalty,
                outside_penalty=outside_penalty,
            )
        )
    geo = torch.tensor(geo_scores, dtype=support_scores.dtype)
    total = model_scores.detach().cpu() + weight * geo
    best = int(torch.argmax(total).item())
    return (
        _assignment_from_target_row(sketch, legal_targets[best].detach().cpu(), active),
        int(legal_targets.shape[0]),
        float(model_scores.detach().cpu()[best].item()),
        float(geo[best].item()),
    )


def _gold_in_topk(
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    top_k: int,
) -> bool:
    legal_targets = base._topk_legal_assignment_targets(
        support_scores,
        active_mask,
        sketch,
        top_k,
    ).detach().cpu()
    valid_variant_ids = variant_mask.detach().cpu().nonzero(as_tuple=True)[0].tolist()
    for vi in valid_variant_ids:
        gold = support_target_variants[vi].detach().cpu()
        active = gold >= 0
        if active.sum().item() == 0:
            continue
        matches = (legal_targets[:, active] == gold[active]).all(dim=1)
        if bool(matches.any().item()):
            return True
    return False


def _support_facts(assignment: dict[str, str], active_parts: list[str], sketch: AriacPlacementSketch) -> set[str]:
    active = set(active_parts)
    facts = set()
    for part in active_parts:
        support = assignment[part]
        if support in active:
            facts.add(f"(on {part} {support})")
        else:
            facts.add(f"(part_at {part} {support})")
    return facts


def _bbox_part_centers(
    sample: base.AriacSample,
    label_dir: Path,
    class_names: list[str],
    sketch: AriacPlacementSketch,
    fallback_centers: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, bool]]:
    centers = fallback_centers.clone()
    available = {part: False for part in sketch.parts}
    label_path = label_dir / f"{sample.sample_id}.txt"
    if not label_path.exists():
        return centers, available
    assigned, _ = base._assign_label_boxes(sample, label_path, class_names)
    for part, box in assigned.items():
        if part not in sketch.parts:
            continue
        pi = sketch.part_index(part)
        cx, cy, _, _ = box
        centers[pi, 0] = float(cx)
        centers[pi, 1] = float(cy)
        available[part] = True
    return centers, available


def _prediction_row(
    sketch: AriacPlacementSketch,
    assignment: dict[str, str],
    active_parts: list[str],
) -> torch.Tensor:
    return torch.tensor(sketch.assignment_to_vector(assignment, active_parts), dtype=torch.float32)


def _evaluate_predictions(
    pred_assignments: list[dict[str, str]],
    sample_indices: np.ndarray,
    samples: list[base.AriacSample],
    sketch: AriacPlacementSketch,
    active_part_masks: torch.Tensor,
    labels: torch.Tensor,
    active_atom_masks: torch.Tensor,
    label_variants: torch.Tensor,
    variant_masks: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, list[bool]]:
    rows = []
    exact_flags: list[bool] = []
    for assignment, si in zip(pred_assignments, sample_indices):
        active = _active_parts(sketch, active_part_masks[si])
        rows.append(_prediction_row(sketch, assignment, active))
    preds = torch.stack(rows)
    metrics = base.compute_metrics(
        preds,
        labels[sample_indices],
        active_atom_masks[sample_indices],
        sketch,
        active_part_masks[sample_indices],
        label_variants[sample_indices],
        variant_masks[sample_indices],
    )
    pred_v = preds.float().cpu().unsqueeze(1)
    label_v = label_variants[sample_indices].float().cpu() * active_atom_masks[sample_indices].cpu().unsqueeze(1)
    mask_v = active_atom_masks[sample_indices].cpu().unsqueeze(1)
    mismatches = ((pred_v != label_v) & (mask_v == 1)).sum(dim=2)
    mismatches = mismatches.masked_fill(~variant_masks[sample_indices].cpu(), 10**9)
    exact_flags = (mismatches.min(dim=1).values == 0).tolist()
    return metrics, preds, [bool(x) for x in exact_flags]


def _write_wrong_samples(
    lines: list[str],
    method_name: str,
    pred_assignments: list[dict[str, str]],
    exact_flags: list[bool],
    sample_indices: np.ndarray,
    samples: list[base.AriacSample],
    sketch: AriacPlacementSketch,
    active_part_masks: torch.Tensor,
    gold_in_topk: list[bool],
    max_samples: int,
):
    wrong_ids = [
        samples[int(si)].sample_id
        for si, ok in zip(sample_indices, exact_flags)
        if not ok
    ]
    lines.append(f"\n## Wrong Samples - {method_name}\n")
    lines.append(f"wrong_count: {len(wrong_ids)}")
    lines.append("")
    for assignment, ok, si in zip(pred_assignments, exact_flags, sample_indices):
        if ok:
            continue
        sample = samples[int(si)]
        active = _active_parts(sketch, active_part_masks[si])
        gold_assignment = sample.assignment or {}
        gold_facts = _support_facts(gold_assignment, active, sketch)
        pred_facts = _support_facts(assignment, active, sketch)
        sample_pos = list(sample_indices).index(si)
        lines.append(f"### {sample.sample_id}  gold_in_topK={gold_in_topk[sample_pos]}")
        lines.append("missing:")
        for fact in sorted(gold_facts - pred_facts):
            lines.append(f"  {fact}")
        lines.append("extra:")
        for fact in sorted(pred_facts - gold_facts):
            lines.append(f"  {fact}")
        lines.append("")
        if sum(not x for x in exact_flags[: sample_pos + 1]) >= max_samples:
            remaining = len(wrong_ids) - max_samples
            if remaining > 0:
                lines.append(f"... {remaining} more wrong samples omitted")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "ariac")
    parser.add_argument("--workspace-label-csv", type=Path, default=ROOT / "data" / "ariac" / "labels.csv")
    parser.add_argument("--part-label-dir", type=Path, default=ROOT / "data" / "ariac" / "labels")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--workspace-weight", type=float, default=5.0)
    parser.add_argument("--table-residual-penalty", type=float, default=1.0)
    parser.add_argument("--outside-penalty", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "ariac_workspace_pure_eval_diagnostic_20260603.md")
    parser.add_argument("--max-wrong-samples", type=int, default=30)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("metadata", {})
    samples = base.load_samples(args.data_dir, strict_valid=bool(meta.get("strict_valid_factor_states", True)))
    if bool(meta.get("exclude_duplicate_parts", False)):
        samples = [s for s in samples if not base.has_duplicate_active_parts(s)]

    all_parts = ckpt.get("parts") or sorted({p for s in samples for p in s.active_parts})
    all_locations = ckpt.get("locations") or [
        loc for loc in base.DEFAULT_LOCATIONS
        if any(loc in s.active_locations for s in samples)
    ]
    domain_info = base.build_domain_info(all_parts, all_locations)
    sketch = AriacPlacementSketch.from_domain_info(domain_info)

    labels, active_part_masks, active_atom_masks = [], [], []
    for sample in samples:
        y, part_mask, atom_mask = base.sample_labels(sample, sketch)
        labels.append(y)
        active_part_masks.append(part_mask)
        active_atom_masks.append(atom_mask)
    labels_t = torch.stack(labels)
    active_part_masks_t = torch.stack(active_part_masks)
    active_atom_masks_t = torch.stack(active_atom_masks)
    label_variants_t, support_target_variants_t, variant_masks_t, _ = base.build_duplicate_label_variants(
        samples=samples,
        labels=labels_t,
        active_part_masks=active_part_masks_t,
        sketch=sketch,
        duplicate_mode=meta.get("duplicate_mode", "exchangeable"),
    )

    feature_obj = torch.load(args.feature_cache, map_location="cpu")
    features = feature_obj["features"]
    cache_ids = feature_obj.get("metadata", {}).get("sample_ids")
    sample_ids = [s.sample_id for s in samples]
    if cache_ids is not None and list(cache_ids) != sample_ids:
        raise RuntimeError("Feature cache sample_ids do not match current sample order")

    rng = np.random.default_rng(int(meta.get("split_seed", 42)))
    perm = rng.permutation(len(samples))
    n_test = int(meta.get("n_test", 100))
    k = int(ckpt.get("k", meta.get("k", 52)))
    test_idx = perm[:n_test]
    train_pool = perm[n_test:]
    train_idx = train_pool[:k]

    model = _build_model(ckpt, domain_info, int(ckpt.get("input_feature_dim", features.shape[-1]))).to(args.device)
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long, device=args.device)
    slot_init = ckpt.get("slot_init")
    if slot_init is None:
        slot_init = base.object_slot_init(all_parts, all_locations, int(ckpt.get("d_slot", 256)))
    slot_init = slot_init.to(args.device)

    support_chunks, center_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(test_idx), args.batch_size):
            idx = test_idx[start:start + args.batch_size]
            feats = features[idx].to(args.device)
            batch = feats.shape[0]
            out = model(
                feats,
                object_type_ids=type_ids.unsqueeze(0).expand(batch, -1),
                slot_init=slot_init.unsqueeze(0).expand(batch, -1, -1),
            )
            support_chunks.append(out["support_scores"].cpu())
            center_chunks.append(base._object_mask_part_centers(out["obj_masks"], sketch).cpu())
    support_scores = torch.cat(support_chunks, dim=0)
    attention_centers = torch.cat(center_chunks, dim=0)

    workspace_boxes = base.load_workspace_boxes(args.workspace_label_csv, all_locations)
    class_names = [x.strip() for x in (args.part_label_dir / "classes.txt").read_text().splitlines() if x.strip()]

    normal_assignments = []
    workspace_attention_assignments = []
    workspace_bbox_assignments = []
    gold_in_topk = []
    bbox_complete = []
    bbox_root_complete = []
    bbox_available_parts = 0
    active_parts_total = 0
    for local_i, si in enumerate(test_idx):
        active = _active_parts(sketch, active_part_masks_t[si])
        active_parts_total += len(active)
        decoded = sketch.decode(support_scores[local_i], active)
        normal_assignments.append(decoded.assignment)

        att_assignment, _, _, _ = _workspace_decode_assignment(
            sketch,
            support_scores[local_i],
            active_part_masks_t[si],
            attention_centers[local_i],
            workspace_boxes,
            top_k=args.top_k,
            weight=args.workspace_weight,
            table_residual_penalty=args.table_residual_penalty,
            outside_penalty=args.outside_penalty,
        )
        workspace_attention_assignments.append(att_assignment)

        bbox_centers, available = _bbox_part_centers(
            samples[int(si)],
            args.part_label_dir,
            class_names,
            sketch,
            attention_centers[local_i],
        )
        bbox_available_parts += sum(1 for p in active if available.get(p, False))
        bbox_complete.append(all(available.get(p, False) for p in active))
        root_parts = [
            p for p in active
            if (samples[int(si)].assignment or {}).get(p) in sketch.locations
        ]
        bbox_root_complete.append(all(available.get(p, False) for p in root_parts))
        bbox_assignment, _, _, _ = _workspace_decode_assignment(
            sketch,
            support_scores[local_i],
            active_part_masks_t[si],
            bbox_centers,
            workspace_boxes,
            top_k=args.top_k,
            weight=args.workspace_weight,
            table_residual_penalty=args.table_residual_penalty,
            outside_penalty=args.outside_penalty,
        )
        workspace_bbox_assignments.append(bbox_assignment)

        gold_in_topk.append(
            _gold_in_topk(
                support_scores[local_i],
                active_part_masks_t[si],
                support_target_variants_t[si],
                variant_masks_t[si],
                sketch,
                args.top_k,
            )
        )

    methods = {
        "A_normal": normal_assignments,
        "B_workspace_attention_center": workspace_attention_assignments,
        "C_workspace_bbox_center": workspace_bbox_assignments,
    }
    metrics = {}
    preds = {}
    exact = {}
    for name, assignments in methods.items():
        metrics[name], preds[name], exact[name] = _evaluate_predictions(
            assignments,
            test_idx,
            samples,
            sketch,
            active_part_masks_t,
            labels_t,
            active_atom_masks_t,
            label_variants_t,
            variant_masks_t,
        )

    def changed_summary(src: str, dst: str) -> dict[str, list[tuple[str, bool]]]:
        out = {"changed": [], "good_to_bad": [], "bad_to_good": [], "bad_to_bad": []}
        for i, si in enumerate(test_idx):
            if methods[src][i] == methods[dst][i]:
                continue
            sid = samples[int(si)].sample_id
            item = (sid, gold_in_topk[i])
            out["changed"].append(item)
            if exact[src][i] and not exact[dst][i]:
                out["good_to_bad"].append(item)
            elif not exact[src][i] and exact[dst][i]:
                out["bad_to_good"].append(item)
            elif not exact[src][i] and not exact[dst][i]:
                out["bad_to_bad"].append(item)
        return out

    def fmt_changed(items: list[tuple[str, bool]]) -> str:
        return ", ".join(f"{sid}(gold_topK={topk})" for sid, topk in items[:80])

    lines: list[str] = []
    lines.append("# ARIAC Workspace Pure Evaluation Diagnostic")
    lines.append("")
    lines.append("This diagnostic loads one checkpoint and does not train.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- feature_cache: `{args.feature_cache}`")
    lines.append(f"- train K from checkpoint: `{k}`")
    lines.append(f"- test size from checkpoint metadata: `{n_test}`")
    lines.append(f"- workspace topK: `{args.top_k}`")
    lines.append(f"- workspace weight: `{args.workspace_weight}`")
    lines.append(f"- workspace boxes: `{args.workspace_label_csv}`")
    lines.append(f"- part bbox labels: `{args.part_label_dir}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| decode | EM | F1 | precision | recall | legal |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['exact_match']:.4f} | {m['f1']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['legal']:.4f} |"
        )
    lines.append("")
    lines.append("## Top-K Gold Coverage")
    lines.append("")
    lines.append(
        f"gold legal state in top{args.top_k}: "
        f"{sum(gold_in_topk)}/{len(gold_in_topk)} = {sum(gold_in_topk)/max(len(gold_in_topk),1):.4f}"
    )
    lines.append("")
    lines.append("## BBox Center Coverage")
    lines.append("")
    lines.append(
        f"assigned active part bbox centers: {bbox_available_parts}/{active_parts_total} = "
        f"{bbox_available_parts/max(active_parts_total,1):.4f}"
    )
    lines.append(
        f"samples with all active part bbox centers: {sum(bbox_complete)}/{len(bbox_complete)} = "
        f"{sum(bbox_complete)/max(len(bbox_complete),1):.4f}"
    )
    lines.append(
        f"samples with all gold chain-root bbox centers: {sum(bbox_root_complete)}/{len(bbox_root_complete)} = "
        f"{sum(bbox_root_complete)/max(len(bbox_root_complete),1):.4f}"
    )
    lines.append("")
    lines.append("## Changed Images")
    for dst in ("B_workspace_attention_center", "C_workspace_bbox_center"):
        ch = changed_summary("A_normal", dst)
        lines.append("")
        lines.append(f"### A_normal -> {dst}")
        for key in ("changed", "bad_to_good", "good_to_bad", "bad_to_bad"):
            lines.append(f"- {key}: {len(ch[key])}")
            if ch[key]:
                lines.append("  " + fmt_changed(ch[key]))
    for name, assignments in methods.items():
        _write_wrong_samples(
            lines,
            name,
            assignments,
            exact[name],
            test_idx,
            samples,
            sketch,
            active_part_masks_t,
            gold_in_topk,
            args.max_wrong_samples,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))
    print(f"\nSaved diagnostic report to {args.out}")


if __name__ == "__main__":
    main()
