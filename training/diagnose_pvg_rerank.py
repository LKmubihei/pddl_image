#!/usr/bin/env python3
"""PVG: Proposal-Verified Geometry reranking diagnostic for ARIAC.

This is a pure-evaluation script.  It keeps the H+640 placement checkpoint as
the legal-state candidate generator, builds dynamic part centers from proposal
boxes, and reranks only the baseline top-K legal states with conservative
geometry/atom verification.

The first implementation intentionally uses only low-cost proposal evidence:
existing YOLO class-only boxes in ``data/ariac/labels`` plus the repository's
color-based active-part assignment.  No new neural localizer is trained.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
            dinov3_last_n_layers=int(_metadata_get(meta, "dinov3_last_n_layers", 1)),
            dinov3_base_dim=int(_metadata_get(meta, "dinov3_base_dim", 1280)),
        )
        model.feat_proj = base.build_feature_projector(input_feature_dim, d_slot, fp_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def parse_float_values(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_int_values(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_str_values(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def _active_parts(sketch: AriacPlacementSketch, active_mask: torch.Tensor) -> list[str]:
    return [p for p, flag in zip(sketch.parts, active_mask.tolist()) if flag > 0]


def _assignment_from_target_row(
    sketch: AriacPlacementSketch,
    target_row: torch.Tensor,
    active_parts: list[str],
) -> dict[str, str]:
    row = target_row.detach().cpu()
    assignment: dict[str, str] = {}
    for part in active_parts:
        pi = sketch.part_index(part)
        ci = int(row[pi].item())
        assignment[part] = sketch.place_candidates[part][ci]
    return assignment


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
) -> tuple[dict[str, float], list[bool]]:
    rows = []
    for assignment, si in zip(pred_assignments, sample_indices):
        active = _active_parts(sketch, active_part_masks[int(si)])
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
    return metrics, [bool(x) for x in exact_flags]


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
    for vi in variant_mask.detach().cpu().nonzero(as_tuple=True)[0].tolist():
        gold = support_target_variants[vi].detach().cpu()
        active = gold >= 0
        if active.sum().item() == 0:
            continue
        matches = (legal_targets[:, active] == gold[active]).all(dim=1)
        if bool(matches.any().item()):
            return True
    return False


def _part_kind(name: str) -> str | None:
    for kind in ("battery", "pump", "regulator"):
        if f"_{kind}" in name:
            return kind
    return None


def _part_color(name: str) -> str | None:
    for color in ("blue", "green", "red"):
        if name.startswith(color + "_"):
            return color
    return None


@dataclass(frozen=True)
class PartBox:
    cx: float
    cy: float
    w: float
    h: float
    confidence: float
    source: str

    @property
    def x1(self) -> float:
        return self.cx - 0.5 * self.w

    @property
    def y1(self) -> float:
        return self.cy - 0.5 * self.h

    @property
    def x2(self) -> float:
        return self.cx + 0.5 * self.w

    @property
    def y2(self) -> float:
        return self.cy + 0.5 * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.cx, self.y2)

    @property
    def top_center(self) -> tuple[float, float]:
        return (self.cx, self.y1)


@dataclass
class ProposalEvidence:
    boxes: dict[str, PartBox]
    label_file_present: bool
    assigned_parts: int
    source_counts: dict[str, int]


def _box_iou(a: PartBox, b: PartBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = max(a.w * a.h + b.w * b.h - inter, 1e-8)
    return float(inter / union)


def _hsv_component_boxes(
    image_path: Path,
    min_area: int,
    max_area_frac: float,
    hsv_confidence: float,
) -> dict[str, list[PartBox]]:
    import cv2

    arr = np.asarray(Image.open(image_path).convert("RGB"))
    h, w = arr.shape[:2]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    color_masks = {
        "red": (((hue <= 10) | (hue >= 170)) & (sat >= 55) & (val >= 35)),
        "green": ((hue >= 35) & (hue <= 90) & (sat >= 45) & (val >= 35)),
        "blue": ((hue >= 90) & (hue <= 135) & (sat >= 45) & (val >= 35)),
    }
    kernel = np.ones((5, 5), dtype=np.uint8)
    max_area = float(h * w) * max_area_frac
    out: dict[str, list[PartBox]] = {}
    for color, mask_bool in color_masks.items():
        mask = (mask_bool.astype(np.uint8) * 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: list[PartBox] = []
        for label in range(1, n_labels):
            x, y, bw, bh, area = stats[label]
            if area < min_area or area > max_area:
                continue
            if bw <= 3 or bh <= 3:
                continue
            # Reject very thin color fragments that are usually highlights.
            if min(bw / max(bh, 1), bh / max(bw, 1)) < 0.12:
                continue
            boxes.append(
                PartBox(
                    cx=(x + 0.5 * bw) / float(w),
                    cy=(y + 0.5 * bh) / float(h),
                    w=bw / float(w),
                    h=bh / float(h),
                    confidence=hsv_confidence,
                    source="hsv_component",
                )
            )
        out[color] = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
    return out


def _add_hsv_missing_boxes(
    sample: base.AriacSample,
    boxes: dict[str, PartBox],
    min_area: int,
    max_area_frac: float,
    hsv_confidence: float,
    overlap_threshold: float,
) -> int:
    components = _hsv_component_boxes(
        sample.image_path,
        min_area=min_area,
        max_area_frac=max_area_frac,
        hsv_confidence=hsv_confidence,
    )
    added = 0
    used_components: set[tuple[str, int]] = set()
    for color in ("blue", "green", "red"):
        missing_parts = [
            p for p in sample.active_parts
            if p not in boxes and _part_color(p) == color
        ]
        if not missing_parts:
            continue
        candidates = []
        for idx, cand in enumerate(components.get(color, [])):
            if any(_box_iou(cand, existing) > overlap_threshold for existing in boxes.values()):
                continue
            candidates.append((idx, cand))
        if not candidates:
            continue
        # The HSV source has color but no reliable kind classifier.  Assign
        # largest remaining same-color components to missing same-color parts.
        for part, (idx, cand) in zip(missing_parts, candidates):
            if (color, idx) in used_components:
                continue
            boxes[part] = cand
            used_components.add((color, idx))
            added += 1
    return added


def _proposal_evidence_from_sources(
    sample: base.AriacSample,
    label_dir: Path,
    class_names: list[str],
    proposal_sources: set[str],
    hsv_min_area: int,
    hsv_max_area_frac: float,
    hsv_confidence: float,
    hsv_overlap_threshold: float,
) -> ProposalEvidence:
    boxes: dict[str, PartBox] = {}
    source_counts: dict[str, int] = {}
    label_path = label_dir / f"{sample.sample_id}.txt"
    label_present = label_path.exists()
    if "label" in proposal_sources and label_present:
        assigned, _ = base._assign_label_boxes(sample, label_path, class_names)
        boxes.update({
            part: PartBox(
                cx=float(box[0]),
                cy=float(box[1]),
                w=float(box[2]),
                h=float(box[3]),
                confidence=1.0,
                source="yolo_label_color_assign",
            )
            for part, box in assigned.items()
        })
        source_counts["label"] = len(boxes)
    if "hsv" in proposal_sources:
        added = _add_hsv_missing_boxes(
            sample,
            boxes,
            min_area=hsv_min_area,
            max_area_frac=hsv_max_area_frac,
            hsv_confidence=hsv_confidence,
            overlap_threshold=hsv_overlap_threshold,
        )
        source_counts["hsv"] = added
    return ProposalEvidence(boxes, label_present, len(boxes), source_counts)


def _point_in_box(point: tuple[float, float], box: tuple[float, float, float, float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x >= x1 and x <= x2 and y >= y1 and y <= y2


def _box_center_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    sx = max(x2 - x1, 1e-6)
    sy = max(y2 - y1, 1e-6)
    return float(np.sqrt(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def _inside_workspace(
    pbox: PartBox,
    wbox: tuple[float, float, float, float],
) -> bool:
    return _point_in_box(pbox.center, wbox) or _point_in_box(pbox.bottom_center, wbox)


def _x_overlap_ratio(a: PartBox, b: PartBox) -> float:
    inter = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    denom = max(min(a.w, b.w), 1e-6)
    return float(inter / denom)


def _center_distance(a: PartBox, b: PartBox) -> float:
    return float(np.sqrt((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2))


def _geometry_scores(
    assignment: dict[str, str],
    evidence: ProposalEvidence,
    active_parts: list[str],
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
) -> tuple[float, float, dict[str, float]]:
    active_set = set(active_parts)
    named_regions = [
        loc for loc in sketch.locations
        if loc != sketch.table_location and loc in workspace_boxes
    ]

    region_scores: list[float] = []
    stack_scores: list[float] = []
    for part in active_parts:
        support = assignment[part]
        pbox = evidence.boxes.get(part)
        if pbox is None:
            continue

        if support in sketch.locations:
            score = 0.0
            if support == sketch.table_location:
                table_box = workspace_boxes.get(support)
                in_table = table_box is None or _inside_workspace(pbox, table_box)
                in_named = any(_inside_workspace(pbox, workspace_boxes[loc]) for loc in named_regions)
                if in_table and not in_named:
                    score += 1.0
                else:
                    score -= 1.0 if in_named else 0.5
            else:
                target = workspace_boxes.get(support)
                if target is not None and _inside_workspace(pbox, target):
                    score += 1.0
                elif target is not None:
                    dist = min(
                        _box_center_distance(pbox.center, target),
                        _box_center_distance(pbox.bottom_center, target),
                    )
                    score -= min(dist, 2.0)
                for other in named_regions:
                    if other != support and _inside_workspace(pbox, workspace_boxes[other]):
                        score -= 1.0
                        break
            region_scores.append(score)
            continue

        if support in active_set:
            qbox = evidence.boxes.get(support)
            if qbox is None:
                continue
            overlap = _x_overlap_ratio(pbox, qbox)
            vertical_gap = abs(pbox.y2 - qbox.y1)
            above = pbox.cy < qbox.cy
            dist = _center_distance(pbox, qbox)
            score = 0.0
            score += 1.0 if overlap >= 0.20 else -1.0
            score += 1.0 if vertical_gap <= 0.12 else -min(vertical_gap / 0.12, 2.0)
            score += 1.0 if above else -1.0
            if dist > 0.40:
                score -= 1.0
            stack_scores.append(score / 3.0)

    region_score = float(np.mean(region_scores)) if region_scores else 0.0
    stack_score = float(np.mean(stack_scores)) if stack_scores else 0.0
    return region_score, stack_score, {
        "region_checked": float(len(region_scores)),
        "stack_checked": float(len(stack_scores)),
    }


def _geometry_contradictions(
    assignment: dict[str, str],
    evidence: ProposalEvidence,
    active_parts: list[str],
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
    min_conf: float,
) -> list[str]:
    active_set = set(active_parts)
    named_regions = [
        loc for loc in sketch.locations
        if loc != sketch.table_location and loc in workspace_boxes
    ]
    contradictions: list[str] = []
    for part in active_parts:
        support = assignment[part]
        pbox = evidence.boxes.get(part)
        if pbox is None or pbox.confidence < min_conf:
            continue
        if support in sketch.locations:
            if support == sketch.table_location:
                if any(_inside_workspace(pbox, workspace_boxes[loc]) for loc in named_regions):
                    contradictions.append(f"{part}->table but proposal is inside named region")
            else:
                target = workspace_boxes.get(support)
                in_target = target is not None and _inside_workspace(pbox, target)
                in_other = [
                    loc for loc in named_regions
                    if loc != support and _inside_workspace(pbox, workspace_boxes[loc])
                ]
                if in_other and not in_target:
                    contradictions.append(f"{part}->{support} but proposal is inside {in_other[0]}")
            continue

        if support in active_set:
            qbox = evidence.boxes.get(support)
            if qbox is None or qbox.confidence < min_conf:
                continue
            overlap = _x_overlap_ratio(pbox, qbox)
            vertical_gap = abs(pbox.y2 - qbox.y1)
            if pbox.cy >= qbox.cy + 0.02:
                contradictions.append(f"{part}->{support} but top center is below support center")
            if overlap < 0.10:
                contradictions.append(f"{part}->{support} but horizontal overlap is too low")
            if vertical_gap > 0.25:
                contradictions.append(f"{part}->{support} but vertical gap is too large")
    return contradictions


def _edge_has_evidence(
    part: str,
    support: str,
    evidence: ProposalEvidence,
    sketch: AriacPlacementSketch,
    min_conf: float,
) -> bool:
    pbox = evidence.boxes.get(part)
    if pbox is None or pbox.confidence < min_conf:
        return False
    if support in sketch.locations:
        return True
    qbox = evidence.boxes.get(support)
    return qbox is not None and qbox.confidence >= min_conf


def _changed_edges_have_evidence(
    baseline: dict[str, str],
    candidate: dict[str, str],
    evidence: ProposalEvidence,
    active_parts: list[str],
    sketch: AriacPlacementSketch,
    min_conf: float,
    changed_only: str,
) -> bool:
    changed = [part for part in active_parts if baseline.get(part) != candidate.get(part)]
    if not changed:
        return False
    for part in changed:
        old_support = baseline[part]
        new_support = candidate[part]
        if changed_only == "root_only":
            if old_support not in sketch.locations or new_support not in sketch.locations:
                return False
            if not _edge_has_evidence(part, new_support, evidence, sketch, min_conf):
                return False
        elif changed_only in {"root_and_stack", "all_changed_edges"}:
            if not _edge_has_evidence(part, new_support, evidence, sketch, min_conf):
                return False
        else:
            raise ValueError(f"Unknown changed_only mode: {changed_only}")
    return True


def _atom_scores_for_targets(
    sketch: AriacPlacementSketch,
    targets: torch.Tensor,
    atom_logits: torch.Tensor,
    active_parts: list[str],
    temperatures: list[float],
) -> dict[float, torch.Tensor]:
    logits = atom_logits.detach().cpu().float()
    active_mask = sketch.active_atom_mask(active_parts).float()
    denom = torch.clamp(active_mask.sum(), min=1.0)
    rows = []
    for row in targets.detach().cpu():
        assignment = _assignment_from_target_row(sketch, row, active_parts)
        vec = torch.tensor(sketch.assignment_to_vector(assignment, active_parts), dtype=torch.float32)
        rows.append(vec)
    vectors = torch.stack(rows)
    out: dict[float, torch.Tensor] = {}
    for temp in temperatures:
        scaled = logits / float(temp)
        logprob = vectors * F.logsigmoid(scaled).unsqueeze(0) + (1.0 - vectors) * F.logsigmoid(-scaled).unsqueeze(0)
        out[float(temp)] = (logprob * active_mask.unsqueeze(0)).sum(dim=1) / denom
    return out


@dataclass(frozen=True)
class CandidatePack:
    sample_index: int
    top_k: int
    active_parts: list[str]
    targets: torch.Tensor
    assignments: list[dict[str, str]]
    base_scores: torch.Tensor
    region_scores: torch.Tensor
    stack_scores: torch.Tensor
    atom_scores: dict[float, torch.Tensor]
    baseline_index: int


@dataclass(frozen=True)
class PVGConfig:
    method: str
    top_k: int
    lambda_region: float
    lambda_stack: float
    lambda_atom: float
    atom_temperature: float
    tau: float
    min_conf: float
    changed_only: str

    def label(self) -> str:
        return (
            f"topK={self.top_k}, region={self.lambda_region:g}, "
            f"stack={self.lambda_stack:g}, atom={self.lambda_atom:g}, "
            f"temp={self.atom_temperature:g}, tau={self.tau:g}, "
            f"min_conf={self.min_conf:g}, changed={self.changed_only}"
        )


def _build_candidate_pack(
    sample_index: int,
    support_scores: torch.Tensor,
    atom_logits: torch.Tensor,
    active_mask: torch.Tensor,
    evidence: ProposalEvidence,
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
    top_k: int,
    atom_temperatures: list[float],
) -> CandidatePack:
    active_parts = _active_parts(sketch, active_mask.detach().cpu())
    targets = base._topk_legal_assignment_targets(
        support_scores,
        active_mask,
        sketch,
        top_k,
    ).detach().cpu()
    base_scores = base._assignment_scores_from_targets(support_scores, targets).detach().cpu().float()
    assignments = [
        _assignment_from_target_row(sketch, row, active_parts)
        for row in targets
    ]
    region_scores: list[float] = []
    stack_scores: list[float] = []
    for assignment in assignments:
        region, stack, _ = _geometry_scores(
            assignment,
            evidence,
            active_parts,
            sketch,
            workspace_boxes,
        )
        region_scores.append(region)
        stack_scores.append(stack)
    return CandidatePack(
        sample_index=sample_index,
        top_k=top_k,
        active_parts=active_parts,
        targets=targets,
        assignments=assignments,
        base_scores=base_scores,
        region_scores=torch.tensor(region_scores, dtype=torch.float32),
        stack_scores=torch.tensor(stack_scores, dtype=torch.float32),
        atom_scores=_atom_scores_for_targets(
            sketch,
            targets,
            atom_logits,
            active_parts,
            atom_temperatures,
        ),
        baseline_index=int(torch.argmax(base_scores).item()),
    )


def _decode_pack(
    pack: CandidatePack,
    cfg: PVGConfig,
    evidence: ProposalEvidence,
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
) -> tuple[dict[str, str], dict[str, object]]:
    atom = pack.atom_scores.get(float(cfg.atom_temperature))
    if atom is None:
        raise KeyError(f"Missing atom scores for temperature={cfg.atom_temperature}")
    total = (
        pack.base_scores
        + cfg.lambda_region * pack.region_scores
        + cfg.lambda_stack * pack.stack_scores
        + cfg.lambda_atom * atom
    )
    baseline_idx = pack.baseline_index
    best_idx = int(torch.argmax(total).item())
    baseline = pack.assignments[baseline_idx]
    candidate = pack.assignments[best_idx]
    accepted = False
    reject_reason = "same_as_baseline"

    if best_idx != baseline_idx:
        margin = float((total[best_idx] - total[baseline_idx]).item())
        if margin <= cfg.tau:
            reject_reason = "below_tau"
        elif cfg.method in {"C_geometry", "D_pvg"} and not _changed_edges_have_evidence(
            baseline,
            candidate,
            evidence,
            pack.active_parts,
            sketch,
            cfg.min_conf,
            cfg.changed_only,
        ):
            reject_reason = "missing_changed_edge_evidence"
        else:
            contradictions = (
                _geometry_contradictions(
                    candidate,
                    evidence,
                    pack.active_parts,
                    sketch,
                    workspace_boxes,
                    cfg.min_conf,
                )
                if cfg.method in {"C_geometry", "D_pvg"}
                else []
            )
            if contradictions:
                reject_reason = "geometry_contradiction"
            else:
                accepted = True
                reject_reason = ""

    final_idx = best_idx if accepted else baseline_idx
    diagnostics = {
        "accepted": accepted,
        "reject_reason": reject_reason,
        "best_index": best_idx,
        "baseline_index": baseline_idx,
        "base_score": float(pack.base_scores[final_idx].item()),
        "region_score": float(pack.region_scores[final_idx].item()),
        "stack_score": float(pack.stack_scores[final_idx].item()),
        "atom_score": float(atom[final_idx].item()),
        "total_score": float(total[final_idx].item()),
        "candidate_margin": float((total[best_idx] - total[baseline_idx]).item()),
    }
    return pack.assignments[final_idx], diagnostics


def _forward_indices(
    model: PaQModel,
    features: torch.Tensor,
    indices: np.ndarray,
    type_ids: torch.Tensor,
    slot_init: torch.Tensor,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_chunks: list[torch.Tensor] = []
    atom_chunks: list[torch.Tensor] = []
    type_ids = type_ids.to(device)
    slot_init = slot_init.to(device)
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            idx = indices[start:start + batch_size]
            feats = features[idx].to(device)
            batch = feats.shape[0]
            out = model(
                feats,
                object_type_ids=type_ids.unsqueeze(0).expand(batch, -1),
                slot_init=slot_init.unsqueeze(0).expand(batch, -1, -1),
            )
            support_chunks.append(out["support_scores"].cpu())
            atom_chunks.append(out["canonical_scores"].cpu())
    return torch.cat(support_chunks, dim=0), torch.cat(atom_chunks, dim=0)


def _prepare_split_and_data(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("metadata", {})
    samples = base.load_samples(
        args.data_dir,
        strict_valid=bool(meta.get("strict_valid_factor_states", True)),
    )
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
    label_variants_t, support_target_variants_t, variant_masks_t, duplicate_meta = base.build_duplicate_label_variants(
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

    model = _build_model(
        ckpt,
        domain_info,
        int(ckpt.get("input_feature_dim", features.shape[-1])),
    ).to(args.device)
    type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long)
    slot_init = ckpt.get("slot_init")
    if slot_init is None:
        slot_init = base.object_slot_init(all_parts, all_locations, int(ckpt.get("d_slot", 256)))

    return {
        "ckpt": ckpt,
        "meta": meta,
        "samples": samples,
        "domain_info": domain_info,
        "sketch": sketch,
        "features": features,
        "labels": labels_t,
        "active_part_masks": active_part_masks_t,
        "active_atom_masks": active_atom_masks_t,
        "label_variants": label_variants_t,
        "support_target_variants": support_target_variants_t,
        "variant_masks": variant_masks_t,
        "duplicate_meta": duplicate_meta,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "model": model,
        "type_ids": type_ids,
        "slot_init": slot_init,
    }


def _build_evidence_bank(
    samples: list[base.AriacSample],
    label_dir: Path,
    proposal_sources: set[str],
    hsv_min_area: int,
    hsv_max_area_frac: float,
    hsv_confidence: float,
    hsv_overlap_threshold: float,
) -> dict[int, ProposalEvidence]:
    class_names: list[str] = []
    if "label" in proposal_sources:
        class_path = label_dir / "classes.txt"
        if not class_path.exists():
            raise FileNotFoundError(f"Missing class file: {class_path}")
        class_names = [line.strip() for line in class_path.read_text().splitlines() if line.strip()]
    return {
        i: _proposal_evidence_from_sources(
            sample,
            label_dir,
            class_names,
            proposal_sources=proposal_sources,
            hsv_min_area=hsv_min_area,
            hsv_max_area_frac=hsv_max_area_frac,
            hsv_confidence=hsv_confidence,
            hsv_overlap_threshold=hsv_overlap_threshold,
        )
        for i, sample in enumerate(samples)
    }


def _coverage_summary(
    sample_indices: np.ndarray,
    samples: list[base.AriacSample],
    active_part_masks: torch.Tensor,
    evidence_bank: dict[int, ProposalEvidence],
    sketch: AriacPlacementSketch,
) -> dict[str, float]:
    active_total = 0
    active_covered = 0
    root_total = 0
    root_covered = 0
    stack_edge_total = 0
    stack_edge_covered = 0
    label_files = 0
    source_counts: dict[str, int] = {}
    for si_np in sample_indices:
        si = int(si_np)
        evidence = evidence_bank[si]
        if evidence.label_file_present:
            label_files += 1
        for source, count in evidence.source_counts.items():
            source_counts[source] = source_counts.get(source, 0) + count
        active = _active_parts(sketch, active_part_masks[si])
        assignment = samples[si].assignment or {}
        for part in active:
            active_total += 1
            if part in evidence.boxes:
                active_covered += 1
            support = assignment.get(part)
            if support in sketch.locations:
                root_total += 1
                if part in evidence.boxes:
                    root_covered += 1
            elif support in set(active):
                stack_edge_total += 1
                if part in evidence.boxes and support in evidence.boxes:
                    stack_edge_covered += 1
    return {
        "label_file_coverage": float(label_files / max(len(sample_indices), 1)),
        "active_part_center_coverage": float(active_covered / max(active_total, 1)),
        "active_part_center_covered": float(active_covered),
        "active_part_center_total": float(active_total),
        "root_part_center_coverage": float(root_covered / max(root_total, 1)),
        "root_part_center_covered": float(root_covered),
        "root_part_center_total": float(root_total),
        "stack_edge_box_pair_coverage": float(stack_edge_covered / max(stack_edge_total, 1)),
        "stack_edge_box_pair_covered": float(stack_edge_covered),
        "stack_edge_box_pair_total": float(stack_edge_total),
        **{f"{source}_assigned_parts": float(count) for source, count in source_counts.items()},
    }


def _build_packs(
    sample_indices: np.ndarray,
    support_scores: torch.Tensor,
    atom_scores: torch.Tensor,
    active_part_masks: torch.Tensor,
    evidence_bank: dict[int, ProposalEvidence],
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
    top_k_values: list[int],
    atom_temperatures: list[float],
) -> dict[tuple[int, int], CandidatePack]:
    packs: dict[tuple[int, int], CandidatePack] = {}
    for local_i, si_np in enumerate(sample_indices):
        si = int(si_np)
        for top_k in top_k_values:
            packs[(si, top_k)] = _build_candidate_pack(
                sample_index=si,
                support_scores=support_scores[local_i],
                atom_logits=atom_scores[local_i],
                active_mask=active_part_masks[si],
                evidence=evidence_bank[si],
                sketch=sketch,
                workspace_boxes=workspace_boxes,
                top_k=top_k,
                atom_temperatures=atom_temperatures,
            )
    return packs


def _normal_assignments(
    sample_indices: np.ndarray,
    support_scores: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> list[dict[str, str]]:
    assignments = []
    for local_i, si_np in enumerate(sample_indices):
        si = int(si_np)
        active = _active_parts(sketch, active_part_masks[si])
        assignments.append(sketch.decode(support_scores[local_i], active).assignment)
    return assignments


def _configs_for_method(
    method: str,
    top_k_values: list[int],
    lambda_region_values: list[float],
    lambda_stack_values: list[float],
    lambda_atom_values: list[float],
    atom_temperatures: list[float],
    tau_values: list[float],
    min_conf_values: list[float],
    changed_only_values: list[str],
) -> list[PVGConfig]:
    configs: list[PVGConfig] = []
    if method == "B_atom":
        for top_k, lam_atom, temp, tau in product(
            top_k_values,
            [x for x in lambda_atom_values if x > 0],
            atom_temperatures,
            tau_values,
        ):
            configs.append(PVGConfig(method, top_k, 0.0, 0.0, lam_atom, temp, tau, 0.0, "none"))
        return configs
    if method == "C_geometry":
        for top_k, lam_region, lam_stack, tau, min_conf, changed in product(
            top_k_values,
            lambda_region_values,
            lambda_stack_values,
            tau_values,
            min_conf_values,
            changed_only_values,
        ):
            configs.append(PVGConfig(method, top_k, lam_region, lam_stack, 0.0, atom_temperatures[0], tau, min_conf, changed))
        return configs
    if method == "D_pvg":
        for top_k, lam_region, lam_stack, lam_atom, temp, tau, min_conf, changed in product(
            top_k_values,
            lambda_region_values,
            lambda_stack_values,
            lambda_atom_values,
            atom_temperatures,
            tau_values,
            min_conf_values,
            changed_only_values,
        ):
            configs.append(PVGConfig(method, top_k, lam_region, lam_stack, lam_atom, temp, tau, min_conf, changed))
        return configs
    raise ValueError(f"Unknown method: {method}")


def _select_sort_key(item: tuple[PVGConfig, dict[str, float], int]) -> tuple[float, float, int, float, float, float]:
    cfg, metrics, changed_count = item
    return (
        metrics["exact_match"],
        metrics["f1"],
        -changed_count,
        -abs(cfg.lambda_region) - abs(cfg.lambda_stack) - abs(cfg.lambda_atom),
        -cfg.top_k,
        -cfg.tau,
    )


def _predict_for_config(
    sample_indices: np.ndarray,
    packs: dict[tuple[int, int], CandidatePack],
    evidence_bank: dict[int, ProposalEvidence],
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
    cfg: PVGConfig,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    assignments: list[dict[str, str]] = []
    diagnostics: list[dict[str, object]] = []
    for si_np in sample_indices:
        si = int(si_np)
        assignment, diag = _decode_pack(
            packs[(si, cfg.top_k)],
            cfg,
            evidence_bank[si],
            sketch,
            workspace_boxes,
        )
        assignments.append(assignment)
        diagnostics.append(diag)
    return assignments, diagnostics


def _select_config(
    method: str,
    configs: list[PVGConfig],
    train_idx: np.ndarray,
    train_packs: dict[tuple[int, int], CandidatePack],
    evidence_bank: dict[int, ProposalEvidence],
    samples: list[base.AriacSample],
    sketch: AriacPlacementSketch,
    workspace_boxes: dict[str, tuple[float, float, float, float]],
    active_part_masks: torch.Tensor,
    labels: torch.Tensor,
    active_atom_masks: torch.Tensor,
    label_variants: torch.Tensor,
    variant_masks: torch.Tensor,
) -> tuple[PVGConfig, dict[str, float], dict[str, object]]:
    ranked: list[tuple[PVGConfig, dict[str, float], int]] = []
    for cfg in configs:
        assignments, diagnostics = _predict_for_config(
            train_idx,
            train_packs,
            evidence_bank,
            sketch,
            workspace_boxes,
            cfg,
        )
        metrics, _ = _evaluate_predictions(
            assignments,
            train_idx,
            samples,
            sketch,
            active_part_masks,
            labels,
            active_atom_masks,
            label_variants,
            variant_masks,
        )
        changed_count = sum(1 for diag in diagnostics if bool(diag["accepted"]))
        ranked.append((cfg, metrics, changed_count))
    best_cfg, best_metrics, best_changed = max(ranked, key=_select_sort_key)
    top = sorted(ranked, key=_select_sort_key, reverse=True)[:20]
    selection = {
        "method": method,
        "best_config": best_cfg.__dict__,
        "best_train_metrics": best_metrics,
        "best_train_changed": best_changed,
        "top_configs": [
            {
                "config": cfg.__dict__,
                "metrics": metrics,
                "changed": changed,
            }
            for cfg, metrics, changed in top
        ],
    }
    return best_cfg, best_metrics, selection


def _support_facts(
    assignment: dict[str, str],
    active_parts: list[str],
    sketch: AriacPlacementSketch,
) -> set[str]:
    active = set(active_parts)
    facts = set()
    for part in active_parts:
        support = assignment[part]
        if support in active:
            facts.add(f"(on {part} {support})")
        else:
            facts.add(f"(part_at {part} {support})")
    return facts


def _changed_summary(
    sample_indices: np.ndarray,
    samples: list[base.AriacSample],
    src_assignments: list[dict[str, str]],
    dst_assignments: list[dict[str, str]],
    src_exact: list[bool],
    dst_exact: list[bool],
    gold_in_topk: list[bool],
) -> dict[str, list[tuple[str, bool]]]:
    out = {"changed": [], "good_to_bad": [], "bad_to_good": [], "bad_to_bad": []}
    for i, si_np in enumerate(sample_indices):
        if src_assignments[i] == dst_assignments[i]:
            continue
        item = (samples[int(si_np)].sample_id, gold_in_topk[i])
        out["changed"].append(item)
        if src_exact[i] and not dst_exact[i]:
            out["good_to_bad"].append(item)
        elif not src_exact[i] and dst_exact[i]:
            out["bad_to_good"].append(item)
        elif not src_exact[i] and not dst_exact[i]:
            out["bad_to_bad"].append(item)
    return out


def _changed_edge_coverage(
    sample_indices: np.ndarray,
    baseline_assignments: list[dict[str, str]],
    method_assignments: list[dict[str, str]],
    active_part_masks: torch.Tensor,
    evidence_bank: dict[int, ProposalEvidence],
    sketch: AriacPlacementSketch,
) -> dict[str, float]:
    total = 0
    covered = 0
    for si_np, base_a, meth_a in zip(sample_indices, baseline_assignments, method_assignments):
        si = int(si_np)
        active = _active_parts(sketch, active_part_masks[si])
        evidence = evidence_bank[si]
        for part in active:
            if base_a.get(part) == meth_a.get(part):
                continue
            total += 1
            if _edge_has_evidence(part, meth_a[part], evidence, sketch, 0.0):
                covered += 1
    return {
        "changed_edge_evidence_coverage": float(covered / max(total, 1)),
        "changed_edge_evidence_covered": float(covered),
        "changed_edge_evidence_total": float(total),
    }


def _prediction_error_counts(
    sample_indices: np.ndarray,
    assignments: list[dict[str, str]],
    samples: list[base.AriacSample],
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> dict[str, float]:
    missed_stack = 0
    location_region = 0
    wrong_support_part = 0
    false_stack = 0
    wrong_edges = 0
    for si_np, pred in zip(sample_indices, assignments):
        si = int(si_np)
        gold = samples[si].assignment or {}
        active = _active_parts(sketch, active_part_masks[si])
        for part in active:
            if pred.get(part) == gold.get(part):
                continue
            wrong_edges += 1
            pred_support = pred.get(part)
            gold_support = gold.get(part)
            pred_is_part = pred_support in sketch.parts
            gold_is_part = gold_support in sketch.parts
            gold_is_location = gold_support in sketch.locations
            if gold_is_part and not pred_is_part:
                missed_stack += 1
            elif gold_is_location and pred_support in sketch.locations:
                location_region += 1
            elif gold_is_part and pred_is_part:
                wrong_support_part += 1
            elif gold_is_location and pred_is_part:
                false_stack += 1
    return {
        "wrong_edges": float(wrong_edges),
        "missed_stack": float(missed_stack),
        "location_region": float(location_region),
        "wrong_support_part": float(wrong_support_part),
        "false_stack": float(false_stack),
    }


def _fmt_changed(items: list[tuple[str, bool]], limit: int = 80) -> str:
    return ", ".join(f"{sid}(gold_topK={topk})" for sid, topk in items[:limit])


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
    shown = 0
    for assignment, ok, si_np in zip(pred_assignments, exact_flags, sample_indices):
        if ok:
            continue
        si = int(si_np)
        sample = samples[si]
        active = _active_parts(sketch, active_part_masks[si])
        gold_assignment = sample.assignment or {}
        gold_facts = _support_facts(gold_assignment, active, sketch)
        pred_facts = _support_facts(assignment, active, sketch)
        sample_pos = list(sample_indices).index(si_np)
        lines.append(f"### {sample.sample_id}  gold_in_topK={gold_in_topk[sample_pos]}")
        lines.append("missing:")
        for fact in sorted(gold_facts - pred_facts):
            lines.append(f"  {fact}")
        lines.append("extra:")
        for fact in sorted(pred_facts - gold_facts):
            lines.append(f"  {fact}")
        lines.append("")
        shown += 1
        if shown >= max_samples:
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
    parser.add_argument(
        "--proposal-sources",
        type=str,
        default="hsv",
        help="Comma-separated proposal sources. Supported: label,hsv.",
    )
    parser.add_argument("--hsv-min-area", type=int, default=80)
    parser.add_argument("--hsv-max-area-frac", type=float, default=0.05)
    parser.add_argument("--hsv-confidence", type=float, default=0.6)
    parser.add_argument("--hsv-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--top-k-grid", type=str, default="10,25")
    parser.add_argument("--lambda-region-grid", type=str, default="1,2,5,8")
    parser.add_argument("--lambda-stack-grid", type=str, default="1,2,5,8")
    parser.add_argument("--lambda-atom-grid", type=str, default="0,0.25,0.5,1,2")
    parser.add_argument("--atom-temperature-grid", type=str, default="1,2,4")
    parser.add_argument("--tau-grid", type=str, default="0,0.25,0.5,1")
    parser.add_argument("--min-conf-grid", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--changed-only-grid", type=str, default="root_only,root_and_stack,all_changed_edges")
    parser.add_argument("--methods", type=str, default="B_atom,C_geometry,D_pvg")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "ariac_pvg_rerank_diagnostic_20260603.md")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--max-wrong-samples", type=int, default=30)
    args = parser.parse_args()

    top_k_values = parse_int_values(args.top_k_grid)
    lambda_region_values = parse_float_values(args.lambda_region_grid)
    lambda_stack_values = parse_float_values(args.lambda_stack_grid)
    lambda_atom_values = parse_float_values(args.lambda_atom_grid)
    atom_temperatures = parse_float_values(args.atom_temperature_grid)
    tau_values = parse_float_values(args.tau_grid)
    min_conf_values = parse_float_values(args.min_conf_grid)
    changed_only_values = parse_str_values(args.changed_only_grid)
    methods = parse_str_values(args.methods)
    proposal_sources = set(parse_str_values(args.proposal_sources))
    unknown_sources = proposal_sources - {"label", "hsv"}
    if unknown_sources:
        raise ValueError(f"Unknown proposal sources: {sorted(unknown_sources)}")
    for top_k in top_k_values:
        if top_k <= 0:
            raise ValueError("--top-k-grid values must be positive")
    for changed in changed_only_values:
        if changed not in {"root_only", "root_and_stack", "all_changed_edges"}:
            raise ValueError(f"Unknown changed-only value: {changed}")

    data = _prepare_split_and_data(args)
    ckpt = data["ckpt"]
    meta = data["meta"]
    samples = data["samples"]
    sketch = data["sketch"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    model = data["model"]
    workspace_boxes = base.load_workspace_boxes(args.workspace_label_csv, ckpt.get("locations"))
    evidence_bank = _build_evidence_bank(
        samples,
        args.part_label_dir,
        proposal_sources=proposal_sources,
        hsv_min_area=args.hsv_min_area,
        hsv_max_area_frac=args.hsv_max_area_frac,
        hsv_confidence=args.hsv_confidence,
        hsv_overlap_threshold=args.hsv_overlap_threshold,
    )

    print("=" * 72)
    print("ARIAC PVG rerank diagnostic")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  feature cache: {args.feature_cache}")
    print(f"  train={len(train_idx)} test={len(test_idx)} topK={top_k_values}")
    print(f"  methods={methods}")
    print("=" * 72)

    train_support, train_atom = _forward_indices(
        model,
        data["features"],
        train_idx,
        data["type_ids"],
        data["slot_init"],
        args.batch_size,
        args.device,
    )
    test_support, test_atom = _forward_indices(
        model,
        data["features"],
        test_idx,
        data["type_ids"],
        data["slot_init"],
        args.batch_size,
        args.device,
    )

    train_packs = _build_packs(
        train_idx,
        train_support,
        train_atom,
        data["active_part_masks"],
        evidence_bank,
        sketch,
        workspace_boxes,
        top_k_values,
        atom_temperatures,
    )
    test_packs = _build_packs(
        test_idx,
        test_support,
        test_atom,
        data["active_part_masks"],
        evidence_bank,
        sketch,
        workspace_boxes,
        top_k_values,
        atom_temperatures,
    )

    train_normal = _normal_assignments(train_idx, train_support, data["active_part_masks"], sketch)
    test_normal = _normal_assignments(test_idx, test_support, data["active_part_masks"], sketch)
    train_metrics_normal, train_exact_normal = _evaluate_predictions(
        train_normal,
        train_idx,
        samples,
        sketch,
        data["active_part_masks"],
        data["labels"],
        data["active_atom_masks"],
        data["label_variants"],
        data["variant_masks"],
    )
    test_metrics_normal, test_exact_normal = _evaluate_predictions(
        test_normal,
        test_idx,
        samples,
        sketch,
        data["active_part_masks"],
        data["labels"],
        data["active_atom_masks"],
        data["label_variants"],
        data["variant_masks"],
    )

    train_gold_top10 = [
        _gold_in_topk(train_support[pos], data["active_part_masks"][int(si)], data["support_target_variants"][int(si)], data["variant_masks"][int(si)], sketch, 10)
        for pos, si in enumerate(train_idx)
    ]
    train_gold_top25 = [
        _gold_in_topk(train_support[pos], data["active_part_masks"][int(si)], data["support_target_variants"][int(si)], data["variant_masks"][int(si)], sketch, 25)
        for pos, si in enumerate(train_idx)
    ]
    test_gold_top10 = [
        _gold_in_topk(test_support[pos], data["active_part_masks"][int(si)], data["support_target_variants"][int(si)], data["variant_masks"][int(si)], sketch, 10)
        for pos, si in enumerate(test_idx)
    ]
    test_gold_top25 = [
        _gold_in_topk(test_support[pos], data["active_part_masks"][int(si)], data["support_target_variants"][int(si)], data["variant_masks"][int(si)], sketch, 25)
        for pos, si in enumerate(test_idx)
    ]

    train_coverage = _coverage_summary(train_idx, samples, data["active_part_masks"], evidence_bank, sketch)
    test_coverage = _coverage_summary(test_idx, samples, data["active_part_masks"], evidence_bank, sketch)

    results: dict[str, dict[str, object]] = {}
    for method in methods:
        configs = _configs_for_method(
            method,
            top_k_values,
            lambda_region_values,
            lambda_stack_values,
            lambda_atom_values,
            atom_temperatures,
            tau_values,
            min_conf_values,
            changed_only_values,
        )
        print(f"  selecting {method}: {len(configs)} configs")
        best_cfg, train_metrics, selection = _select_config(
            method,
            configs,
            train_idx,
            train_packs,
            evidence_bank,
            samples,
            sketch,
            workspace_boxes,
            data["active_part_masks"],
            data["labels"],
            data["active_atom_masks"],
            data["label_variants"],
            data["variant_masks"],
        )
        test_assignments, test_diag = _predict_for_config(
            test_idx,
            test_packs,
            evidence_bank,
            sketch,
            workspace_boxes,
            best_cfg,
        )
        test_metrics, test_exact = _evaluate_predictions(
            test_assignments,
            test_idx,
            samples,
            sketch,
            data["active_part_masks"],
            data["labels"],
            data["active_atom_masks"],
            data["label_variants"],
            data["variant_masks"],
        )
        changed = _changed_summary(
            test_idx,
            samples,
            test_normal,
            test_assignments,
            test_exact_normal,
            test_exact,
            test_gold_top25,
        )
        changed_cov = _changed_edge_coverage(
            test_idx,
            test_normal,
            test_assignments,
            data["active_part_masks"],
            evidence_bank,
            sketch,
        )
        results[method] = {
            "best_config": best_cfg.__dict__,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "selection": selection,
            "test_changed": {k: [{"sample_id": sid, "gold_top25": topk} for sid, topk in v] for k, v in changed.items()},
            "test_changed_edge_coverage": changed_cov,
            "test_error_counts": _prediction_error_counts(test_idx, test_assignments, samples, data["active_part_masks"], sketch),
            "test_diagnostics": [
                {"sample_id": samples[int(si)].sample_id, **diag}
                for si, diag in zip(test_idx, test_diag)
            ],
            "test_assignments": test_assignments,
            "test_exact": test_exact,
        }
        print(
            f"    {method}: train EM={train_metrics['exact_match']:.4f} "
            f"test EM={test_metrics['exact_match']:.4f} "
            f"changed={len(changed['changed'])} "
            f"bad_to_good={len(changed['bad_to_good'])} "
            f"good_to_bad={len(changed['good_to_bad'])}"
        )

    result_json = {
        "checkpoint": str(args.checkpoint),
        "feature_cache": str(args.feature_cache),
        "metadata": {
            "train": len(train_idx),
            "test": len(test_idx),
            "checkpoint_k": int(ckpt.get("k", meta.get("k", len(train_idx)))),
            "checkpoint_n_test": int(meta.get("n_test", len(test_idx))),
            "split_seed": int(meta.get("split_seed", 42)),
            "top_k_grid": top_k_values,
            "methods": methods,
            "proposal_sources": sorted(proposal_sources),
            "hsv_min_area": args.hsv_min_area,
            "hsv_max_area_frac": args.hsv_max_area_frac,
            "hsv_confidence": args.hsv_confidence,
            "hsv_overlap_threshold": args.hsv_overlap_threshold,
        },
        "coverage": {
            "train": train_coverage,
            "test": test_coverage,
        },
        "oracle": {
            "train_top10": float(sum(train_gold_top10) / max(len(train_gold_top10), 1)),
            "train_top25": float(sum(train_gold_top25) / max(len(train_gold_top25), 1)),
            "test_top10": float(sum(test_gold_top10) / max(len(test_gold_top10), 1)),
            "test_top25": float(sum(test_gold_top25) / max(len(test_gold_top25), 1)),
        },
        "normal": {
            "train_metrics": train_metrics_normal,
            "test_metrics": test_metrics_normal,
            "test_error_counts": _prediction_error_counts(test_idx, test_normal, samples, data["active_part_masks"], sketch),
        },
        "results": {
            method: {
                k: v for k, v in result.items()
                if k not in {"test_assignments", "test_exact"}
            }
            for method, result in results.items()
        },
    }

    lines: list[str] = []
    lines.append("# ARIAC PVG Rerank Diagnostic")
    lines.append("")
    lines.append("This diagnostic does not train a new model. It reranks baseline top-K legal states with proposal geometry and atom likelihood.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- feature_cache: `{args.feature_cache}`")
    lines.append(f"- workspace boxes: `{args.workspace_label_csv}`")
    lines.append(f"- part proposal labels: `{args.part_label_dir}`")
    lines.append(f"- proposal sources: `{sorted(proposal_sources)}`")
    lines.append(f"- train/test: `{len(train_idx)}/{len(test_idx)}`")
    lines.append(f"- topK grid: `{top_k_values}`")
    lines.append("")
    lines.append("## Proposal Coverage")
    lines.append("")
    lines.append("| split | label files | active centers | root centers | stack bbox pairs |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for split, cov in (("train", train_coverage), ("test", test_coverage)):
        source_text = ", ".join(
            f"{k.replace('_assigned_parts', '')}={v:.0f}"
            for k, v in sorted(cov.items())
            if k.endswith("_assigned_parts")
        )
        lines.append(
            f"| {split} | {cov['label_file_coverage']:.4f} | "
            f"{cov['active_part_center_covered']:.0f}/{cov['active_part_center_total']:.0f} = {cov['active_part_center_coverage']:.4f} | "
            f"{cov['root_part_center_covered']:.0f}/{cov['root_part_center_total']:.0f} = {cov['root_part_center_coverage']:.4f} | "
            f"{cov['stack_edge_box_pair_covered']:.0f}/{cov['stack_edge_box_pair_total']:.0f} = {cov['stack_edge_box_pair_coverage']:.4f} |"
        )
        if source_text:
            lines.append(f"`{split}` source counts: {source_text}")
    lines.append("")
    lines.append("## Oracle Coverage")
    lines.append("")
    lines.append(
        f"train top10/top25: {sum(train_gold_top10)}/{len(train_gold_top10)} = {sum(train_gold_top10)/max(len(train_gold_top10),1):.4f} / "
        f"{sum(train_gold_top25)}/{len(train_gold_top25)} = {sum(train_gold_top25)/max(len(train_gold_top25),1):.4f}"
    )
    lines.append(
        f"test top10/top25: {sum(test_gold_top10)}/{len(test_gold_top10)} = {sum(test_gold_top10)/max(len(test_gold_top10),1):.4f} / "
        f"{sum(test_gold_top25)}/{len(test_gold_top25)} = {sum(test_gold_top25)/max(len(test_gold_top25),1):.4f}"
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| decode | train EM | test EM | test F1 | P | R | changed | bad->good | good->bad | bad->bad |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        f"| A_normal | {train_metrics_normal['exact_match']:.4f} | "
        f"{test_metrics_normal['exact_match']:.4f} | {test_metrics_normal['f1']:.4f} | "
        f"{test_metrics_normal['precision']:.4f} | {test_metrics_normal['recall']:.4f} | "
        "0 | 0 | 0 | 0 |"
    )
    for method in methods:
        result = results[method]
        test_m = result["test_metrics"]
        train_m = result["train_metrics"]
        changed = result["test_changed"]
        lines.append(
            f"| {method} | {train_m['exact_match']:.4f} | "
            f"{test_m['exact_match']:.4f} | {test_m['f1']:.4f} | "
            f"{test_m['precision']:.4f} | {test_m['recall']:.4f} | "
            f"{len(changed['changed'])} | {len(changed['bad_to_good'])} | "
            f"{len(changed['good_to_bad'])} | {len(changed['bad_to_bad'])} |"
        )
    lines.append("")
    lines.append("## Selected Configs")
    lines.append("")
    for method in methods:
        cfg = PVGConfig(**results[method]["best_config"])
        lines.append(f"- `{method}`: `{cfg.label()}`")
    lines.append("")
    lines.append("## Error Counts")
    lines.append("")
    lines.append("| decode | wrong edges | missed_stack | location_region | wrong_support_part | false_stack |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    normal_err = result_json["normal"]["test_error_counts"]
    lines.append(
        f"| A_normal | {normal_err['wrong_edges']:.0f} | {normal_err['missed_stack']:.0f} | "
        f"{normal_err['location_region']:.0f} | {normal_err['wrong_support_part']:.0f} | {normal_err['false_stack']:.0f} |"
    )
    for method in methods:
        err = results[method]["test_error_counts"]
        lines.append(
            f"| {method} | {err['wrong_edges']:.0f} | {err['missed_stack']:.0f} | "
            f"{err['location_region']:.0f} | {err['wrong_support_part']:.0f} | {err['false_stack']:.0f} |"
        )
    lines.append("")
    lines.append("## Changed Images")
    for method in methods:
        changed = results[method]["test_changed"]
        cov = results[method]["test_changed_edge_coverage"]
        lines.append("")
        lines.append(f"### {method}")
        lines.append(
            f"changed-edge evidence: {cov['changed_edge_evidence_covered']:.0f}/{cov['changed_edge_evidence_total']:.0f} = "
            f"{cov['changed_edge_evidence_coverage']:.4f}"
        )
        for key in ("changed", "bad_to_good", "good_to_bad", "bad_to_bad"):
            items = [(x["sample_id"], x["gold_top25"]) for x in changed[key]]
            lines.append(f"- {key}: {len(items)}")
            if items:
                lines.append("  " + _fmt_changed(items))

    _write_wrong_samples(
        lines,
        "A_normal",
        test_normal,
        test_exact_normal,
        test_idx,
        samples,
        sketch,
        data["active_part_masks"],
        test_gold_top25,
        args.max_wrong_samples,
    )
    for method in methods:
        _write_wrong_samples(
            lines,
            method,
            results[method]["test_assignments"],
            results[method]["test_exact"],
            test_idx,
            samples,
            sketch,
            data["active_part_masks"],
            test_gold_top25,
            args.max_wrong_samples,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    json_path = args.json_out or args.out.with_suffix(".json")
    json_path.write_text(json.dumps(result_json, indent=2) + "\n")
    print("\n".join(lines[:120]))
    print(f"\nSaved PVG report to {args.out}")
    print(f"Saved PVG JSON to {json_path}")


if __name__ == "__main__":
    main()
