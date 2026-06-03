#!/usr/bin/env python3
"""Pure evaluation diagnostic for a PDDL-conditioned cache verifier.

The baseline placement model is kept as the candidate generator.  This script
does not train a new neural module.  It extracts the model's object slots and
support logits, builds a non-parametric memory from the training split, selects
cache hyperparameters by train leave-one-out, and reranks the baseline top-K
legal placement states on the held-out split.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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


def parse_int_values(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_float_values(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def _active_parts(sketch: AriacPlacementSketch, active_mask: torch.Tensor) -> list[str]:
    return [p for p, flag in zip(sketch.parts, active_mask.tolist()) if flag > 0]


def _active_indices(active_mask: torch.Tensor) -> list[int]:
    return [i for i, flag in enumerate(active_mask.tolist()) if flag > 0]


def _valid_candidate_indices(
    sketch: AriacPlacementSketch,
    part: str,
    active_set: set[str],
) -> list[int]:
    return [
        ci
        for ci, cand in enumerate(sketch.place_candidates[part])
        if cand in sketch.locations or cand in active_set
    ]


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
) -> tuple[dict[str, float], torch.Tensor, list[bool]]:
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
    return metrics, preds, [bool(x) for x in exact_flags]


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


def _part_kind(name: str) -> str:
    for kind in ("battery", "pump", "regulator"):
        if f"_{kind}" in name:
            return kind
    return "other"


def _part_color(name: str) -> str:
    for color in ("blue", "green", "red"):
        if name.startswith(color + "_"):
            return color
    return "none"


def _candidate_relation_type(
    sketch: AriacPlacementSketch,
    candidate: str,
) -> str:
    if candidate == sketch.table_location:
        return "table"
    if candidate in sketch.locations:
        return "region"
    return "support_part"


def _bucket_name(
    sketch: AriacPlacementSketch,
    part: str,
    candidate: str,
    bucket_mode: str,
) -> str:
    relation = _candidate_relation_type(sketch, candidate)
    if bucket_mode == "coarse":
        return relation
    if bucket_mode == "part_kind":
        cand_kind = candidate if candidate in sketch.locations else _part_kind(candidate)
        return f"{_part_kind(part)}->{relation}:{cand_kind}"
    if bucket_mode == "candidate_name":
        return f"{_part_kind(part)}->{candidate if candidate in sketch.locations else relation}"
    raise ValueError(f"Unknown bucket_mode: {bucket_mode}")


def _one_hot(name: str, vocab: list[str]) -> list[float]:
    return [1.0 if name == item else 0.0 for item in vocab]


def _edge_key(
    slots: torch.Tensor,
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    part_idx: int,
    cand_idx: int,
    meta_feature_scale: float,
) -> torch.Tensor:
    part = sketch.parts[part_idx]
    cand = sketch.place_candidates[part][cand_idx]
    active = set(_active_parts(sketch, active_mask.detach().cpu()))
    valid = _valid_candidate_indices(sketch, part, active)
    row = support_scores[part_idx].detach().float().cpu()
    valid_scores = row[valid]
    chosen = row[cand_idx]
    mean = valid_scores.mean()
    std = valid_scores.std(unbiased=False).clamp_min(1e-6)
    alt = [ci for ci in valid if ci != cand_idx]
    best_alt = row[alt].max() if alt else chosen
    table_ci = (
        sketch.place_candidates[part].index(sketch.table_location)
        if sketch.table_location in sketch.place_candidates[part]
        else cand_idx
    )
    rank = 1 + sum(1 for ci in valid if float(row[ci].item()) > float(chosen.item()))

    part_slot = F.normalize(
        slots[sketch.part_object_indices[part_idx]].detach().float().cpu(),
        dim=0,
    )
    cand_obj_idx = sketch.place_candidate_object_indices[part_idx][cand_idx]
    cand_slot = F.normalize(slots[cand_obj_idx].detach().float().cpu(), dim=0)

    relation = _candidate_relation_type(sketch, cand)
    scalars = torch.tensor(
        [
            float(((chosen - mean) / std).item()),
            float(((chosen - best_alt) / std).item()),
            float(((chosen - row[table_ci]) / std).item()),
            float(rank) / float(max(len(valid), 1)),
            1.0 if cand in active else 0.0,
        ],
        dtype=torch.float32,
    )
    type_bits = torch.tensor(
        _one_hot(_part_kind(part), ["battery", "pump", "regulator", "other"])
        + _one_hot(_part_color(part), ["blue", "green", "red", "none"])
        + _one_hot(relation, ["table", "region", "support_part"])
        + _one_hot(cand if cand in sketch.locations else "part", sketch.locations + ["part"])
        + _one_hot(_part_kind(cand) if cand in sketch.parts else "location", ["battery", "pump", "regulator", "location", "other"]),
        dtype=torch.float32,
    )
    key = torch.cat(
        [
            part_slot,
            cand_slot,
            part_slot * cand_slot,
            torch.abs(part_slot - cand_slot),
            meta_feature_scale * scalars,
            meta_feature_scale * type_bits,
        ],
        dim=0,
    )
    return F.normalize(key, dim=0)


@dataclass(frozen=True)
class EdgeExample:
    sample_index: int
    part_idx: int
    cand_idx: int
    bucket: str
    key: torch.Tensor
    label: int


@dataclass(frozen=True)
class StateExample:
    sample_index: int
    key: torch.Tensor
    label: int


@dataclass(frozen=True)
class CacheConfig:
    k: int
    beta: float
    edge_lambda: float
    state_lambda: float

    def name(self) -> str:
        return (
            f"k={self.k}, beta={self.beta:g}, "
            f"edge_lambda={self.edge_lambda:g}, state_lambda={self.state_lambda:g}"
        )


def _gold_candidate_indices(
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    part_idx: int,
) -> set[int]:
    golds: set[int] = set()
    for vi in variant_mask.detach().cpu().nonzero(as_tuple=True)[0].tolist():
        ci = int(support_target_variants[vi, part_idx].detach().cpu().item())
        if ci >= 0:
            golds.add(ci)
    return golds


def _sample_edge_examples(
    sample_index: int,
    slots: torch.Tensor,
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    bucket_mode: str,
    meta_feature_scale: float,
) -> list[EdgeExample]:
    examples: list[EdgeExample] = []
    active = set(_active_parts(sketch, active_mask.detach().cpu()))
    for part_idx in _active_indices(active_mask.detach().cpu()):
        part = sketch.parts[part_idx]
        golds = _gold_candidate_indices(support_target_variants, variant_mask, part_idx)
        for cand_idx in _valid_candidate_indices(sketch, part, active):
            cand = sketch.place_candidates[part][cand_idx]
            examples.append(
                EdgeExample(
                    sample_index=sample_index,
                    part_idx=part_idx,
                    cand_idx=cand_idx,
                    bucket=_bucket_name(sketch, part, cand, bucket_mode),
                    key=_edge_key(
                        slots,
                        support_scores,
                        active_mask,
                        sketch,
                        part_idx,
                        cand_idx,
                        meta_feature_scale,
                    ),
                    label=1 if cand_idx in golds else 0,
                )
            )
    return examples


def _state_key(
    support_scores: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    meta_feature_scale: float,
) -> torch.Tensor:
    features = base._legal_assignment_features(
        support_scores,
        target.unsqueeze(0),
        active_mask,
        sketch,
    ).squeeze(0).detach().float().cpu()
    bits: list[float] = []
    active = set(_active_parts(sketch, active_mask.detach().cpu()))
    for part_idx, part in enumerate(sketch.parts):
        ci = int(target.detach().cpu()[part_idx].item())
        if part not in active or ci < 0:
            bits.extend([1.0, 0.0, 0.0, 0.0])
            bits.extend([0.0] * (len(sketch.locations) + 1))
            bits.extend([0.0] * 4)
            continue
        cand = sketch.place_candidates[part][ci]
        relation = _candidate_relation_type(sketch, cand)
        bits.extend([0.0] + _one_hot(relation, ["table", "region", "support_part"]))
        bits.extend(_one_hot(cand if cand in sketch.locations else "part", sketch.locations + ["part"]))
        bits.extend(_one_hot(_part_kind(cand) if cand in sketch.parts else "location", ["battery", "pump", "regulator", "location"]))
    key = torch.cat([meta_feature_scale * features, torch.tensor(bits, dtype=torch.float32)], dim=0)
    return F.normalize(key, dim=0)


def _is_gold_target(
    target: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
) -> bool:
    row = target.detach().cpu()
    for vi in variant_mask.detach().cpu().nonzero(as_tuple=True)[0].tolist():
        gold = support_target_variants[vi].detach().cpu()
        active = gold >= 0
        if active.sum().item() and bool((row[active] == gold[active]).all().item()):
            return True
    return False


def _sample_state_examples(
    sample_index: int,
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
    top_k: int,
    meta_feature_scale: float,
) -> list[StateExample]:
    rows: list[tuple[torch.Tensor, int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(row: torch.Tensor, label: int):
        key = tuple(int(x) for x in row.detach().cpu().tolist())
        if key in seen:
            return
        seen.add(key)
        rows.append((row.detach().cpu().long(), label))

    for vi in variant_mask.detach().cpu().nonzero(as_tuple=True)[0].tolist():
        gold = support_target_variants[vi]
        if (gold >= 0).sum().item():
            add(gold, 1)

    legal_targets = base._topk_legal_assignment_targets(
        support_scores,
        active_mask,
        sketch,
        top_k,
    ).detach().cpu()
    for row in legal_targets:
        if not _is_gold_target(row, support_target_variants, variant_mask):
            add(row, 0)

    return [
        StateExample(
            sample_index=sample_index,
            key=_state_key(support_scores, row, active_mask, sketch, meta_feature_scale),
            label=label,
        )
        for row, label in rows
    ]


def _build_edge_memory(
    examples_by_sample: dict[int, list[EdgeExample]],
    memory_indices: set[int],
) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    buckets: dict[str, list[EdgeExample]] = {}
    for si in memory_indices:
        for ex in examples_by_sample[si]:
            buckets.setdefault(ex.bucket, []).append(ex)
    memory: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for bucket, examples in buckets.items():
        keys = torch.stack([ex.key for ex in examples]).float()
        labels = torch.tensor([ex.label for ex in examples], dtype=torch.float32)
        prior = float(labels.mean().item()) if labels.numel() else 0.5
        memory[bucket] = (keys, labels, prior)
    return memory


def _build_state_memory(
    examples_by_sample: dict[int, list[StateExample]],
    memory_indices: set[int],
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    examples: list[StateExample] = []
    for si in memory_indices:
        examples.extend(examples_by_sample[si])
    if not examples:
        return None
    keys = torch.stack([ex.key for ex in examples]).float()
    labels = torch.tensor([ex.label for ex in examples], dtype=torch.float32)
    prior = float(labels.mean().item()) if labels.numel() else 0.5
    return keys, labels, prior


def _memory_scores_from_similarity(
    sims: torch.Tensor,
    labels: torch.Tensor,
    prior: float,
    k: int,
    beta: float,
    mode: str,
    prior_smoothing: float,
) -> torch.Tensor:
    if sims.numel() == 0:
        return torch.zeros(sims.shape[0], dtype=torch.float32)
    keep_k = min(int(k), sims.shape[1])
    if keep_k <= 0:
        keep_k = sims.shape[1]
    top_vals, top_idx = torch.topk(sims, k=keep_k, dim=1)
    top_labels = labels[top_idx]
    if mode == "knn_logit":
        weights = torch.exp(beta * top_vals)
        pos = (weights * top_labels).sum(dim=1)
        denom = weights.sum(dim=1)
        prob = (pos + prior_smoothing * prior) / (denom + prior_smoothing)
        prob = prob.clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(prob)
    if mode == "tip_logsumexp":
        scaled = beta * top_vals
        pos_scaled = scaled.masked_fill(top_labels <= 0.5, -float("inf"))
        neg_scaled = scaled.masked_fill(top_labels > 0.5, -float("inf"))
        has_pos = (top_labels > 0.5).any(dim=1)
        has_neg = (top_labels <= 0.5).any(dim=1)
        scores = torch.zeros(sims.shape[0], dtype=torch.float32)
        valid = has_pos & has_neg
        if valid.any():
            scores[valid] = torch.logsumexp(pos_scaled[valid], dim=1) - torch.logsumexp(neg_scaled[valid], dim=1)
        return scores
    raise ValueError(f"Unknown cache mode: {mode}")


def _edge_score_tables_for_sample(
    query_examples: list[EdgeExample],
    memory: dict[str, tuple[torch.Tensor, torch.Tensor, float]],
    sketch: AriacPlacementSketch,
    k_values: list[int],
    betas: list[float],
    mode: str,
    prior_smoothing: float,
) -> dict[tuple[int, float], torch.Tensor]:
    tables = {
        (k, beta): torch.zeros(sketch.n_parts, sketch.n_candidates, dtype=torch.float32)
        for k in k_values
        for beta in betas
    }
    by_bucket: dict[str, list[EdgeExample]] = {}
    for ex in query_examples:
        by_bucket.setdefault(ex.bucket, []).append(ex)
    for bucket, examples in by_bucket.items():
        if bucket not in memory:
            continue
        mem_keys, labels, prior = memory[bucket]
        q_keys = torch.stack([ex.key for ex in examples]).float()
        sims = q_keys.matmul(mem_keys.T)
        for k in k_values:
            for beta in betas:
                scores = _memory_scores_from_similarity(
                    sims,
                    labels,
                    prior,
                    k=k,
                    beta=beta,
                    mode=mode,
                    prior_smoothing=prior_smoothing,
                )
                table = tables[(k, beta)]
                for ex, score in zip(examples, scores.tolist()):
                    table[ex.part_idx, ex.cand_idx] = float(score)
    return tables


def _state_scores_for_targets(
    support_scores: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor,
    state_memory: tuple[torch.Tensor, torch.Tensor, float] | None,
    sketch: AriacPlacementSketch,
    k_values: list[int],
    betas: list[float],
    mode: str,
    prior_smoothing: float,
    meta_feature_scale: float,
) -> dict[tuple[int, float], torch.Tensor]:
    out = {
        (k, beta): torch.zeros(targets.shape[0], dtype=torch.float32)
        for k in k_values
        for beta in betas
    }
    if state_memory is None or targets.numel() == 0:
        return out
    mem_keys, labels, prior = state_memory
    q_keys = torch.stack([
        _state_key(support_scores, row, active_mask, sketch, meta_feature_scale)
        for row in targets.detach().cpu()
    ]).float()
    sims = q_keys.matmul(mem_keys.T)
    for k in k_values:
        for beta in betas:
            out[(k, beta)] = _memory_scores_from_similarity(
                sims,
                labels,
                prior,
                k=k,
                beta=beta,
                mode=mode,
                prior_smoothing=prior_smoothing,
            )
    return out


def _decode_with_cache(
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    edge_score_table: torch.Tensor,
    state_scores: torch.Tensor,
    sketch: AriacPlacementSketch,
    top_k: int,
    edge_lambda: float,
    state_lambda: float,
) -> tuple[dict[str, str], dict[str, float]]:
    legal_targets = base._topk_legal_assignment_targets(
        support_scores,
        active_mask,
        sketch,
        top_k,
    )
    base_scores = base._assignment_scores_from_targets(support_scores, legal_targets).detach().cpu()
    edge_scores: list[float] = []
    for row in legal_targets.detach().cpu():
        total = 0.0
        for pi, ci in enumerate(row.tolist()):
            if ci >= 0:
                total += float(edge_score_table[pi, ci].item())
        edge_scores.append(total)
    edge_scores_t = torch.tensor(edge_scores, dtype=torch.float32)
    state_scores = state_scores.detach().cpu()
    total_scores = base_scores + edge_lambda * edge_scores_t + state_lambda * state_scores
    best = int(torch.argmax(total_scores).item())
    active = _active_parts(sketch, active_mask.detach().cpu())
    assignment = _assignment_from_target_row(sketch, legal_targets[best], active)
    diagnostics = {
        "base_score": float(base_scores[best].item()),
        "edge_score": float(edge_scores_t[best].item()),
        "state_score": float(state_scores[best].item()) if state_scores.numel() else 0.0,
        "total_score": float(total_scores[best].item()),
        "num_candidates": float(legal_targets.shape[0]),
    }
    return assignment, diagnostics


def _normal_decode(
    support_scores: torch.Tensor,
    active_mask: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> dict[str, str]:
    active = _active_parts(sketch, active_mask.detach().cpu())
    return sketch.decode(support_scores, active).assignment


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
    slot_chunks: list[torch.Tensor] = []
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
            slot_chunks.append(out["object_slots"].cpu())
    return torch.cat(support_chunks, dim=0), torch.cat(slot_chunks, dim=0)


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


def _precompute_examples(
    sample_indices: np.ndarray,
    support_scores: torch.Tensor,
    object_slots: torch.Tensor,
    active_part_masks: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    args,
) -> tuple[dict[int, list[EdgeExample]], dict[int, list[StateExample]]]:
    edge_examples: dict[int, list[EdgeExample]] = {}
    state_examples: dict[int, list[StateExample]] = {}
    for local_i, si_np in enumerate(sample_indices):
        si = int(si_np)
        edge_examples[si] = _sample_edge_examples(
            sample_index=si,
            slots=object_slots[local_i],
            support_scores=support_scores[local_i],
            active_mask=active_part_masks[si],
            support_target_variants=support_target_variants[si],
            variant_mask=variant_masks[si],
            sketch=sketch,
            bucket_mode=args.bucket_mode,
            meta_feature_scale=args.meta_feature_scale,
        )
        state_examples[si] = _sample_state_examples(
            sample_index=si,
            support_scores=support_scores[local_i],
            active_mask=active_part_masks[si],
            support_target_variants=support_target_variants[si],
            variant_mask=variant_masks[si],
            sketch=sketch,
            top_k=args.state_negative_top_k,
            meta_feature_scale=args.meta_feature_scale,
        )
    return edge_examples, state_examples


def _score_config_sort_key(item: tuple[CacheConfig, dict[str, float]]) -> tuple[float, float, float, float, float]:
    cfg, metrics = item
    # Prefer exact match, then F1, then smaller total cache correction.
    return (
        metrics["exact_match"],
        metrics["f1"],
        -abs(cfg.edge_lambda),
        -abs(cfg.state_lambda),
        -float(cfg.k),
    )


def _loo_select_config(
    train_idx: np.ndarray,
    train_support: torch.Tensor,
    train_edge_examples: dict[int, list[EdgeExample]],
    train_state_examples: dict[int, list[StateExample]],
    active_part_masks: torch.Tensor,
    labels: torch.Tensor,
    active_atom_masks: torch.Tensor,
    label_variants: torch.Tensor,
    support_target_variants: torch.Tensor,
    variant_masks: torch.Tensor,
    samples: list[base.AriacSample],
    sketch: AriacPlacementSketch,
    k_values: list[int],
    betas: list[float],
    edge_lambdas: list[float],
    state_lambdas: list[float],
    args,
) -> tuple[CacheConfig, dict[str, float], dict[str, object]]:
    configs = [
        CacheConfig(k=k, beta=beta, edge_lambda=edge_l, state_lambda=state_l)
        for k in k_values
        for beta in betas
        for edge_l in edge_lambdas
        for state_l in state_lambdas
    ]
    pred_by_config: dict[CacheConfig, list[dict[str, str]]] = {cfg: [] for cfg in configs}
    train_pos = {int(si): pos for pos, si in enumerate(train_idx)}
    train_set = {int(si) for si in train_idx}

    for heldout_si_np in train_idx:
        heldout_si = int(heldout_si_np)
        memory_indices = train_set - {heldout_si}
        edge_memory = _build_edge_memory(train_edge_examples, memory_indices)
        state_memory = _build_state_memory(train_state_examples, memory_indices)
        local_pos = train_pos[heldout_si]
        support = train_support[local_pos]
        active_mask = active_part_masks[heldout_si]
        legal_targets = base._topk_legal_assignment_targets(
            support,
            active_mask,
            sketch,
            args.legal_top_k,
        ).detach().cpu()
        edge_tables = _edge_score_tables_for_sample(
            train_edge_examples[heldout_si],
            edge_memory,
            sketch,
            k_values,
            betas,
            mode=args.cache_mode,
            prior_smoothing=args.prior_smoothing,
        )
        state_tables = _state_scores_for_targets(
            support,
            legal_targets,
            active_mask,
            state_memory,
            sketch,
            k_values,
            betas,
            mode=args.cache_mode,
            prior_smoothing=args.prior_smoothing,
            meta_feature_scale=args.meta_feature_scale,
        )
        for cfg in configs:
            assignment, _ = _decode_with_cache(
                support,
                active_mask,
                edge_tables[(cfg.k, cfg.beta)],
                state_tables[(cfg.k, cfg.beta)],
                sketch,
                top_k=args.legal_top_k,
                edge_lambda=cfg.edge_lambda,
                state_lambda=cfg.state_lambda,
            )
            pred_by_config[cfg].append(assignment)

    metrics_by_config: dict[CacheConfig, dict[str, float]] = {}
    for cfg, assignments in pred_by_config.items():
        metrics, _, _ = _evaluate_predictions(
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
        metrics_by_config[cfg] = metrics
    best_cfg, best_metrics = max(metrics_by_config.items(), key=_score_config_sort_key)

    top_rows = sorted(metrics_by_config.items(), key=_score_config_sort_key, reverse=True)[:20]
    selection = {
        "best_config": best_cfg.__dict__,
        "best_train_loo_metrics": best_metrics,
        "top_configs": [
            {"config": cfg.__dict__, "metrics": metrics}
            for cfg, metrics in top_rows
        ],
    }
    return best_cfg, best_metrics, selection


def _predict_with_config(
    sample_indices: np.ndarray,
    support_scores: torch.Tensor,
    query_edge_examples: dict[int, list[EdgeExample]],
    memory_edge_examples: dict[int, list[EdgeExample]],
    state_examples_memory: dict[int, list[StateExample]],
    memory_indices: set[int],
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
    cfg: CacheConfig,
    args,
) -> tuple[list[dict[str, str]], list[dict[str, float]]]:
    edge_memory = _build_edge_memory(memory_edge_examples, memory_indices)
    state_memory = _build_state_memory(state_examples_memory, memory_indices)
    sample_pos = {int(si): pos for pos, si in enumerate(sample_indices)}
    assignments: list[dict[str, str]] = []
    diagnostics: list[dict[str, float]] = []
    for si_np in sample_indices:
        si = int(si_np)
        local_pos = sample_pos[si]
        support = support_scores[local_pos]
        active_mask = active_part_masks[si]
        legal_targets = base._topk_legal_assignment_targets(
            support,
            active_mask,
            sketch,
            args.legal_top_k,
        ).detach().cpu()
        edge_tables = _edge_score_tables_for_sample(
            query_edge_examples[si],
            edge_memory,
            sketch,
            [cfg.k],
            [cfg.beta],
            mode=args.cache_mode,
            prior_smoothing=args.prior_smoothing,
        )
        state_tables = _state_scores_for_targets(
            support,
            legal_targets,
            active_mask,
            state_memory,
            sketch,
            [cfg.k],
            [cfg.beta],
            mode=args.cache_mode,
            prior_smoothing=args.prior_smoothing,
            meta_feature_scale=args.meta_feature_scale,
        )
        assignment, diag = _decode_with_cache(
            support,
            active_mask,
            edge_tables[(cfg.k, cfg.beta)],
            state_tables[(cfg.k, cfg.beta)],
            sketch,
            top_k=args.legal_top_k,
            edge_lambda=cfg.edge_lambda,
            state_lambda=cfg.state_lambda,
        )
        assignments.append(assignment)
        diagnostics.append(diag)
    return assignments, diagnostics


def _normal_assignments(
    sample_indices: np.ndarray,
    support_scores: torch.Tensor,
    active_part_masks: torch.Tensor,
    sketch: AriacPlacementSketch,
) -> list[dict[str, str]]:
    return [
        _normal_decode(support_scores[pos], active_part_masks[int(si)], sketch)
        for pos, si in enumerate(sample_indices)
    ]


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
    for i, si in enumerate(sample_indices):
        if src_assignments[i] == dst_assignments[i]:
            continue
        item = (samples[int(si)].sample_id, gold_in_topk[i])
        out["changed"].append(item)
        if src_exact[i] and not dst_exact[i]:
            out["good_to_bad"].append(item)
        elif not src_exact[i] and dst_exact[i]:
            out["bad_to_good"].append(item)
        elif not src_exact[i] and not dst_exact[i]:
            out["bad_to_bad"].append(item)
    return out


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
    for assignment, ok, si in zip(pred_assignments, exact_flags, sample_indices):
        if ok:
            continue
        sample = samples[int(si)]
        active = _active_parts(sketch, active_part_masks[int(si)])
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
    parser.add_argument("--legal-top-k", type=int, default=10)
    parser.add_argument("--knn-k-grid", type=str, default="3,5,10,20")
    parser.add_argument("--beta-grid", type=str, default="5,10,20")
    parser.add_argument("--edge-lambda-grid", type=str, default="0.2,0.5,1.0,2.0")
    parser.add_argument("--state-lambda-grid", type=str, default="0,0.2,0.5")
    parser.add_argument("--state-negative-top-k", type=int, default=10)
    parser.add_argument("--bucket-mode", type=str, default="coarse", choices=["coarse", "part_kind", "candidate_name"])
    parser.add_argument("--cache-mode", type=str, default="knn_logit", choices=["knn_logit", "tip_logsumexp"])
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
    parser.add_argument("--meta-feature-scale", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "ariac_cache_verifier_diagnostic_20260603.md")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--max-wrong-samples", type=int, default=30)
    args = parser.parse_args()

    if args.legal_top_k <= 0:
        raise ValueError("--legal-top-k must be positive")
    if args.state_negative_top_k <= 0:
        raise ValueError("--state-negative-top-k must be positive")
    if args.prior_smoothing < 0:
        raise ValueError("--prior-smoothing must be non-negative")
    if args.meta_feature_scale < 0:
        raise ValueError("--meta-feature-scale must be non-negative")

    k_values = parse_int_values(args.knn_k_grid)
    betas = parse_float_values(args.beta_grid)
    edge_lambdas = parse_float_values(args.edge_lambda_grid)
    state_lambdas = parse_float_values(args.state_lambda_grid)

    data = _prepare_split_and_data(args)
    ckpt = data["ckpt"]
    meta = data["meta"]
    samples = data["samples"]
    sketch = data["sketch"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    model = data["model"]

    print("=" * 72)
    print("ARIAC PDDL-cache verifier diagnostic")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  feature cache: {args.feature_cache}")
    print(f"  train={len(train_idx)} test={len(test_idx)} topK={args.legal_top_k}")
    print(f"  cache mode={args.cache_mode} bucket={args.bucket_mode}")
    print("=" * 72)

    train_support, train_slots = _forward_indices(
        model,
        data["features"],
        train_idx,
        data["type_ids"],
        data["slot_init"],
        args.batch_size,
        args.device,
    )
    test_support, test_slots = _forward_indices(
        model,
        data["features"],
        test_idx,
        data["type_ids"],
        data["slot_init"],
        args.batch_size,
        args.device,
    )

    train_edge_examples, train_state_examples = _precompute_examples(
        train_idx,
        train_support,
        train_slots,
        data["active_part_masks"],
        data["support_target_variants"],
        data["variant_masks"],
        sketch,
        args,
    )
    test_edge_examples, _ = _precompute_examples(
        test_idx,
        test_support,
        test_slots,
        data["active_part_masks"],
        data["support_target_variants"],
        data["variant_masks"],
        sketch,
        args,
    )

    print("  selecting cache hyperparameters by train leave-one-out...")
    best_cfg, best_loo_metrics, selection = _loo_select_config(
        train_idx,
        train_support,
        train_edge_examples,
        train_state_examples,
        data["active_part_masks"],
        data["labels"],
        data["active_atom_masks"],
        data["label_variants"],
        data["support_target_variants"],
        data["variant_masks"],
        samples,
        sketch,
        k_values,
        betas,
        edge_lambdas,
        state_lambdas,
        args,
    )
    print(f"  best LOO: {best_cfg.name()} EM={best_loo_metrics['exact_match']:.4f} F1={best_loo_metrics['f1']:.4f}")

    train_normal = _normal_assignments(train_idx, train_support, data["active_part_masks"], sketch)
    test_normal = _normal_assignments(test_idx, test_support, data["active_part_masks"], sketch)

    test_cache, test_diag = _predict_with_config(
        test_idx,
        test_support,
        test_edge_examples,
        train_edge_examples,
        train_state_examples,
        {int(si) for si in train_idx},
        data["active_part_masks"],
        sketch,
        best_cfg,
        args,
    )

    train_metrics_normal, _, train_exact_normal = _evaluate_predictions(
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
    test_metrics_normal, _, test_exact_normal = _evaluate_predictions(
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
    test_metrics_cache, _, test_exact_cache = _evaluate_predictions(
        test_cache,
        test_idx,
        samples,
        sketch,
        data["active_part_masks"],
        data["labels"],
        data["active_atom_masks"],
        data["label_variants"],
        data["variant_masks"],
    )

    train_gold_topk = [
        _gold_in_topk(
            train_support[pos],
            data["active_part_masks"][int(si)],
            data["support_target_variants"][int(si)],
            data["variant_masks"][int(si)],
            sketch,
            args.legal_top_k,
        )
        for pos, si in enumerate(train_idx)
    ]
    test_gold_topk = [
        _gold_in_topk(
            test_support[pos],
            data["active_part_masks"][int(si)],
            data["support_target_variants"][int(si)],
            data["variant_masks"][int(si)],
            sketch,
            args.legal_top_k,
        )
        for pos, si in enumerate(test_idx)
    ]

    test_placement_metrics = base.placement_ranking_metrics(
        test_support,
        data["support_target_variants"][test_idx],
        data["variant_masks"][test_idx],
        data["active_part_masks"][test_idx],
        sketch,
    )

    changed = _changed_summary(
        test_idx,
        samples,
        test_normal,
        test_cache,
        test_exact_normal,
        test_exact_cache,
        test_gold_topk,
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
            "legal_top_k": args.legal_top_k,
            "cache_mode": args.cache_mode,
            "bucket_mode": args.bucket_mode,
            "prior_smoothing": args.prior_smoothing,
            "meta_feature_scale": args.meta_feature_scale,
            "duplicate_meta": data["duplicate_meta"],
        },
        "selection": selection,
        "metrics": {
            "train_normal": train_metrics_normal,
            "train_loo_cache": best_loo_metrics,
            "test_normal": test_metrics_normal,
            "test_cache": test_metrics_cache,
            "test_placement": test_placement_metrics,
        },
        "topk_gold": {
            "train": {
                "count": int(sum(train_gold_topk)),
                "total": len(train_gold_topk),
                "rate": float(sum(train_gold_topk) / max(len(train_gold_topk), 1)),
            },
            "test": {
                "count": int(sum(test_gold_topk)),
                "total": len(test_gold_topk),
                "rate": float(sum(test_gold_topk) / max(len(test_gold_topk), 1)),
            },
        },
        "changed": {k: [{"sample_id": sid, "gold_topk": topk} for sid, topk in v] for k, v in changed.items()},
        "test_diagnostics": [
            {"sample_id": samples[int(si)].sample_id, **diag}
            for si, diag in zip(test_idx, test_diag)
        ],
    }

    lines: list[str] = []
    lines.append("# ARIAC PDDL-Cache Verifier Diagnostic")
    lines.append("")
    lines.append("This diagnostic loads one checkpoint and does not train a neural scorer.")
    lines.append("The cache verifier is calibrated by train leave-one-out, then fixed for held-out evaluation.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- checkpoint: `{args.checkpoint}`")
    lines.append(f"- feature_cache: `{args.feature_cache}`")
    lines.append(f"- train K from checkpoint: `{len(train_idx)}`")
    lines.append(f"- test size from checkpoint metadata: `{len(test_idx)}`")
    lines.append(f"- legal rerank topK: `{args.legal_top_k}`")
    lines.append(f"- cache mode: `{args.cache_mode}`")
    lines.append(f"- bucket mode: `{args.bucket_mode}`")
    lines.append(f"- selected config: `{best_cfg.name()}`")
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| decode | EM | F1 | precision | recall | legal |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, m in (
        ("train normal", train_metrics_normal),
        ("train LOO cache", best_loo_metrics),
        ("test normal", test_metrics_normal),
        ("test cache", test_metrics_cache),
    ):
        lines.append(
            f"| {name} | {m['exact_match']:.4f} | {m['f1']:.4f} | "
            f"{m['precision']:.4f} | {m['recall']:.4f} | {m['legal']:.4f} |"
        )
    lines.append("")
    lines.append("## Placement Ranking")
    lines.append("")
    lines.append(
        "test placement top1/top3/top10: "
        f"{test_placement_metrics['placement_part_top1']:.4f} / "
        f"{test_placement_metrics['placement_part_top3']:.4f} / "
        f"{test_placement_metrics['placement_part_top10']:.4f}"
    )
    lines.append(
        "test top1 error counts: "
        f"missed_stack={test_placement_metrics['missed_stack_top1']:.0f}, "
        f"location_region={test_placement_metrics['location_region_top1']:.0f}, "
        f"wrong_support_part={test_placement_metrics['wrong_support_part_top1']:.0f}, "
        f"false_stack={test_placement_metrics['false_stack_top1']:.0f}"
    )
    lines.append("")
    lines.append("## Top-K Gold Coverage")
    lines.append("")
    lines.append(
        f"train gold legal state in top{args.legal_top_k}: "
        f"{sum(train_gold_topk)}/{len(train_gold_topk)} = "
        f"{sum(train_gold_topk)/max(len(train_gold_topk),1):.4f}"
    )
    lines.append(
        f"test gold legal state in top{args.legal_top_k}: "
        f"{sum(test_gold_topk)}/{len(test_gold_topk)} = "
        f"{sum(test_gold_topk)/max(len(test_gold_topk),1):.4f}"
    )
    lines.append("")
    lines.append("## LOO Selection")
    lines.append("")
    lines.append("| rank | k | beta | edge lambda | state lambda | LOO EM | LOO F1 |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, item in enumerate(selection["top_configs"][:10], start=1):
        cfg = item["config"]
        m = item["metrics"]
        lines.append(
            f"| {rank} | {cfg['k']} | {cfg['beta']:.4g} | "
            f"{cfg['edge_lambda']:.4g} | {cfg['state_lambda']:.4g} | "
            f"{m['exact_match']:.4f} | {m['f1']:.4f} |"
        )
    lines.append("")
    lines.append("## Changed Images")
    lines.append("")
    for key in ("changed", "bad_to_good", "good_to_bad", "bad_to_bad"):
        lines.append(f"- {key}: {len(changed[key])}")
        if changed[key]:
            lines.append("  " + _fmt_changed(changed[key]))

    _write_wrong_samples(
        lines,
        "test normal",
        test_normal,
        test_exact_normal,
        test_idx,
        samples,
        sketch,
        data["active_part_masks"],
        test_gold_topk,
        args.max_wrong_samples,
    )
    _write_wrong_samples(
        lines,
        "test cache",
        test_cache,
        test_exact_cache,
        test_idx,
        samples,
        sketch,
        data["active_part_masks"],
        test_gold_topk,
        args.max_wrong_samples,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    json_path = args.json_out or args.out.with_suffix(".json")
    json_path.write_text(json.dumps(result_json, indent=2) + "\n")
    print("\n".join(lines[:90]))
    print(f"\nSaved cache verifier report to {args.out}")
    print(f"Saved cache verifier JSON to {json_path}")


if __name__ == "__main__":
    main()
