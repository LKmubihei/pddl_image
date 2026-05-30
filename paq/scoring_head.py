"""Predicate Scoring Head for PaQ (Type-Aware, Canonical Ordering).
===================================================================

Key features:
  - Enumerates type-constrained object combinations
  - Supports both hardcoded and predicted (soft) type assignments
  - Returns flat (B, N_canonical) tensor in deterministic canonical order
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredicateScoringHead(nn.Module):
    """Type-aware predicate scoring with canonical ordering.

    For each predicate type, only scores valid type-constrained object
    combinations.  Returns a flat (B, N_canonical) tensor of logits in
    canonical order.

    Supports two modes:
      1. Hard type assignment: object_type_ids (B, N_obj) long tensor
      2. Soft type distribution: type_probs (B, N_obj, N_types) from
         SlotTypeClassifier — uses marginal scoring over type uncertainty

    Canonical order: predicate types sorted by name; within each type,
    groundings follow the natural product order of sorted type-member lists.
    """

    def __init__(
        self,
        d_slot: int = 256,
        predicate_defs: list[dict] | None = None,
        type_names: list[str] | None = None,
        tau_unknown: float = 0.3,
        scorer_type: str = "film",
    ):
        """
        Args:
            d_slot: slot dimension.
            predicate_defs: ordered list of dicts, one per dynamic predicate
                type (sorted by name, matching predicate query/slot order):
                    {"name": str, "arity": int, "param_types": [str, ...]}
            type_names: list of type names where list index == type id.
            tau_unknown: margin for "unknown" classification.
        """
        super().__init__()
        self.d_slot = d_slot
        self.predicate_defs = predicate_defs or []
        self.type_names = type_names or []
        self.tau_unknown = tau_unknown
        if scorer_type not in {"film", "legacy"}:
            raise ValueError(f"Unknown scorer_type: {scorer_type}")
        self.scorer_type = scorer_type

        # Legacy arity-shared scorers, kept as an explicit ablation because
        # older real-DINOv3 oracle-transition runs were calibrated with it.
        self.legacy_nullary_scorer = nn.Sequential(
            nn.Linear(d_slot, d_slot // 2), nn.GELU(), nn.Linear(d_slot // 2, 1)
        )
        self.legacy_unary_scorer = nn.Sequential(
            nn.Linear(d_slot * 2, d_slot), nn.GELU(), nn.Linear(d_slot, 1)
        )
        self.legacy_binary_scorer = nn.Sequential(
            nn.Linear(d_slot * 4, d_slot), nn.GELU(), nn.Linear(d_slot, 1)
        )

        # Predicate-conditioned scoring. Object relations are encoded by
        # arity-specific feature MLPs, then modulated by the predicate slot.
        self.nullary_scorer = nn.Sequential(
            nn.Linear(d_slot * 4, d_slot), nn.GELU(), nn.Linear(d_slot, 1)
        )
        self.unary_feat = nn.Sequential(
            nn.Linear(d_slot, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot)
        )
        self.binary_feat = nn.Sequential(
            nn.Linear(d_slot * 4, d_slot), nn.GELU(), nn.Linear(d_slot, d_slot)
        )
        self.unary_film = nn.Linear(d_slot, d_slot * 2)
        self.binary_film = nn.Linear(d_slot, d_slot * 2)
        self.unary_out = nn.Linear(d_slot, 1)
        self.binary_out = nn.Linear(d_slot, 1)

    def _build_type_map(
        self, object_type_ids: torch.Tensor
    ) -> dict[str, list[int]]:
        """Map type name -> sorted list of object slot indices.

        Uses batch-0 (types are consistent across the batch).
        """
        tm: dict[str, list[int]] = {}
        for type_idx, tname in enumerate(self.type_names):
            mask = object_type_ids[0] == type_idx
            tm[tname] = mask.nonzero(as_tuple=True)[0].tolist()
        return tm

    def _build_soft_type_map(
        self, type_probs: torch.Tensor, threshold: float = 0.1
    ) -> dict[str, list[int]]:
        """Map type name -> list of object slot indices (soft assignment).

        Includes slots where P(type) > threshold.
        Uses batch-0 for the canonical ordering.
        """
        tm: dict[str, list[int]] = {}
        for type_idx, tname in enumerate(self.type_names):
            mask = type_probs[0, :, type_idx] > threshold
            tm[tname] = mask.nonzero(as_tuple=True)[0].tolist()
        return tm

    def forward(
        self,
        predicate_slots: torch.Tensor,
        object_slots: torch.Tensor,
        object_type_ids: torch.Tensor | None = None,
        type_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            predicate_slots: (B, N_pred_types, D) — one per dynamic pred type.
            object_slots:    (B, N_obj, D) — one per object.
            object_type_ids: (B, N_obj) long — type index per object slot.
                             Can be None if type_probs is provided.
            type_probs:      (B, N_obj, N_types) soft type probabilities
                             from SlotTypeClassifier. Used when object_type_ids
                             is None, or to refine the type map.

        Returns:
            canonical_scores: (B, N_canonical) logits in canonical order.
        """
        B = predicate_slots.shape[0]
        device = predicate_slots.device

        # Build type map: prefer predicted types if available
        if type_probs is not None:
            type_map = self._build_soft_type_map(type_probs)
        elif object_type_ids is not None:
            type_map = self._build_type_map(object_type_ids)
        else:
            # Fallback: assign all objects to first type
            type_map = {tname: list(range(object_slots.shape[1]))
                        for tname in self.type_names}

        parts: list[torch.Tensor] = []
        obj_mean = object_slots.mean(dim=1)
        obj_max = object_slots.max(dim=1).values

        for slot_idx, pdef in enumerate(self.predicate_defs):
            arity = pdef["arity"]
            param_types = pdef["param_types"]

            if slot_idx >= predicate_slots.shape[1]:
                n_g = self._n_groundings(param_types, type_map)
                parts.append(torch.zeros(B, max(n_g, 1), device=device))
                continue

            p_slot = predicate_slots[:, slot_idx, :]  # (B, D)

            if arity == 0:
                if self.scorer_type == "legacy":
                    s = self.legacy_nullary_scorer(p_slot).squeeze(-1)
                else:
                    attn = torch.softmax(
                        (object_slots * p_slot.unsqueeze(1)).sum(dim=-1)
                        / (self.d_slot ** 0.5),
                        dim=1,
                    )
                    obj_attn = torch.sum(attn.unsqueeze(-1) * object_slots, dim=1)
                    s = self.nullary_scorer(
                        torch.cat([p_slot, obj_attn, obj_mean, obj_max], dim=-1)
                    ).squeeze(-1)
                parts.append(s.unsqueeze(-1))

            elif arity == 1:
                idx = type_map.get(param_types[0], [])
                if not idx:
                    parts.append(torch.zeros(B, 1, device=device))
                    continue
                objs = object_slots[:, idx, :]
                if self.scorer_type == "legacy":
                    p_exp = p_slot.unsqueeze(1).expand_as(objs)
                    s = self.legacy_unary_scorer(
                        torch.cat([p_exp, objs], dim=-1)
                    ).squeeze(-1)
                else:
                    h = self.unary_feat(objs)
                    gamma, beta = self.unary_film(p_slot).chunk(2, dim=-1)
                    h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
                    s = self.unary_out(F.gelu(h)).squeeze(-1)
                parts.append(s)

            elif arity == 2:
                idx0 = type_map.get(param_types[0], [])
                idx1 = type_map.get(param_types[1], [])
                if not idx0 or not idx1:
                    parts.append(torch.zeros(B, 1, device=device))
                    continue
                s0 = object_slots[:, idx0, :]
                s1 = object_slots[:, idx1, :]
                N0, N1 = len(idx0), len(idx1)
                o1 = s0.unsqueeze(2).expand(-1, -1, N1, -1)
                o2 = s1.unsqueeze(1).expand(-1, N0, -1, -1)
                if self.scorer_type == "legacy":
                    p_exp = p_slot.unsqueeze(1).unsqueeze(1).expand(-1, N0, N1, -1)
                    s = self.legacy_binary_scorer(
                        torch.cat([p_exp, o1, o2, o1 * o2], dim=-1)
                    ).squeeze(-1)
                else:
                    pair = torch.cat([o1, o2, o1 * o2, (o1 - o2).abs()], dim=-1)
                    h = self.binary_feat(pair)
                    gamma, beta = self.binary_film(p_slot).chunk(2, dim=-1)
                    h = (
                        h * (1.0 + gamma.unsqueeze(1).unsqueeze(1))
                        + beta.unsqueeze(1).unsqueeze(1)
                    )
                    s = self.binary_out(F.gelu(h)).squeeze(-1)
                # Filter self-relations when both params are same type
                same_type = param_types[0] == param_types[1]
                if same_type:
                    flat = s.reshape(B, N0 * N1)
                    mask = [i * N1 + j for i in range(N0) for j in range(N1)
                            if idx0[i] != idx1[j]]
                    parts.append(flat[:, mask] if mask
                                 else torch.zeros(B, 1, device=device))
                else:
                    parts.append(s.reshape(B, N0 * N1))

        return torch.cat(parts, dim=1)

    @staticmethod
    def _n_groundings(param_types: list[str], type_map: dict) -> int:
        n = 1
        for pt in param_types:
            n *= len(type_map.get(pt, []))
        return max(n, 1)
