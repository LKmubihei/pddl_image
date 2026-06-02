"""Support-factor prediction heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class BlocksworldSupportHead(nn.Module):
    """Scores support(block) candidates from object slots.

    For every top block b and candidate support s, scores:
        MLP([z_b, z_s, z_b * z_s, z_b - z_s])
    """

    def __init__(
        self,
        d_slot: int,
        block_slot_indices: list[int],
        candidate_slot_indices: list[list[int]],
        hidden_dim: int | None = None,
        scorer_type: str = "legacy",
        temperature: float = 1.0,
        geometry_dim: int = 0,
        candidate_type_ids: list[list[int]] | None = None,
        candidate_prior_xy: list[list[list[float]]] | None = None,
        location_prior_weight: float = 0.0,
        location_prior_sigma: float = 0.2,
        patch_evidence_type: str = "none",
        patch_location_scale_init: float = 0.5,
        patch_table_scale_init: float = 0.5,
        patch_contact_scale_init: float = 0.5,
        patch_location_sigma: float = 0.18,
        patch_temperature: float = 1.0,
        patch_contact_top_k: int = 16,
        patch_contact_sigma_x: float = 0.12,
        patch_contact_sigma_y: float = 0.12,
        patch_contact_gap: float = 0.06,
    ):
        super().__init__()
        if scorer_type not in {
            "legacy",
            "pair",
            "two_stage",
            "typed_two_stage",
            "calibrated_two_stage",
        }:
            raise ValueError(f"Unknown support scorer_type: {scorer_type}")
        if temperature <= 0:
            raise ValueError("Support temperature must be positive")
        if geometry_dim < 0:
            raise ValueError("geometry_dim must be non-negative")
        if location_prior_sigma <= 0:
            raise ValueError("location_prior_sigma must be positive")
        if patch_evidence_type not in {
            "none",
            "location",
            "location_table",
            "location_table_contact",
        }:
            raise ValueError(f"Unknown patch_evidence_type: {patch_evidence_type}")
        if patch_location_sigma <= 0:
            raise ValueError("patch_location_sigma must be positive")
        if patch_temperature <= 0:
            raise ValueError("patch_temperature must be positive")
        if patch_contact_top_k <= 0:
            raise ValueError("patch_contact_top_k must be positive")
        if patch_contact_sigma_x <= 0 or patch_contact_sigma_y <= 0:
            raise ValueError("patch contact sigmas must be positive")
        if not block_slot_indices:
            raise ValueError("BlocksworldSupportHead requires at least one block slot")
        if not candidate_slot_indices:
            raise ValueError("BlocksworldSupportHead requires support candidates")
        n_candidates = len(candidate_slot_indices[0])
        if any(len(cands) != n_candidates for cands in candidate_slot_indices):
            raise ValueError("All blocks must have the same number of support candidates")
        if len(candidate_slot_indices) != len(block_slot_indices):
            raise ValueError(
                "candidate_slot_indices must have one candidate list per block"
            )

        hidden = hidden_dim or d_slot
        self.d_slot = d_slot
        self.n_blocks = len(block_slot_indices)
        self.n_candidates = n_candidates
        self.scorer_type = scorer_type
        self.temperature = float(temperature)
        self.geometry_dim = int(geometry_dim)
        self.location_prior_weight = float(location_prior_weight)
        self.location_prior_sigma = float(location_prior_sigma)
        self.patch_evidence_type = patch_evidence_type
        self.patch_location_sigma = float(patch_location_sigma)
        self.patch_temperature = float(patch_temperature)
        self.patch_contact_top_k = int(patch_contact_top_k)
        self.patch_contact_sigma_x = float(patch_contact_sigma_x)
        self.patch_contact_sigma_y = float(patch_contact_sigma_y)
        self.patch_contact_gap = float(patch_contact_gap)
        self.pair_geometry_dim = 0 if self.geometry_dim == 0 else self.geometry_dim * 2 + 10
        block_set = set(block_slot_indices)
        candidate_is_block = [
            [1.0 if idx in block_set else 0.0 for idx in row]
            for row in candidate_slot_indices
        ]
        if candidate_type_ids is None:
            candidate_type_ids = [
                [0 if idx in block_set else 2 for idx in row]
                for row in candidate_slot_indices
            ]
        if len(candidate_type_ids) != len(block_slot_indices) or any(
            len(row) != n_candidates for row in candidate_type_ids
        ):
            raise ValueError(
                "candidate_type_ids must match candidate_slot_indices shape"
            )
        if candidate_prior_xy is None:
            candidate_prior_xy = [
                [[0.0, 0.0] for _ in row]
                for row in candidate_slot_indices
            ]
        if len(candidate_prior_xy) != len(block_slot_indices) or any(
            len(row) != n_candidates for row in candidate_prior_xy
        ):
            raise ValueError(
                "candidate_prior_xy must match candidate_slot_indices shape"
            )
        self.register_buffer(
            "block_slot_indices",
            torch.tensor(block_slot_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "candidate_slot_indices",
            torch.tensor(candidate_slot_indices, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "candidate_is_block",
            torch.tensor(candidate_is_block, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "candidate_type_ids",
            torch.tensor(candidate_type_ids, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "candidate_prior_xy",
            torch.tensor(candidate_prior_xy, dtype=torch.float32),
            persistent=False,
        )
        visual_pair_dim = d_slot * 4 if scorer_type == "legacy" else d_slot * 5
        pair_dim = visual_pair_dim + self.pair_geometry_dim
        if scorer_type == "legacy":
            self.scorer = nn.Sequential(
                nn.Linear(pair_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        elif scorer_type == "typed_two_stage":
            self.part_scorer = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.location_scorer = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            self.candidate_bias = nn.Parameter(
                torch.zeros(self.n_blocks, self.n_candidates)
            )
        else:
            self.scorer = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            if scorer_type == "calibrated_two_stage":
                # Candidate types: 0=support part, 1=table, 2=placement region.
                self.type_stack_scale = nn.Parameter(
                    torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
                )
                self.type_bias = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.stack_scorer = None
        if scorer_type in {"two_stage", "typed_two_stage", "calibrated_two_stage"}:
            self.stack_scorer = nn.Sequential(
                nn.LayerNorm(d_slot),
                nn.Linear(d_slot, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        if patch_evidence_type != "none":
            self.patch_query_norm = nn.LayerNorm(d_slot)
            self.patch_key_norm = nn.LayerNorm(d_slot)
            self.patch_query_proj = nn.Linear(d_slot, d_slot, bias=False)
            self.patch_key_proj = nn.Linear(d_slot, d_slot, bias=False)
            self.patch_location_scale = nn.Parameter(
                torch.tensor(float(patch_location_scale_init), dtype=torch.float32)
            )
            self.patch_table_scale = nn.Parameter(
                torch.tensor(float(patch_table_scale_init), dtype=torch.float32)
            )
            self.patch_contact_scale = nn.Parameter(
                torch.tensor(float(patch_contact_scale_init), dtype=torch.float32)
            )

    def _pair_geometry_features(
        self,
        object_geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Build relation-level features from per-object soft geometry.

        object_geometry is expected to contain at least normalized center x/y in
        the first two channels and spread x/y in the next two channels.
        """
        if object_geometry.dim() != 3:
            raise ValueError(
                "object_geometry must be (B, N_obj, G), got "
                f"{tuple(object_geometry.shape)}"
            )
        if object_geometry.shape[-1] != self.geometry_dim:
            raise ValueError(
                f"Expected geometry_dim={self.geometry_dim}, got "
                f"{object_geometry.shape[-1]}"
            )
        top_geo = object_geometry[:, self.block_slot_indices, :]
        support_geo = object_geometry[:, self.candidate_slot_indices.reshape(-1), :]
        support_geo = support_geo.reshape(
            object_geometry.shape[0],
            self.n_blocks,
            self.n_candidates,
            self.geometry_dim,
        )
        top_geo = top_geo.unsqueeze(2).expand(-1, -1, self.n_candidates, -1)

        delta_xy = top_geo[..., :2] - support_geo[..., :2]
        abs_delta_xy = delta_xy.abs()
        dist = torch.sqrt((delta_xy.square()).sum(dim=-1, keepdim=True) + 1e-8)
        size_sum = top_geo[..., 2:4].abs() + support_geo[..., 2:4].abs() + 1e-6
        horiz_overlap = torch.exp(-abs_delta_xy[..., 0:1] / size_sum[..., 0:1])
        vert_overlap = torch.exp(-abs_delta_xy[..., 1:2] / size_sum[..., 1:2])
        soft_contact = horiz_overlap * vert_overlap
        top_above = support_geo[..., 1:2] - top_geo[..., 1:2]
        candidate_is_block = self.candidate_is_block.to(object_geometry.device)
        candidate_is_block = candidate_is_block.unsqueeze(0).unsqueeze(-1).expand(
            object_geometry.shape[0], -1, -1, -1
        )
        return torch.cat(
            [
                top_geo,
                support_geo,
                delta_xy,
                abs_delta_xy,
                dist,
                horiz_overlap,
                vert_overlap,
                soft_contact,
                top_above,
                candidate_is_block,
            ],
            dim=-1,
        )

    def _patch_coords(
        self,
        n_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        side = int(n_tokens ** 0.5)
        if side * side != n_tokens:
            raise ValueError(
                f"Patch evidence requires square token grid, got n_tokens={n_tokens}"
            )
        axis = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _object_patch_logits(
        self,
        patch_tokens: torch.Tensor,
        object_queries: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.dim() != 3:
            raise ValueError(
                f"patch_tokens must be (B, T, D), got {tuple(patch_tokens.shape)}"
            )
        if object_queries.dim() != 3:
            raise ValueError(
                "object_queries must be (B, N_obj, D), got "
                f"{tuple(object_queries.shape)}"
            )
        q = self.patch_query_proj(self.patch_query_norm(object_queries))
        k = self.patch_key_proj(self.patch_key_norm(patch_tokens))
        return torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_slot)

    def _location_patch_correction(
        self,
        object_patch_logits: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        part_logits = object_patch_logits[:, self.block_slot_indices, :]
        part_prob = torch.softmax(part_logits / self.patch_temperature, dim=-1)
        prior_xy = self.candidate_prior_xy.to(
            device=object_patch_logits.device,
            dtype=object_patch_logits.dtype,
        )
        delta = coords.view(1, 1, 1, -1, 2) - prior_xy.view(
            1, self.n_blocks, self.n_candidates, 1, 2
        )
        dist2 = delta.square().sum(dim=-1)
        prior = torch.exp(
            -dist2 / (2.0 * self.patch_location_sigma * self.patch_location_sigma)
        )
        mass = (part_prob.unsqueeze(2) * prior).sum(dim=-1).clamp_min(1e-6)
        raw = torch.log(mass)

        cand_types = self.candidate_type_ids.to(object_patch_logits.device)
        region_mask = (cand_types == 2).to(raw.dtype)
        region_count = region_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        region_mean = (raw * region_mask.unsqueeze(0)).sum(dim=-1, keepdim=True)
        region_mean = region_mean / region_count.unsqueeze(0)
        centered = raw - region_mean

        correction = self.patch_location_scale * centered * region_mask.unsqueeze(0)
        if self.patch_evidence_type in {"location_table", "location_table_contact"}:
            table_mask = (cand_types == 1).to(raw.dtype)
            region_centered = centered.masked_fill(
                region_mask.unsqueeze(0) <= 0,
                -1e6,
            )
            max_region = region_centered.max(dim=-1, keepdim=True).values.clamp_min(0.0)
            correction = correction - self.patch_table_scale * max_region * table_mask.unsqueeze(0)
        return correction

    def _contact_patch_correction(
        self,
        object_patch_logits: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        k_top = min(self.patch_contact_top_k, coords.shape[0])
        log_prob = torch.log_softmax(
            object_patch_logits / self.patch_temperature,
            dim=-1,
        )
        part_log_prob = log_prob[:, self.block_slot_indices, :]
        support_log_prob = log_prob[:, self.candidate_slot_indices.reshape(-1), :]
        support_log_prob = support_log_prob.reshape(
            log_prob.shape[0],
            self.n_blocks,
            self.n_candidates,
            log_prob.shape[-1],
        )

        top_vals, top_idx = torch.topk(part_log_prob, k=k_top, dim=-1)
        sup_vals, sup_idx = torch.topk(support_log_prob, k=k_top, dim=-1)
        top_xy = coords[top_idx]
        sup_xy = coords[sup_idx]
        dx = (top_xy.unsqueeze(2).unsqueeze(4)[..., 0] - sup_xy.unsqueeze(3)[..., 0]).abs()
        dy = sup_xy.unsqueeze(3)[..., 1] - top_xy.unsqueeze(2).unsqueeze(4)[..., 1]
        kernel = -dx / self.patch_contact_sigma_x
        kernel = kernel - (dy - self.patch_contact_gap).square() / (
            2.0 * self.patch_contact_sigma_y * self.patch_contact_sigma_y
        )
        kernel = kernel + torch.where(
            dy >= 0,
            torch.zeros_like(dy),
            torch.full_like(dy, -2.0),
        )
        pair = top_vals.unsqueeze(2).unsqueeze(4) + sup_vals.unsqueeze(3) + kernel
        raw = torch.logsumexp(pair.reshape(*pair.shape[:3], -1), dim=-1)

        support_mask = self.candidate_is_block.to(raw.device, raw.dtype)
        support_count = support_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        support_mean = (raw * support_mask.unsqueeze(0)).sum(dim=-1, keepdim=True)
        support_mean = support_mean / support_count.unsqueeze(0)
        centered = raw - support_mean
        return self.patch_contact_scale * centered * support_mask.unsqueeze(0)

    def _patch_evidence_correction(
        self,
        patch_tokens: torch.Tensor,
        object_queries: torch.Tensor,
    ) -> torch.Tensor:
        object_patch_logits = self._object_patch_logits(patch_tokens, object_queries)
        coords = self._patch_coords(
            object_patch_logits.shape[-1],
            object_patch_logits.device,
            object_patch_logits.dtype,
        )
        correction = self._location_patch_correction(object_patch_logits, coords)
        if self.patch_evidence_type == "location_table_contact":
            correction = correction + self._contact_patch_correction(
                object_patch_logits,
                coords,
            )
        return correction

    def forward(
        self,
        object_slots: torch.Tensor,
        object_geometry: torch.Tensor | None = None,
        patch_tokens: torch.Tensor | None = None,
        object_queries: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return support scores with shape (B, n_blocks, n_candidates)."""
        if object_slots.dim() != 3:
            raise ValueError(
                f"object_slots must be (B, N_obj, D), got {tuple(object_slots.shape)}"
            )
        top = object_slots[:, self.block_slot_indices, :]
        support = object_slots[:, self.candidate_slot_indices.reshape(-1), :]
        support = support.reshape(
            object_slots.shape[0], self.n_blocks, self.n_candidates, self.d_slot
        )
        top = top.unsqueeze(2).expand(-1, -1, self.n_candidates, -1)
        if self.scorer_type == "legacy":
            pair = torch.cat([top, support, top * support, top - support], dim=-1)
        else:
            pair = torch.cat(
                [top, support, top * support, top - support, (top - support).abs()],
                dim=-1,
            )
        if self.geometry_dim > 0:
            if object_geometry is None:
                object_geometry = object_slots.new_zeros(
                    object_slots.shape[0], object_slots.shape[1], self.geometry_dim
                )
            pair = torch.cat([pair, self._pair_geometry_features(object_geometry)], dim=-1)
        if self.scorer_type == "typed_two_stage":
            part_scores = self.part_scorer(pair).squeeze(-1)
            location_scores = self.location_scorer(pair).squeeze(-1)
            candidate_is_block = self.candidate_is_block.to(pair.device).unsqueeze(0)
            scores = (
                part_scores * candidate_is_block
                + location_scores * (1.0 - candidate_is_block)
                + self.candidate_bias.to(pair.device).unsqueeze(0)
            )
        else:
            scores = self.scorer(pair).squeeze(-1)
        if self.scorer_type == "calibrated_two_stage":
            if self.stack_scorer is None:
                raise RuntimeError("calibrated_two_stage head missing stack_scorer")
            stack_logit = self.stack_scorer(top[:, :, 0, :]).squeeze(-1)
            cand_types = self.candidate_type_ids.to(scores.device)
            stack_scale = self.type_stack_scale.to(scores.device)[cand_types]
            type_bias = self.type_bias.to(scores.device)[cand_types]
            scores = scores + stack_logit.unsqueeze(-1) * stack_scale.unsqueeze(0)
            scores = scores + type_bias.unsqueeze(0)
        elif self.scorer_type in {"two_stage", "typed_two_stage"}:
            if self.stack_scorer is None:
                raise RuntimeError("two-stage support head missing stack_scorer")
            stack_logit = self.stack_scorer(top[:, :, 0, :]).squeeze(-1)
            stack_sign = self.candidate_is_block.to(scores.device) * 2.0 - 1.0
            scores = scores + stack_logit.unsqueeze(-1) * stack_sign.unsqueeze(0)
        if self.location_prior_weight != 0.0:
            if object_geometry is None:
                raise RuntimeError("location prior requires object_geometry")
            top_xy = object_geometry[:, self.block_slot_indices, :2]
            top_xy = top_xy.unsqueeze(2).expand(-1, -1, self.n_candidates, -1)
            prior_xy = self.candidate_prior_xy.to(
                device=scores.device,
                dtype=top_xy.dtype,
            ).unsqueeze(0)
            dist2 = (top_xy - prior_xy).square().sum(dim=-1)
            region_mask = (self.candidate_type_ids.to(scores.device) == 2).to(scores.dtype)
            prior_score = torch.exp(
                -dist2 / (2.0 * self.location_prior_sigma * self.location_prior_sigma)
            )
            scores = scores + self.location_prior_weight * prior_score * region_mask.unsqueeze(0)
        if self.patch_evidence_type != "none":
            if patch_tokens is None or object_queries is None:
                raise RuntimeError(
                    "patch evidence requires patch_tokens and object_queries"
                )
            scores = scores + self._patch_evidence_correction(
                patch_tokens,
                object_queries,
            )
        return scores / self.temperature
