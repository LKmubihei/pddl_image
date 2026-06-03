"""Dual-Level Slot Attention for PaQ with Slot Type Classification.
===================================================================

Components:
  - ObjectSlotAttention: discovers object-centric slots from visual features
  - SlotTypeClassifier: predicts per-slot type distributions (block, column, etc.)
  - PredicateSlotAttention: query-conditioned attention over typed object slots
  - DualSlotAttention: combines all components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectSlotAttention(nn.Module):
    """Object-centric slot attention with learnable Gaussian initialization."""

    def __init__(self, d_slot=256, n_slots=16, n_iter=3):
        super().__init__()
        self.d_slot = d_slot
        self.n_slots = n_slots
        self.n_iter = n_iter
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, d_slot) * 0.02)
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, n_slots, d_slot))
        self.to_q = nn.Linear(d_slot, d_slot)
        self.to_k = nn.Linear(d_slot, d_slot)
        self.to_v = nn.Linear(d_slot, d_slot)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_input = nn.LayerNorm(d_slot)
        self.mlp = nn.Sequential(
            nn.Linear(d_slot, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, d_slot),
        )
        self.norm_mlp = nn.LayerNorm(d_slot)

    def forward(self, features, slot_init=None):
        """
        Args:
            features: (B, N, D) visual features
            slot_init: optional (B, n_slots, D) initial slot values

        Returns:
            slots: (B, n_slots, D)
            attn: (B, n_slots, N) attention masks
        """
        B = features.shape[0]
        device = features.device

        if slot_init is not None:
            slots = slot_init.clone()
        else:
            mu = self.slots_mu.expand(B, -1, -1)
            sigma = self.slots_log_sigma.exp().expand(B, -1, -1)
            slots = mu + sigma * torch.randn_like(mu)

        inputs = self.norm_input(features)
        k = self.to_k(inputs)
        v = self.to_v(inputs)

        for _ in range(self.n_iter):
            q = self.to_q(self.norm_slots(slots))
            attn = torch.bmm(q, k.transpose(1, 2)) / (self.d_slot ** 0.5)
            attn = F.softmax(attn, dim=1)
            # Normalize attention (competition among slots)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
            updates = torch.bmm(attn_norm, v)
            slots = self.gru(
                updates.reshape(-1, self.d_slot),
                slots.reshape(-1, self.d_slot)
            ).reshape(B, self.n_slots, self.d_slot)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn


class SlotTypeClassifier(nn.Module):
    """Predict per-slot type distributions from learned slot features.

    Each object slot outputs a distribution over PDDL types:
        slot_i -> P(type=block), P(type=column), P(type=screw), ...

    This enables the model to discover object types from visual features
    rather than relying on hardcoded type assignments.

    Can be supervised with cross-entropy loss when type labels are available,
    or used as a soft constraint via KL divergence.
    """

    def __init__(self, d_slot: int = 256, n_types: int = 2):
        super().__init__()
        self.d_slot = d_slot
        self.n_types = n_types
        self.classifier = nn.Sequential(
            nn.Linear(d_slot, d_slot // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_slot // 2, n_types),
        )

    def forward(self, object_slots: torch.Tensor) -> torch.Tensor:
        """
        Args:
            object_slots: (B, N_obj, D) slot features

        Returns:
            type_logits: (B, N_obj, N_types) unnormalized log probabilities
        """
        return self.classifier(object_slots)

    def predict_types(self, object_slots: torch.Tensor) -> torch.Tensor:
        """Return hard type predictions.

        Returns:
            type_ids: (B, N_obj) long tensor of predicted type indices
        """
        logits = self.forward(object_slots)  # (B, N_obj, N_types)
        return logits.argmax(dim=-1)  # (B, N_obj)

    def type_probs(self, object_slots: torch.Tensor) -> torch.Tensor:
        """Return soft type probabilities.

        Returns:
            type_probs: (B, N_obj, N_types) probabilities
        """
        logits = self.forward(object_slots)
        return F.softmax(logits, dim=-1)

    def compute_loss(
        self,
        object_slots: torch.Tensor,
        target_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-entropy loss for slot type classification.

        Args:
            object_slots: (B, N_obj, D)
            target_type_ids: (B, N_obj) or (N_obj,) ground-truth type indices

        Returns:
            scalar loss
        """
        logits = self.forward(object_slots)  # (B, N_obj, N_types)
        # Reshape for cross_entropy: (B*N_obj, N_types) vs (B*N_obj,)
        B, N_obj, N_types = logits.shape
        if target_type_ids.dim() == 1:
            # Expand (N_obj,) -> (B, N_obj)
            target_type_ids = target_type_ids.unsqueeze(0).expand(B, -1)
        return F.cross_entropy(
            logits.reshape(B * N_obj, N_types),
            target_type_ids.reshape(B * N_obj),
        )


class PredicateSlotAttention(nn.Module):
    """Query-conditioned slot attention over object slots."""

    def __init__(self, d_slot=256, n_iter=3):
        super().__init__()
        self.n_iter = n_iter
        self.to_q = nn.Linear(d_slot, d_slot)
        self.to_k = nn.Linear(d_slot, d_slot)
        self.to_v = nn.Linear(d_slot, d_slot)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.norm_slots = nn.LayerNorm(d_slot)
        self.norm_input = nn.LayerNorm(d_slot)

    def forward(self, object_slots, predicate_queries):
        """
        Args:
            object_slots: (B, N_obj, D)
            predicate_queries: (B, N_pred, D) query embeddings from schema encoder

        Returns:
            pred_slots: (B, N_pred, D) refined predicate slots
        """
        slots = predicate_queries
        inputs = self.norm_input(object_slots)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        for _ in range(self.n_iter):
            q = self.to_q(self.norm_slots(slots))
            attn = torch.bmm(q, k.transpose(1, 2)) / (inputs.shape[-1] ** 0.5)
            attn = F.softmax(attn, dim=-1)
            updates = torch.bmm(attn, v)
            slots = self.gru(
                updates.reshape(-1, inputs.shape[-1]),
                slots.reshape(-1, inputs.shape[-1])
            ).reshape(slots.shape)
        return slots


class ObjectQueryExtractor(nn.Module):
    """Extract known PDDL objects/locations as query-conditioned visual slots."""

    def __init__(
        self,
        d_slot: int = 256,
        n_iter: int = 2,
        n_relation_layers: int = 0,
        n_heads: int = 4,
        local_refine: bool = False,
        local_top_k: int = 4,
        local_radius: int = 2,
        pooling_mode: str = "iterative",
    ):
        super().__init__()
        if pooling_mode not in {"iterative", "heatmap"}:
            raise ValueError(f"Unknown object query pooling mode: {pooling_mode}")
        if n_relation_layers < 0:
            raise ValueError("n_relation_layers must be non-negative")
        if local_top_k <= 0:
            raise ValueError("local_top_k must be positive")
        if local_radius < 0:
            raise ValueError("local_radius must be non-negative")
        if d_slot % n_heads != 0:
            n_heads = 1
        self.d_slot = d_slot
        self.n_iter = n_iter
        self.local_refine = local_refine
        self.local_top_k = int(local_top_k)
        self.local_radius = int(local_radius)
        self.pooling_mode = pooling_mode
        self.query_norm = nn.LayerNorm(d_slot)
        self.input_norm = nn.LayerNorm(d_slot)
        self.to_q = nn.Linear(d_slot, d_slot)
        self.to_k = nn.Linear(d_slot, d_slot)
        self.to_v = nn.Linear(d_slot, d_slot)
        self.gru = nn.GRUCell(d_slot, d_slot)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(d_slot),
            nn.Linear(d_slot, d_slot * 2),
            nn.GELU(),
            nn.Linear(d_slot * 2, d_slot),
        )
        self.query_scale = nn.Parameter(torch.ones(1, 1, d_slot))
        self.relation_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_slot,
                nhead=n_heads,
                dim_feedforward=d_slot * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_relation_layers)
        ])

    def forward(
        self,
        visual_features: torch.Tensor,
        object_queries: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if object_queries is None:
            raise ValueError("ObjectQueryExtractor requires object_queries")
        slots = self.query_norm(object_queries) * self.query_scale
        inputs = self.input_norm(visual_features)
        k = self.to_k(inputs)
        v = self.to_v(inputs)
        attn = None

        if self.pooling_mode == "heatmap":
            q = self.to_q(self.query_norm(slots))
            logits = torch.bmm(q, k.transpose(1, 2)) / (self.d_slot ** 0.5)
            attn = F.softmax(logits, dim=-1)
            slots = torch.bmm(attn, v)
            slots = slots + self.update_mlp(slots)
            for layer in self.relation_layers:
                slots = layer(slots)
            return slots, attn

        for _ in range(self.n_iter):
            q = self.to_q(self.query_norm(slots))
            logits = torch.bmm(q, k.transpose(1, 2)) / (self.d_slot ** 0.5)
            attn = F.softmax(logits, dim=-1)
            updates = torch.bmm(attn, v)
            slots = self.gru(
                updates.reshape(-1, self.d_slot),
                slots.reshape(-1, self.d_slot),
            ).reshape_as(slots)
            slots = slots + self.update_mlp(slots)

        if self.local_refine:
            n_tokens = visual_features.shape[1]
            side = int(n_tokens ** 0.5)
            if side * side == n_tokens:
                k_top = min(self.local_top_k, n_tokens)
                ref = torch.topk(attn, k=k_top, dim=-1).indices
                ref_y = ref // side
                ref_x = ref % side
                local_mask = torch.zeros_like(attn, dtype=torch.bool)
                for dy in range(-self.local_radius, self.local_radius + 1):
                    for dx in range(-self.local_radius, self.local_radius + 1):
                        yy = (ref_y + dy).clamp(0, side - 1)
                        xx = (ref_x + dx).clamp(0, side - 1)
                        local_idx = yy * side + xx
                        local_mask.scatter_(2, local_idx, True)
                q = self.to_q(self.query_norm(slots))
                logits = torch.bmm(q, k.transpose(1, 2)) / (self.d_slot ** 0.5)
                logits = logits.masked_fill(~local_mask, -1e4)
                attn = F.softmax(logits, dim=-1)
                updates = torch.bmm(attn, v)
                slots = self.gru(
                    updates.reshape(-1, self.d_slot),
                    slots.reshape(-1, self.d_slot),
                ).reshape_as(slots)
                slots = slots + self.update_mlp(slots)

        for layer in self.relation_layers:
            slots = layer(slots)

        assert attn is not None
        return slots, attn


class DualSlotAttention(nn.Module):
    """Combined object + predicate slot attention with type classification.

    Pipeline:
        visual features -> Object Slot Attention -> object slots
        object slots -> Slot Type Classifier -> type predictions
        object slots + predicate queries -> Predicate Slot Attention -> predicate slots
    """

    def __init__(
        self,
        d_slot: int = 256,
        n_object_slots: int = 16,
        n_iter: int = 3,
        n_types: int = 2,
        predict_types: bool = True,
    ):
        """
        Args:
            d_slot: slot dimension.
            n_object_slots: number of object slots.
            n_iter: slot attention iterations.
            n_types: number of PDDL types for type classification.
            predict_types: whether to include the type classifier.
        """
        super().__init__()
        self.obj_sa = ObjectSlotAttention(d_slot, n_object_slots, n_iter)
        self.pred_sa = PredicateSlotAttention(d_slot, n_iter)
        self.predict_types = predict_types

        if predict_types:
            self.type_classifier = SlotTypeClassifier(d_slot, n_types)

    def forward(
        self,
        visual_features: torch.Tensor,
        predicate_queries: torch.Tensor,
        slot_init: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            visual_features: (B, N_patches, D)
            predicate_queries: (B, N_pred, D)
            slot_init: optional (B, n_obj_slots, D) slot initialization

        Returns:
            dict with:
                'object_slots': (B, N_obj, D)
                'predicate_slots': (B, N_pred, D)
                'obj_masks': (B, N_obj, N_patches) attention masks
                'type_logits': (B, N_obj, N_types) if predict_types
                'type_probs': (B, N_obj, N_types) if predict_types
                'predicted_type_ids': (B, N_obj) if predict_types
        """
        # Object slot attention
        obj_slots, obj_masks = self.obj_sa(visual_features, slot_init=slot_init)

        # Predicate slot attention
        pred_slots = self.pred_sa(obj_slots, predicate_queries)

        result = {
            "object_slots": obj_slots,
            "predicate_slots": pred_slots,
            "obj_masks": obj_masks,
        }

        # Type classification (single forward pass, cache results)
        if self.predict_types:
            type_logits = self.type_classifier(obj_slots)        # (B, N_obj, N_types)
            type_probs = torch.softmax(type_logits, dim=-1)      # (B, N_obj, N_types)
            predicted_type_ids = type_probs.argmax(dim=-1)        # (B, N_obj)
            result["type_logits"] = type_logits
            result["type_probs"] = type_probs
            result["predicted_type_ids"] = predicted_type_ids

        return result
