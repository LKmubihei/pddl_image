"""PaQ Model: Predicate-as-Query for PDDL State Parsing.
========================================================

Architecture:
    Image -> DINOv2/DINOv3 (frozen) -> Patch Features
                                    |
                 PDDL Domain -> Predicate Query Encoder -> Schema Queries
                                    |
                              Object Slot Attention -> Typed Object Slots
                                    |
                        Slot Type Classifier -> Type Predictions (learned)
                                    |
                        Predicate Slot Attention -> Predicate Slots
                                    |
                        Type-Constrained Scoring Head -> Predicate State

Key contributions wired in this module:
  1. Domain-conditioned: takes PDDL domain info, not hardcoded labels
  2. Schema-conditioned queries: PDDL typed predicate signatures as queries
  3. Learned type classification: slots predict their own PDDL types
  4. Type-constrained binding: scoring respects PDDL type signatures
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional


class PaQModel(nn.Module):
    """Full Predicate-as-Query model with PDDL domain conditioning.

    Can be constructed either:
      1. From a DomainInfo object (recommended):
         PaQModel.from_domain_info(domain_info, ...)
      2. From explicit parameters (legacy):
         PaQModel(predicate_names=..., predicate_arities=..., ...)
    """

    def __init__(
        self,
        predicate_names: list[str],
        predicate_arities: dict[str, int],
        type_names: list[str],
        n_object_slots: int = 16,
        d_slot: int = 256,
        n_slot_iters: int = 3,
        use_real_encoder: bool = False,
        tau_unknown: float = 0.3,
        predicate_param_types: dict[str, list[str]] | None = None,
        predicate_schemas: list[dict] | None = None,
        predict_slot_types: bool = True,
        direct_object_tokens: bool = False,
        scoring_head_type: str = "film",
        **kwargs,
    ):
        super().__init__()
        self.predicate_names = predicate_names
        self.predicate_arities = predicate_arities
        self.type_names = type_names
        self.n_object_slots = n_object_slots
        self.d_slot = d_slot
        self.tau_unknown = tau_unknown
        self.predict_slot_types = predict_slot_types
        self.direct_object_tokens = direct_object_tokens

        # --- Component 1: Visual Encoder ---
        if use_real_encoder:
            from .visual_encoder import VisualEncoder, DINOv3VisualEncoder
            encoder = kwargs.get("visual_encoder")
            if encoder is not None:
                self.visual_encoder = encoder
            elif kwargs.get("use_dinov3", False):
                self.visual_encoder = DINOv3VisualEncoder(
                    model_name=kwargs.get("dinov3_model_name", "dinov3_vith16plus"),
                    d_out=d_slot,
                    source=kwargs.get("dinov3_source", "local"),
                    repo_dir=kwargs.get("dinov3_repo_dir"),
                    hf_model_id=kwargs.get("dinov3_hf_model_id"),
                    weights_path=kwargs.get("dinov3_weights_path"),
                )
            else:
                self.visual_encoder = VisualEncoder(d_out=d_slot)
        else:
            from .visual_encoder import MockVisualEncoder
            self.visual_encoder = MockVisualEncoder(d_out=d_slot, n_patches=256)

        # --- Component 2: Schema-Conditioned Predicate Query Encoder ---
        from .predicate_query_encoder import PredicateQueryEncoder
        if predicate_schemas is not None:
            self.query_encoder = PredicateQueryEncoder(
                predicate_schemas=predicate_schemas,
                type_names=type_names,
                d_out=d_slot,
            )
        else:
            # Fallback: construct minimal schemas from names + arities
            _ppt = predicate_param_types or {}
            minimal_schemas = [
                {
                    "name": name,
                    "arity": predicate_arities.get(name, 0),
                    "param_types": _ppt.get(name, []),
                    "action_roles": [],
                    "gloss": f"predicate {name} holds",
                }
                for name in predicate_names
            ]
            self.query_encoder = PredicateQueryEncoder(
                predicate_schemas=minimal_schemas,
                type_names=type_names,
                d_out=d_slot,
            )

        # --- Component 3: Dual Slot Attention with Type Classification ---
        from .slot_attention import DualSlotAttention
        self.slot_attention = DualSlotAttention(
            d_slot=d_slot,
            n_object_slots=n_object_slots,
            n_iter=n_slot_iters,
            n_types=len(type_names),
            predict_types=predict_slot_types,
        )

        # --- Component 4: Scoring Head ---
        from .scoring_head import PredicateScoringHead
        _ppt = predicate_param_types or {}
        predicate_defs = [
            {
                "name": name,
                "arity": predicate_arities[name],
                "param_types": _ppt.get(name, []),
            }
            for name in predicate_names
        ]
        self.scoring_head = PredicateScoringHead(
            d_slot=d_slot,
            predicate_defs=predicate_defs,
            type_names=type_names,
            tau_unknown=tau_unknown,
            scorer_type=scoring_head_type,
        )

        # --- Projection for visual features ---
        self.feat_proj = nn.Linear(d_slot, d_slot)
        with torch.no_grad():
            nn.init.eye_(self.feat_proj.weight)
            nn.init.zeros_(self.feat_proj.bias)
        self.object_slot_init = nn.Parameter(torch.zeros(n_object_slots, d_slot))

    @classmethod
    def from_domain_info(
        cls,
        domain_info,
        n_object_slots: int = 16,
        d_slot: int = 256,
        n_slot_iters: int = 3,
        use_real_encoder: bool = False,
        tau_unknown: float = 0.3,
        predict_slot_types: bool = True,
        **kwargs,
    ) -> "PaQModel":
        """Construct model from a DomainInfo object (recommended).

        Args:
            domain_info: DomainInfo from PDDLDomainCompiler.compile()
            n_object_slots: number of object-centric slots
            d_slot: slot dimension
            n_slot_iters: slot attention iterations
            use_real_encoder: use real DINOv2/v3 encoder
            tau_unknown: margin for unknown classification
            predict_slot_types: enable slot type classifier
            **kwargs: passed to __init__ (e.g., visual_encoder, dinov3 settings)
        """
        predicate_schemas = []
        for s in domain_info.predicate_schemas:
            predicate_schemas.append({
                "name": s.name,
                "arity": s.arity,
                "param_types": s.param_types,
                "action_roles": s.action_roles,
                "gloss": s.gloss,
            })

        return cls(
            predicate_names=domain_info.predicate_names,
            predicate_arities=domain_info.predicate_arities,
            type_names=domain_info.types,
            n_object_slots=n_object_slots,
            d_slot=d_slot,
            n_slot_iters=n_slot_iters,
            use_real_encoder=use_real_encoder,
            tau_unknown=tau_unknown,
            predicate_param_types=domain_info.predicate_param_types,
            predicate_schemas=predicate_schemas,
            predict_slot_types=predict_slot_types,
            **kwargs,
        )

    def forward(
        self,
        images: torch.Tensor,
        object_type_ids: Optional[torch.Tensor] = None,
        slot_init: Optional[torch.Tensor] = None,
        use_soft_types: bool = False,
    ) -> dict:
        """
        Args:
            images: (B, 3, H, W) or (B, N_patches, D) if using mock encoder
            object_type_ids: (B, N_obj) int tensor of type indices per slot.
                             If provided, used for scoring (oracle types).
            slot_init: optional (B, n_obj_slots, D) for slot initialization
                       (e.g., color-grounded init).
            use_soft_types: if True and no oracle types, use soft type probs
                            for scoring (output dim may differ). Default False
                            uses hard argmax predicted types for stable dims.

        Returns:
            dict with keys:
                'canonical_scores': (B, N_canonical) logits
                'object_slots': (B, N_obj, D)
                'predicate_slots': (B, N_pred, D)
                'predicate_queries': (B, N_pred, D)
                'type_logits': (B, N_obj, N_types) if predict_slot_types
                'type_probs': (B, N_obj, N_types) if predict_slot_types
                'predicted_type_ids': (B, N_obj) if predict_slot_types
        """
        B = images.shape[0]
        device = images.device

        # 1. Visual features
        if images.dim() == 4:
            visual_feats = self.visual_encoder(images)
        else:
            visual_feats = images
        visual_feats = self.feat_proj(visual_feats)
        assert visual_feats.dim() == 3 and visual_feats.shape[0] == B, \
            f"visual_feats shape error: expected (B={B}, N, D), got {visual_feats.shape}"

        # 2. Schema-conditioned predicate queries
        pred_queries = self.query_encoder.forward_batch(B).to(device)
        assert pred_queries.dim() == 3 and pred_queries.shape[0] == B, \
            f"pred_queries shape error: expected (B={B}, N_pred, D), got {pred_queries.shape}"

        # 3. Dual slot attention with type classification
        if slot_init is None:
            slot_init = self.object_slot_init.unsqueeze(0).expand(B, -1, -1)
        if self.direct_object_tokens:
            if visual_feats.shape[1] != self.n_object_slots:
                raise ValueError(
                    "direct_object_tokens=True requires one feature token per "
                    f"object slot: got {visual_feats.shape[1]} tokens for "
                    f"{self.n_object_slots} slots"
                )
            obj_slots = visual_feats + slot_init
            pred_slots = self.slot_attention.pred_sa(obj_slots, pred_queries)
            slot_out = {
                "object_slots": obj_slots,
                "predicate_slots": pred_slots,
                "obj_masks": torch.eye(
                    self.n_object_slots, device=device, dtype=visual_feats.dtype,
                ).unsqueeze(0).expand(B, -1, -1),
            }
            if self.predict_slot_types:
                type_logits = self.slot_attention.type_classifier(obj_slots)
                slot_out["type_logits"] = type_logits
                slot_out["type_probs"] = torch.softmax(type_logits, dim=-1)
                slot_out["predicted_type_ids"] = type_logits.argmax(dim=-1)
        else:
            slot_out = self.slot_attention(
                visual_feats, pred_queries, slot_init=slot_init
            )
        obj_slots = slot_out["object_slots"]
        pred_slots = slot_out["predicate_slots"]
        assert obj_slots.shape[:2] == (B, self.n_object_slots), \
            f"obj_slots shape error: expected (B={B}, N_obj={self.n_object_slots}, D), got {obj_slots.shape}"
        assert pred_slots.shape[0] == B, \
            f"pred_slots shape error: expected (B={B}, ...), got {pred_slots.shape}"

        # 4. Get type assignments for scoring
        # During training: use ground-truth type_ids for stable canonical dims
        # During inference: use predicted type_ids from the classifier
        if self.predict_slot_types:
            type_probs = slot_out["type_probs"]
            predicted_ids = slot_out["predicted_type_ids"]
            if object_type_ids is not None:
                # Training: use oracle types for stable canonical dims
                canonical_scores = self.scoring_head(
                    pred_slots, obj_slots,
                    object_type_ids=object_type_ids,
                )
            elif use_soft_types:
                # Soft scoring via type_probs (output dim may differ)
                canonical_scores = self.scoring_head(
                    pred_slots, obj_slots,
                    type_probs=type_probs,
                )
            else:
                # Inference default: use hard predicted types (stable dims)
                canonical_scores = self.scoring_head(
                    pred_slots, obj_slots,
                    object_type_ids=predicted_ids,
                )
        else:
            type_probs = None
            predicted_ids = None
            if object_type_ids is None:
                n_types = len(self.type_names)
                object_type_ids = torch.zeros(
                    B, self.n_object_slots, dtype=torch.long, device=device
                )
                for i in range(self.n_object_slots):
                    object_type_ids[:, i] = i % n_types
            canonical_scores = self.scoring_head(
                pred_slots, obj_slots, object_type_ids=object_type_ids
            )

        result = {
            "canonical_scores": canonical_scores,
            "object_slots": obj_slots,
            "predicate_slots": pred_slots,
            "predicate_queries": pred_queries,
        }
        if self.predict_slot_types:
            result["type_logits"] = slot_out["type_logits"]
            result["type_probs"] = type_probs
            result["predicted_type_ids"] = predicted_ids

        return result

    def compute_type_loss(
        self,
        object_type_ids: torch.Tensor,
        forward_output: dict | None = None,
        images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute slot type classification loss.

        Args:
            object_type_ids: (B, N_obj) or (N_obj,) ground-truth type indices.
            forward_output: output from forward() if already computed.
            images: images to run forward() if forward_output not provided.

        Returns:
            scalar cross-entropy loss.
        """
        if forward_output is None:
            if images is None:
                raise ValueError("Need either forward_output or images")
            forward_output = self.forward(images)

        type_logits = forward_output["type_logits"]  # (B, N_obj, N_types)
        B, N_obj, N_types = type_logits.shape

        if object_type_ids.dim() == 1:
            object_type_ids = object_type_ids.unsqueeze(0).expand(B, -1)

        return nn.functional.cross_entropy(
            type_logits.reshape(B * N_obj, N_types),
            object_type_ids.reshape(B * N_obj),
        )

    def predict_state(
        self,
        images: torch.Tensor,
        object_type_ids: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Predict discrete predicate state (binary).

        Returns:
            (B, N_canonical) long tensor in {0, 1}.
        """
        out = self.forward(images, object_type_ids)
        probs = torch.sigmoid(out["canonical_scores"])
        return (probs >= threshold).long()

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
