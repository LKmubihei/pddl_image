"""Predicate Query Encoder: Schema-Conditioned PDDL Typed Predicate Queries.
===========================================================================

Encodes PDDL typed predicate signatures into dense query embeddings.

Each predicate query is conditioned on its **full PDDL schema**:
  - predicate name (char-level encoding)
  - arity (learned positional embedding)
  - argument types (type name embeddings)
  - action roles (precondition / effect embedding)
  - natural language gloss (char-level encoding)

This replaces the previous approach of learned ID embeddings or simple
description-only encoding, enabling the claim:

    "We treat PDDL typed predicate signatures as queries, rather than
     fixed label IDs."
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Action role vocabulary
# ======================================================================
ACTION_ROLES = ["precondition", "effect", "state_constraint", "static"]
ROLE_TO_IDX = {r: i for i, r in enumerate(ACTION_ROLES)}


class PredicateQueryEncoder(nn.Module):
    """Encode PDDL predicate schemas into query embeddings.

    Each predicate query is conditioned on:
      1. predicate name → char-LSTM encoding
      2. arity → learned embedding
      3. argument types → per-type embeddings, max-pooled
      4. action roles → role embedding, mean-pooled
      5. gloss (natural language) → char-LSTM encoding

    All components are projected to d_out and summed with learned gates.
    """

    def __init__(
        self,
        predicate_schemas: list[dict] | None = None,
        predicate_names: list[str] | None = None,
        type_names: list[str] | None = None,
        d_out: int = 256,
        max_name_len: int = 32,
        max_gloss_len: int = 64,
        max_arity: int = 4,
    ):
        """
        Args:
            predicate_schemas: list of dicts from DomainInfo.predicate_schemas,
                each with keys: name, arity, param_types, action_roles, gloss.
                If None, falls back to predicate_names + predicate_param_types.
            predicate_names: fallback list of predicate names (legacy API).
            type_names: list of type names for type embedding.
            d_out: output dimension.
            max_name_len: max characters for predicate name encoding.
            max_gloss_len: max characters for gloss encoding.
            max_arity: maximum predicate arity supported.
        """
        super().__init__()
        self.d_out = d_out
        self.max_name_len = max_name_len
        self.max_gloss_len = max_gloss_len
        self.max_arity = max_arity

        # Normalize schema input
        if predicate_schemas is not None:
            self._schemas = predicate_schemas
        elif predicate_names is not None:
            # Legacy fallback: create minimal schemas from names
            self._schemas = [
                {"name": n, "arity": 0, "param_types": [],
                 "action_roles": [], "gloss": f"predicate {n} holds"}
                for n in predicate_names
            ]
        else:
            self._schemas = []

        self.n_predicates = len(self._schemas)
        self.type_names = type_names or []
        self.type_to_idx = {t: i for i, t in enumerate(self.type_names)}

        # --- Component 1: Predicate name encoder (char-level) ---
        self.char_embed = nn.Embedding(128, 64)
        self.name_lstm = nn.LSTM(64, 128, batch_first=True, num_layers=1)
        self.name_proj = nn.Sequential(nn.Linear(128, d_out), nn.GELU())

        # --- Component 2: Arity embedding ---
        self.arity_embed = nn.Embedding(max_arity + 1, d_out)

        # --- Component 3: Type embedding ---
        n_types = max(len(self.type_names), 8)
        self.type_embed = nn.Embedding(n_types, d_out)

        # --- Component 4: Action role embedding ---
        self.role_embed = nn.Embedding(len(ACTION_ROLES), d_out)

        # --- Component 5: Gloss encoder (char-level) ---
        self.gloss_lstm = nn.LSTM(64, 128, batch_first=True, num_layers=1)
        self.gloss_proj = nn.Sequential(nn.Linear(128, d_out), nn.GELU())

        # --- Fusion: gated sum of all components ---
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_out * 2, d_out),
            nn.Sigmoid(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_out, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out),
        )

        # --- Pre-compute and cache all encodings as buffers ---
        self._precompute_buffers()

    def _precompute_buffers(self):
        """Pre-compute all static tensor representations and register as buffers."""
        # Name char indices
        name_matrix = torch.zeros(self.n_predicates, self.max_name_len, dtype=torch.long)
        for i, schema in enumerate(self._schemas):
            chars = [min(ord(c), 127) for c in schema["name"][:self.max_name_len]]
            name_matrix[i, :len(chars)] = torch.tensor(chars)
        self.register_buffer("name_matrix", name_matrix)

        # Arity indices
        arities = torch.zeros(self.n_predicates, dtype=torch.long)
        for i, schema in enumerate(self._schemas):
            arities[i] = min(schema["arity"], self.max_arity)
        self.register_buffer("arity_ids", arities)

        # Type indices per argument position: (N_pred, max_arity)
        type_matrix = torch.zeros(self.n_predicates, self.max_arity, dtype=torch.long)
        for i, schema in enumerate(self._schemas):
            for j, tname in enumerate(schema.get("param_types", [])[:self.max_arity]):
                type_matrix[i, j] = self.type_to_idx.get(tname, 0)
        self.register_buffer("type_matrix", type_matrix)

        # Argument mask (which positions have valid types)
        arg_mask = torch.zeros(self.n_predicates, self.max_arity, dtype=torch.float)
        for i, schema in enumerate(self._schemas):
            for j in range(len(schema.get("param_types", []))):
                arg_mask[i, j] = 1.0
        self.register_buffer("arg_mask", arg_mask)

        # Action role indices per predicate: (N_pred, max_roles)
        max_roles = 4
        role_matrix = torch.zeros(self.n_predicates, max_roles, dtype=torch.long)
        role_mask = torch.zeros(self.n_predicates, max_roles, dtype=torch.float)
        for i, schema in enumerate(self._schemas):
            roles = schema.get("action_roles", [])
            for j, r in enumerate(roles[:max_roles]):
                role_matrix[i, j] = ROLE_TO_IDX.get(r, 0)
                role_mask[i, j] = 1.0
        self.register_buffer("role_matrix", role_matrix)
        self.register_buffer("role_mask", role_mask)

        # Gloss char indices
        gloss_matrix = torch.zeros(self.n_predicates, self.max_gloss_len, dtype=torch.long)
        for i, schema in enumerate(self._schemas):
            gloss = schema.get("gloss", schema["name"])
            chars = [min(ord(c), 127) for c in gloss[:self.max_gloss_len]]
            gloss_matrix[i, :len(chars)] = torch.tensor(chars)
        self.register_buffer("gloss_matrix", gloss_matrix)

    def forward(self) -> torch.Tensor:
        """Encode all predicate schemas into query embeddings.

        Returns:
            queries: (N_predicates, d_out)
        """
        N = self.n_predicates
        if N == 0:
            return torch.zeros(0, self.d_out, device=self.char_embed.weight.device)

        # 1. Name encoding: char-LSTM
        name_chars = self.char_embed(self.name_matrix)  # (N, max_name, 64)
        _, (h_n, _) = self.name_lstm(name_chars)  # (1, N, 128)
        name_feat = self.name_proj(h_n.squeeze(0))  # (N, d_out)

        # 2. Arity embedding
        arity_feat = self.arity_embed(self.arity_ids)  # (N, d_out)

        # 3. Type embedding: embed each arg position, mask-pool
        type_embs = self.type_embed(self.type_matrix)  # (N, max_arity, d_out)
        mask_exp = self.arg_mask.unsqueeze(-1)  # (N, max_arity, 1)
        type_feat = (type_embs * mask_exp).sum(dim=1)  # (N, d_out)
        # Normalize by number of valid args (avoid div by 0)
        n_args = self.arg_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
        type_feat = type_feat / n_args

        # 4. Action role embedding: embed roles, mask-pool
        role_embs = self.role_embed(self.role_matrix)  # (N, max_roles, d_out)
        role_mask_exp = self.role_mask.unsqueeze(-1)  # (N, max_roles, 1)
        role_feat = (role_embs * role_mask_exp).sum(dim=1)  # (N, d_out)
        n_roles = self.role_mask.sum(dim=1, keepdim=True).clamp(min=1)
        role_feat = role_feat / n_roles

        # 5. Gloss encoding: char-LSTM
        gloss_chars = self.char_embed(self.gloss_matrix)  # (N, max_gloss, 64)
        _, (h_g, _) = self.gloss_lstm(gloss_chars)  # (1, N, 128)
        gloss_feat = self.gloss_proj(h_g.squeeze(0))  # (N, d_out)

        # 6. Fusion: gated combination of structured features and text features
        structured = name_feat + arity_feat + type_feat + role_feat  # (N, d_out)
        textual = gloss_feat  # (N, d_out)

        gate = self.fusion_gate(torch.cat([structured, textual], dim=-1))  # (N, d_out)
        fused = gate * structured + (1 - gate) * textual
        queries = self.fusion_proj(fused)  # (N, d_out)

        return queries

    def forward_batch(self, B: int) -> torch.Tensor:
        """Return queries expanded for batch.

        Returns:
            queries: (B, N_predicates, d_out)
        """
        queries = self.forward()  # (N, d_out)
        return queries.unsqueeze(0).expand(B, -1, -1)

    @classmethod
    def from_domain_info(cls, domain_info, d_out: int = 256) -> "PredicateQueryEncoder":
        """Create encoder directly from a DomainInfo object.

        Args:
            domain_info: DomainInfo from PDDLDomainCompiler.
            d_out: output dimension.

        Returns:
            PredicateQueryEncoder configured with domain's predicate schemas.
        """
        schemas = []
        for s in domain_info.predicate_schemas:
            schemas.append({
                "name": s.name,
                "arity": s.arity,
                "param_types": s.param_types,
                "action_roles": s.action_roles,
                "gloss": s.gloss,
            })
        return cls(
            predicate_schemas=schemas,
            type_names=domain_info.types,
            d_out=d_out,
        )
