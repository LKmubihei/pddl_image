"""AE-PaQ Loss Functions — Three-Constraint Framework
=====================================================

The grounding function G: visual state -> symbolic state is trained under
three structural constraints:

  Constraint 1 — Pointwise Correctness:
      G(I_t) ≈ S_t  (standard BCE on true fact sets)

  Constraint 2 — Action Equivariance:
      G(I_{t+1}) ≈ Γ_a(G(I_t))
      The visual grounding must commute with the symbolic transition.
      After a real-world action, the grounded state must match what the
      PDDL transition function Γ_a predicts.

  Constraint 3 — Counterfactual Discriminability:
      Score(G(I_t), a_true, G(I_{t+1}))  >  Score(G(I_t), a_false, G(I_{t+1}))
      The grounded states must carry enough action-semantics to distinguish
      the true action from wrong ones. This prevents the model from learning
      pixel-diff shortcuts.

This framing positions action transitions not as "extra supervision" but as
a definition of what makes a grounding *planning-grade*: the grounded states
must serve as valid operands for the PDDL action model.

Component losses (used internally by the three constraints):
  - PredicateStateLoss (L_seed): BCE for ground predicate scoring
  - PredicateContrastiveLoss: InfoNCE slot-query alignment
  - PreconditionConsistencyLoss (L_pre): preconditions True before action
  - EffectConsistencyLoss (L_eff): add/del effects after action
  - FrameConsistencyLoss (L_frame): frame axioms
  - ActionEquivarianceLoss: unified Γ_a consistency
  - CounterfactualActionLoss: energy-based ranking loss

Legacy losses kept for backward compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Constraint 1: Pointwise Correctness
# ======================================================================

class PredicateStateLoss(nn.Module):
    """Binary cross-entropy for ground predicate scoring.

    This is the standard supervised loss: for each image, push predicted
    scores toward the ground-truth true fact set.

    Supports ignoring unknown labels (label == -1) by masking them out.
    """

    def __init__(self, reduction: str = "mean", pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.reduction = reduction
        if pos_weight is None:
            self.register_buffer("pos_weight", None)
        else:
            self.register_buffer("pos_weight", pos_weight.float())

    def forward(
        self,
        predicted_scores: torch.Tensor,
        ground_truth_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            predicted_scores: (B, N_canonical) raw logits.
            ground_truth_labels: (B, N_canonical) float labels.
                1 = True, 0 = False, -1 = unknown (ignored).

        Returns:
            scalar loss.
        """
        valid_mask = (ground_truth_labels != -1).float()
        labels_clamped = ground_truth_labels.clamp(min=0.0, max=1.0)
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=predicted_scores.device, dtype=predicted_scores.dtype)
        bce_per_elem = F.binary_cross_entropy_with_logits(
            predicted_scores, labels_clamped, reduction="none", pos_weight=pos_weight
        )
        masked_bce = bce_per_elem * valid_mask

        if self.reduction == "mean":
            num_valid = valid_mask.sum()
            if num_valid > 0:
                return masked_bce.sum() / num_valid
            return torch.tensor(0.0, device=predicted_scores.device)
        elif self.reduction == "sum":
            return masked_bce.sum()
        else:
            return masked_bce


class PredicateContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss between predicate slots and queries."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        predicate_slots: torch.Tensor,
        predicate_query_embs: torch.Tensor,
    ) -> torch.Tensor:
        B, N_pred, D = predicate_slots.shape
        slots = F.normalize(predicate_slots, dim=-1)
        queries = F.normalize(predicate_query_embs, dim=-1)

        sim_s2q = torch.bmm(slots, queries.transpose(1, 2)) / self.temperature
        labels = torch.arange(N_pred, device=slots.device).unsqueeze(0).expand(B, -1)
        loss_s2q = F.cross_entropy(
            sim_s2q.reshape(B * N_pred, N_pred),
            labels.reshape(B * N_pred),
        )

        sim_q2s = torch.bmm(queries, slots.transpose(1, 2)) / self.temperature
        loss_q2s = F.cross_entropy(
            sim_q2s.reshape(B * N_pred, N_pred),
            labels.reshape(B * N_pred),
        )

        return (loss_s2q + loss_q2s) / 2.0


# ======================================================================
# Constraint 2: Action Equivariance
# ======================================================================

class PreconditionConsistencyLoss(nn.Module):
    """Precondition atoms should be True before action.

    For transition (I_t, a_t, I_{t+1}):
        f in Pre(a_t) => p(f | I_t) should be high.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        state_t_scores: torch.Tensor,
        precondition_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_pre = precondition_mask.sum()
        if num_pre > 0:
            bce = F.binary_cross_entropy_with_logits(
                state_t_scores, torch.ones_like(state_t_scores), reduction="none"
            )
            loss = (bce * precondition_mask).sum() / num_pre
        else:
            loss = torch.tensor(0.0, device=state_t_scores.device)

        if self.reduction == "sum":
            loss = loss * precondition_mask.sum().clamp(min=1)
        return loss


class EffectConsistencyLoss(nn.Module):
    """Add/del effect atoms after action.

    For transition (I_t, a_t, I_{t+1}):
        f in Add(a_t) => p(f | I_{t+1}) should be high
        f in Del(a_t) => p(f | I_{t+1}) should be low
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        state_t1_scores: torch.Tensor,
        add_mask: torch.Tensor,
        del_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_add = add_mask.sum()
        num_del = del_mask.sum()
        total = num_add + num_del

        if total > 0:
            if num_add > 0:
                bce_add = F.binary_cross_entropy_with_logits(
                    state_t1_scores, torch.ones_like(state_t1_scores), reduction="none"
                )
                L_add = (bce_add * add_mask).sum() / num_add
            else:
                L_add = torch.tensor(0.0, device=state_t1_scores.device)

            if num_del > 0:
                bce_del = F.binary_cross_entropy_with_logits(
                    state_t1_scores, torch.zeros_like(state_t1_scores), reduction="none"
                )
                L_del = (bce_del * del_mask).sum() / num_del
            else:
                L_del = torch.tensor(0.0, device=state_t1_scores.device)

            loss = (L_add * num_add + L_del * num_del) / total
        else:
            loss = torch.tensor(0.0, device=state_t1_scores.device)

        if self.reduction == "sum":
            loss = loss * total.clamp(min=1)
        return loss


class FrameConsistencyLoss(nn.Module):
    """Unaffected predicates stay unchanged.

    For transition (I_t, a_t, I_{t+1}):
        f in Frame(a_t) => p(f | I_t) ~ p(f | I_{t+1})
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        state_t_probs: torch.Tensor,
        state_t1_probs: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_frame = frame_mask.sum()
        if num_frame > 0:
            diff = (state_t1_probs - state_t_probs).abs() * frame_mask
            loss = diff.sum() / num_frame
        else:
            loss = torch.tensor(0.0, device=state_t_probs.device)

        if self.reduction == "sum":
            loss = loss * num_frame.clamp(min=1)
        return loss


class ActionEquivarianceLoss(nn.Module):
    """Unified action-equivariance constraint.

    Implements:  G(I_{t+1}) ≈ Γ_a(G(I_t))

    Where Γ_a is the PDDL symbolic transition:
        Γ_a(s)[f] = 1         if f in Add(a)
        Γ_a(s)[f] = 0         if f in Del(a)
        Γ_a(s)[f] = s[f]      if f in Frame(a)

    This loss decomposes into three components:
        L_pre:   precondition consistency (Γ_a applicable => pre satisfied)
        L_eff:   effect consistency (post-state matches Γ_a's add/del)
        L_frame: frame axioms (unchanged predicates preserved)

    These are NOT independent loss terms. They jointly enforce that the
    predicted states form a valid PDDL transition. This is what distinguishes
    action-equivariant grounding from static classification.

    Args:
        w_pre: weight for precondition consistency
        w_eff: weight for effect consistency
        w_frame: weight for frame axioms
    """

    def __init__(self, w_pre: float = 0.5, w_eff: float = 1.0, w_frame: float = 0.5):
        super().__init__()
        self.w_pre = w_pre
        self.w_eff = w_eff
        self.w_frame = w_frame
        self._pre_loss = PreconditionConsistencyLoss()
        self._eff_loss = EffectConsistencyLoss()
        self._frame_loss = FrameConsistencyLoss()

    def forward(
        self,
        state_t_scores: torch.Tensor,
        state_t1_scores: torch.Tensor,
        pre_mask: torch.Tensor,
        add_mask: torch.Tensor,
        del_mask: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            state_t_scores:  (B, N_canonical) logits from G(I_t)
            state_t1_scores: (B, N_canonical) logits from G(I_{t+1})
            pre_mask:        (B, N_canonical) precondition mask
            add_mask:        (B, N_canonical) add effect mask
            del_mask:        (B, N_canonical) delete effect mask
            frame_mask:      (B, N_canonical) frame axiom mask

        Returns:
            dict with:
                'total': scalar, weighted sum of all components
                'L_pre': scalar, precondition consistency
                'L_eff': scalar, effect consistency
                'L_frame': scalar, frame consistency
        """
        state_t_probs = torch.sigmoid(state_t_scores)
        state_t1_probs = torch.sigmoid(state_t1_scores)

        L_pre = self._pre_loss(state_t_scores, pre_mask)
        L_eff = self._eff_loss(state_t1_scores, add_mask, del_mask)
        L_frame = self._frame_loss(state_t_probs, state_t1_probs, frame_mask)

        total = self.w_pre * L_pre + self.w_eff * L_eff + self.w_frame * L_frame

        return {
            "total": total,
            "L_pre": L_pre,
            "L_eff": L_eff,
            "L_frame": L_frame,
        }


# ======================================================================
# Constraint 3: Counterfactual Discriminability
# ======================================================================

class TransitionEnergyScorer(nn.Module):
    """Energy-based transition verifier: E(S_t, a, S_{t+1}).

    Computes an energy score that measures how well the predicted state
    transition (S_t -> S_{t+1}) is explained by action a under the PDDL
    model. Lower energy = better explanation.

    Energy = PreViolation(S_t, a) + AddViolation(S_{t+1}, a)
           + DelViolation(S_{t+1}, a) + FrameViolation(S_t, S_{t+1}, a)

    This is used by CounterfactualDiscriminabilityLoss for ranking.
    """

    def forward(
        self,
        s_t: torch.Tensor,
        s_t1: torch.Tensor,
        pre_mask: torch.Tensor,
        add_mask: torch.Tensor,
        del_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute energy score (lower = better explanation).

        Args:
            s_t, s_t1: (*, N_canonical) probability tensors
            pre_mask, add_mask, del_mask: (*, N_canonical) binary masks
                Must have compatible leading dims with s_t, s_t1.

        Returns:
            (*) energy per sample (scalar per leading-dim element).
        """
        # Precondition violation: pre atoms should be True in s_t
        n_pre = pre_mask.sum(dim=-1).clamp(min=1)
        pre_violation = ((1 - s_t) * pre_mask).sum(dim=-1) / n_pre

        # Add effect violation: add atoms should be True in s_t1
        n_add = add_mask.sum(dim=-1).clamp(min=1)
        add_violation = ((1 - s_t1) * add_mask).sum(dim=-1) / n_add

        # Delete effect violation: del atoms should be False in s_t1
        n_del = del_mask.sum(dim=-1).clamp(min=1)
        del_violation = (s_t1 * del_mask).sum(dim=-1) / n_del

        # Frame violation: frame atoms should be unchanged
        frame_mask = (1 - add_mask - del_mask).clamp(min=0)
        n_frame = frame_mask.sum(dim=-1).clamp(min=1)
        frame_violation = ((s_t1 - s_t).abs() * frame_mask).sum(dim=-1) / n_frame

        return pre_violation + add_violation + del_violation + frame_violation


class CounterfactualDiscriminabilityLoss(nn.Module):
    """Constraint 3: counterfactual action discrimination.

    The grounded states must carry enough structural information to
    distinguish the true action from wrong actions.

    Formally:
        E(G(I_t), a_true, G(I_{t+1})) < E(G(I_t), a_false, G(I_{t+1}))

    Where E is the transition energy scorer. Trained via margin ranking:
        L = max(0, margin + E(pos) - E(neg))

    This is NOT just "more supervision" — it forces the grounding to
    encode action-relevant semantics, not just visual appearance.

    Args:
        margin: margin for the ranking loss (default 1.0).
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        self.energy_scorer = TransitionEnergyScorer()

    def forward(
        self,
        state_t_probs: torch.Tensor,
        state_t1_probs: torch.Tensor,
        pos_pre_mask: torch.Tensor,
        pos_add_mask: torch.Tensor,
        pos_del_mask: torch.Tensor,
        neg_pre_masks: torch.Tensor,
        neg_add_masks: torch.Tensor,
        neg_del_masks: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            state_t_probs:   (B, N_canonical)
            state_t1_probs:  (B, N_canonical)
            pos_pre_mask:    (B, N_canonical) positive action pre mask
            pos_add_mask:    (B, N_canonical) positive action add mask
            pos_del_mask:    (B, N_canonical) positive action del mask
            neg_pre_masks:   (B, K, N_canonical) negative action pre masks
            neg_add_masks:   (B, K, N_canonical) negative action add masks
            neg_del_masks:   (B, K, N_canonical) negative action del masks

        Returns:
            dict with:
                'total': scalar ranking loss
                'pos_energy': (B,) positive action energy
                'neg_energy': (B, K) negative action energies
                'violation_rate': fraction of violated margins
        """
        # Positive action energy (lower = better)
        pos_energy = self.energy_scorer(
            state_t_probs, state_t1_probs,
            pos_pre_mask, pos_add_mask, pos_del_mask,
        )  # (B,)

        # Negative action energies
        K = neg_pre_masks.shape[1]
        neg_energy = self._compute_neg_energies(
            state_t_probs, state_t1_probs,
            neg_pre_masks, neg_add_masks, neg_del_masks,
        )  # (B, K)

        # Margin ranking: max(0, margin + E(pos) - E(neg))
        violations = F.relu(self.margin + pos_energy.unsqueeze(-1) - neg_energy)
        violation_rate = (violations > 0).float().mean()

        return {
            "total": violations.mean(),
            "pos_energy": pos_energy.detach(),
            "neg_energy": neg_energy.detach(),
            "violation_rate": violation_rate.detach(),
        }

    def _compute_neg_energies(
        self,
        s_t: torch.Tensor,
        s_t1: torch.Tensor,
        neg_pre_masks: torch.Tensor,
        neg_add_masks: torch.Tensor,
        neg_del_masks: torch.Tensor,
    ) -> torch.Tensor:
        B, K, N = neg_pre_masks.shape
        s_t_exp = s_t.unsqueeze(1).expand(-1, K, -1)
        s_t1_exp = s_t1.unsqueeze(1).expand(-1, K, -1)

        n_pre = neg_pre_masks.sum(dim=-1).clamp(min=1)
        pre_viol = ((1 - s_t_exp) * neg_pre_masks).sum(dim=-1) / n_pre

        n_add = neg_add_masks.sum(dim=-1).clamp(min=1)
        add_viol = ((1 - s_t1_exp) * neg_add_masks).sum(dim=-1) / n_add

        n_del = neg_del_masks.sum(dim=-1).clamp(min=1)
        del_viol = (s_t1_exp * neg_del_masks).sum(dim=-1) / n_del

        frame_masks = (1 - neg_add_masks - neg_del_masks).clamp(min=0)
        n_frame = frame_masks.sum(dim=-1).clamp(min=1)
        frame_viol = ((s_t1_exp - s_t_exp).abs() * frame_masks).sum(dim=-1) / n_frame

        return pre_viol + add_viol + del_viol + frame_viol


# Keep old name as alias
CounterfactualActionLoss = CounterfactualDiscriminabilityLoss


# ======================================================================
# Legacy losses (backward compatibility)
# ======================================================================

class ActionSemanticsLoss(nn.Module):
    """Legacy: precondition consistency loss."""

    def __init__(self):
        super().__init__()

    def forward(self, state_t: torch.Tensor, action_info: dict) -> torch.Tensor:
        pre_mask = action_info["precondition_mask"]
        num_pre = pre_mask.sum()
        if num_pre > 0:
            bce = F.binary_cross_entropy_with_logits(
                state_t, torch.ones_like(state_t), reduction="none"
            )
            L_pre = (bce * pre_mask).sum() / num_pre
        else:
            L_pre = torch.tensor(0.0, device=state_t.device)
        return L_pre


class ReconstructionLoss(nn.Module):
    """Slot-mask-weighted MSE reconstruction loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, reconstructed, original, masks):
        sq_error = (reconstructed - original) ** 2
        mask_weights = masks.max(dim=1).values
        weighted_error = sq_error * mask_weights.unsqueeze(-1)
        if self.reduction == "mean":
            return weighted_error.mean()
        elif self.reduction == "sum":
            return weighted_error.sum()
        return weighted_error


class TemporalConsistencyLoss(nn.Module):
    """Legacy: temporal consistency loss."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, state_t, state_t1, action_effects):
        no_change_mask = (1.0 - action_effects)
        sq_diff = (state_t1 - state_t) ** 2 * no_change_mask
        if self.reduction == "mean":
            num_no_change = no_change_mask.sum()
            if num_no_change > 0:
                return sq_diff.sum() / num_no_change
            return torch.tensor(0.0, device=state_t.device)
        elif self.reduction == "sum":
            return sq_diff.sum()
        return sq_diff


# ======================================================================
# Self-test
# ======================================================================
if __name__ == "__main__":
    B, N_canon, D = 4, 32, 64

    print("=" * 60)
    print("Constraint 1: Pointwise Correctness")
    print("=" * 60)
    scores = torch.randn(B, N_canon)
    labels = torch.randint(-1, 2, (B, N_canon)).float()
    loss_seed = PredicateStateLoss()
    print(f"  L_seed = {loss_seed(scores, labels):.4f}")

    slots = torch.randn(B, 5, D)
    queries = torch.randn(B, 5, D)
    loss_contrast = PredicateContrastiveLoss(0.1)
    print(f"  L_contrast = {loss_contrast(slots, queries):.4f}")

    print()
    print("=" * 60)
    print("Constraint 2: Action Equivariance")
    print("=" * 60)
    st = torch.randn(B, N_canon)
    st1 = torch.randn(B, N_canon)
    pre_mask = torch.zeros(B, N_canon).scatter_(1, torch.randint(0, N_canon, (B, 3)), 1.0)
    add_mask = torch.zeros(B, N_canon).scatter_(1, torch.randint(0, N_canon, (B, 2)), 1.0)
    del_mask = torch.zeros(B, N_canon).scatter_(1, torch.randint(0, N_canon, (B, 2)), 1.0)
    frame_mask = (1 - add_mask - del_mask).clamp(min=0)

    equiv_loss = ActionEquivarianceLoss(w_pre=0.5, w_eff=1.0, w_frame=0.5)
    equiv_result = equiv_loss(st, st1, pre_mask, add_mask, del_mask, frame_mask)
    print(f"  L_equiv total = {equiv_result['total']:.4f}")
    print(f"    L_pre   = {equiv_result['L_pre']:.4f}")
    print(f"    L_eff   = {equiv_result['L_eff']:.4f}")
    print(f"    L_frame = {equiv_result['L_frame']:.4f}")

    print()
    print("=" * 60)
    print("Constraint 3: Counterfactual Discriminability")
    print("=" * 60)
    K = 3
    st_prob = torch.sigmoid(st)
    st1_prob = torch.sigmoid(st1)
    pos_pre = pre_mask
    pos_add = add_mask
    pos_del = del_mask
    neg_pre = torch.zeros(B, K, N_canon).scatter_(2, torch.randint(0, N_canon, (B, K, 2)), 1.0)
    neg_add = torch.zeros(B, K, N_canon).scatter_(2, torch.randint(0, N_canon, (B, K, 2)), 1.0)
    neg_del = torch.zeros(B, K, N_canon).scatter_(2, torch.randint(0, N_canon, (B, K, 2)), 1.0)

    cf_loss = CounterfactualDiscriminabilityLoss(margin=1.0)
    cf_result = cf_loss(st_prob, st1_prob, pos_pre, pos_add, pos_del, neg_pre, neg_add, neg_del)
    print(f"  L_cf total = {cf_result['total']:.4f}")
    print(f"  pos_energy = {cf_result['pos_energy'].mean():.4f}")
    print(f"  neg_energy = {cf_result['neg_energy'].mean():.4f}")
    print(f"  violation_rate = {cf_result['violation_rate']:.4f}")

    print()
    print("=== TransitionEnergyScorer standalone ===")
    scorer = TransitionEnergyScorer()
    energy = scorer(st_prob, st1_prob, pre_mask, add_mask, del_mask)
    print(f"  energy = {energy}")
    print(f"  shape  = {energy.shape}")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
