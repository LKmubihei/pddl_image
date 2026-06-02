# AE-PaQ: Action-Equivariant Visual Predicate Grounding
# Three constraints:
#   C1: Pointwise Correctness    G(I) ≈ S
#   C2: Action Equivariance      G(I_{t+1}) ≈ Γ_a(G(I_t))
#   C3: Counterfactual Discrim.  E(G(I_t), a_true, G(I_{t+1})) < E(..., a_false, ...)
from .model import PaQModel
from .domain_compiler import PDDLDomainCompiler, DomainInfo, ActionSemantics
from .blocksworld_support import BlocksworldSupportSketch
from .ariac_support import AriacPlacementSketch
from .losses import (
    # Constraint 1: Pointwise Correctness
    PredicateStateLoss,
    SupportStateLoss,
    PredicateContrastiveLoss,
    # Constraint 2: Action Equivariance
    ActionEquivarianceLoss,
    PreconditionConsistencyLoss,
    EffectConsistencyLoss,
    FrameConsistencyLoss,
    # Constraint 3: Counterfactual Discriminability
    CounterfactualDiscriminabilityLoss,
    CounterfactualActionLoss,  # alias
    TransitionEnergyScorer,
    # Legacy
    ActionSemanticsLoss,
    ReconstructionLoss,
    TemporalConsistencyLoss,
)
