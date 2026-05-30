"""Synthetic Data Generator for TV Assembly Domain
=================================================
Generates synthetic training data for the PaQ framework based on the
PDDL domain definition.  No real images are required -- features are
random vectors whose distribution is conditioned on the predicate state.

Classes:
    TVAssemblyDomain       - reads domain.pddl, exposes grounding helpers
    SyntheticDataGenerator - produces (visual_features, types, state) triples
    PredicateStateDataset  - torch.utils.data.Dataset wrapper
"""

from __future__ import annotations

import sys
import os
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset

# Ensure the project root is importable so ``pddl_parser`` resolves.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from pddl_parser import PDDLParser, PDDLProblemParser


# ======================================================================
# TVAssemblyDomain
# ======================================================================

class TVAssemblyDomain:
    """High-level wrapper around the PDDL domain & problem files.

    Reads *domain.pddl* (and optionally a problem file for object
    definitions) and provides convenience methods for querying types,
    predicates, actions, and enumerating all ground predicates / states.
    """

    def __init__(
        self,
        domain_path: str | None = None,
        problem_path: str | None = None,
    ):
        if domain_path is None:
            domain_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "solver", "domain.pddl"
            )
        self.domain_path = os.path.abspath(domain_path)
        self.parser = PDDLParser(self.domain_path)

        self.problem_parser: PDDLProblemParser | None = None
        self.objects: dict[str, list[str]] = {}
        if problem_path is not None:
            self.problem_parser = PDDLProblemParser(problem_path)
            self.objects = self.problem_parser.objects

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_types(self) -> list[str]:
        """Return the list of type names defined in the domain."""
        return list(self.parser.types)

    def get_predicates(self) -> list[dict]:
        """Return predicate definitions (each has 'name' and 'params')."""
        return list(self.parser.predicates)

    def get_actions(self) -> list[dict]:
        """Return action definitions (each has 'name', 'parameters', ...)."""
        return list(self.parser.actions)

    def get_ground_predicates(self, objects: dict[str, list[str]] | None = None) -> list[str]:
        """Return all possible ground predicate strings.

        Parameters
        ----------
        objects : if *None*, uses the objects loaded from a problem file.
        """
        obj = objects if objects is not None else self.objects
        if not obj:
            raise ValueError(
                "No objects available.  Pass a problem file or supply objects."
            )
        return self.parser.get_all_ground_predicates(obj)

    # ------------------------------------------------------------------
    # State enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def state_to_vector(
        state: set[str],
        all_ground_preds: list[str],
    ) -> np.ndarray:
        """Convert a set of true predicate strings to a binary vector."""
        pred_index = {p: i for i, p in enumerate(all_ground_preds)}
        vec = np.zeros(len(all_ground_preds), dtype=np.float32)
        for p in state:
            if p in pred_index:
                vec[pred_index[p]] = 1.0
        return vec

    def get_all_possible_states(self) -> list[set[str]]:
        """Enumerate **all** valid (non-contradictory) states.

        Because the full combinatorial space is enormous (2^N for N ground
        predicates), this method only enumerates states reachable from the
        initial state by applying actions -- up to a configurable depth.

        For the synthetic-data generator the full enumeration is *not*
        required; :class:`SyntheticDataGenerator` samples random valid
        states instead.  This method is provided for completeness and
        small test cases.

        Returns
        -------
        list of sets of predicate strings (each set is one state).
        """
        if self.problem_parser is None:
            raise ValueError("A problem file is required for state enumeration.")

        all_gp = self.get_ground_predicates()
        init_state = set(self.problem_parser.init_state)

        # Build a quick mapping: predicate name -> set of grounded strings
        def pred_name(gp: str) -> str:
            return gp.strip("()").split()[0]

        # For the TV assembly domain we know the states well enough.
        # We will do a BFS over actions up to max_depth.
        visited: list[set[str]] = [init_state]
        visited_frozen: set[frozenset] = {frozenset(init_state)}
        queue: list[set[str]] = [init_state]
        max_depth = 6  # keep tractable

        for _ in range(max_depth):
            next_queue: list[set[str]] = []
            for state in queue:
                for action in self.get_actions():
                    for grounded in self._apply_action(action, state):
                        fs = frozenset(grounded)
                        if fs not in visited_frozen:
                            visited_frozen.add(fs)
                            visited.append(grounded)
                            next_queue.append(grounded)
            queue = next_queue
            if not queue:
                break

        return visited

    # ------------------------------------------------------------------
    # Internal: grounded action application
    # ------------------------------------------------------------------

    def _apply_action(
        self,
        action: dict,
        state: set[str],
    ) -> list[set[str]]:
        """Try to apply *action* in *state*; return list of resulting states.

        For each possible parameter binding that satisfies preconditions,
        produce the successor state by applying effects.
        """
        import re

        params = action.get("parameters", [])
        if not params:
            # Nullary action
            if self._check_precondition(action.get("precondition"), state, {}):
                new_state = set(state)
                self._apply_effects(action.get("effect"), new_state, {})
                return [new_state]
            return []

        # Gather candidate objects per parameter type
        param_choices: list[list[str]] = []
        for p in params:
            ptype = p["type"]
            candidates = self.objects.get(ptype, [])
            if not candidates:
                return []
            param_choices.append(candidates)

        results: list[set[str]] = []
        for combo in product(*param_choices):
            binding = {
                p["name"]: obj for p, obj in zip(params, combo)
            }
            if self._check_precondition(action.get("precondition"), state, binding):
                new_state = set(state)
                self._apply_effects(action.get("effect"), new_state, binding)
                results.append(new_state)
        return results

    # ------------------------------------------------------------------
    # Precondition / effect helpers (simplified)
    # ------------------------------------------------------------------

    @staticmethod
    def _subst(expr, binding: dict) -> list:
        """Replace ?variables with bound values in an S-expression."""
        if isinstance(expr, str):
            if expr.startswith("?") and expr in binding:
                return binding[expr]
            return expr
        return [TVAssemblyDomain._subst(e, binding) for e in expr]

    @staticmethod
    def _expr_to_str(expr) -> str:
        if isinstance(expr, str):
            return f"({expr})" if not expr.startswith("(") else expr
        parts = [str(e) for e in expr]
        return f"({' '.join(parts)})"

    def _check_precondition(self, precond, state: set[str], binding: dict) -> bool:
        """Evaluate a simplified precondition expression against *state*."""
        if precond is None:
            return True
        precond = self._subst(precond, binding)

        tag = precond[0] if isinstance(precond, list) else None

        if tag == "and":
            return all(self._check_precondition(p, state, {}) for p in precond[1:])
        if tag == "or":
            return any(self._check_precondition(p, state, {}) for p in precond[1:])
        if tag == "not":
            return not self._check_precondition(precond[1], state, {})
        if tag == "forall":
            # Very simplified: skip universal quantification for speed
            return True

        # Atom
        atom_str = self._expr_to_str(precond)
        return atom_str in state

    def _apply_effects(self, effect, state: set[str], binding: dict):
        """Mutate *state* according to *effect*."""
        if effect is None:
            return
        effect = self._subst(effect, binding)

        tag = effect[0] if isinstance(effect, list) else None
        if tag == "and":
            for sub in effect[1:]:
                self._apply_effects(sub, state, {})
            return
        if tag == "not":
            atom_str = self._expr_to_str(effect[1])
            state.discard(atom_str)
            return
        if tag == "forall":
            return  # skip for simplicity

        # Positive atom
        atom_str = self._expr_to_str(effect)
        state.add(atom_str)


# ======================================================================
# SyntheticDataGenerator
# ======================================================================

class SyntheticDataGenerator:
    """Generate synthetic (visual_features, object_types, ground_truth_state) triples.

    Each object receives a base feature vector (D=256) sampled from a normal
    distribution.  The feature vector is then perturbed depending on which
    ground predicates involving that object are True in the sampled state.

    This allows a classifier to learn the mapping ``features -> predicates``
    purely from the statistical correlation we inject.

    Parameters
    ----------
    domain : TVAssemblyDomain
        An initialised domain object (must have a problem file loaded).
    feature_dim : int
        Dimensionality of per-object feature vectors (default 256).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        domain: TVAssemblyDomain,
        feature_dim: int = 256,
        seed: int | None = 42,
    ):
        self.domain = domain
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)

        # Pre-compute grounding information
        self.all_ground_preds = domain.get_ground_predicates()
        self.n_ground_preds = len(self.all_ground_preds)

        # Build object catalogue
        self.object_names: list[str] = []
        self.object_types: list[str] = []
        self.type_to_id: dict[str, int] = {}
        for i, t in enumerate(domain.get_types()):
            self.type_to_id[t] = i

        for t in domain.get_types():
            for obj_name in domain.objects.get(t, []):
                self.object_names.append(obj_name)
                self.object_types.append(t)
        self.n_objects = len(self.object_names)

        # Build a map: ground_pred_index -> set of object indices involved
        self._build_pred_object_map()

        # Base features (will be shared, then perturbed per sample)
        self._base_features = self.rng.normal(
            0, 1, (self.n_objects, self.feature_dim)
        ).astype(np.float32)

        # Per-predicate perturbation vectors (so that each predicate
        # leaves a unique "fingerprint" on the object features)
        self._pred_perturbation = self.rng.normal(
            0, 0.3, (self.n_ground_preds, self.feature_dim)
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_pred_object_map(self):
        """For each ground predicate, find which object indices are involved."""
        self.pred_to_obj_ids: list[list[int]] = []
        obj_name_to_idx = {name: i for i, name in enumerate(self.object_names)}

        for gp in self.all_ground_preds:
            parts = gp.strip("()").split()
            # parts[0] = predicate name, parts[1:] = object arguments
            involved: list[int] = []
            for arg in parts[1:]:
                if arg in obj_name_to_idx:
                    involved.append(obj_name_to_idx[arg])
            self.pred_to_obj_ids.append(involved)

    # ------------------------------------------------------------------
    # State sampling
    # ------------------------------------------------------------------

    def sample_random_state(self) -> set[str]:
        """Sample a random (not necessarily valid) state.

        Each ground predicate is True with probability 0.3.
        Static predicates (e.g. screw-for-hole) are always True.
        """
        state: set[str] = set()
        static_names = {"screw-for-hole", "requires-predecessor"}
        for i, gp in enumerate(self.all_ground_preds):
            pred_name = gp.strip("()").split()[0]
            if pred_name in static_names:
                state.add(gp)
            elif self.rng.random() < 0.3:
                state.add(gp)
        return state

    def sample_valid_state_pair(
        self,
    ) -> tuple[set[str], set[str], dict] | None:
        """Try to produce a (state_t, state_t1, action_info) transition pair.

        Picks a random state, tries random actions, and returns the first
        successful application.  Returns *None* after max_attempts failures.
        """
        max_attempts = 200
        for _ in range(max_attempts):
            state_t = self.sample_random_state()
            actions = self.domain.get_actions()
            order = self.rng.permutation(len(actions))
            for ai in order:
                action = actions[ai]
                successors = self.domain._apply_action(action, state_t)
                if successors:
                    state_t1 = successors[0]
                    action_info = self._build_action_info(
                        action, state_t, state_t1
                    )
                    return state_t, state_t1, action_info
        return None

    def _build_action_info(
        self,
        action: dict,
        state_t: set[str],
        state_t1: set[str],
    ) -> dict:
        """Build action_info dict compatible with ActionSemanticsLoss."""
        # Precondition mask: which ground preds appear as preconditions
        pre_mask = np.zeros(self.n_ground_preds, dtype=np.float32)
        # Effect delta: +1 for newly True, -1 for newly False
        eff_delta = np.zeros(self.n_ground_preds, dtype=np.float32)

        for i, gp in enumerate(self.all_ground_preds):
            if gp not in state_t and gp in state_t1:
                eff_delta[i] = 1.0
            elif gp in state_t and gp not in state_t1:
                eff_delta[i] = -1.0

        return {
            "action_name": action["name"],
            "precondition_mask": pre_mask,   # simplified; filled as effect mask proxy
            "effect_delta": eff_delta,
        }

    # ------------------------------------------------------------------
    # Feature generation
    # ------------------------------------------------------------------

    def state_to_features(
        self,
        state: set[str],
    ) -> np.ndarray:
        """Generate per-object feature vectors conditioned on *state*.

        Returns
        -------
        features : (n_objects, feature_dim) float32 array
        """
        features = self._base_features.copy()
        for i, gp in enumerate(self.all_ground_preds):
            if gp in state:
                for obj_idx in self.pred_to_obj_ids[i]:
                    features[obj_idx] += self._pred_perturbation[i]
        return features

    def state_to_label_vector(self, state: set[str]) -> np.ndarray:
        """Convert a state (set of true atoms) to a binary label vector."""
        vec = np.zeros(self.n_ground_preds, dtype=np.float32)
        for i, gp in enumerate(self.all_ground_preds):
            if gp in state:
                vec[i] = 1.0
        return vec

    def state_to_object_type_ids(self) -> np.ndarray:
        """Return integer type-id for each object."""
        return np.array(
            [self.type_to_id[t] for t in self.object_types], dtype=np.int64
        )

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_samples: int,
        include_trajectory: bool = False,
    ) -> dict[str, np.ndarray]:
        """Generate a full dataset of synthetic samples.

        Parameters
        ----------
        n_samples : int
            Number of samples to produce.
        include_trajectory : bool
            If True, also generate (state_t, action, state_t1) pairs.

        Returns
        -------
        dict with keys:
            "visual_features" : (n_samples, n_objects, feature_dim)
            "object_type_ids" : (n_samples, n_objects)
            "predicate_labels": (n_samples, n_ground_preds)
        and optionally:
            "next_state"      : (n_samples, n_ground_preds)
            "action_info"     : list[dict]  (action_name, effect_delta, ...)
        """
        all_features = np.zeros(
            (n_samples, self.n_objects, self.feature_dim), dtype=np.float32
        )
        all_types = np.zeros((n_samples, self.n_objects), dtype=np.int64)
        all_labels = np.zeros((n_samples, self.n_ground_preds), dtype=np.float32)

        next_states: list[np.ndarray] | None = [] if include_trajectory else None
        action_infos: list[dict] | None = [] if include_trajectory else None

        type_ids = self.state_to_object_type_ids()

        for s in range(n_samples):
            if include_trajectory:
                result = self.sample_valid_state_pair()
                if result is not None:
                    st, st1, ainfo = result
                    all_features[s] = self.state_to_features(st)
                    all_types[s] = type_ids
                    all_labels[s] = self.state_to_label_vector(st)
                    next_states.append(self.state_to_label_vector(st1))  # type: ignore
                    action_infos.append(ainfo)  # type: ignore
                else:
                    # Fallback: random state
                    st = self.sample_random_state()
                    all_features[s] = self.state_to_features(st)
                    all_types[s] = type_ids
                    all_labels[s] = self.state_to_label_vector(st)
                    next_states.append(all_labels[s].copy())  # type: ignore
                    action_infos.append({  # type: ignore
                        "action_name": "noop",
                        "precondition_mask": np.zeros(self.n_ground_preds, np.float32),
                        "effect_delta": np.zeros(self.n_ground_preds, np.float32),
                    })
            else:
                st = self.sample_random_state()
                all_features[s] = self.state_to_features(st)
                all_types[s] = type_ids
                all_labels[s] = self.state_to_label_vector(st)

        out: dict[str, object] = {
            "visual_features": all_features,
            "object_type_ids": all_types,
            "predicate_labels": all_labels,
        }
        if include_trajectory and next_states is not None and action_infos is not None:
            out["next_state"] = np.stack(next_states)
            out["action_info"] = action_infos
        return out

    def generate_all_splits(
        self,
        train: int = 10_000,
        val: int = 1_000,
        test: int = 1_000,
        include_trajectory: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Generate train / val / test splits at once.

        Returns
        -------
        dict with keys ``"train"``, ``"val"``, ``"test"``, each mapping to
        the output of :meth:`generate_dataset`.
        """
        splits: dict[str, dict] = {}
        for name, n in [("train", train), ("val", val), ("test", test)]:
            print(f"Generating {name} split ({n} samples) ...")
            splits[name] = self.generate_dataset(n, include_trajectory)
        return splits


# ======================================================================
# PredicateStateDataset  (torch.utils.data.Dataset)
# ======================================================================

class PredicateStateDataset(Dataset):
    """PyTorch Dataset wrapping synthetic data for predicate-state prediction.

    Each item returns:
        - visual_features  : (n_objects, D)
        - object_type_ids  : (n_objects,)
        - predicate_labels : (n_ground_preds,)
    And optionally (when trajectory data is present):
        - next_state       : (n_ground_preds,)
        - action_info      : dict  (action_name, effect_delta, ...)

    Parameters
    ----------
    data : dict
        Output of :meth:`SyntheticDataGenerator.generate_dataset`.
    include_trajectory : bool
        Whether to return next_state and action_info.
    """

    def __init__(
        self,
        data: dict[str, np.ndarray],
        include_trajectory: bool = False,
    ):
        self.visual_features = data["visual_features"]       # (N, n_obj, D)
        self.object_type_ids = data["object_type_ids"]       # (N, n_obj)
        self.predicate_labels = data["predicate_labels"]     # (N, n_gp)

        self.include_trajectory = include_trajectory
        self.next_state: np.ndarray | None = data.get("next_state")  # (N, n_gp)
        self.action_info: list[dict] | None = data.get("action_info")  # list[dict]

        assert len(self.visual_features) == len(self.predicate_labels)

    def __len__(self) -> int:
        return len(self.visual_features)

    def __getitem__(self, idx: int):
        vf = torch.from_numpy(self.visual_features[idx])
        oti = torch.from_numpy(self.object_type_ids[idx])
        pl = torch.from_numpy(self.predicate_labels[idx])

        if self.include_trajectory and self.next_state is not None and self.action_info is not None:
            ns = torch.from_numpy(self.next_state[idx])
            ai = {
                "action_name": self.action_info[idx]["action_name"],
                "effect_delta": torch.from_numpy(self.action_info[idx]["effect_delta"]),
                "precondition_mask": torch.from_numpy(
                    self.action_info[idx]["precondition_mask"]
                ),
            }
            return vf, oti, pl, ns, ai
        else:
            return vf, oti, pl


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    domain_path = os.path.join(os.path.dirname(__file__), "..", "..", "solver", "domain.pddl")
    problem_path = os.path.join(os.path.dirname(__file__), "..", "..", "solver", "p_real.pddl")

    print("=== TVAssemblyDomain ===")
    domain = TVAssemblyDomain(domain_path, problem_path)
    print(f"  Types:      {domain.get_types()}")
    print(f"  Predicates: {len(domain.get_predicates())}")
    print(f"  Actions:    {[a['name'] for a in domain.get_actions()]}")

    gp = domain.get_ground_predicates()
    print(f"  Ground predicates ({len(gp)}):")
    for p in gp[:5]:
        print(f"    {p}")
    if len(gp) > 5:
        print(f"    ... and {len(gp) - 5} more")

    print("\n=== SyntheticDataGenerator ===")
    gen = SyntheticDataGenerator(domain, feature_dim=256, seed=0)
    print(f"  Objects ({gen.n_objects}): {gen.object_names}")
    print(f"  Type IDs: {gen.type_to_id}")
    print(f"  Ground predicates: {gen.n_ground_preds}")

    # Quick single-sample test
    state = gen.sample_random_state()
    print(f"\n  Sampled state ({len(state)} true preds):")
    for p in sorted(state)[:5]:
        print(f"    {p}")

    feats = gen.state_to_features(state)
    print(f"  Features shape: {feats.shape}")

    labels = gen.state_to_label_vector(state)
    print(f"  Labels shape: {labels.shape}, sum={labels.sum():.0f}")

    # Generate a small dataset
    print("\n=== Generate small dataset (train=32, val=8, test=8) ===")
    splits = gen.generate_all_splits(train=32, val=8, test=8, include_trajectory=True)
    for split_name, split_data in splits.items():
        print(f"  {split_name}: features {split_data['visual_features'].shape}, "
              f"labels {split_data['predicate_labels'].shape}")
        if "next_state" in split_data:
            print(f"          next_state {split_data['next_state'].shape}, "
                  f"actions {len(split_data['action_info'])}")

    # Test the Dataset wrapper
    print("\n=== PredicateStateDataset ===")
    ds = PredicateStateDataset(splits["train"], include_trajectory=True)
    print(f"  Dataset length: {len(ds)}")
    sample = ds[0]
    print(f"  Sample tuple length: {len(sample)}")
    vf, oti, pl, ns, ai = sample
    print(f"    visual_features : {vf.shape}")
    print(f"    object_type_ids : {oti.shape}")
    print(f"    predicate_labels: {pl.shape}")
    print(f"    next_state      : {ns.shape}")
    print(f"    action_name     : {ai['action_name']}")
    print(f"    effect_delta    : {ai['effect_delta'].shape}")

    # DataLoader smoke test
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\n  DataLoader batch: visual_features={batch[0].shape}, "
          f"labels={batch[2].shape}")

    # Test without trajectory
    print("\n=== Dataset without trajectory ===")
    ds_plain = PredicateStateDataset(splits["train"], include_trajectory=False)
    vf2, oti2, pl2 = ds_plain[0]
    print(f"  visual_features : {vf2.shape}")
    print(f"  object_type_ids : {oti2.shape}")
    print(f"  predicate_labels: {pl2.shape}")

    print("\nAll synthetic data tests passed.")
