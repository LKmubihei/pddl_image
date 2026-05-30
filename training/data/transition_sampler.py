"""Transition Sampler for AE-PaQ
================================
Generates (I_t, a_t, I_{t+1}) transition samples from a PDDL domain by
BFS over reachable states, applying grounded actions to produce successor
states with full action masks (precondition, add, delete, frame).

This replaces the synthetic-only data generation with PDDL-grounded
transition samples that carry explicit action semantics.
"""
from __future__ import annotations

from collections import deque
from itertools import product as iter_product
from typing import Optional

import numpy as np
import torch

from paq.domain_compiler import DomainInfo, PDDLDomainCompiler


class TransitionSampler:
    """Generate grounded transition samples from PDDL domain semantics.

    Given a compiled DomainInfo, this sampler:
    1. Enumerates reachable states via BFS from an initial state
    2. For each state transition, records full action masks
    3. Returns transitions as tensors ready for training

    The transition samples carry:
        - action_idx: index into domain_info.action_semantics
        - state_t_label: binary vector of true facts before action
        - state_t1_label: binary vector of true facts after action
        - pre_mask, add_mask, del_mask, frame_mask
    """

    def __init__(
        self,
        domain_info: DomainInfo,
        initial_state: set[str] | None = None,
        max_states: int = 5000,
        seed: int = 42,
    ):
        """
        Args:
            domain_info: compiled DomainInfo from PDDLDomainCompiler.
            initial_state: set of true atom strings for the initial state.
                          If None, uses empty state.
            max_states: maximum number of reachable states to enumerate.
            seed: random seed.
        """
        self.domain_info = domain_info
        self.max_states = max_states
        self.rng = np.random.default_rng(seed)

        # Atom string -> index mapping
        self.atom_to_idx = {a.str_repr: i for i, a in enumerate(domain_info.canonical_atoms)}
        self.idx_to_atom = {i: a.str_repr for i, a in enumerate(domain_info.canonical_atoms)}
        self.n_canonical = domain_info.n_canonical

        # Action name -> index mapping
        self.action_name_to_idx = domain_info.get_action_name_to_idx()
        self.n_actions = len(domain_info.action_semantics)

        # Pre-compute action masks as numpy arrays
        self._precompute_action_masks()

        # Enumerate reachable states
        self.initial_state = initial_state or set()
        self.transitions = self._enumerate_transitions()

    def _precompute_action_masks(self):
        """Pre-compute action masks as numpy arrays for fast lookup."""
        sems = self.domain_info.action_semantics
        self._pre_masks = np.array([s.precondition_mask for s in sems], dtype=np.float32)
        self._add_masks = np.array([s.add_mask for s in sems], dtype=np.float32)
        self._del_masks = np.array([s.del_mask for s in sems], dtype=np.float32)
        self._frame_masks = np.array([s.frame_mask for s in sems], dtype=np.float32)
        self._eff_deltas = np.array([s.effect_delta for s in sems], dtype=np.float32)

    def state_to_vector(self, state: set[str]) -> np.ndarray:
        """Convert a set of true atom strings to a binary vector."""
        vec = np.zeros(self.n_canonical, dtype=np.float32)
        for atom_str in state:
            if atom_str in self.atom_to_idx:
                vec[self.atom_to_idx[atom_str]] = 1.0
        return vec

    def vector_to_state(self, vec: np.ndarray) -> set[str]:
        """Convert a binary vector back to a set of true atom strings."""
        return {self.idx_to_atom[i] for i in range(self.n_canonical) if vec[i] > 0.5}

    def apply_action(self, state_vec: np.ndarray, action_idx: int) -> Optional[np.ndarray]:
        """Apply a grounded action to a state vector.

        Checks preconditions; returns None if not satisfied.
        Otherwise returns the successor state vector.
        """
        pre_mask = self._pre_masks[action_idx]
        eff_delta = self._eff_deltas[action_idx]

        # Check preconditions: all precondition atoms must be True
        if not np.all(state_vec[pre_mask > 0] > 0.5):
            return None

        # Apply effects
        new_state = state_vec.copy()
        # Add effects
        add_positions = eff_delta > 0
        new_state[add_positions] = 1.0
        # Delete effects
        del_positions = eff_delta < 0
        new_state[del_positions] = 0.0

        return new_state

    def _enumerate_transitions(self) -> list[dict]:
        """Enumerate reachable states and all valid transitions via BFS.

        Returns:
            list of transition dicts with keys:
                state_t_vec, state_t1_vec, action_idx,
                pre_mask, add_mask, del_mask, frame_mask
        """
        init_vec = self.state_to_vector(self.initial_state)
        init_key = init_vec.tobytes()

        visited = {init_key}
        queue = deque([init_vec])
        transitions = []

        while queue and len(visited) < self.max_states:
            state_vec = queue.popleft()

            # Try all grounded actions
            for action_idx in range(self.n_actions):
                next_vec = self.apply_action(state_vec, action_idx)
                if next_vec is None:
                    continue

                transitions.append({
                    "state_t_vec": state_vec,
                    "state_t1_vec": next_vec,
                    "action_idx": action_idx,
                    "pre_mask": self._pre_masks[action_idx],
                    "add_mask": self._add_masks[action_idx],
                    "del_mask": self._del_masks[action_idx],
                    "frame_mask": self._frame_masks[action_idx],
                })

                next_key = next_vec.tobytes()
                if next_key not in visited:
                    visited.add(next_key)
                    queue.append(next_vec)

        return transitions

    def sample_transitions(
        self,
        n_samples: int,
        n_negatives: int = 3,
    ) -> dict[str, torch.Tensor]:
        """Sample transition batches for training.

        For each sample, also generates n_negatives counterfactual actions
        (wrong actions from the same state).

        Args:
            n_samples: number of transition samples to generate.
            n_negatives: number of negative (counterfactual) actions per sample.

        Returns:
            dict with tensor values:
                state_t_labels:  (n_samples, N_canonical)
                state_t1_labels: (n_samples, N_canonical)
                action_idx:      (n_samples,) long
                pre_mask:        (n_samples, N_canonical)
                add_mask:        (n_samples, N_canonical)
                del_mask:        (n_samples, N_canonical)
                frame_mask:      (n_samples, N_canonical)
                neg_pre_masks:   (n_samples, n_negatives, N_canonical)
                neg_add_masks:   (n_samples, n_negatives, N_canonical)
                neg_del_masks:   (n_samples, n_negatives, N_canonical)
        """
        if not self.transitions:
            return {}

        N = len(self.transitions)
        indices = self.rng.integers(0, N, size=n_samples)

        state_t = np.stack([self.transitions[i]["state_t_vec"] for i in indices])
        state_t1 = np.stack([self.transitions[i]["state_t1_vec"] for i in indices])
        act_idx = np.array([self.transitions[i]["action_idx"] for i in indices], dtype=np.int64)
        pre_m = np.stack([self.transitions[i]["pre_mask"] for i in indices])
        add_m = np.stack([self.transitions[i]["add_mask"] for i in indices])
        del_m = np.stack([self.transitions[i]["del_mask"] for i in indices])
        frame_m = np.stack([self.transitions[i]["frame_mask"] for i in indices])

        # Generate negative actions
        neg_pre = np.zeros((n_samples, n_negatives, self.n_canonical), dtype=np.float32)
        neg_add = np.zeros((n_samples, n_negatives, self.n_canonical), dtype=np.float32)
        neg_del = np.zeros((n_samples, n_negatives, self.n_canonical), dtype=np.float32)

        for s in range(n_samples):
            pos_action = act_idx[s]
            # Sample negative actions (different from the positive one)
            candidates = [a for a in range(self.n_actions) if a != pos_action]
            if not candidates:
                continue
            neg_indices = self.rng.choice(candidates, size=min(n_negatives, len(candidates)), replace=False)
            for k, neg_idx in enumerate(neg_indices):
                neg_pre[s, k] = self._pre_masks[neg_idx]
                neg_add[s, k] = self._add_masks[neg_idx]
                neg_del[s, k] = self._del_masks[neg_idx]

        return {
            "state_t_labels": torch.from_numpy(state_t),
            "state_t1_labels": torch.from_numpy(state_t1),
            "action_idx": torch.from_numpy(act_idx),
            "pre_mask": torch.from_numpy(pre_m),
            "add_mask": torch.from_numpy(add_m),
            "del_mask": torch.from_numpy(del_m),
            "frame_mask": torch.from_numpy(frame_m),
            "neg_pre_masks": torch.from_numpy(neg_pre),
            "neg_add_masks": torch.from_numpy(neg_add),
            "neg_del_masks": torch.from_numpy(neg_del),
        }

    @property
    def n_transitions(self) -> int:
        return len(self.transitions)

    def summary(self) -> dict:
        return {
            "n_canonical": self.n_canonical,
            "n_actions": self.n_actions,
            "n_transitions": len(self.transitions),
        }
