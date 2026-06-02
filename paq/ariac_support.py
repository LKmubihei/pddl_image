"""ARIAC placement-factor sketch and constrained decoder.

The ARIAC labels in ``data/ariac`` describe stable states with true init atoms.
For these states the useful low-dimensional factor is:

    place(part) in location | other_part

The factor derives:
  - ``part_at(p, l)`` when ``place(p) == l``
  - ``on(p, q)`` when ``place(p) == q``
  - ``clear(p)`` when no active part is directly on ``p``
  - fixed stable atoms ``handempty`` and ``robot_at(table)``

The decoder is generated from the problem objects for each image.  Inactive
global parts are ignored and produce no atoms.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch


@dataclass(frozen=True)
class DecodedAriacState:
    assignment: dict[str, str]
    score: float
    atom_vector: list[int]
    atoms: set[str]
    diagnostics: dict[str, object]


class AriacPlacementSketch:
    """Domain sketch for ARIAC stable-state placement variables."""

    def __init__(
        self,
        parts: list[str],
        locations: list[str],
        part_object_indices: list[int],
        location_object_indices: list[int],
        canonical_atom_strings: list[str],
        table_location: str = "table",
    ):
        if not parts:
            raise ValueError("ARIAC placement sketch requires at least one part")
        if not locations:
            raise ValueError("ARIAC placement sketch requires at least one location")

        self.parts = list(parts)
        self.locations = list(locations)
        self.part_object_indices = list(part_object_indices)
        self.location_object_indices = list(location_object_indices)
        self.canonical_atom_strings = list(canonical_atom_strings)
        self.atom_to_idx = {a: i for i, a in enumerate(self.canonical_atom_strings)}
        self.table_location = table_location

        self.place_candidates: dict[str, list[str]] = {
            p: [x for x in self.parts if x != p] + self.locations
            for p in self.parts
        }
        obj_idx = {
            **{p: i for p, i in zip(self.parts, self.part_object_indices)},
            **{l: i for l, i in zip(self.locations, self.location_object_indices)},
        }
        self.place_candidate_object_indices: list[list[int]] = []
        for p in self.parts:
            self.place_candidate_object_indices.append(
                [obj_idx[c] for c in self.place_candidates[p]]
            )

        self.n_parts = len(self.parts)
        self.n_locations = len(self.locations)
        self.n_candidates = len(self.place_candidates[self.parts[0]])

        self._on_atom_indices = torch.full(
            (self.n_parts, self.n_parts), -1, dtype=torch.long
        )
        self._on_candidate_indices = torch.full(
            (self.n_parts, self.n_parts), -1, dtype=torch.long
        )
        self._part_at_atom_indices = torch.full(
            (self.n_parts, self.n_locations), -1, dtype=torch.long
        )
        self._location_candidate_indices = torch.full(
            (self.n_parts, self.n_locations), -1, dtype=torch.long
        )
        for pi, part in enumerate(self.parts):
            cand_to_idx = {
                cand: ci for ci, cand in enumerate(self.place_candidates[part])
            }
            for qj, other in enumerate(self.parts):
                atom = f"(on {part} {other})"
                if atom in self.atom_to_idx:
                    self._on_atom_indices[pi, qj] = self.atom_to_idx[atom]
                if other in cand_to_idx:
                    self._on_candidate_indices[pi, qj] = cand_to_idx[other]
            for lj, loc in enumerate(self.locations):
                atom = f"(part_at {part} {loc})"
                if atom in self.atom_to_idx:
                    self._part_at_atom_indices[pi, lj] = self.atom_to_idx[atom]
                self._location_candidate_indices[pi, lj] = cand_to_idx[loc]

    @classmethod
    def from_domain_info(
        cls,
        domain_info,
        part_type: str = "part",
        location_type: str = "location",
    ) -> "AriacPlacementSketch":
        parts, locations = [], []
        part_indices, location_indices = [], []
        for i, obj in enumerate(domain_info.objects):
            if obj.type_name == part_type:
                parts.append(obj.name)
                part_indices.append(i)
            elif obj.type_name == location_type:
                locations.append(obj.name)
                location_indices.append(i)
        return cls(
            parts=parts,
            locations=locations,
            part_object_indices=part_indices,
            location_object_indices=location_indices,
            canonical_atom_strings=domain_info.canonical_atom_strings,
        )

    def part_index(self, part: str) -> int:
        return self.parts.index(part)

    def is_valid_assignment(
        self,
        assignment: dict[str, str],
        active_parts: Iterable[str] | None = None,
    ) -> bool:
        active = list(active_parts) if active_parts is not None else list(assignment)
        active_set = set(active)
        loc_set = set(self.locations)

        for p in active:
            s = assignment.get(p)
            if s is None or s == p:
                return False
            if s not in active_set and s not in loc_set:
                return False

        support_count = {p: 0 for p in active}
        for p in active:
            s = assignment[p]
            if s in support_count:
                support_count[s] += 1
                if support_count[s] > 1:
                    return False

        for p in active:
            seen: set[str] = set()
            cur = p
            while cur in active_set:
                if cur in seen:
                    return False
                seen.add(cur)
                cur = assignment[cur]
            if cur not in loc_set:
                return False
        return True

    def derive_atoms(
        self,
        assignment: dict[str, str],
        active_parts: Iterable[str] | None = None,
    ) -> set[str]:
        active = list(active_parts) if active_parts is not None else list(assignment)
        if not self.is_valid_assignment(assignment, active):
            raise ValueError(f"Invalid ARIAC placement assignment: {assignment}")

        canonical = set(self.canonical_atom_strings)
        atoms: set[str] = set()
        if "(handempty)" in canonical:
            atoms.add("(handempty)")
        robot_table = f"(robot_at {self.table_location})"
        if robot_table in canonical:
            atoms.add(robot_table)

        active_set = set(active)
        for p in active:
            s = assignment[p]
            if s in active_set:
                atom = f"(on {p} {s})"
            else:
                atom = f"(part_at {p} {s})"
            if atom in canonical:
                atoms.add(atom)

        for p in active:
            if not any(assignment[q] == p for q in active):
                atom = f"(clear {p})"
                if atom in canonical:
                    atoms.add(atom)
        return atoms

    def atoms_to_vector(self, atoms: Iterable[str]) -> list[int]:
        labels = [0] * len(self.canonical_atom_strings)
        for atom in atoms:
            idx = self.atom_to_idx.get(atom)
            if idx is not None:
                labels[idx] = 1
        return labels

    def assignment_to_vector(
        self,
        assignment: dict[str, str],
        active_parts: Iterable[str] | None = None,
    ) -> list[int]:
        return self.atoms_to_vector(self.derive_atoms(assignment, active_parts))

    def active_atom_mask(self, active_parts: Iterable[str]) -> torch.Tensor:
        """Mask canonical atoms that belong to a problem's active object set."""
        active = set(active_parts)
        mask = torch.zeros(len(self.canonical_atom_strings), dtype=torch.float32)
        for i, atom in enumerate(self.canonical_atom_strings):
            toks = atom.strip("()").split()
            if not toks:
                continue
            pred = toks[0]
            if pred == "handempty":
                mask[i] = 1.0
            elif pred == "robot_at":
                mask[i] = 1.0
            elif pred == "clear" and len(toks) == 2 and toks[1] in active:
                mask[i] = 1.0
            elif pred == "part_at" and len(toks) == 3 and toks[1] in active:
                mask[i] = 1.0
            elif (
                pred == "on"
                and len(toks) == 3
                and toks[1] in active
                and toks[2] in active
            ):
                mask[i] = 1.0
        return mask

    def labels_to_support_targets(
        self,
        labels: torch.Tensor,
        active_part_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Convert canonical init labels to placement targets.

        Active parts with no unique support are assigned ``-1`` and ignored by
        CE.  Inactive parts are also ``-1``.
        """
        squeeze = False
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
            active_part_mask = active_part_mask.unsqueeze(0)
            squeeze = True
        if labels.dim() != 2:
            raise ValueError(f"labels must be 1D/2D, got {tuple(labels.shape)}")

        device = labels.device
        targets = torch.full(
            (labels.shape[0], self.n_parts), -1, dtype=torch.long, device=device
        )
        on_atom = self._on_atom_indices.to(device)
        on_cand = self._on_candidate_indices.to(device)
        part_at_atom = self._part_at_atom_indices.to(device)
        loc_cand = self._location_candidate_indices.to(device)

        for bi in range(labels.shape[0]):
            for pi in range(self.n_parts):
                if active_part_mask[bi, pi].item() <= 0:
                    continue
                found: list[int] = []
                for sj in range(self.n_parts):
                    ai = on_atom[pi, sj].item()
                    ci = on_cand[pi, sj].item()
                    if ai >= 0 and ci >= 0 and labels[bi, ai].item() > 0.5:
                        found.append(ci)
                for lj in range(self.n_locations):
                    ai = part_at_atom[pi, lj].item()
                    ci = loc_cand[pi, lj].item()
                    if ai >= 0 and labels[bi, ai].item() > 0.5:
                        found.append(ci)
                if len(found) == 1:
                    targets[bi, pi] = found[0]

        return targets.squeeze(0) if squeeze else targets

    def decode(
        self,
        placement_scores: torch.Tensor,
        active_parts: Iterable[str],
    ) -> DecodedAriacState:
        """Decode one score table into the best legal active placement."""
        if placement_scores.dim() != 2:
            raise ValueError(
                "placement_scores must have shape (n_parts, n_candidates), "
                f"got {tuple(placement_scores.shape)}"
            )
        if tuple(placement_scores.shape) != (self.n_parts, self.n_candidates):
            raise ValueError(
                f"Expected placement_scores {(self.n_parts, self.n_candidates)}, "
                f"got {tuple(placement_scores.shape)}"
            )

        active = [p for p in self.parts if p in set(active_parts)]
        if not active:
            atoms = self.derive_atoms({}, [])
            vec = self.atoms_to_vector(atoms)
            return DecodedAriacState({}, 0.0, vec, atoms, {"num_valid": 1})

        scores = placement_scores.detach().cpu()
        part_to_idx = {p: i for i, p in enumerate(self.parts)}
        choices: list[list[tuple[str, int, float]]] = []
        for p in active:
            pi = part_to_idx[p]
            row = []
            for ci, cand in enumerate(self.place_candidates[p]):
                if cand in self.locations or cand in active:
                    row.append((cand, ci, float(scores[pi, ci].item())))
            choices.append(row)

        best_assignment: dict[str, str] | None = None
        best_score = -float("inf")
        num_valid = 0
        num_total = 0
        for combo in product(*choices):
            num_total += 1
            assignment = {p: cand for p, (cand, _, _) in zip(active, combo)}
            if not self.is_valid_assignment(assignment, active):
                continue
            num_valid += 1
            total = sum(val for _, _, val in combo)
            if total > best_score:
                best_score = total
                best_assignment = assignment

        if best_assignment is None:
            raise RuntimeError(f"No legal ARIAC placement assignment for {active}")

        atoms = self.derive_atoms(best_assignment, active)
        return DecodedAriacState(
            assignment=best_assignment,
            score=best_score,
            atom_vector=self.atoms_to_vector(atoms),
            atoms=atoms,
            diagnostics={"num_total": num_total, "num_valid": num_valid},
        )

    def decode_batch(
        self,
        placement_scores: torch.Tensor,
        active_part_mask: torch.Tensor,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, list[DecodedAriacState]]:
        if placement_scores.dim() != 3:
            raise ValueError(
                "placement_scores must have shape (B, n_parts, n_candidates), "
                f"got {tuple(placement_scores.shape)}"
            )
        decoded = []
        for bi, scores in enumerate(placement_scores):
            active = [
                p for p, flag in zip(self.parts, active_part_mask[bi].tolist())
                if flag > 0
            ]
            decoded.append(self.decode(scores, active))
        vecs = torch.tensor(
            [d.atom_vector for d in decoded],
            dtype=torch.float32,
            device=device if device is not None else placement_scores.device,
        )
        return vecs, decoded
