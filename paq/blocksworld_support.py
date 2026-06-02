"""Blocksworld support-factor sketch and constrained decoder.

This module is the executable domain sketch for the structural Blocksworld
state factor:

    support(block) in other_block | column

It converts between the existing canonical atom vector and support labels,
checks Blocksworld invariants, and decodes neural support scores into the best
legal support assignment by exhaustive search. The search space is generated
from PDDL-compiled objects, not from a table of training states.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import torch


@dataclass(frozen=True)
class DecodedSupportState:
    """Result of constrained support decoding."""

    assignment: dict[str, str]
    score: float
    atom_vector: list[int]
    atoms: set[str]
    diagnostics: dict[str, object]


class BlocksworldSupportSketch:
    """Domain sketch for Blocksworld support variables.

    The candidate order for each block is:
        all other blocks in PDDL object order, then all columns in object order.
    """

    def __init__(
        self,
        blocks: list[str],
        columns: list[str],
        block_object_indices: list[int],
        column_object_indices: list[int],
        canonical_atom_strings: list[str],
    ):
        if not blocks:
            raise ValueError("Blocksworld support sketch requires at least one block")
        if not columns:
            raise ValueError("Blocksworld support sketch requires at least one column")

        self.blocks = list(blocks)
        self.columns = list(columns)
        self.block_object_indices = list(block_object_indices)
        self.column_object_indices = list(column_object_indices)
        self.canonical_atom_strings = list(canonical_atom_strings)
        self.atom_to_idx = {a: i for i, a in enumerate(self.canonical_atom_strings)}

        self.support_candidates: dict[str, list[str]] = {
            b: [x for x in self.blocks if x != b] + self.columns
            for b in self.blocks
        }
        self.support_candidate_object_indices: list[list[int]] = []
        obj_idx = {
            **{b: i for b, i in zip(self.blocks, self.block_object_indices)},
            **{c: i for c, i in zip(self.columns, self.column_object_indices)},
        }
        for b in self.blocks:
            self.support_candidate_object_indices.append(
                [obj_idx[cand] for cand in self.support_candidates[b]]
            )

        self.n_blocks = len(self.blocks)
        self.n_columns = len(self.columns)
        self.n_candidates = len(self.support_candidates[self.blocks[0]])

        self._on_atom_indices = torch.full(
            (self.n_blocks, self.n_blocks), -1, dtype=torch.long
        )
        self._on_candidate_indices = torch.full(
            (self.n_blocks, self.n_blocks), -1, dtype=torch.long
        )
        self._in_column_atom_indices = torch.full(
            (self.n_blocks, self.n_columns), -1, dtype=torch.long
        )
        self._column_candidate_indices = torch.full(
            (self.n_blocks, self.n_columns), -1, dtype=torch.long
        )

        for bi, b in enumerate(self.blocks):
            cand_to_idx = {cand: ci for ci, cand in enumerate(self.support_candidates[b])}
            for sj, s in enumerate(self.blocks):
                atom = f"(on {b} {s})"
                if atom in self.atom_to_idx:
                    self._on_atom_indices[bi, sj] = self.atom_to_idx[atom]
                if s in cand_to_idx:
                    self._on_candidate_indices[bi, sj] = cand_to_idx[s]
            for cj, c in enumerate(self.columns):
                atom = f"(inColumn {b} {c})"
                if atom in self.atom_to_idx:
                    self._in_column_atom_indices[bi, cj] = self.atom_to_idx[atom]
                self._column_candidate_indices[bi, cj] = cand_to_idx[c]

    @classmethod
    def from_domain_info(
        cls,
        domain_info,
        block_type: str = "block",
        column_type: str = "column",
    ) -> "BlocksworldSupportSketch":
        blocks, columns = [], []
        block_indices, column_indices = [], []
        for i, obj in enumerate(domain_info.objects):
            if obj.type_name == block_type:
                blocks.append(obj.name)
                block_indices.append(i)
            elif obj.type_name == column_type:
                columns.append(obj.name)
                column_indices.append(i)
        return cls(
            blocks=blocks,
            columns=columns,
            block_object_indices=block_indices,
            column_object_indices=column_indices,
            canonical_atom_strings=domain_info.canonical_atom_strings,
        )

    def is_valid_assignment(self, assignment: dict[str, str]) -> bool:
        """Check Blocksworld support invariants."""
        blocks = set(self.blocks)
        columns = set(self.columns)

        for b in self.blocks:
            s = assignment.get(b)
            if s is None or s == b:
                return False
            if s not in blocks and s not in columns:
                return False

        support_count = {b: 0 for b in self.blocks}
        for b in self.blocks:
            s = assignment[b]
            if s in support_count:
                support_count[s] += 1
                if support_count[s] > 1:
                    return False

        for b in self.blocks:
            seen: set[str] = set()
            cur = b
            while True:
                if cur in seen:
                    return False
                seen.add(cur)
                nxt = assignment[cur]
                if nxt in columns:
                    break
                if nxt not in blocks:
                    return False
                cur = nxt
        return True

    def chain_end_column(self, block: str, assignment: dict[str, str]) -> str:
        """Return the column reached by following the support chain."""
        seen: set[str] = set()
        cur = block
        while True:
            if cur in seen:
                raise ValueError(f"Support assignment has a cycle at {cur}: {assignment}")
            seen.add(cur)
            nxt = assignment[cur]
            if nxt in self.columns:
                return nxt
            cur = nxt

    def derive_atoms(self, assignment: dict[str, str]) -> set[str]:
        """Derive canonical dynamic atoms from a legal support assignment."""
        if not self.is_valid_assignment(assignment):
            raise ValueError(f"Invalid Blocksworld support assignment: {assignment}")

        atoms: set[str] = set()
        canonical = set(self.canonical_atom_strings)

        for b in self.blocks:
            s = assignment[b]
            if s in self.blocks:
                atom = f"(on {b} {s})"
                if atom in canonical:
                    atoms.add(atom)

        for x in self.blocks:
            if not any(assignment[b] == x for b in self.blocks):
                atom = f"(clear {x})"
                if atom in canonical:
                    atoms.add(atom)

        for b in self.blocks:
            end_col = self.chain_end_column(b, assignment)
            atom = f"(inColumn {b} {end_col})"
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

    def assignment_to_vector(self, assignment: dict[str, str]) -> list[int]:
        return self.atoms_to_vector(self.derive_atoms(assignment))

    def labels_to_support_targets(self, labels: torch.Tensor) -> torch.Tensor:
        """Convert canonical atom labels to per-block support target indices.

        Args:
            labels: (B, N_canonical) or (N_canonical,) closed-world labels.

        Returns:
            Long tensor shaped (B, n_blocks). Unknown/unrecoverable supports are
            marked as -1 for CE ignore_index compatibility.
        """
        squeeze = False
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
            squeeze = True
        if labels.dim() != 2:
            raise ValueError(f"labels must be 1D or 2D, got {tuple(labels.shape)}")

        device = labels.device
        targets = torch.full(
            (labels.shape[0], self.n_blocks), -1, dtype=torch.long, device=device
        )

        on_atom = self._on_atom_indices.to(device)
        on_cand = self._on_candidate_indices.to(device)
        in_col_atom = self._in_column_atom_indices.to(device)
        col_cand = self._column_candidate_indices.to(device)

        for bi in range(self.n_blocks):
            for sj in range(self.n_blocks):
                atom_idx = int(on_atom[bi, sj].item())
                cand_idx = int(on_cand[bi, sj].item())
                if atom_idx < 0 or cand_idx < 0:
                    continue
                mask = (labels[:, atom_idx] > 0.5) & (targets[:, bi] < 0)
                targets[mask, bi] = cand_idx

            no_block_support = targets[:, bi] < 0
            for cj in range(self.n_columns):
                atom_idx = int(in_col_atom[bi, cj].item())
                cand_idx = int(col_cand[bi, cj].item())
                if atom_idx < 0 or cand_idx < 0:
                    continue
                mask = (labels[:, atom_idx] > 0.5) & no_block_support & (targets[:, bi] < 0)
                targets[mask, bi] = cand_idx

        return targets.squeeze(0) if squeeze else targets

    def decode(self, support_scores: torch.Tensor) -> DecodedSupportState:
        """Decode one score table to the best legal support assignment."""
        if support_scores.dim() != 2:
            raise ValueError(
                "support_scores must have shape (n_blocks, n_candidates), "
                f"got {tuple(support_scores.shape)}"
            )
        if tuple(support_scores.shape) != (self.n_blocks, self.n_candidates):
            raise ValueError(
                f"Expected support_scores shape {(self.n_blocks, self.n_candidates)}, "
                f"got {tuple(support_scores.shape)}"
            )

        scores = support_scores.detach().cpu()
        best_assignment: dict[str, str] | None = None
        best_indices: tuple[int, ...] | None = None
        best_score = float("-inf")
        n_total = 0
        n_valid = 0

        candidate_ranges = [range(len(self.support_candidates[b])) for b in self.blocks]
        for indices in product(*candidate_ranges):
            n_total += 1
            assignment = {
                b: self.support_candidates[b][ci]
                for b, ci in zip(self.blocks, indices)
            }
            if not self.is_valid_assignment(assignment):
                continue
            n_valid += 1
            total = sum(float(scores[bi, ci].item()) for bi, ci in enumerate(indices))
            if total > best_score:
                best_score = total
                best_assignment = assignment
                best_indices = tuple(int(i) for i in indices)

        if best_assignment is None or best_indices is None:
            raise RuntimeError("No legal Blocksworld support assignment found")

        atoms = self.derive_atoms(best_assignment)
        return DecodedSupportState(
            assignment=best_assignment,
            score=best_score,
            atom_vector=self.atoms_to_vector(atoms),
            atoms=atoms,
            diagnostics={
                "enumerated_assignments": n_total,
                "valid_assignments": n_valid,
                "best_score": best_score,
                "candidate_indices": list(best_indices),
            },
        )

    def decode_batch(
        self,
        support_scores: torch.Tensor,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, list[DecodedSupportState]]:
        """Decode a batch of support scores into canonical atom vectors."""
        if support_scores.dim() != 3:
            raise ValueError(
                "support_scores must have shape (B, n_blocks, n_candidates), "
                f"got {tuple(support_scores.shape)}"
            )
        decoded = [self.decode(s) for s in support_scores]
        vectors = torch.tensor(
            [d.atom_vector for d in decoded],
            dtype=torch.long,
            device=device if device is not None else support_scores.device,
        )
        return vectors, decoded
