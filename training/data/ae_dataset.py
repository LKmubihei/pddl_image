"""AE-PaQ Dataset
================
Dataset classes for the Action-Effect Predicate-as-Query framework.

Two data modalities:
  1. StateDataset: (I_i, S_i^+) — image + true fact set (strong labels)
  2. TransitionDataset: (I_t, a_t, I_{t+1}) — visual transitions with action labels
  3. AEPaQDataset: mixed dataset combining both modalities with batch sampling

The key insight: state samples provide the seed supervision (L_seed),
while transition samples provide action-effect supervision (L_pre + L_eff + L_frame + L_cf).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Iterator

from paq.domain_compiler import DomainInfo


class StateDataset(Dataset):
    """Strong-labeled state dataset: (image, true_fact_set).

    Each item returns:
        - features: (N_patches, D) visual features or (3, H, W) images
        - state_label: (N_canonical) binary, 1=true, 0=false (closed-world)
        - object_type_ids: (N_obj,) type indices

    Supports few-shot: can subsample to K labeled visual samples.
    """

    def __init__(
        self,
        features: np.ndarray | torch.Tensor,
        state_labels: np.ndarray | torch.Tensor,
        object_type_ids: np.ndarray | torch.Tensor,
    ):
        """
        Args:
            features: (N, ...) visual features or images
            state_labels: (N, N_canonical) binary state vectors
            object_type_ids: (N_obj,) or (N, N_obj) type indices
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        if isinstance(state_labels, np.ndarray):
            state_labels = torch.from_numpy(state_labels)
        if isinstance(object_type_ids, np.ndarray):
            object_type_ids = torch.from_numpy(object_type_ids)

        self.features = features.float()
        self.state_labels = state_labels.float()

        if object_type_ids.dim() == 1:
            # Broadcast same types for all samples
            self.object_type_ids = object_type_ids.long().unsqueeze(0).expand(len(features), -1)
        else:
            self.object_type_ids = object_type_ids.long()

        assert len(self.features) == len(self.state_labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[idx], self.state_labels[idx], self.object_type_ids[idx]

    def subsample(self, k: int, seed: int = 42) -> "StateDataset":
        """Return a subsample of K dataset items / visual samples."""
        rng = np.random.default_rng(seed)
        if k >= len(self):
            return self
        indices = rng.choice(len(self), size=k, replace=False)
        return StateDataset(
            self.features[indices],
            self.state_labels[indices],
            self.object_type_ids[indices],
        )


class TransitionDataset(Dataset):
    """Transition dataset: (I_t, action_info, I_{t+1}).

    Each item returns:
        - features_t: (N_patches, D) features at time t
        - features_t1: (N_patches, D) features at time t+1
        - action_idx: scalar int — action index
        - pre_mask: (N_canonical,) precondition mask
        - add_mask: (N_canonical,) add effect mask
        - del_mask: (N_canonical,) delete effect mask
        - frame_mask: (N_canonical,) frame axiom mask
        - neg_pre_masks: (K, N_canonical) negative action pre masks
        - neg_add_masks: (K, N_canonical) negative action add masks
        - neg_del_masks: (K, N_canonical) negative action del masks
        - state_label_t: optional (N_canonical,) label at time t
        - state_label_t1: optional (N_canonical,) label at time t+1
        - object_type_ids: (N_obj,) type indices
    """

    def __init__(
        self,
        features_t: np.ndarray | torch.Tensor,
        features_t1: np.ndarray | torch.Tensor,
        action_idx: np.ndarray | torch.Tensor,
        pre_masks: np.ndarray | torch.Tensor,
        add_masks: np.ndarray | torch.Tensor,
        del_masks: np.ndarray | torch.Tensor,
        frame_masks: np.ndarray | torch.Tensor,
        neg_pre_masks: np.ndarray | torch.Tensor,
        neg_add_masks: np.ndarray | torch.Tensor,
        neg_del_masks: np.ndarray | torch.Tensor,
        object_type_ids: np.ndarray | torch.Tensor,
        state_labels_t: np.ndarray | torch.Tensor | None = None,
        state_labels_t1: np.ndarray | torch.Tensor | None = None,
    ):
        def _to_tensor(x):
            return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x.float()

        self.features_t = _to_tensor(features_t)
        self.features_t1 = _to_tensor(features_t1)
        self.action_idx = torch.from_numpy(action_idx).long() if isinstance(action_idx, np.ndarray) else action_idx.long()
        self.pre_masks = _to_tensor(pre_masks)
        self.add_masks = _to_tensor(add_masks)
        self.del_masks = _to_tensor(del_masks)
        self.frame_masks = _to_tensor(frame_masks)
        self.neg_pre_masks = _to_tensor(neg_pre_masks)
        self.neg_add_masks = _to_tensor(neg_add_masks)
        self.neg_del_masks = _to_tensor(neg_del_masks)
        self.state_labels_t = _to_tensor(state_labels_t) if state_labels_t is not None else None
        self.state_labels_t1 = _to_tensor(state_labels_t1) if state_labels_t1 is not None else None

        if isinstance(object_type_ids, np.ndarray):
            object_type_ids = torch.from_numpy(object_type_ids)
        if object_type_ids.dim() == 1:
            self.object_type_ids = object_type_ids.long().unsqueeze(0).expand(len(features_t), -1)
        else:
            self.object_type_ids = object_type_ids.long()

        if self.state_labels_t is not None:
            assert len(self.state_labels_t) == len(self.features_t)
        if self.state_labels_t1 is not None:
            assert len(self.state_labels_t1) == len(self.features_t1)

    def __len__(self) -> int:
        return len(self.features_t)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "features_t": self.features_t[idx],
            "features_t1": self.features_t1[idx],
            "action_idx": self.action_idx[idx],
            "pre_mask": self.pre_masks[idx],
            "add_mask": self.add_masks[idx],
            "del_mask": self.del_masks[idx],
            "frame_mask": self.frame_masks[idx],
            "neg_pre_masks": self.neg_pre_masks[idx],
            "neg_add_masks": self.neg_add_masks[idx],
            "neg_del_masks": self.neg_del_masks[idx],
            "object_type_ids": self.object_type_ids[idx],
        }
        if self.state_labels_t is not None:
            item["state_label_t"] = self.state_labels_t[idx]
        if self.state_labels_t1 is not None:
            item["state_label_t1"] = self.state_labels_t1[idx]
        return item


class MixedBatchSampler(Sampler):
    """Sampler that produces mixed batches of state and transition indices.

    For each batch, samples `state_ratio` fraction from the state dataset
    and the rest from the transition dataset.
    """

    def __init__(
        self,
        state_dataset_len: int,
        trans_dataset_len: int,
        batch_size: int = 32,
        state_ratio: float = 0.3,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.state_len = state_dataset_len
        self.trans_len = trans_dataset_len
        self.batch_size = batch_size
        self.state_ratio = state_ratio
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        n_state_per_batch = max(1, int(batch_size * state_ratio))
        n_trans_per_batch = batch_size - n_state_per_batch
        self.n_state_per_batch = n_state_per_batch
        self.n_trans_per_batch = n_trans_per_batch

    def __iter__(self) -> Iterator[tuple[str, list[int]]]:
        # Generate all batches
        n_state_batches = max(1, self.state_len // self.n_state_per_batch)
        n_trans_batches = max(1, self.trans_len // self.n_trans_per_batch)
        n_batches = max(n_state_batches, n_trans_batches)

        state_indices = np.arange(self.state_len)
        trans_indices = np.arange(self.trans_len)

        if self.shuffle:
            self.rng.shuffle(state_indices)
            self.rng.shuffle(trans_indices)

        for i in range(n_batches):
            # Cycle through indices if needed
            s_start = (i * self.n_state_per_batch) % self.state_len
            t_start = (i * self.n_trans_per_batch) % self.trans_len

            s_idx = state_indices[s_start:s_start + self.n_state_per_batch].tolist()
            t_idx = trans_indices[t_start:t_start + self.n_trans_per_batch].tolist()

            # Pad if necessary
            while len(s_idx) < self.n_state_per_batch:
                s_idx.append(state_indices[self.rng.integers(0, self.state_len)])
            while len(t_idx) < self.n_trans_per_batch:
                t_idx.append(trans_indices[self.rng.integers(0, self.trans_len)])

            yield ("state", s_idx)
            yield ("trans", t_idx)

    def __len__(self) -> int:
        n_state_batches = max(1, self.state_len // self.n_state_per_batch)
        n_trans_batches = max(1, self.trans_len // self.n_trans_per_batch)
        return 2 * max(n_state_batches, n_trans_batches)


def collate_state_batch(batch):
    """Collate function for StateDataset."""
    features, labels, types = zip(*batch)
    return {
        "features": torch.stack(features),
        "state_labels": torch.stack(labels),
        "object_type_ids": torch.stack(types),
    }


def collate_trans_batch(batch):
    """Collate function for TransitionDataset (items are dicts)."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
