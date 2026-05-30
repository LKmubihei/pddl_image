#!/usr/bin/env python3
"""
PaQ Self-Optimizing Training Loop (6h)
========================================
Automated training loop that:
1. Generates synthetic data from TV screw assembly domain
2. Trains PaQ model with progressive difficulty
3. Evaluates periodically
4. Self-adjusts hyperparameters based on performance
5. Saves best model and final report

Usage:
    python3 training/train_auto.py --hours 6
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pddl_parser import PDDLParser, PDDLProblemParser
from paq.domain_compiler import PDDLDomainCompiler
from itertools import product as itertools_product


# =========================================================================
# Synthetic Data Generator (uses PDDLDomainCompiler)
# =========================================================================

class AssemblyStateGenerator:
    """Generate synthetic training data for TV screw assembly domain.

    Uses PDDLDomainCompiler for domain-conditioned setup.
    """

    def __init__(self, domain_path: str, problem_path: str, d_feat: int = 256):
        self.domain = PDDLParser(domain_path)
        self.problem = PDDLProblemParser(problem_path)
        self.d_feat = d_feat
        self.objects = self.problem.objects
        self.types = self.domain.types

        # Use PDDLDomainCompiler for domain-conditioned setup
        self.static_preds = {"screw-for-hole", "requires-predecessor"}
        compiler = PDDLDomainCompiler(domain_path)
        self.domain_info = compiler.compile(
            objects=self.objects,
            static_predicates=self.static_preds,
        )

        # Build object list with types
        self.obj_list = []  # [(name, type_idx)]
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        for t in self.types:
            for obj_name in self.objects.get(t, []):
                self.obj_list.append((obj_name, self.type_to_idx[t]))

        self.n_objects = len(self.obj_list)

        # Build ground predicates
        self.ground_preds = self.domain.get_all_ground_predicates(self.objects)

        # Separate static vs dynamic
        self.static_set = set()
        for gp in self.ground_preds:
            pname = gp.strip("()").split()[0]
            if pname in self.static_preds:
                self.static_set.add(gp)

        # Build obj name -> index mapping
        self.obj_name_to_idx = {name: i for i, (name, _) in enumerate(self.obj_list)}

        # From domain_info (Gap 1: domain-conditioned)
        self.pred_param_types = self.domain_info.predicate_param_types
        self.pred_names = self.domain_info.predicate_names
        self.canonical_ground_preds = self.domain_info.canonical_atom_strings
        self.pred_arities = self.domain_info.predicate_arities
        self.action_semantics = self.domain_info.action_semantics

        # Generate base features for each object (fixed per run)
        np.random.seed(42)
        self.base_features = np.random.randn(self.n_objects, d_feat).astype(np.float32) * 0.3
        # Each type gets a distinct cluster center
        self.type_centers = np.random.randn(len(self.types), d_feat).astype(np.float32) * 2.0
        for i, (name, tidx) in enumerate(self.obj_list):
            self.base_features[i] += self.type_centers[tidx]

    def _build_canonical_ground_preds(self) -> list[str]:
        """Build canonical ordered list of dynamic ground predicates.

        Order: predicate types sorted by name, within each type the
        groundings follow itertools.product order of type-member lists.
        This MUST match the scoring head output order exactly.
        """
        type_to_objs = {}
        for t in self.types:
            type_to_objs[t] = self.objects.get(t, [])

        canonical = []
        for pname in self.pred_names:  # sorted
            param_types = self.pred_param_types[pname]
            arity = len(param_types)

            if arity == 0:
                canonical.append(f"({pname})")
            else:
                valid_names = [type_to_objs[pt] for pt in param_types]
                for combo in itertools_product(*valid_names):
                    args = " ".join(combo)
                    canonical.append(f"({pname} {args})")

        return canonical

    def _state_to_labels(self, state_true: set) -> torch.Tensor:
        """Convert a set of true predicates to label tensor in canonical order."""
        labels = torch.zeros(len(self.canonical_ground_preds), dtype=torch.float32)
        for i, gp in enumerate(self.canonical_ground_preds):
            if gp in state_true:
                labels[i] = 1.0
        return labels

    def generate_initial_state(self) -> set:
        """Generate the initial state."""
        s = set()
        s.add("(initial-state)")
        s.add("(comp-grasp-free)")
        s.add("(screw-grasp-free)")
        s.add("(in-material-box power_com material_box)")
        screws = self.objects.get("screw", [])
        holes = self.objects.get("hole", [])
        for s_name in screws:
            s.add(f"(screw-unused {s_name})")
        for h_name in holes:
            s.add(f"(hole-empty {h_name})")
        return s | self.static_set

    def generate_random_state(self) -> tuple[set, str]:
        """Generate a random valid state with description."""
        import random
        screws = self.objects.get("screw", [])
        holes = self.objects.get("hole", [])

        # Random progress level
        phase = random.choice(["initial", "inspected", "placed", "screwing"])
        s = set()

        if phase == "initial":
            s.add("(initial-state)")
            s.add("(comp-grasp-free)")
            s.add("(screw-grasp-free)")
            s.add("(in-material-box power_com material_box)")
            for sn in screws:
                s.add(f"(screw-unused {sn})")
            for hn in holes:
                s.add(f"(hole-empty {hn})")
            desc = "Initial state: power_com in box, all screws unused, all holes empty"

        elif phase == "inspected":
            s.add("(power-com-inspected)")
            s.add("(comp-grasp-free)")
            s.add("(screw-grasp-free)")
            s.add("(in-material-box power_com material_box)")
            for sn in screws:
                s.add(f"(screw-unused {sn})")
            for hn in holes:
                s.add(f"(hole-empty {hn})")
            desc = "Power component inspected, still in material box"

        elif phase == "placed":
            s.add("(power-com-inspected)")
            s.add("(power-com-placement-done)")
            s.add("(comp-grasp-free)")
            s.add("(screw-grasp-free)")
            s.add("(comp-on-panel power_com TV_panel)")
            for sn in screws:
                s.add(f"(screw-unused {sn})")
            for hn in holes:
                s.add(f"(hole-empty {hn})")
            desc = "Power component placed on panel, no screws yet"

        else:  # screwing
            s.add("(power-com-inspected)")
            s.add("(power-com-placement-done)")
            s.add("(comp-grasp-free)")
            s.add("(screw-grasp-free)")
            s.add("(comp-on-panel power_com TV_panel)")

            n_fastened = random.randint(0, len(screws))
            for i in range(n_fastened):
                s.add(f"(screw-fastened {screws[i]} {holes[i]})")
                s.add(f"(hole-done {holes[i]})")
            for i in range(n_fastened, len(screws)):
                s.add(f"(screw-unused {screws[i]})")
            for i in range(n_fastened, len(holes)):
                s.add(f"(hole-empty {holes[i]})")
            desc = f"{n_fastened} screws fastened, {len(screws)-n_fastened} remaining"

        s = s | self.static_set
        return s, desc

    def state_to_features(self, state: set, noise: float = 0.1) -> torch.Tensor:
        """Convert a state to visual feature tensor.

        Simulates what DINOv2 would produce for this scene.
        Objects in different states have different features.
        """
        features = self.base_features.copy()
        noise_feat = np.random.randn(*features.shape).astype(np.float32) * noise

        # Modify features based on state
        for i, (obj_name, tidx) in enumerate(self.obj_list):
            # Add state-dependent offsets
            if f"(comp-on-panel {obj_name} TV_panel)" in state:
                features[i] += np.random.randn(self.d_feat).astype(np.float32) * 0.5
            if f"(comp-in-hand {obj_name})" in state:
                features[i] += np.ones(self.d_feat, dtype=np.float32) * 0.8
            for h in self.objects.get("hole", []):
                if f"(screw-fastened {obj_name} {h})" in state:
                    features[i] += np.ones(self.d_feat, dtype=np.float32) * 1.0
                if f"(screw-inserted {obj_name} {h})" in state:
                    features[i] += np.ones(self.d_feat, dtype=np.float32) * 0.6

        features += noise_feat
        return torch.from_numpy(features)

    def generate_dataset(
        self, n_samples: int, noise: float = 0.1
    ) -> list[dict]:
        """Generate a dataset of (features, labels) pairs."""
        dataset = []
        for _ in range(n_samples):
            state, desc = self.generate_random_state()
            features = self.state_to_features(state, noise=noise)
            labels = self._state_to_labels(state)

            # Object type IDs
            type_ids = torch.tensor([tidx for _, tidx in self.obj_list], dtype=torch.long)

            dataset.append({
                "features": features,      # (N_obj, D)
                "type_ids": type_ids,       # (N_obj,)
                "labels": labels,           # dict: {ground_pred_str: float}
                "state": state,             # set of true predicates
                "desc": desc,
            })
        return dataset

    def generate_trajectory(
        self, n_steps: int = 10, noise: float = 0.1
    ) -> list[dict]:
        """Generate a trajectory of states with transitions."""
        screws = self.objects.get("screw", [])
        holes = self.objects.get("hole", [])

        # Build trajectory: initial → inspect → pick → move → locate → place → fetch → ...
        states = []

        # State 0: Initial
        s = self.generate_initial_state()
        states.append(("initial", s))

        # State 1: Inspected
        s1 = set(s)
        s1.discard("(initial-state)")
        s1.add("(power-com-inspected)")
        states.append(("inspect_power_com", s1))

        # State 2: Picked
        s2 = set(s1)
        s2.discard("(in-material-box power_com material_box)")
        s2.discard("(comp-grasp-free)")
        s2.add("(comp-in-hand power_com)")
        states.append(("pick_power_com", s2))

        # State 3: Moved
        s3 = set(s2)
        s3.add("(comp-at-panel-area power_com TV_panel)")
        states.append(("move_power_com", s3))

        # State 4: Located
        s4 = set(s3)
        s4.add("(comp-aligned power_com TV_panel)")
        states.append(("locating_power_com", s4))

        # State 5: Placed
        s5 = set(s4)
        s5.discard("(comp-in-hand power_com)")
        s5.discard("(comp-at-panel-area power_com TV_panel)")
        s5.discard("(comp-aligned power_com TV_panel)")
        s5.add("(comp-on-panel power_com TV_panel)")
        s5.add("(power-com-placement-done)")
        s5.add("(comp-grasp-free)")
        states.append(("place_power_com", s5))

        # States 6+: Screw operations
        for i in range(min(n_steps - 6, len(screws))):
            s_prev = states[-1][1]
            s_new = set(s_prev)
            # fetch → locate → insert → fasten
            s_new.add(f"(screw-fastened {screws[i]} {holes[i]})")
            s_new.add(f"(hole-done {holes[i]})")
            s_new.discard(f"(screw-unused {screws[i]})")
            s_new.discard(f"(hole-empty {holes[i]})")
            states.append((f"fasten_{screws[i]}", s_new))

        # Convert to feature/label pairs
        trajectory = []
        for action_name, state in states:
            features = self.state_to_features(state, noise=noise)
            labels = self._state_to_labels(state)
            type_ids = torch.tensor([tidx for _, tidx in self.obj_list], dtype=torch.long)
            trajectory.append({
                "features": features,
                "type_ids": type_ids,
                "labels": labels,
                "state": state,
                "action": action_name,
            })
        return trajectory


# =========================================================================
# Training Dataset wrapper
# =========================================================================

class PaQDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "features": s["features"],     # (N_obj, D)
            "type_ids": s["type_ids"],      # (N_obj,)
            "labels": s["labels"],          # (N_canonical,) tensor
        }


# =========================================================================
# Training Loop with Self-Optimization
# =========================================================================

def train_auto(hours: float = 6.0, output_dir: str = None):
    """Run the full self-optimizing training loop."""
    start_time = time.time()
    deadline = start_time + hours * 3600
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if output_dir is None:
        output_dir = ROOT / "experiments" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"  PaQ Self-Optimizing Training Loop ({hours}h)")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"  Deadline: {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')}")
    print("=" * 70)

    # ---- Step 1: Setup Domain ----
    domain_path = str(ROOT / "solver" / "domain.pddl")
    problem_path = str(ROOT / "solver" / "p_real.pddl")

    gen = AssemblyStateGenerator(domain_path, problem_path, d_feat=256)
    print(f"\nDomain: {gen.domain.domain_name}")
    print(f"Objects: {gen.n_objects}, Types: {len(gen.types)}")
    print(f"Dynamic predicates: {len(gen.canonical_ground_preds)}")
    print(f"Static predicates: {len(gen.static_set)}")

    # ---- Step 2: Generate Data ----
    print("\n[Phase 1] Generating synthetic data...")
    # Progressive difficulty
    noise_schedule = [0.05, 0.10, 0.15, 0.20, 0.25]

    # Initial training data
    train_data = gen.generate_dataset(5000, noise=0.05)
    val_data = gen.generate_dataset(500, noise=0.10)
    test_data = gen.generate_dataset(500, noise=0.15)

    train_ds = PaQDataset(train_data)
    val_ds = PaQDataset(val_data)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_data)}")

    # ---- Step 3: Build Model ----
    print("\n[Phase 2] Building PaQ model (domain-conditioned)...")

    sys.path.insert(0, str(ROOT))

    model = PaQModel.from_domain_info(
        gen.domain_info,
        n_object_slots=gen.n_objects,
        d_slot=256,
        n_slot_iters=3,
        use_real_encoder=False,
        tau_unknown=0.3,
        predict_slot_types=True,
    ).to(device)

    params = model.count_parameters()
    print(f"  Parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    # ---- Step 4: Training Config ----
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # Loss weights (self-adjustable)
    loss_weights = {
        "state": 1.0,
        "contrastive": 0.5,
        "recon": 0.2,
    }

    best_val_f1 = 0.0
    epoch = 0
    history = []
    difficulty_level = 0

    # ---- Step 5: Training Loop ----
    print("\n[Phase 3] Starting training loop...")
    print(f"  Auto-stopping at: {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')}")

    while time.time() < deadline:
        epoch += 1
        epoch_start = time.time()

        # --- Progressive difficulty ---
        if epoch % 100 == 0 and difficulty_level < len(noise_schedule) - 1:
            difficulty_level += 1
            new_noise = noise_schedule[difficulty_level]
            print(f"\n  >> Increasing difficulty: noise={new_noise}")
            train_data = gen.generate_dataset(5000, noise=new_noise)
            train_ds = PaQDataset(train_data)
            train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)

        # --- Train one epoch ---
        model.train()
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0

        for batch in train_loader:
            features = batch["features"].to(device)  # (B, N_obj, D)
            type_ids = batch["type_ids"].to(device)
            labels = batch["labels"].to(device)       # (B, N_canonical)

            optimizer.zero_grad()

            # Forward (using object features directly as "visual features")
            # Pass ground-truth type_ids for stable scoring dimensions
            out = model(features, object_type_ids=type_ids)

            # Compute loss
            canonical_scores = out["canonical_scores"]  # (B, N_canonical) logits
            pred_slots = out["predicate_slots"]
            pred_queries = out["predicate_queries"]
            obj_slots = out["object_slots"]

            # State loss — logits vs labels, already aligned by canonical order
            loss_state = nn.functional.binary_cross_entropy_with_logits(
                canonical_scores, labels
            )

            # Contrastive loss
            if pred_slots.shape[1] > 1 and pred_queries.shape[1] > 1:
                ps = nn.functional.normalize(pred_slots, dim=-1)
                pq = nn.functional.normalize(pred_queries, dim=-1)
                sim = torch.bmm(ps, pq.transpose(1, 2))  # (B, N, N)
                labels_contrast = torch.arange(sim.shape[1], device=device).unsqueeze(0).expand(sim.shape[0], -1)
                loss_contrast = nn.functional.cross_entropy(sim / 0.1, labels_contrast)
            else:
                loss_contrast = torch.tensor(0.0, device=device)

            # Reconstruction loss
            if obj_slots.shape[1] > 0:
                recon = torch.bmm(
                    nn.functional.softmax(torch.bmm(obj_slots, features.transpose(1, 2)), dim=-1),
                    features
                )
                loss_recon = nn.functional.mse_loss(recon, obj_slots.detach())
            else:
                loss_recon = torch.tensor(0.0, device=device)

            loss = (loss_weights["state"] * loss_state
                    + loss_weights["contrastive"] * loss_contrast
                    + loss_weights["recon"] * loss_recon)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            # Accuracy tracking
            with torch.no_grad():
                preds = (canonical_scores > 0).float()  # logit>0 ↔ prob>0.5
                n_correct += (preds == labels).sum().item()
                n_total += labels.numel()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = n_correct / max(n_total, 1)

        # --- Validation ---
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            tp, fp, fn = 0, 0, 0

            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(device)
                    type_ids = batch["type_ids"].to(device)
                    labels = batch["labels"].to(device)

                    out = model(features, object_type_ids=type_ids)
                    canonical_scores = out["canonical_scores"]  # (B, N_canonical)
                    preds = (canonical_scores > 0).float()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.numel()
                    tp += ((preds == 1) & (labels == 1)).sum().item()
                    fp += ((preds == 1) & (labels == 0)).sum().item()
                    fn += ((preds == 0) & (labels == 1)).sum().item()

            val_acc = val_correct / max(val_total, 1)
            val_prec = tp / max(tp + fp, 1)
            val_recall = tp / max(tp + fn, 1)
            val_f1 = 2 * val_prec * val_recall / max(val_prec + val_recall, 1e-8)

            elapsed = time.time() - start_time
            remaining = max(0, deadline - time.time())
            epoch_time = time.time() - epoch_start

            record = {
                "epoch": epoch,
                "train_loss": round(avg_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "val_f1": round(val_f1, 4),
                "val_prec": round(val_prec, 4),
                "val_recall": round(val_recall, 4),
                "difficulty": difficulty_level,
                "lr": round(scheduler.get_last_lr()[0], 6),
                "loss_weights": {k: round(v, 3) for k, v in loss_weights.items()},
                "elapsed_min": round(elapsed / 60, 1),
                "remaining_min": round(remaining / 60, 1),
            }
            history.append(record)

            # Log
            print(f"  [{epoch:4d}] loss={avg_loss:.4f} train_acc={train_acc:.3f} "
                  f"val_f1={val_f1:.3f} P={val_prec:.3f} R={val_recall:.3f} "
                  f"diff={difficulty_level} time={epoch_time:.1f}s "
                  f"elapsed={elapsed/60:.0f}m remain={remaining/60:.0f}m")

            # --- Save best model ---
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                    "val_prec": val_prec,
                    "val_recall": val_recall,
                    "loss_weights": loss_weights,
                }, os.path.join(output_dir, "best_model.pt"))
                print(f"  >> New best F1: {val_f1:.4f} (saved)")

            # --- Self-Optimization: Adjust hyperparameters ---
            if epoch % 50 == 0 and len(history) >= 10:
                recent = history[-10:]
                f1_trend = recent[-1]["val_f1"] - recent[0]["val_f1"]

                if f1_trend < 0.01:
                    # Not improving — increase contrastive weight
                    loss_weights["contrastive"] = min(2.0, loss_weights["contrastive"] * 1.2)
                    print(f"  >> Self-opt: increasing contrastive to {loss_weights['contrastive']:.3f}")

                if val_recall < 0.5 and val_prec > 0.8:
                    # High precision, low recall — reduce threshold or increase state weight
                    loss_weights["state"] = min(3.0, loss_weights["state"] * 1.1)
                    print(f"  >> Self-opt: increasing state weight to {loss_weights['state']:.3f}")

                if val_prec < 0.5 and val_recall > 0.8:
                    # High recall, low precision — increase contrastive
                    loss_weights["contrastive"] = min(2.0, loss_weights["contrastive"] * 1.3)
                    print(f"  >> Self-opt: rebalancing for precision")

            # --- Save checkpoint ---
            if epoch % 100 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "history": history,
                }, os.path.join(output_dir, f"checkpoint_{epoch}.pt"))

        # --- Time check ---
        if time.time() >= deadline:
            break

    # ---- Final Report ----
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total epochs: {epoch}")
    print(f"  Best val F1: {best_val_f1:.4f}")
    print(f"  Total time: {(time.time() - start_time)/3600:.2f}h")

    # Save history
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history,
        "best_val_f1": best_val_f1,
    }, os.path.join(output_dir, "final_model.pt"))

    # Generate final report
    report = {
        "total_epochs": epoch,
        "total_time_hours": round((time.time() - start_time) / 3600, 2),
        "best_val_f1": round(best_val_f1, 4),
        "final_loss_weights": {k: round(v, 3) for k, v in loss_weights.items()},
        "final_difficulty": difficulty_level,
        "history_last_10": history[-10:],
        "model_params": model.count_parameters(),
    }
    with open(os.path.join(output_dir, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Files: best_model.pt, final_model.pt, training_history.json, final_report.json")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    report = train_auto(hours=args.hours, output_dir=args.output_dir)
    print("\nFinal Report:")
    print(json.dumps(report, indent=2, default=str))
