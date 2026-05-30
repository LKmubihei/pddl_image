#!/usr/bin/env python3
"""
Run Baseline -- VLM Predicate State Parsing Pipeline
=====================================================
End-to-end script that:
1. Loads & parses the PDDL domain
2. Defines test scenarios (text descriptions + ground truth)
3. Runs VLM predicate scoring for each scenario
4. Compares predictions against ground truth
5. Outputs evaluation metrics
"""

from __future__ import annotations

import json
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pddl_parser import PDDLParser, PDDLProblemParser
from vlm_predicate_scorer import VLMPredicateScorer
from state_evaluator import StateEvaluator

# Static predicates (domain facts, not scene-dependent)
STATIC_PREDICATES = {"screw-for-hole", "requires-predecessor"}

DOMAIN_PATH = "/home/pc/PDDL/solver/domain.pddl"
PROBLEM_PATH = "/home/pc/PDDL/solver/p_real.pddl"


# ======================================================================
# Test scenario definitions
# ======================================================================

def get_objects() -> dict:
    """Standard object set for the TV screw assembly domain."""
    return {
        "component": ["power_com"],
        "panel": ["TV_panel"],
        "box": ["material_box"],
        "screw": [f"screw_{c}" for c in "ABCDEFG"],
        "hole": [f"hole_{c}" for c in "ABCDEFG"],
    }


def get_static_predicates() -> set[str]:
    """Return the set of static (always-true) ground predicates."""
    objects = get_objects()
    statics = set()
    # screw-for-hole: each screw assigned to its matching hole
    for c in "ABCDEFG":
        statics.add(f"(screw-for-hole screw_{c} hole_{c})")
    # requires-predecessor: fixed order B->A, C->B, ...
    order = list("ABCDEFG")
    for i in range(1, len(order)):
        statics.add(f"(requires-predecessor hole_{order[i]} hole_{order[i-1]})")
    return statics


def define_scenarios() -> list[dict]:
    """Define 3 test scenarios with text descriptions and ground truth states."""
    screws = [f"screw_{c}" for c in "ABCDEFG"]
    holes = [f"hole_{c}" for c in "ABCDEFG"]
    statics = get_static_predicates()

    # ------------------------------------------------------------------
    # Scenario 1: Initial state
    # ------------------------------------------------------------------
    s1_true = set()
    # Phase
    s1_true.add("(initial-state)")
    # Resources
    s1_true.add("(comp-grasp-free)")
    s1_true.add("(screw-grasp-free)")
    # Material
    s1_true.add("(in-material-box power_com material_box)")
    # All screws unused
    for s in screws:
        s1_true.add(f"(screw-unused {s})")
    # All holes empty
    for h in holes:
        s1_true.add(f"(hole-empty {h})")
    # Statics
    s1_true |= statics

    s1_description = (
        "This is a photo of a TV panel assembly workbench at the very beginning. "
        "A power component (power_com) is still inside a material box on the table. "
        "Seven screws (screw_A through screw_G) are laid out on the bench, all unused. "
        "Seven holes (hole_A through hole_G) on the TV panel are all empty. "
        "The robotic gripper is free (not holding anything). "
        "No assembly steps have been performed yet."
    )

    # ------------------------------------------------------------------
    # Scenario 2: Component placed, screw_A fastened
    # ------------------------------------------------------------------
    s2_true = set()
    # Phase
    s2_true.add("(power-com-inspected)")
    s2_true.add("(power-com-placement-done)")
    # Resources
    s2_true.add("(comp-grasp-free)")
    s2_true.add("(screw-grasp-free)")
    # Component state
    s2_true.add("(comp-on-panel power_com TV_panel)")
    # Screw states
    s2_true.add("(screw-fastened screw_A hole_A)")
    s2_true.add("(hole-done hole_A)")
    for s in screws[1:]:  # B through G unused
        s2_true.add(f"(screw-unused {s})")
    for h in holes[1:]:  # B through G empty
        s2_true.add(f"(hole-empty {h})")
    # Statics
    s2_true |= statics

    s2_description = (
        "A photo of the TV panel assembly workbench after several steps. "
        "The power component (power_com) has been inspected, picked from the material box, "
        "moved to the panel, aligned, and placed on the TV panel. It is now securely mounted. "
        "Screw_A has been fetched, positioned over hole_A, inserted, and fully fastened (tightened). "
        "Hole_A is done. "
        "The remaining screws (screw_B through screw_G) are still unused on the bench. "
        "Holes hole_B through hole_G are empty. "
        "The gripper is currently free."
    )

    # ------------------------------------------------------------------
    # Scenario 3: Mid-assembly -- screws A/B/C fastened
    # ------------------------------------------------------------------
    s3_true = set()
    # Phase
    s3_true.add("(power-com-inspected)")
    s3_true.add("(power-com-placement-done)")
    # Resources
    s3_true.add("(comp-grasp-free)")
    s3_true.add("(screw-grasp-free)")
    # Component state
    s3_true.add("(comp-on-panel power_com TV_panel)")
    # Screws A, B, C fastened
    for s, h in zip(screws[:3], holes[:3]):
        s3_true.add(f"(screw-fastened {s} {h})")
        s3_true.add(f"(hole-done {h})")
    # Remaining screws unused
    for s in screws[3:]:
        s3_true.add(f"(screw-unused {s})")
    # Remaining holes empty
    for h in holes[3:]:
        s3_true.add(f"(hole-empty {h})")
    # Statics
    s3_true |= statics

    s3_description = (
        "A photo of the TV panel assembly workbench at a mid-assembly stage. "
        "The power component (power_com) has been fully placed on the TV panel. "
        "Screws screw_A, screw_B, and screw_C have each been inserted and fastened "
        "into their respective holes hole_A, hole_B, and hole_C. "
        "Holes hole_A, hole_B, hole_C are done. "
        "The remaining screws (screw_D through screw_G) are unused on the bench. "
        "Holes hole_D through hole_G are empty. "
        "The robotic gripper is free."
    )

    return [
        {
            "name": "scenario_1_initial",
            "description": s1_description,
            "ground_truth": s1_true,
        },
        {
            "name": "scenario_2_placed_A_fastened",
            "description": s2_description,
            "ground_truth": s2_true,
        },
        {
            "name": "scenario_3_ABC_fastened",
            "description": s3_description,
            "ground_truth": s3_true,
        },
    ]


# ======================================================================
# Main pipeline
# ======================================================================

def main():
    print("=" * 70)
    print("  VLM -> PDDL Predicate State Parsing  (Baseline Pipeline)")
    print("=" * 70)

    # ---- Step 1: Parse domain ----
    print("\n[Step 1] Parsing PDDL domain ...")
    domain = PDDLParser(DOMAIN_PATH)
    print(f"  Domain: {domain.domain_name}")
    print(f"  Types: {domain.types}")
    print(f"  Predicates: {len(domain.predicates)}")
    print(f"  Actions: {len(domain.actions)}")

    # ---- Step 2: Parse problem (for objects) ----
    print("\n[Step 2] Parsing problem file for objects ...")
    problem = PDDLProblemParser(PROBLEM_PATH)
    objects = problem.objects
    print(f"  Objects: {json.dumps(objects, indent=4)}")

    # ---- Step 3: Generate all ground predicates ----
    print("\n[Step 3] Generating all ground predicates ...")
    all_ground = domain.get_all_ground_predicates(objects)
    print(f"  Total ground predicates: {len(all_ground)}")

    # ---- Step 4: Define test scenarios ----
    print("\n[Step 4] Loading test scenarios ...")
    scenarios = define_scenarios()
    for sc in scenarios:
        print(f"  - {sc['name']}: {len(sc['ground_truth'])} true atoms")

    # ---- Step 5: VLM scoring ----
    print("\n[Step 5] Running VLM predicate scoring ...")
    scorer = VLMPredicateScorer(
        domain_predicates=domain.predicates,
        model="gemini-2.5-pro-official",
        batch_size=15,
        verbose=True,
    )

    predictions: list[set[str]] = []
    ground_truths: list[set[str]] = []
    scene_names: list[str] = []

    for sc in scenarios:
        print(f"\n>>> Scoring scene: {sc['name']}")
        print(f"    Description: {sc['description'][:100]}...")
        scores = scorer.score_scene(
            ground_preds=all_ground,
            scene_description=sc["description"],
        )

        predicted_state = VLMPredicateScorer.scores_to_state(scores)
        predictions.append(predicted_state)
        ground_truths.append(sc["ground_truth"])
        scene_names.append(sc["name"])

        # Print prediction summary
        print(scorer.format_scores(scores))

    # ---- Step 6: Evaluation ----
    print("\n[Step 6] Evaluating predictions ...")
    evaluator = StateEvaluator(
        all_ground_preds=all_ground,
        static_preds=STATIC_PREDICATES,
    )

    report = evaluator.evaluate_multi(predictions, ground_truths, scene_names)
    print(evaluator.format_report(report))

    # ---- Save results ----
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert sets to sorted lists for JSON serialization
        serializable = {
            "scenarios": [
                {
                    "name": sc["name"],
                    "ground_truth": sorted(sc["ground_truth"]),
                }
                for sc in scenarios
            ],
            "predictions": [sorted(p) for p in predictions],
            "metrics": {
                k: v for k, v in report.items()
                if k != "per_scene"
            },
            "per_scene_summary": {
                name: {
                    "tp": res["tp"],
                    "fp": res["fp"],
                    "fn": res["fn"],
                    "tn": res["tn"],
                    "precision": res["precision"],
                    "recall": res["recall"],
                    "f1": res["f1"],
                    "exact_match": res["exact_match"],
                    "unsupported_atom_rate": res["unsupported_atom_rate"],
                }
                for name, res in report["per_scene"].items()
            },
        }
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
