"""
State Evaluator
================
Compare a predicted predicate state against a ground-truth state and compute
metrics: predicate-level precision / recall / F1 (micro & macro),
state-level exact match rate, and unsupported-atom rate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any


class StateEvaluator:
    """Evaluate predicted predicate states against ground truth."""

    def __init__(
        self,
        all_ground_preds: list[str],
        static_preds: set[str] | None = None,
    ):
        """
        Parameters
        ----------
        all_ground_preds : master list of every possible ground predicate
        static_preds : predicate names (not grounded strings) that are
            static facts and should be excluded from evaluation
        """
        self.all_ground_preds = set(all_ground_preds)
        self.static_pred_names = static_preds or set()

        # Separate static from evaluable
        self.evaluable_preds = {
            gp for gp in self.all_ground_preds
            if gp.strip("()").split()[0] not in self.static_pred_names
        }

    # ------------------------------------------------------------------
    # Core comparison
    # ------------------------------------------------------------------

    def evaluate(
        self,
        predicted_state: set[str],
        ground_truth_state: set[str],
    ) -> dict[str, Any]:
        """Compute all metrics for a single scene.

        Parameters
        ----------
        predicted_state : set of predicate strings predicted True
        ground_truth_state : set of predicate strings actually True

        Returns
        -------
        dict with keys: tp, fp, fn, tn, precision, recall, f1,
                        exact_match, unsupported_atom_rate,
                        per_predicate_detail
        """
        # Restrict to evaluable predicates
        pred = predicted_state & self.evaluable_preds
        gt = ground_truth_state & self.evaluable_preds

        tp = pred & gt
        fp = pred - gt
        fn = gt - pred
        tn = self.evaluable_preds - pred - gt

        tp_count = len(tp)
        fp_count = len(fp)
        fn_count = len(fn)
        tn_count = len(tn)

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # State-level exact match
        exact_match = (pred == gt)

        # Unsupported atom rate: fraction of predicted True atoms that are
        # actually False (i.e., "hallucinations")
        unsupported_rate = fp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0

        # Per-predicate detail
        per_pred_detail: dict[str, dict[str, Any]] = {}
        for gp in self.evaluable_preds:
            p_true = gp in pred
            g_true = gp in gt
            if p_true and g_true:
                label = "TP"
            elif p_true and not g_true:
                label = "FP"
            elif not p_true and g_true:
                label = "FN"
            else:
                label = "TN"
            per_pred_detail[gp] = {
                "predicted": p_true,
                "ground_truth": g_true,
                "label": label,
            }

        return {
            "tp": tp_count,
            "fp": fp_count,
            "fn": fn_count,
            "tn": tn_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": exact_match,
            "unsupported_atom_rate": unsupported_rate,
            "per_predicate_detail": per_pred_detail,
        }

    # ------------------------------------------------------------------
    # Aggregate evaluation across multiple scenes
    # ------------------------------------------------------------------

    def evaluate_multi(
        self,
        predictions: list[set[str]],
        ground_truths: list[set[str]],
        scene_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate multiple scenes and compute micro/macro averages."""
        assert len(predictions) == len(ground_truths)
        n = len(predictions)
        if scene_names is None:
            scene_names = [f"scene_{i}" for i in range(n)]

        per_scene: dict[str, dict] = {}
        all_tp, all_fp, all_fn = 0, 0, 0
        f1_list: list[float] = []
        exact_match_count = 0

        for name, pred, gt in zip(scene_names, predictions, ground_truths):
            result = self.evaluate(pred, gt)
            per_scene[name] = result
            all_tp += result["tp"]
            all_fp += result["fp"]
            all_fn += result["fn"]
            f1_list.append(result["f1"])
            if result["exact_match"]:
                exact_match_count += 1

        # Micro-average
        micro_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 1.0
        micro_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 1.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Macro-average
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
        exact_match_rate = exact_match_count / n if n > 0 else 0.0

        return {
            "per_scene": per_scene,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "exact_match_rate": exact_match_rate,
            "total_scenes": n,
        }

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: dict[str, Any]) -> str:
        """Format an aggregate report for display."""
        lines = [
            "=" * 60,
            "         PREDICT STATE EVALUATION REPORT",
            "=" * 60,
            "",
            f"Total scenes evaluated: {report['total_scenes']}",
            "",
            "--- Aggregate Metrics ---",
            f"  Micro Precision : {report['micro_precision']:.4f}",
            f"  Micro Recall    : {report['micro_recall']:.4f}",
            f"  Micro F1        : {report['micro_f1']:.4f}",
            f"  Macro F1        : {report['macro_f1']:.4f}",
            f"  Exact Match Rate: {report['exact_match_rate']:.4f}",
            "",
        ]

        for scene_name, result in report["per_scene"].items():
            lines.append(f"--- {scene_name} ---")
            lines.append(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  TN={result['tn']}")
            lines.append(f"  Precision : {result['precision']:.4f}")
            lines.append(f"  Recall    : {result['recall']:.4f}")
            lines.append(f"  F1        : {result['f1']:.4f}")
            lines.append(f"  Exact Match: {result['exact_match']}")
            lines.append(f"  Unsupported Atom Rate: {result['unsupported_atom_rate']:.4f}")

            # Show FP and FN
            detail = result["per_predicate_detail"]
            fps = [gp for gp, d in detail.items() if d["label"] == "FP"]
            fns = [gp for gp, d in detail.items() if d["label"] == "FN"]
            if fps:
                lines.append("  False Positives (predicted True, actually False):")
                for fp in fps:
                    lines.append(f"    + {fp}")
            if fns:
                lines.append("  False Negatives (predicted False, actually True):")
                for fn in fns:
                    lines.append(f"    - {fn}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


# ======================================================================
# Quick self-test
# ======================================================================

if __name__ == "__main__":
    all_preds = [
        "(initial-state)", "(comp-grasp-free)", "(screw-grasp-free)",
        "(in-material-box power_com material_box)",
        "(screw-unused screw_A)", "(screw-unused screw_B)",
        "(hole-empty hole_A)", "(hole-empty hole_B)",
    ]
    evaluator = StateEvaluator(all_preds, static_preds={"screw-for-hole"})

    gt = {
        "(initial-state)", "(comp-grasp-free)", "(screw-grasp-free)",
        "(in-material-box power_com material_box)",
        "(screw-unused screw_A)", "(screw-unused screw_B)",
        "(hole-empty hole_A)", "(hole-empty hole_B)",
    }
    pred = {
        "(initial-state)", "(comp-grasp-free)",
        "(in-material-box power_com material_box)",
        "(screw-unused screw_A)",
        "(hole-empty hole_A)", "(hole-empty hole_B)",
        # FP: this one was not in GT
        "(screw-fastened screw_A hole_A)",
    }

    result = evaluator.evaluate(pred, gt)
    print(json.dumps(result, indent=2, default=str))
