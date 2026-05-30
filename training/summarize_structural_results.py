#!/usr/bin/env python3
"""Summarize PaQ structural experiment JSON files as CSV/Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "exact_match",
    "type_acc",
    "pred_pos_rate",
    "label_pos_rate",
    "avg_prob",
]


def _iter_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict[str, Any]] = []
    for k, kres in data.items():
        meta = dict(kres.get("_metadata", {}))
        for condition, result in kres.items():
            if condition == "_metadata":
                continue
            test = result.get("test", {})
            row = {
                "source_file": str(path),
                "K": k,
                "condition": condition,
                "feature_source": meta.get("feature_source", "unknown"),
                "transition_mask_source": result.get(
                    "transition_mask_source",
                    meta.get("transition_mask_source", "unknown"),
                ),
                "transition_supervision": meta.get("transition_supervision", "unknown"),
                "direct_object_tokens": meta.get("direct_object_tokens", "unknown"),
                "object_type_source": meta.get("object_type_source", "unknown"),
                "scoring_head_type": meta.get("scoring_head_type", "unknown"),
                "n_labeled_samples": meta.get("n_labeled_samples", ""),
                "n_labeled_states": meta.get("n_labeled_states", ""),
                "best_val_f1": result.get("best_val_f1", ""),
                "best_threshold": result.get("best_threshold", ""),
            }
            for key in METRIC_KEYS:
                row[key] = test.get(key, "")
            rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    cols = [
        "feature_source",
        "transition_mask_source",
        "transition_supervision",
        "direct_object_tokens",
        "object_type_source",
        "scoring_head_type",
        "K",
        "condition",
        "f1",
        "exact_match",
        "precision",
        "recall",
        "pred_pos_rate",
        "label_pos_rate",
        "best_threshold",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Result JSON files or experiment directories.")
    parser.add_argument("--csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--md", default=None, help="Optional Markdown output path.")
    args = parser.parse_args()

    result_files: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            result_files.extend(sorted(path.glob("*results*.json")))
        else:
            result_files.append(path)

    rows: list[dict[str, Any]] = []
    for path in result_files:
        rows.extend(_iter_rows(path))

    rows.sort(key=lambda r: (
        str(r["feature_source"]),
        str(r["transition_mask_source"]),
        int(r["K"]) if str(r["K"]).isdigit() else str(r["K"]),
        str(r["condition"]),
    ))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)

    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(rows, out)

    if not args.csv and not args.md:
        _write_markdown(rows, Path("/dev/stdout"))


if __name__ == "__main__":
    main()
