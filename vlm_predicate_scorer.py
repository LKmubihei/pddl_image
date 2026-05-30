"""
VLM Predicate State Scorer
===========================
Given a PDDL domain description and a scene description (text or image),
query a Vision-Language Model for each ground predicate to obtain a
yes / no / unknown judgement.  Outputs a distribution over {True, False, Unknown}
for every predicate instance.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# Re-use the same API configuration as solver/llm_api.py
# ---------------------------------------------------------------------------
API_KEY = "sk-Akxfqns3yu893SxhzeAM4nzM528IcJoI107ZiroMYiaGKb1W"
BASE_URL = "https://toapis.com/v1"

_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# Predicate semantics -- human-readable descriptions for each predicate name
# ---------------------------------------------------------------------------
PREDICATE_DESCRIPTIONS: dict[str, str] = {
    "initial-state": "the assembly process is in its initial state (nothing has been done yet)",
    "power-com-inspected": "the power component has been inspected / verified",
    "power-com-placement-done": "the power component has been fully placed onto the panel",
    "in-material-box": "{0} is inside the material box {1}",
    "comp-grasp-free": "the gripper/robot hand is free (not holding any component)",
    "screw-grasp-free": "the gripper/robot hand is free (not holding any screw)",
    "comp-in-hand": "the component {0} is currently held in the gripper",
    "comp-at-panel-area": "the component {0} has been moved to the panel area {1} (coarse positioning)",
    "comp-aligned": "the component {0} is precisely aligned on the panel {1}",
    "comp-on-panel": "the component {0} has been placed (mounted) on the panel {1}",
    "screw-fetched": "a screw has been picked up and is ready for positioning",
    "screw-positioned": "a screw has been positioned over its target hole",
    "screw-unused": "the screw {0} has not yet been used (still available)",
    "screw-for-hole": "screw {0} is assigned to hole {1} (static process assignment)",
    "hole-empty": "hole {0} is empty (no screw inserted)",
    "screw-aligned": "screw {0} is aligned with hole {1}",
    "screw-inserted": "screw {0} has been inserted into hole {1}",
    "screw-fastened": "screw {0} has been fully fastened / tightened in hole {1}",
    "hole-done": "hole {0} has been fully completed (screw fastened)",
    "requires-predecessor": "hole {0} requires hole {1} to be completed first (process ordering)",
}

# Predicates that are *static facts* about the domain (always True by design)
# and should not be queried from the VLM.
STATIC_PREDICATES = {"screw-for-hole", "requires-predecessor"}


# ---------------------------------------------------------------------------
# VLM Predicate Scorer
# ---------------------------------------------------------------------------

class VLMPredicateScorer:
    """Score each ground predicate against a scene using a VLM."""

    def __init__(
        self,
        domain_predicates: list[dict[str, Any]],
        model: str = "gemini-2.5-pro",
        batch_size: int = 10,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        domain_predicates : list of predicate dicts from PDDLParser
        model : VLM model name
        batch_size : how many predicates to include per VLM call
        verbose : print progress
        """
        self.domain_predicates = {p["name"]: p for p in domain_predicates}
        self.model = model
        self.batch_size = batch_size
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _describe_predicate(self, ground_pred: str) -> str:
        """Turn ``(screw-fastened screw_A hole_A)`` into a natural language question."""
        inner = ground_pred.strip().strip("()")
        parts = inner.split()
        pred_name = parts[0]
        args = parts[1:]

        template = PREDICATE_DESCRIPTIONS.get(pred_name)
        if template:
            try:
                return template.format(*args)
            except (IndexError, KeyError):
                pass
        # fallback
        return f"Predicate '{pred_name}' holds for objects {', '.join(args) if args else '(no arguments)'}"

    def _build_batch_prompt(
        self,
        scene_description: str,
        ground_preds: list[str],
    ) -> str:
        """Build the VLM prompt for a batch of predicates."""
        pred_lines: list[str] = []
        for idx, gp in enumerate(ground_preds):
            desc = self._describe_predicate(gp)
            pred_lines.append(f"  {idx + 1}. {gp}  --  Meaning: {desc}")

        prompt = (
            "You are an expert assembly-line vision inspector.  You are given a description "
            "of an assembly scene and a list of logical predicates (state atoms).  For EACH "
            "predicate, judge whether it is TRUE, FALSE, or UNKNOWN in the described scene.\n\n"
            "SCENE DESCRIPTION:\n"
            f"{scene_description}\n\n"
            "PREDICATES TO JUDGE:\n"
            + "\n".join(pred_lines)
            + "\n\n"
            'Respond ONLY with a JSON object where each key is the predicate string exactly as '
            'given (with parentheses), and each value is one of: "yes", "no", "unknown".\n'
            'Example: {"(screw-fastened screw_A hole_A)": "yes", "(hole-empty hole_B)": "no"}\n'
            "Do NOT include any other text."
        )
        return prompt

    # ------------------------------------------------------------------
    # VLM call
    # ------------------------------------------------------------------

    def _call_vlm(self, prompt: str, image_url: str | None = None) -> str:
        """Call the VLM with the given prompt (optionally with an image)."""
        content_parts: list[dict] = [{"type": "text", "text": prompt}]
        if image_url:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url},
            })

        messages = [{"role": "user", "content": content_parts}]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = _client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                return response.choices[0].message.content
            except Exception as e:
                if self.verbose:
                    print(f"  [Retry {attempt + 1}/{max_retries}] API error: {e}")
                time.sleep(2 ** attempt)
        return ""

    # ------------------------------------------------------------------
    # Parse VLM response
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vlm_response(raw: str) -> dict[str, str]:
        """Extract JSON mapping from VLM response text."""
        # Try to find JSON block (may be wrapped in ```json ... ```)
        json_match = re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        # fallback: try the whole string
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------
    # Main scoring method
    # ------------------------------------------------------------------

    def score_scene(
        self,
        ground_preds: list[str],
        scene_description: str,
        image_url: str | None = None,
        skip_static: bool = True,
    ) -> dict[str, dict[str, float]]:
        """Score every ground predicate for the given scene.

        Parameters
        ----------
        ground_preds : list of grounded predicate strings
        scene_description : text describing the scene
        image_url : optional URL of an image
        skip_static : skip predicates like ``screw-for-hole`` that are static

        Returns
        -------
        dict mapping predicate_string -> {"True": p, "False": p, "Unknown": p}
        where probabilities sum to 1.
        """
        # Filter out static predicates
        eval_preds = []
        static_preds = []
        for gp in ground_preds:
            pred_name = gp.strip("()").split()[0]
            if skip_static and pred_name in STATIC_PREDICATES:
                static_preds.append(gp)
            else:
                eval_preds.append(gp)

        if self.verbose:
            print(f"Scoring {len(eval_preds)} predicates "
                  f"({len(static_preds)} static skipped) ...")

        # Batch calls
        all_results: dict[str, str] = {}

        for batch_start in range(0, len(eval_preds), self.batch_size):
            batch = eval_preds[batch_start: batch_start + self.batch_size]
            prompt = self._build_batch_prompt(scene_description, batch)
            raw = self._call_vlm(prompt, image_url)
            parsed = self._parse_vlm_response(raw)

            if self.verbose:
                print(f"  Batch {batch_start // self.batch_size + 1}: "
                      f"got {len(parsed)} judgments (expected {len(batch)})")

            for gp in batch:
                all_results[gp] = parsed.get(gp, "unknown")

            # Rate limiting
            time.sleep(1)

        # Convert to probability distribution
        scores: dict[str, dict[str, float]] = {}
        for gp in eval_preds:
            judgement = all_results.get(gp, "unknown").lower().strip()
            if judgement in ("yes", "true"):
                scores[gp] = {"True": 1.0, "False": 0.0, "Unknown": 0.0}
            elif judgement in ("no", "false"):
                scores[gp] = {"True": 0.0, "False": 1.0, "Unknown": 0.0}
            else:
                scores[gp] = {"True": 0.0, "False": 0.0, "Unknown": 1.0}

        # Add static predicates as always-True
        for gp in static_preds:
            scores[gp] = {"True": 1.0, "False": 0.0, "Unknown": 0.0}

        return scores

    # ------------------------------------------------------------------
    # Convenience: extract hard prediction from scores
    # ------------------------------------------------------------------

    @staticmethod
    def scores_to_state(scores: dict[str, dict[str, float]]) -> set[str]:
        """Return the set of predicates predicted True (threshold > 0.5)."""
        return {gp for gp, dist in scores.items() if dist["True"] > 0.5}

    @staticmethod
    def format_scores(scores: dict[str, dict[str, float]]) -> str:
        """Pretty-print the scoring results."""
        lines = []
        true_preds = []
        false_preds = []
        unknown_preds = []
        for gp, dist in sorted(scores.items()):
            label = max(dist, key=lambda k: dist[k])
            if label == "True":
                true_preds.append(gp)
            elif label == "False":
                false_preds.append(gp)
            else:
                unknown_preds.append(gp)

        lines.append(f"=== PREDICTED TRUE ({len(true_preds)}) ===")
        for p in true_preds:
            lines.append(f"  {p}")
        lines.append(f"\n=== PREDICTED FALSE ({len(false_preds)}) ===")
        for p in false_preds:
            lines.append(f"  {p}")
        if unknown_preds:
            lines.append(f"\n=== UNKNOWN ({len(unknown_preds)}) ===")
            for p in unknown_preds:
                lines.append(f"  {p}")
        return "\n".join(lines)
