# ARIAC PVG Rerank Diagnostic

Date: 2026-06-03

## Method

Implemented:

```text
training/diagnose_pvg_rerank.py
```

This is a pure-evaluation implementation of PVG:

```text
H+640 baseline checkpoint
-> support_scores + canonical_scores
-> top10/top25 legal assignments
-> proposal-derived part boxes/centers
-> workspace/table/root geometry
-> stack bbox geometry
-> optional atom-branch likelihood
-> conservative gated rerank
```

Proposal sources:

```text
hsv: deterministic HSV connected components; current default and main low-cost proposal source
label: existing data/ariac/labels YOLO class-only boxes + color assignment; ablation only, not part of the proposed method
```

The reranker gates every change:

```text
1. candidate must beat baseline by tau
2. changed edges must have proposal evidence
3. candidate must have no high-confidence geometry contradiction
4. candidate must be inside baseline topK legal states
```

## 52/100 Results

### Label-only proposal source ablation

The existing YOLO-label proposal source has low coverage:

```text
active center coverage: 149/333 = 0.4474
root center coverage:   141/306 = 0.4608
stack bbox-pair cov:      8/27  = 0.2963
```

Forced top25 high-geometry diagnostic:

```text
normal:      EM 0.7900, F1 0.9603
C_geometry:  EM 0.8000, F1 0.9627
changed:     1
bad_to_good: 1  (picture_307)
good_to_bad: 0
```

This confirms the geometry verifier has some real signal, but label-only
coverage is far below the 80-85% go/no-go threshold.

### HSV-only proposal source

After removing YOLO label boxes entirely, HSV-only still gives high coverage:

```text
active center coverage: 326/333 = 0.9790
root center coverage:   299/306 = 0.9771
stack bbox-pair cov:     26/27  = 0.9630
source counts: hsv=326
```

LOO/narrow-grid selection remains unchanged:

```text
A_normal:    EM 0.7900, F1 0.9603
B_atom:      EM 0.7900, F1 0.9603, changed 0
C_geometry:  EM 0.7900, F1 0.9603, changed 0
D_pvg:       EM 0.7900, F1 0.9603, changed 0
```

Forced top25 high-geometry with conservative `tau=0.5`:

```text
C_geometry:  EM 0.8000, F1 0.9620
changed:     2
bad_to_good: 1  (picture_307)
good_to_bad: 0
bad_to_bad:  1  (picture_63)
```

So the safe exact-match gain remains +1 without using YOLO labels.

### Label + HSV diagnostic ablation

This mixed-source run is retained only to diagnose whether better coverage
alone changes behavior:

```text
active center coverage: 332/333 = 0.9970
root center coverage:   305/306 = 0.9967
stack bbox-pair cov:     27/27  = 1.0000
source counts: label=149, hsv=183
```

LOO/narrow-grid selection remains unchanged:

```text
A_normal:    EM 0.7900, F1 0.9603
B_atom:      EM 0.7900, F1 0.9603, changed 0
C_geometry:  EM 0.7900, F1 0.9603, changed 0
D_pvg:       EM 0.7900, F1 0.9603, changed 0
```

Reason: the train split is already exact (`train EM=1.0`), so LOO EM/F1 cannot
distinguish useful rerank configs. The specified tie-breaker "fewer changed"
therefore selects no-op conservative configs.

Forced top25 high-geometry with `tau=0`:

```text
C_geometry:  EM 0.7900, F1 0.9615
changed:     2
bad_to_good: 1  (picture_307)
good_to_bad: 1  (picture_336)
```

The good-to-bad sample had a tiny score margin (`0.037`) and all proposal
evidence came from HSV components.

Forced top25 high-geometry with conservative `tau=0.5`:

```text
C_geometry:  EM 0.8000, F1 0.9627
changed:     1
bad_to_good: 1  (picture_307)
good_to_bad: 0
```

This recovers the label-only safe improvement while using HSV coverage.

## Interpretation

PVG is a better direction than object-slot cache in one important sense:

```text
when geometry changes top1, it can fix a real top25 error
```

But the current low-cost proposal bank is not enough to approach 0.9:

```text
target for 0.90: about +11 net bad_to_good on 52/100
current best low-cost PVG: +1 net bad_to_good
```

HSV-only solves coverage but not reliable part identity/object-region matching. The
new failure is no longer "missing centers"; it is "some centers are weakly
matched or geometry score is not discriminative enough to safely accept more
changes."

## Next Go/No-Go

Do not tune lambda further on current HSV/Yolo proposals. The next useful step
is better proposal identity:

```text
GroundingDINO / OWL-ViT kind boxes
or SAM2 masks prompted by reliable boxes
or DINO crop prototypes for kind disambiguation
```

The gate should stay conservative:

```text
topK=25
tau >= 0.5
changed edge proposal evidence required
reject high-confidence geometry contradictions
```

If stronger proposals still only produce +1 net fix, then the top25 oracle gap
is not reachable with 2D proposal geometry alone.

## Artifacts

```text
training/diagnose_pvg_rerank.py
experiments/ariac_pvg_rerank_52_100_narrow_20260603.md
experiments/ariac_pvg_rerank_52_100_forced_geometry_20260603.md
experiments/ariac_pvg_rerank_52_100_hsv_narrow_20260603.md
experiments/ariac_pvg_rerank_52_100_hsv_forced_geometry_20260603.md
experiments/ariac_pvg_rerank_52_100_hsv_forced_geometry_tau05_20260603.md
```
