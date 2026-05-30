#!/usr/bin/env python3
"""Run PaQ v3 inference on one image.

Usage:
  python3 training/run_paq_infer.py \
    --checkpoint experiments/viplan_dinov3_1779445991/best_paq_model.pt \
    --image /path/to/image.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paq.domain_compiler import ActionSemantics, DomainInfo, PredicateSchema
from paq.model import PaQModel


def _load_domain_info(payload: dict) -> DomainInfo:
    pred_specs = []
    for p in payload["predicates"]:
        schema = p["schema"]
        name, rest = schema.split("(", 1)
        arg_text = rest.rstrip(")")
        param_names = []
        param_types = []
        if arg_text:
            for part in arg_text.split(","):
                part = part.strip()
                if ":" in part:
                    n, t = part.split(":", 1)
                    param_names.append(n.strip())
                    param_types.append(t.strip())
        pred_specs.append(PredicateSchema(
            name=name,
            arity=len(param_types),
            param_types=param_types,
            param_names=param_names,
            action_roles=p.get("roles", []),
            gloss=p.get("gloss", schema),
        ))

    return DomainInfo(
        domain_name=payload["domain"],
        types=payload["types"],
        type_to_idx={t: i for i, t in enumerate(payload["types"])},
        objects=[],
        obj_name_to_idx={},
        predicate_schemas=pred_specs,
        canonical_atoms=[],
        action_semantics=[],
        obj_type_ids=[],
        static_predicates=set(),
        n_canonical=payload["n_canonical"],
    )


def _build_model(ckpt: dict) -> PaQModel:
    domain_info = _load_domain_info(ckpt["domain_info"])
    model = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=len(ckpt["obj_type_ids"]),
        d_slot=256,
        n_slot_iters=3,
        use_real_encoder=False,
        predict_slot_types=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _build_model(ckpt).to(device)

    color_init = ckpt["color_init"].to(device)
    obj_type_ids = ckpt["obj_type_ids"].unsqueeze(0).to(device)
    canonical_preds = ckpt["canonical_preds"]

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(
            x,
            object_type_ids=obj_type_ids,
            slot_init=color_init.unsqueeze(0),
        )
        probs = torch.sigmoid(out["canonical_scores"])[0].detach().cpu().tolist()

    true_preds = []
    for name, prob in zip(canonical_preds, probs):
        label = "true" if prob >= args.threshold else "false"
        print(f"{name}\t{prob:.4f}\t{label}")
        if label == "true":
            true_preds.append(name)

    print("\nTRUE predicates:")
    if true_preds:
        print("  " + ", ".join(true_preds))
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
