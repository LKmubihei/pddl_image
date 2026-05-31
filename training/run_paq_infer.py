#!/usr/bin/env python3
"""Run PaQ inference on one image using a trained checkpoint.

The model was trained on cached DINOv3 features (not raw images), so this
script extracts DINOv3 features from the input image on-the-fly, then runs
the PaQ model on those features.

Usage:
  python3 training/run_paq_infer.py \
    --checkpoint experiments/oracle_state_diff_legacy_full_k200_e20_alltrans_20260530_230420/k_200/model_full.pt \
    --image experiments/aepaq_1779595345/images/state_00000_v0.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paq.domain_compiler import PDDLDomainCompiler
from paq.model import PaQModel

BLOCKS = ["Y", "P", "R", "O"]
COLUMNS = ["C1", "C2", "C3", "C4"]
STATIC_PREDS = {"rightof", "leftof"}
D_SLOT = 256
N_SLOT_ITERS = 3
IMAGE_SIZE = 224
DINOV3_WEIGHTS = ROOT / "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_domain_info():
    bws_domain = (
        Path("/home/claudeuser/ViPlan")
        / "data" / "planning" / "blocksworld" / "domain.pddl"
    )
    compiler = PDDLDomainCompiler(str(bws_domain))
    return compiler.compile(
        objects={"block": BLOCKS, "column": COLUMNS},
        static_predicates=STATIC_PREDS,
    )


def _build_object_slot_init(domain_info, d_slot, device):
    init = torch.zeros(domain_info.n_objects, d_slot, device=device)
    block_colors = {
        "Y": (1.0, 1.0, 0.0), "P": (0.8, 0.0, 0.5),
        "R": (1.0, 0.0, 0.0), "O": (1.0, 0.5, 0.0),
        "G": (0.0, 0.8, 0.0), "B": (0.0, 0.0, 1.0),
    }
    columns = [o.name for o in domain_info.objects if o.type_name == "column"]
    col_to_pos = {
        c: (i / max(len(columns) - 1, 1)) * 2.0 - 1.0
        for i, c in enumerate(columns)
    }
    for i, obj in enumerate(domain_info.objects):
        if obj.type_name == "block":
            rgb = block_colors.get(obj.name.upper(), (0.5, 0.5, 0.5))
            init[i, 0:3] = torch.tensor(rgb, device=device)
            init[i, 3] = 1.0
        elif obj.type_name == "column":
            init[i, 0] = col_to_pos.get(obj.name, 0.0)
            init[i, 4] = 1.0
        identity_dim = 8 + i
        if identity_dim < d_slot:
            init[i, identity_dim] = 1.0
    return init


def _detect_scoring_head_type(state_dict):
    if any(k.startswith("scoring_head.legacy_") for k in state_dict):
        return "legacy"
    return "film"


DINOV3_REPO = Path("/home/claudeuser/facebookresearch/dinov3")


def _extract_dinov3_features(image_path, device):
    import sys
    if str(DINOV3_REPO) not in sys.path:
        sys.path.insert(0, str(DINOV3_REPO))
    from dinov3.models.vision_transformer import DinoVisionTransformer

    backbone = DinoVisionTransformer(
        img_size=224, patch_size=16, in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280, depth=32, num_heads=20,
        ffn_ratio=6.0, qkv_bias=True, drop_path_rate=0.0,
        layerscale_init=1e-5, norm_layer="layernormbf16",
        ffn_layer="swiglu", ffn_bias=True, proj_bias=True,
        n_storage_tokens=4, mask_k_bias=True,
    )
    sd = torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=True)
    backbone.load_state_dict(sd, strict=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    proj = torch.nn.Linear(1280, D_SLOT).to(device)
    backbone = backbone.to(device)

    tfm = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = backbone.forward_features(x)
        patches = out["x_norm_patchtokens"] if isinstance(out, dict) else out[:, 1:, :]
        features = proj(patches.float())
    return features


def main():
    parser = argparse.ArgumentParser(description="PaQ single-image inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model_full.pt")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use-oracle-types", action="store_true", default=True,
                        help="Use oracle object type IDs (default: True)")
    parser.add_argument("--no-oracle-types", dest="use_oracle_types", action="store_false",
                        help="Use predicted object types instead of oracle")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (flat state dict)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scoring_head_type = _detect_scoring_head_type(state_dict)
    print(f"Detected scoring head: {scoring_head_type}")

    # Build domain info and model
    domain_info = _build_domain_info()
    print(f"Domain: {domain_info.objects}")
    print(f"Canonical atoms ({domain_info.n_canonical}):")
    for atom in domain_info.canonical_atom_strings:
        print(f"  {atom}")

    model = PaQModel.from_domain_info(
        domain_info,
        n_object_slots=domain_info.n_objects,
        d_slot=D_SLOT,
        n_slot_iters=N_SLOT_ITERS,
        use_real_encoder=False,
        predict_slot_types=True,
        scoring_head_type=scoring_head_type,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Set object slot init
    slot_init = _build_object_slot_init(domain_info, D_SLOT, device)
    with torch.no_grad():
        model.object_slot_init.copy_(slot_init)

    # Extract DINOv3 features
    print(f"\nExtracting DINOv3 features from: {args.image}")
    features = _extract_dinov3_features(args.image, device)
    print(f"Feature shape: {tuple(features.shape)}")

    # Run inference
    obj_type_ids = torch.tensor(domain_info.obj_type_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(
            features,
            object_type_ids=obj_type_ids if args.use_oracle_types else None,
            slot_init=slot_init.unsqueeze(0),
        )
        probs = torch.sigmoid(out["canonical_scores"])[0].cpu().tolist()

    # Print results
    canonical_preds = domain_info.canonical_atom_strings
    true_preds = []
    print(f"\n{'Atom':<35s} {'Prob':>6s}  {'Label'}")
    print("-" * 55)
    for name, prob in zip(canonical_preds, probs):
        label = "TRUE" if prob >= args.threshold else "false"
        print(f"{name:<35s} {prob:>6.4f}  {label}")
        if prob >= args.threshold:
            true_preds.append(name)

    print(f"\nTRUE predicates (threshold={args.threshold}):")
    if true_preds:
        for p in true_preds:
            print(f"  {p}")
    else:
        print("  (none)")

    # Type predictions
    if "predicted_type_ids" in out:
        pred_types = out["predicted_type_ids"][0].cpu().tolist()
        type_names = domain_info.types
        print(f"\nPredicted slot types: {[type_names[t] for t in pred_types]}")


if __name__ == "__main__":
    main()
