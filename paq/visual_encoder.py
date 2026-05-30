"""Visual Encoder: DINOv2 / DINOv3 (frozen) + learnable projection."""
from __future__ import annotations
import logging
from typing import Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Normalization constants shared across DINOv2/v3
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisualEncoder(nn.Module):
    """DINOv2 ViT-S/14 frozen backbone + learnable linear projection.

    Output: (B, N_patches, D_proj) patch-level features.
    """

    def __init__(self, d_out: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=pretrained
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        d_backbone = 384  # ViT-S/14 embed dim
        self.proj = nn.Linear(d_backbone, d_out)
        self.d_out = d_out
        self.patch_size = 14

    @torch.no_grad()
    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            patch_tokens = out["x_norm_patchtokens"]
        else:
            patch_tokens = out[:, 1:, :]
        return patch_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_feats = self._extract_patch_features(x)
        return self.proj(patch_feats)


class DINOv3VisualEncoder(nn.Module):
    """DINOv3 ViT-H+/16 frozen backbone + learnable linear projection.

    Supports loading from:
      - Local repo checkout  (source='local')
      - torch.hub            (source='github')
      - HuggingFace          (source='huggingface')

    Output: (B, N_patches, D_proj) patch-level features (NOT CLS).
    """

    # Known embed dims for each variant
    EMBED_DIMS = {
        "dinov3_vits16": 384,
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
        "dinov3_vith16plus": 1536,   # 840M params, ViT-H+/16
    }

    def __init__(
        self,
        model_name: str = "dinov3_vith16plus",
        d_out: int = 256,
        source: Literal["local", "github", "huggingface"] = "local",
        repo_dir: str | None = None,
        hf_model_id: str | None = None,
        weights_path: str | None = None,
    ):
        """Args:
            model_name: one of the keys in EMBED_DIMS (e.g. 'dinov3_vith16plus').
            d_out: projection output dimension.
            source: how to load the model.
                'local'       — torch.hub.load(..., source='local', repo_dir=...)
                'github'      — torch.hub.load(..., source='github')
                'huggingface' — load from HF AutoModel
            repo_dir: required when source='local'. Path to the cloned DINOv3 repo.
            hf_model_id: required when source='huggingface'. E.g.
                'facebook/dinov3-vith16plus-pretrain-lvd1689m'
            weights_path: optional path to a local .pt / .pth checkpoint.
        """
        super().__init__()
        self.model_name = model_name
        self.patch_size = 16

        # --- Load backbone ---
        if source == "huggingface":
            backbone = self._load_hf(hf_model_id or f"facebook/{model_name}-pretrain-lvd1689m")
        else:
            backbone = self._load_torch_hub(model_name, source, repo_dir, weights_path)

        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone

        # --- Auto-detect embed dim ---
        d_backbone = self.EMBED_DIMS.get(model_name)
        if d_backbone is None:
            # Infer from the model's embed_dim attribute or first linear layer
            d_backbone = self._infer_embed_dim(backbone)
            logger.info(f"Auto-detected embed dim for {model_name}: {d_backbone}")

        self.proj = nn.Linear(d_backbone, d_out)
        self.d_out = d_out
        logger.info(
            f"DINOv3VisualEncoder: {model_name}, embed_dim={d_backbone}, "
            f"proj→{d_out}, source={source}"
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_torch_hub(model_name, source, repo_dir, weights_path):
        kwargs = dict(source=source)
        if repo_dir:
            kwargs["repo_dir"] = repo_dir
        if weights_path:
            kwargs["weights"] = weights_path
        return torch.hub.load("facebookresearch/dinov3", model_name, **kwargs)

    @staticmethod
    def _load_hf(model_id):
        from transformers import AutoModel
        return AutoModel.from_pretrained(model_id, trust_remote_code=True)

    @staticmethod
    def _infer_embed_dim(model) -> int:
        """Try to read embed_dim from model config or first patch_embed layer."""
        if hasattr(model, "embed_dim"):
            return model.embed_dim
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            return model.config.hidden_size
        # Walk modules for the first Linear in patch_embed
        for name, mod in model.named_modules():
            if "patch_embed" in name and isinstance(mod, nn.Linear):
                return mod.out_features
        raise RuntimeError(
            f"Cannot infer embed dim for model. "
            f"Please pass model_name from EMBED_DIMS or inspect manually."
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-patch token features (drop CLS token).

        Args:
            x: (B, 3, H, W) — size must be divisible by patch_size (16).

        Returns:
            (B, N_patches, d_backbone)
        """
        out = self.backbone.forward_features(x)
        if isinstance(out, dict):
            # DINOv3 uses the same key convention as DINOv2
            if "x_norm_patchtokens" in out:
                return out["x_norm_patchtokens"]
            # Some HF wrappers return different keys
            for key in ("patch_tokens", "last_hidden_state"):
                if key in out:
                    tokens = out[key]
                    # HF last_hidden_state includes CLS at position 0
                    if tokens.shape[1] > 1:
                        return tokens[:, 1:, :]
                    return tokens
            raise KeyError(f"Unexpected output keys: {list(out.keys())}")
        else:
            # Tensor output: (B, 1+N_patches, D) — drop CLS
            return out[:, 1:, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)

        Returns:
            features: (B, N_patches, d_out)
        """
        patch_feats = self._extract_patch_features(x)
        return self.proj(patch_feats)


class MockVisualEncoder(nn.Module):
    """Mock encoder for testing without real backbone. Generates random features."""

    def __init__(self, d_out: int = 256, n_patches: int = 256):
        super().__init__()
        self.d_out = d_out
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        device = x.device
        return torch.randn(B, self.n_patches, self.d_out, device=device)
