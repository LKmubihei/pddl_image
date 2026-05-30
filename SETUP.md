# PaQ Server Setup Guide

## 1. System Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA GPU with ≥24GB VRAM (A100/A6000 recommended) |
| CUDA | ≥11.8 |
| Python | 3.10+ |
| PyTorch | ≥2.7.1 |
| Disk | ≥10GB free (models + data) |
| Blender | 3.0.0 (for ViPlan rendering) |

## 2. Clone Repositories

```bash
# Main project
git clone <your-repo-url> ~/PDDL
cd ~/PDDL

# ViPlan (Blocksworld dataset + renderer)
git clone https://github.com/your-org/ViPlan.git
# Follow ViPlan's own setup for Blender rendering if needed
```

## 3. Python Environment

```bash
conda create -n paq python=3.10 -y
conda activate paq

# PyTorch with CUDA (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Verify torch >= 2.7.1
python -c "import torch; print(torch.__version__)"
```

## 4. DINOv3 ViT-H+/16 Setup

DINOv3 weights require access approval. Follow these steps:

### a) Request Access

1. Go to https://github.com/facebookresearch/dinov3
2. Find the model weights link (e.g., HuggingFace or direct download)
3. Request access and download the `dinov3_vith16plus` checkpoint

### b) Option A: Local Repo (Recommended)

```bash
# Clone DINOv3 repo
git clone https://github.com/facebookresearch/dinov3.git ~/dinov3

# Place the downloaded checkpoint somewhere accessible
# e.g., ~/dinov3/weights/dinov3_vith16plus.pt
```

Then pass `--dinov3-repo-dir ~/dinov3` when training.

### c) Option B: HuggingFace

```bash
pip install transformers

# The HF model ID is:
# facebook/dinov3-vith16plus-pretrain-lvd1689m
# Access may need to be requested on the HF model page.
```

### d) Option C: torch.hub (GitHub)

```python
# This will auto-download if the repo is public:
model = torch.hub.load("facebookresearch/dinov3", "dinov3_vith16plus")
```

### DINOv3 Model Specs

| Property | Value |
|----------|-------|
| Architecture | ViT-H+/16 |
| Parameters | 840M |
| Embed dim | 1536 |
| Patch size | 16×16 |
| Input size | 224×224 (resize 256, center crop) |
| Pretraining | LVD-1689M dataset |
| Output | 196 patch tokens (14×14) |

## 5. Install PaQ Dependencies

```bash
cd ~/PDDL
pip install -e .
# Or install manually:
pip install numpy Pillow torchvision tqdm
```

## 6. ViPlan Blender Setup (for Rendering Only)

If you need to render new images (not using pre-rendered data):

```bash
# Download Blender 3.0.0
cd ~/PDDL/ViPlan
wget https://download.blender.org/release/3.0/blender-3.0.0-linux-x64.tar.xz
tar xf blender-3.0.0-linux-x64.tar.xz

# Verify
./blender-3.0.0-linux-x64/blender --version
```

> If you already have pre-extracted features, you can skip rendering entirely.

## 7. Running Training

```bash
cd ~/PDDL
conda activate paq

# Full pipeline: render + extract + train
python training/train_viplan.py \
    --dinov3-repo-dir ~/dinov3 \
    --epochs 200 \
    --max-render 0 \
    --output-dir experiments/viplan_dinov3_v1

# With pre-rendered images (skip rendering if images exist)
python training/train_viplan.py \
    --dinov3-repo-dir ~/dinov3 \
    --epochs 200

# Quick test with fewer renders
python training/train_viplan.py \
    --dinov3-repo-dir ~/dinov3 \
    --max-render 50 \
    --epochs 50
```

## 8. Project Structure

```
PDDL/
├── paq/
│   ├── model.py              # PaQModel (supports DINOv2/DINOv3/Mock encoders)
│   ├── visual_encoder.py     # VisualEncoder (DINOv2), DINOv3VisualEncoder, MockVisualEncoder
│   ├── scoring_head.py       # Type-aware canonical predicate scoring
│   ├── slot_attention.py     # Dual-level slot attention (object + predicate)
│   ├── predicate_query_encoder.py
│   └── losses.py
├── training/
│   ├── train_viplan.py       # ViPlan Blocksworld pipeline (DINOv3 + slot attention)
│   └── train_auto.py         # TV screw assembly pipeline
├── solver/
│   ├── domain.pddl
│   └── p_real.pddl
├── SETUP.md
└── experiments/              # Output directory
```

## 9. Troubleshooting

### DINOv3 import error
```
ModuleNotFoundError: No module named 'dinov3'
```
→ Ensure you cloned the DINOv3 repo and passed `--dinov3-repo-dir` correctly.

### CUDA OOM with ViT-H+/16
The 840M backbone uses ~3.4GB in fp32. If OOM:
- Use `batch_size=4` or lower in `extract_dinov3_patch_features`
- Use fp16: wrap encoder in `torch.amp.autocast('cuda')`

### Blender rendering fails
- Verify Blender 3.0.0 is in `ViPlan/blender-3.0.0-linux-x64/`
- Check GPU is available for CYCLES: `./blender --background --python-expr "import bpy; print(bpy.context.preferences.addons)"`
