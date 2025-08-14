# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Video-BLADE is a data-free framework for efficient video generation through joint training of adaptive sparse attention and step distillation. The project implements block-sparse attention mechanisms combined with step distillation techniques to accelerate video generation models while maintaining quality.

## Project Structure

The repository contains two main video generation model implementations:

- **CogVideoX**: Located in `cogvideox/` - Implementation for CogVideoX-5B models with adaptive sparse attention
- **WanX**: Located in `wanx/` - Implementation for Wan2.1-1.3B models with similar optimizations

## Key Training Commands

### CogVideoX Training

```bash
# Navigate to CogVideoX training directory
cd cogvideox/train

# Run TDM (Trajectory Distillation Method) training
bash train_tdm_1.sh

# Or use accelerate directly
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29556 --config_file config.yaml train_tdm_tcd.py \
    --pretrained_model_name_or_path CogVideoX-5b \
    --use_sparsity true
```

### WanX Training

```bash
# Navigate to WanX directory
cd wanx

# Run WanX training
bash train_wanx_1.3b.sh
```

## Development Environment

### Dependencies Installation

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Key dependencies include:
# - SwissArmyTransformer>=0.4.12
# - pytorch_lightning>=2.4.0
# - diffusers (custom version with Video-BLADE modifications)
# - deepspeed>=0.15.3
# - wandb>=0.18.5
```

### GPU Requirements

The project requires significant GPU memory:
- Uses DeepSpeed ZeRO-3 optimization with CPU offloading
- Configured for multi-GPU training (typically 2+ GPUs)
- Mixed precision training (bf16) enabled by default

## Architecture Components

### Block Sparse Attention

Core sparse attention implementations located in:
- `cogvideox/train/special_attenions/src/special_attentions/TrainRelated/blocksparseattn.py`
- `wanx/train/special_attentions_local/TrainRelated/blocksparseattn.py`

Key parameters:
- `max_retain_ratio=0.15`, `min_retain_ratio=0.05`
- Block size and sampling configurations for efficient attention computation

### Model Modifications

- `cogvideox/train/modify_cogvideo.py` - Modifications to CogVideoX transformer
- `wanx/train/modify_wan.py` - Modifications to WanX transformer

### Training Pipeline

Both models use TDM (Trajectory Distillation Method) with:
- LoRA fine-tuning for efficient parameter updates
- Step distillation from 50 steps to 8 steps
- Adaptive sparse attention during training

## Configuration Files

### DeepSpeed Configuration
- `ds_config.json` - DeepSpeed ZeRO-3 configuration with CPU offloading
- `cogvideox/train/config.yaml` - Accelerate configuration for distributed training

### Training Parameters
Key training settings across models:
- Learning rates: 1e-4 to 5e-4 depending on component
- Batch size: 1 with gradient accumulation (typically 4 steps)
- Mixed precision: bf16
- Gradient clipping: 1.0

## Inference

### Video Generation

```bash
# Run inference using trained models
cd utils
python inference.py

# Models support:
# - 8-step generation (vs 50-step baseline)
# - Configurable guidance scales
# - Multiple output formats
```

### Model Outputs
Generated videos are saved in structured directories under `All_videos/`:
- `CogvideoX-5B_extracted_videos/` - CogVideoX results
- `wan2.1_extracted_videos/` - WanX results
- Subdirectories for different methods (ASA, TDM, baseline)

## Development Notes

### Memory Optimization
- VAE slicing and tiling enabled for memory efficiency
- Gradient checkpointing activated
- CPU offloading for optimizer and parameters

### Attention Mechanisms
The repository implements several attention variants:
- Block sparse attention with adaptive retention ratios
- Gilbert curve-based 3D attention patterns
- Pooling-based attention kernels for efficiency

### Model Modifications
When modifying attention mechanisms:
1. Update the corresponding `modify_*.py` files
2. Rebuild the `special_attentions` package if needed
3. Ensure compatibility with both training and inference pipelines

## Testing and Validation

No automated testing framework is currently implemented. Validation is performed through:
- Visual inspection of generated videos
- Comparison with baseline 50-step generation
- Quality metrics evaluation during training