# Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation

<div align="center">

[📖 Paper](https://arxiv.org/abs/2508.10774) | [🚀 Homepage](http://ziplab.co/BLADE-Homepage/) | [💾 Models](https://huggingface.co/GYP666/VIDEO-BLADE) | [📖 中文阅读](README_zh.md)  

</div>

Video-BLADE is a data-free framework for efficient video generation. By jointly training an adaptive sparse attention mechanism with a step distillation technique, it achieves a significant acceleration in video generation models. This project combines a block-sparse attention mechanism with step distillation, reducing the number of inference steps from 50 to just 8 while maintaining high-quality generation.

## 📢 News

  - **[Aug 2024]** 🎉 The code and pre-trained models for Video-BLADE have been released\!
  - **[Aug 2024]** 📝 Support for two mainstream video generation models, CogVideoX-5B and WanX-1.3B, is now available.
  - **[Aug 2024]** ⚡ Achieved high-quality video generation in just 8 steps, a significant speedup compared to the 50-step baseline.

## ✨ Key Features

  - 🚀 **Efficient Inference**: Reduces the number of inference steps from 50 to 8 while preserving generation quality.
  - 🎯 **Adaptive Sparse Attention**: Employs a block-sparse attention mechanism to significantly reduce computational complexity.
  - 📈 **Step Distillation**: Utilizes the Trajectory Distillation Method (TDM), enabling training without the need for video data.
  - 🎮 **Plug-and-Play**: Supports CogVideoX-5B and WanX-1.3B models without requiring modifications to their original architectures.

## 🛠️ Environment Setup

### System Requirements

  - Python \>= 3.11 (Recommended)
  - CUDA \>= 11.6 (Recommended)
  - GPU Memory \>= 24GB (for Inference)
  - GPU Memory \>= 80GB (for Training)

### Installation Steps

1.  **Clone the repository**

    ```bash
    git clone https://github.com/Tacossp/VIDEO-BLADE
    cd VIDEO-BLADE
    ```

2.  **Install dependencies**

    ```bash
    # Install using uv (Recommended)
    uv pip install -r requirements.txt

    # Or use pip
    pip install -r requirements.txt
    ```

3.  **Compile the Block-Sparse-Attention library**

    ```bash
    git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
    cd Block-Sparse-Attention
    pip install packaging
    pip install ninja
    python setup.py install
    cd ..
    ```

## 📥 Model Weights Download

### Base Model Weights

Please download the following base model weights and place them in the specified directories:

1.  **CogVideoX-5B Model**

    ```bash
    # Download from Hugging Face
    git lfs install
    git clone https://huggingface.co/zai-org/CogVideoX-5b cogvideox/CogVideoX-5b
    ```

2.  **WanX-1.3B Model**

    ```bash
    # Download from Hugging Face
    git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers wanx/wan1.3b
    ```

### Pre-trained Video-BLADE Weights

We provide pre-trained weights for Video-BLADE:

```bash
# Download pre-trained weights
git clone https://huggingface.co/GYP666/VIDEO-BLADE pretrained_weights
```

### Weight Directory Structure

Ensure your directory structure for weights is as follows:

```
VIDEO-BLADE/
├── cogvideox/
│   └── CogVideoX-5b/           # Base model weights for CogVideoX
├── wanx/
│   └── wan1.3b/               # Base model weights for WanX
└── pretrained_weights/         # Pre-trained weights for Video-BLADE
    ├── BLADE_cogvideox_weight/
    └── BLADE_wanx_weight/
```

## 🚀 Quick Start - Inference

### CogVideoX Inference

```bash
cd cogvideox
python train/inference.py \
    --lora_path ../pretrained_weights/cogvideox_checkpoints/your_checkpoint \
    --gpu 0
```

**Argument Descriptions**:

  - `--lora_path`: Path to the LoRA weights file.
  - `--gpu`: The ID of the GPU device to use (Default: 0).

**Output**: The generated videos will be saved in the `cogvideox/outputs/inference/` directory.

### WanX Inference

```bash
cd wanx
python train/inference.py \
    --lora_path ../pretrained_weights/wanx_checkpoints/your_checkpoint \
    --gpu 0
```

**Output**: The generated videos will be saved in the `wanx/outputs/` directory.

## 🔧 Training Process

### Step 1: Prompt Preprocessing

Before training, you need to preprocess the text prompts to generate embeddings.

#### CogVideoX Preprocessing

```bash
cd utils
python process_prompts_cogvideox.py \
    --input_file your_prompts.txt \
    --output_dir ../cogvideox/prompts \
    --model_path ../cogvideox/CogVideoX-5b \
    --batch_size 32 \
    --save_separate
```

**Argument Descriptions**:

  - `--input_file`: A `.txt` file containing prompts, with one prompt per line.
  - `--output_dir`: The directory to save the output embeddings.
  - `--model_path`: Path to the CogVideoX model.
  - `--batch_size`: The batch size for processing.
  - `--save_separate`: Whether to save each embedding as a separate file.

#### WanX Preprocessing

```bash
cd utils
python process_prompts_wanx.py
```

This script will automatically process the prompts in `utils/all_dimension_aug_wanx.txt` and generate the corresponding embeddings.

### Step 2: Start Training

#### CogVideoX Training

```bash
cd cogvideox
bash train_tdm_1.sh
```

**Core Training Parameters**:

```bash
# If not training with 8 GPUs, you must modify CUDA_VISIBLE_DEVICES and the num_processes in config.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch \
    --config_file train/config.yaml \
    train/train_cogvideo_tdm.py \
    --pretrained_model_name_or_path CogVideoX-5b \        # Path to the base model
    --mixed_precision bf16 \                              # Use mixed-precision for reduced memory usage
    --train_batch_size 5 \                                # Training batch size
    --gradient_accumulation_steps 4 \                     # Number of gradient accumulation steps
    --learning_rate 1e-4 \                                # Learning rate for the student model
    --learning_rate_g 1e-4 \                              
    --learning_rate_fake 5e-4 \                           # Learning rate for the fake model
    --lambda_reg 0.5 \                                    # Regularization weight
    --k_step 8 \                                          # Target number of steps for distillation
    --cfg 3.5 \                                           # Classifier-Free Guidance scale
    --eta 0.9 \                                           # ETA parameter for DDIM
    --use_sparsity true \                                 # Enable sparse attention
    --rank 64 \
    --lora_alpha 64 \                                     # LoRA configuration
    --max_train_steps 300 \                               # Maximum number of training steps
    --checkpointing_steps 15 \                            # Interval for saving checkpoints
    --gradient_checkpointing \                            # Use gradient checkpointing to save memory
    --enable_slicing \
    --enable_tiling                                       # VAE memory optimization
```

#### WanX Training

```bash
cd wanx
bash train_wanx_tdm.sh
```

## 📊 Project Structure

```
VIDEO-BLADE/
├── README.md                   # Project documentation
├── requirements.txt           # List of Python dependencies
│
├── cogvideox/                 # Code related to CogVideoX
│   ├── CogVideoX-5b/         # Directory for base model weights
│   ├── train/                # Training scripts
│   │   ├── inference.py      # Inference script
│   │   ├── train_cogvideo_tdm.py  # Training script
│   │   ├── train_tdm_1.sh    # Script to launch training
│   │   ├── modify_cogvideo.py # Model modification script
│   │   └── config.yaml       # Training configuration file
│   ├── prompts/              # Preprocessed prompts and embeddings
│   └── outputs/              # Output from training and inference
│
├── wanx/                     # Code related to WanX
│   ├── wan1.3b/             # Directory for base model weights
│   ├── train/               # Training scripts
│   │   ├── inference.py     # Inference script
│   │   ├── train_wanx_tdm.py # Training script
│   │   ├── train_wanx_tdm.sh # Script to launch training
│   │   └── modify_wan.py    # Model modification script
│   ├── prompts/             # Preprocessed prompts and embeddings
│   └── outputs/             # Output from training and inference
│
├── utils/                   # Utility scripts
│   ├── process_prompts_cogvideox.py  # Data preprocessing for CogVideoX
│   ├── process_prompts_wanx.py       # Data preprocessing for WanX
│   └── all_dimension_aug_wanx.txt    # Training prompts for WanX
│
├── Block-Sparse-Attention/  # Sparse attention library
│   ├── setup.py            # Compilation and installation script
│   ├── block_sparse_attn/  # Core library code
│   └── README.md           # Library usage instructions
│
└── ds_config.json          # DeepSpeed configuration file
```

## 🤝 Acknowledgements

  - [FlashAttention](https://github.com/Dao-AILab/flash-attention), [Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention): For the foundational work on sparse attention.
  - [CogVideoX](https://github.com/THUDM/CogVideo), [Wan2.1](https://github.com/Wan-Video/Wan2.1): For the supported models.
  - [TDM](https://www.google.com/search?q=https://github.com/Luo-Yihong/TDM): For the foundational work on distillation implementation.
  - [Diffusers](https://github.com/huggingface/diffusers): For the invaluable diffusion models library.

## 📄 Citation

If you use Video-BLADE in your research, please cite our work:

```bibtex
@misc{gu2025videobladeblocksparseattentionmeets,
    title={Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation},
    author={Youping Gu and Xiaolong Li and Yuhao Hu and Bohan Zhuang},
    year={2025},
    eprint={2508.10774},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2508.10774},
}
```

## 📧 Contact

For any questions or suggestions, feel free to:

  - Contact Youping Gu at youpgu71@gmail.com.
  - Submit an issue on our [Github page](https://github.com/Tacossp/VIDEO-BLADE/issues).