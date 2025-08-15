# Video-BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation

<div align="center">

[📖 Paper](https://arxiv.org/abs/2508.10774) |  [🚀 Homepage](http://ziplab.co/BLADE-Homepage/) | [💾 Models](https://huggingface.co/GYP666/VIDEO-BLADE)

</div>

Video-BLADE是一个无需视频数据训练的高效视频生成框架，通过联合训练自适应稀疏注意力和步数蒸馏技术，实现了视频生成模型的显著加速。该项目实现了块稀疏注意力机制与步数蒸馏技术的结合，在保持生成质量的同时将推理步数从50步降低到8步。

## 📢 News

- **[2024-08]** 🎉 Video-BLADE代码和预训练模型发布！
- **[2024-08]** 📝 支持CogVideoX-5B和WanX-1.3B两种主流视频生成模型
- **[2024-08]** ⚡ 实现8步高质量视频生成，相比50步baseline显著提速

## ✨ 主要特性

- 🚀 **高效推理**: 将推理步数从50步减少到8步，保持生成质量
- 🎯 **自适应稀疏注意力**: 块稀疏注意力机制，显著降低计算复杂度
- 📈 **步数蒸馏**: TDM(Trajectory Distillation Method)技术，无需视频数据即可训练
- 🎮 **即插即用**: 支持CogVideoX-5B和WanX-1.3B模型，无需修改原始架构

## 🛠️ 环境配置

### 系统要求
- Python >= 3.11 (建议)
- CUDA >= 11.6 (建议)
- GPU内存 >= 24GB (推理)
- GPU内存 >= 80GB (训练)


### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/Tacossp/VIDEO-BLADE
cd VIDEO-BLADE
```

2. **安装依赖包**
```bash
# 使用uv安装依赖 (推荐)
uv pip install -r requirements.txt

# 或使用pip
pip install -r requirements.txt
```

3. **编译Block-Sparse-Attention库**
```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
cd ..
```

## 📥 模型权重下载

### 基础模型权重

请下载以下基础模型权重并放置在指定目录：

1. **CogVideoX-5B模型**
```bash
# 从Hugging Face下载
git lfs install
git clone https://huggingface.co/zai-org/CogVideoX-5b cogvideox/CogVideoX-5b
```

2. **WanX-1.3B模型**
```bash
# 从Hugging Face下载
git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers wanx/wan1.3b
```

### 预训练的Video-BLADE权重

我们提供了已经训练好的Video-BLADE权重：

```bash
# 下载预训练权重
git clone https://huggingface.co/GYP666/VIDEO-BLADE pretrained_weights
```

### 权重目录结构

确保您的权重目录结构如下：
```
VIDEO-BLADE/
├── cogvideox/
│   └── CogVideoX-5b/           # CogVideoX基础模型权重
├── wanx/
│   └── wan1.3b/               # WanX基础模型权重
└── pretrained_weights/         # Video-BLADE预训练权重
    ├── BLADE_cogvideox_weight/
    └── BLADe_wanx_weight/
```

## 🚀 快速开始 - 推理使用

### CogVideoX推理

```bash
cd cogvideox
python train/inference.py \
    --lora_path ../pretrained_weights/ 
    cogvideox_checkpoints/your_checkpoint \
    --gpu 0
```

**参数说明**:
- `--lora_path`: LoRA权重文件路径
- `--gpu`: 使用的GPU设备ID (默认: 0)

**输出**: 生成的视频将保存在 `cogvideox/outputs/inference/` 目录

### WanX推理

```bash
cd wanx
python train/inference.py \
    --lora_path ../pretrained_weights/wanx_checkpoints/your_checkpoint \
    --gpu 0
```

**输出**: 生成的视频将保存在 `wanx/outputs/` 目录

## 🔧 训练流程

### 第一步: Prompts预处理

在训练前，需要预处理提示词生成embeddings：

#### CogVideoX预处理
```bash
cd utils
python process_prompts_cogvideox.py \
    --input_file your_prompts.txt \
    --output_dir ../cogvideox/prompts \
    --model_path ../cogvideox/CogVideoX-5b \
    --batch_size 32 \
    --save_separate
```

**参数说明**:
- `--input_file`: 包含prompts的txt文件，每行一个prompt
- `--output_dir`: 输出embeddings的目录
- `--model_path`: CogVideoX模型路径
- `--batch_size`: 处理批次大小
- `--save_separate`: 是否将每个embedding单独保存

#### WanX预处理
```bash
cd utils
python process_prompts_wanx.py
```

此脚本会自动处理 `utils/all_dimension_aug_wanx.txt` 中的prompts并生成相应的embeddings。

### 第二步: 启动训练

#### CogVideoX训练

```bash
cd cogvideox
bash train_tdm_1.sh
```

**核心训练参数**:
```bash
#如果不是8卡训练需要修改CUDA_VISIBLE_DEVICES和config.yaml的num_processes
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 accelerate launch \
    --config_file train/config.yaml \
    train/train_cogvideo_tdm.py \
    --pretrained_model_name_or_path CogVideoX-5b \        # 基础模型路径
    --mixed_precision bf16 \                              # 混合精度训练，减少显存使用
    --train_batch_size 5 \                                # 训练批次大小
    --gradient_accumulation_steps 4 \                     # 梯度累积步数
    --learning_rate 1e-4 \                                # student学习率
    --learning_rate_g 1e-4 \                              
    --learning_rate_fake 5e-4 \                           # fake model学习率
    --lambda_reg 0.5 \                                    # 正则化权重
    --k_step 8 \                                          # 蒸馏目标步数
    --cfg 3.5 \                                           # CFG引导强度
    --eta 0.9 \                                           # ETA参数
    --use_sparsity true \                                 # 启用稀疏注意力
    --rank 64 \
    --lora_alpha 64 \                                     # LoRA配置
    --max_train_steps 300 \                               # 最大训练步数
    --checkpointing_steps 15 \                            # 检查点保存间隔
    --gradient_checkpointing \                            # 梯度检查点，节省显存
    --enable_slicing \
    --enable_tiling   \                                   # VAE内存优化
```

#### WanX训练

```bash
cd wanx
bash train_wanx_tdm.sh
```

## 📊 项目结构

```
VIDEO-BLADE/
├── README.md                   # 项目说明文档
├── requirements.txt           # Python依赖列表
│
├── cogvideox/                 # CogVideoX相关代码
│   ├── CogVideoX-5b/         # 基础模型权重目录
│   ├── train/                # 训练脚本
│   │   ├── inference.py      # 推理脚本
│   │   ├── train_cogvideo_tdm.py  # 训练脚本
│   │   ├── train_tdm_1.sh    # 训练启动脚本
│   │   ├── modify_cogvideo.py # 模型修改脚本
│   │   └── config.yaml       # 训练配置文件
│   ├── prompts/              # 预处理的prompts和embeddings
│   └── outputs/              # 训练和推理输出
│
├── wanx/                     # WanX相关代码  
│   ├── wan1.3b/             # 基础模型权重目录
│   ├── train/               # 训练脚本
│   │   ├── inference.py     # 推理脚本
│   │   ├── train_wanx_tdm.py # 训练脚本
│   │   ├── train_wanx_tdm.sh # 训练启动脚本
│   │   └── modify_wan.py    # 模型修改脚本
│   ├── prompts/             # 预处理的prompts和embeddings
│   └── outputs/             # 训练和推理输出
│
├── utils/                   # 工具脚本
│   ├── process_prompts_cogvideox.py  # CogVideoX数据预处理
│   ├── process_prompts_wanx.py       # WanX数据预处理
│   └── all_dimension_aug_wanx.txt    # WanX训练prompts
│
├── Block-Sparse-Attention/  # 稀疏注意力库
│   ├── setup.py            # 编译安装脚本
│   ├── block_sparse_attn/  # 核心库代码
│   └── README.md           # 库使用说明
│
└── ds_config.json          # DeepSpeed配置文件
```


## 🤝 致谢

- [FlashAttention](https://github.com/Dao-AILab/flash-attention),[Block-Sparse-Attention](https://github.com/mit-han-lab/Block-Sparse-Attention): 稀疏注意力实现基础
- [CogVideoX](https://github.com/THUDM/CogVideo),[Wan2.1](https://github.com/Wan-Video/Wan2.1): 模型支持
- [TDM](https://github.com/Luo-Yihong/TDM):蒸馏实现基础
- [Diffusers](https://github.com/huggingface/diffusers): 扩散模型工具库

## 📄 引用

如果您在研究中使用了Video-BLADE，请引用我们的工作：

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

## 📧 联系方式

如有问题或建议，欢迎：
- Please contact Youping Gu (youpgu71@gmail.com) if you have any questions about this work.
- 提交issue: [Github issue](https://github.com/Tacossp/VIDEO-BLADE/issues)
