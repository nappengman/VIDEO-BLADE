#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29556 --config_file train/config.yaml train/train_cogvideo_tdm.py  \
    --pretrained_model_name_or_path CogVideoX-5b \
    --validation_prompt "<ID_TOKEN> Spiderman swinging over buildings::: A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance" \
    --validation_prompt_separator ::: \
    --num_validation_videos 1 \
    --validation_epochs 10 \
    --seed 42 \
    --rank 64 \
    --lora_alpha 64 \
    --mixed_precision bf16 \
    --output_dir outputs/cogvideox/8.11/bs80-lr=1e-4 \
    --height 480 --width 720 --fps 8 --max_num_frames 49 --skip_frames_start 0 --skip_frames_end 0 \
    --train_batch_size 5 \
    --max_train_steps 300 \
    --num_train_epochs 30 \
    --checkpointing_steps 15 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --learning_rate_g 1e-4 \
    --learning_rate_fake 5e-4 \
    --lr_scheduler cosine_with_restarts \
    --lr_warmup_steps 5 \
    --lr_num_cycles 1 \
    --enable_slicing \
    --enable_tiling \
    --optimizer AdamW \
    --adam_beta1 0. \
    --adam_beta2 0.95 \
    --max_grad_norm 1.0 \
    --lambda_reg 0.5 \
    --k_step 8 \
    --cfg 3.5 \
    --eta 0.9 \
    --use_sparsity true \
   

# 以下参数已被注释掉
# 如需使用，请取消注释并将上面的命令末尾加上 \ 符号
# --pretrained_lora_model_name_or_path /root/autodl-tmp/tdm_weight/5.21/cogvideox5b-tdm-base-rearrenge=True-lr=1e-4_lambda-reg_0.5_cfg_3.5_eta_0.9_K_8/checkpoint-250
# --resume_from_checkpoint /root/autodl-tmp/tdm_weight/5.21/cogvideox5b-tdm-base-rearrenge=True-lr=1e-4_lambda-reg_0.5_cfg_3.5_eta_0.9_K_8/checkpoint-250
    