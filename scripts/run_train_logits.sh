#!/bin/bash
cd /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/watermark-learnability

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate llmwatermark
# export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 nohup torchrun --nproc_per_node=6 train_logits_distill.py \
#   --model_name_or_path gpt2 \
#   --train_file /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/watermark-learnability/data/openwebtext_train_200000.txt \
#   --validation_file /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/watermark-learnability/data/openwebtext_test_100000.txt \
#   --do_train \
#   --do_eval \
#   --output_dir ./output \
#   --per_device_train_batch_size 4 \
#   --per_device_eval_batch_size 4 \
#   --num_train_epochs 1 \
#   --save_steps 500 \
#   --watermark_type kgw \
#   --argmax_watermark False \
#   --fsdp no_shard \
# > logs/train_logits_distill_torchrun.log 2>&1 &

nohup torchrun --nproc_per_node=6 train_logits_distill.py \
   --model_name_or_path /home/yijiexu/LLMWatermark/WM-Open-Source-LLM/models/Llama-2-7b-hf/ \
   --dataset_name openwebtext \
   --streaming \
   --per_device_train_batch_size 24 \
   --gradient_accumulation_steps 1 \
   --do_train \
   --max_steps 5000 \
   --logging_steps 1 \
   --output_dir output_fsdp/ \
   --learning_rate 1e-5 \
   --custom_cosine_lr_scheduler False \
   --lr_scheduler_type "cosine" \
   --warmup_steps 500 \
   --block_size 512 \
   --save_steps 1000 \
   --save_total_limit 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
   --tf32 True \
   --bf16 True \
   --gradient_checkpointing \
   --watermark_type "kgw" \
   --kgw_watermark_gamma 0.25 \
   --kgw_watermark_delta 2.0 \
   --argmax_watermark False \
  > logs/train_logits_distill_torchrun.log 2>&1 &