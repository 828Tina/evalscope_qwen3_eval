# 24GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /data/nvme1/weights/Qwen3-0.6B \
    --train_type lora \
    --dataset ./data/train.jsonl\
    --val_dataset ./data/eval.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir /data/nvme1/weights/Qwen3_sft_eval \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name qwen3 \
    --report_to swanlab \
    --swanlab_project swift-qwen3
