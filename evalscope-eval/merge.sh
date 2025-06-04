CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters /data/nvme1/weights/Qwen3_sft_eval/v5-20250603-151411/checkpoint-604 \
    --model /data/nvme1/weights/Qwen3-0.6B \
    --output_dir /data/nvme1/weights/Qwen3_sft_eval/output \
    --merge_lora true