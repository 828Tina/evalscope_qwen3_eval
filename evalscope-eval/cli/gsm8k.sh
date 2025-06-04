CUDA_VISIBLE_DEVICES=1
evalscope eval \
 --model /data/nvme1/weights/Qwen3_sft_eval/output \
 --generation-config '{"max_new_tokens":2048,"chat_template_kwargs":{"enable_thinking": false}}' \
 --datasets gsm8k \
 --dataset-args '{"gsm8k":{"local_path":"/data/nvme0/evaldata/data/gsm8k"}}' \
 --work-dir /data/nvme0/evaldata/qwen3-110k \
 --limit 100