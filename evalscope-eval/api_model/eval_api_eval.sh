evalscope eval \
    --model qwen3-sft \
    --generation-config '{"max_tokens":4096,"temperature":0.1}'\
    --eval-type service \
    --api-url http://127.0.0.1:25001/v1/chat/completions \
    --datasets  gsm8k \
    --dataset-args '{"gsm8k":{"local_path":"/data/nvme0/evaldata/data/gsm8k"}}' \
    --limit 20 \
    --work-dir /data/nvme0/evaldata/qwen3-api-eval