evalscope eval \
 --model /data/nvme1/weights/Qwen3_sft_eval/output \
 --generation-config '{"max_new_tokens":2048,"chat_template_kwargs":{"enable_thinking": false}}' \
 --datasets gsm8k \
 --limit 5