evalscope eval \
 --model /data/nvme1/weights/Qwen3_sft_eval/output \
 --generation-config '{"max_new_tokens":2048,"chat_template_kwargs":{"enable_thinking": true}}' \
 --datasets ceval bbh \
 --dataset-args '{"ceval":{"local_path":"/data/nvme0/evaldata/data/ceval"},"bbh":{"local_path":"/data/nvme0/evaldata/data/bbh"}}' \
 --work-dir /data/nvme0/evaldata/qwen3-110k \
 --limit 3