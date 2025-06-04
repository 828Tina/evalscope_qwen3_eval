export CUDA_VISIBLE_DEVICES=0
evalscope perf \
 --model '/data/nvme1/weights/Qwen3_sft_eval/output' \
 --dataset openqa \
 --number 20 \
 --parallel 2 \
 --swanlab-api-key 'your swanlab api key' \
 --name 'qwen3-openqa' \
 --temperature 0.9 \
 --outputs-dir '/data/nvme0/evaldata/qwen3-perf' \
 --api local