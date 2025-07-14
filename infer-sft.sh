base_model='Qwen3-8B'
job="sft2"
source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=0 \
python ./infer-sft.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "model/${base_model}/$job" \
    --job_name "$job"