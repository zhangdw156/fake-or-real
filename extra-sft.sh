base_model="Qwen3-0.6B"
job="merged-sft"
data_path="data/train_processed.json"

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./extra-sft.py \
    --model_path "/data/download-model/${base_model}" \
    --data_path "${data_path}" \
    --job_name "$job" \
    --epochs 50 \
    --batch_size 8 \
    --lora \
    --lora_path "model/${base_model}/extra-sft"