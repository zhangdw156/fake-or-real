base_model="Qwen3-8B"

job="sft2"
data_path="data/train_processed2.json"

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./sft.py \
    --model_path "/data/download-model/${base_model}" \
    --data_path "${data_path}" \
    --job_name "$job" \
    --epochs 10 \
    --batch_size 1

./evaluate-sft.sh "$job" 1 "${data_path}" "${base_model}"

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./infer-sft.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "model/${base_model}/$job" \
    --job_name "$job"