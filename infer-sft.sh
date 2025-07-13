base_model='Qwen3-4B'
source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./infer-sft.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "model/${base_model}/sft"