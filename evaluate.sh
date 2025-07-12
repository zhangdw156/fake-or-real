source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./evaluate.py \
    --model_path "/data/download-model/Qwen3-0.6B" \
    --lora_path "mode/Qwen3-0.6B/sft"