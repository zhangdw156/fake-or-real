base_model='Qwen3-8B'

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./sft.py \
    --model_path "/data/download-model/${base_model}"

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./evaluate-sft.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "model/${base_model}/sft"