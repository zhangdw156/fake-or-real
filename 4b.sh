base_model='Qwen3-4B'

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./sft.py \
    --model_path "/data/download-model/${base_model}"

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./evaluate.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "mode/${base_model}/sft"