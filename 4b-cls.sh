base_model='Qwen3-4B'

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./cls.py \
    --model_path "/data/download-model/${base_model}" \
    --max_length 4096

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./evaluate-cls.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "model/${base_model}/cls"