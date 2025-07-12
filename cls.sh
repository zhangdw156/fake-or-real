source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=7 \
python ./cls.py \
    --model_path "/data/download-model/Qwen3-0.6B" \
    --max_length 4096