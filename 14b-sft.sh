base_model='Qwen3-14B'

source /home/fine/uv/transformers/bin/activate && \
accelerate launch \
    --config_file "accelerate_zero2_config.yaml" \
    --main_process_port 0 \
    ./sft.py \
        --model_path "/data/download-model/${base_model}"

source /home/fine/uv/transformers/bin/activate && \
accelerate launch \
    --config_file "accelerate_zero2_config.yaml" \
    --main_process_port 0 \
    ./evaluate-sft.py \
        --model_path "/data/download-model/${base_model}" \
        --lora_path "model/${base_model}/sft"