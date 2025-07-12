base_model='Qwen3-4B'
evaluate(){
    source /home/fine/uv/transformers/bin/activate && \
    CUDA_VISIBLE_DEVICES=7 \
    python ./evaluate-cls.py \
        --model_path "/data/download-model/${base_model}" \
        --lora_path "model/${base_model}/cls" 
}

for i in {1..5};do
    echo "第${i}次验证"
    evaluate $1
done