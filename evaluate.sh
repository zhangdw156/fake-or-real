base_model='Qwen3-0.6B'

evaluate(){
    source /home/fine/uv/transformers/bin/activate && \
    CUDA_VISIBLE_DEVICES=7 \
    python ./evaluate-$1.py \
        --model_path "/data/download-model/${base_model}" \
        --lora_path "model/${base_model}/$1" 
}

for i in {1..5};do
    echo "第${i}次验证"
    evaluate $1
done