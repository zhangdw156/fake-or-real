base_model='Qwen3-0.6B'

evaluate(){
    source /home/fine/uv/transformers/bin/activate && \
    CUDA_VISIBLE_DEVICES=7 \
    python ./evaluate-$1.py \
        --model_path "/data/download-model/${base_model}" \
        --lora_path "model/${base_model}/$1" \
        --data_path 'data/extra_train_processed.json'
}


job="${1:=sft}"
count="${2:=1}"

for i in $(seq 1 $count);do
    echo "第${i}次验证"
    evaluate "$job"
done