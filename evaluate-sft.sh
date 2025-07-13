base_model="Qwen3-4B"
data_path="data/train_processed.json"

evaluate(){
    source /home/fine/uv/transformers/bin/activate && \
    CUDA_VISIBLE_DEVICES=7 \
    python ./evaluate-sft.py \
        --model_path "/data/download-model/${base_model}" \
        --lora_path "model/${base_model}/$job" \
        --data_path "${data_path}"
}


job="${1:=sft}"
count="${2:=1}"

for i in $(seq 1 $count);do
    echo "第${i}次验证"
    evaluate "$job"
done