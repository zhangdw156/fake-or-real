#!/bin/bash 



evaluate(){
    source /home/fine/uv/transformers/bin/activate && \
    CUDA_VISIBLE_DEVICES=7 \
    python ./evaluate-sft.py \
        --model_path "/data/download-model/${base_model}" \
        --lora_path "model/${base_model}/$job" \
        --data_path "${data_path}"
}


if [ -z "$1" ];then
    job="sft"
else
    job="$1"
fi

if [ -z "$2" ];then
    count=1
else
    count="$2"
fi

if [ -z "$3" ];then
    data_path="data/train_processed.json"
else
    data_path="$3"
fi

if [ -z "$4" ];then
    base_model="Qwen3-8B"
else
    base_model="$4"
fi

for i in $(seq 1 $count);do
    echo "第${i}次验证"
    evaluate "$job"
done