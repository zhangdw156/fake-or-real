set -x

base_model="Qwen3-8B"

job="sft"
  
# for i in $(seq 1 3);do
#     data_path="data/train_processed${i}.json"
#     source /home/fine/uv/transformers/bin/activate && \
#     CUDA_VISIBLE_DEVICES=0 \
#     python ./sft.py \
#         --model_path "/data/download-model/${base_model}" \
#         --data_path "${data_path}" \
#         --job_name "${job}-${i}" \
#         --epochs 10 \
#         --batch_size 1 \
#         --lr 5e-5

#     ./evaluate-sft.sh "${job}-${i}" 1 "${data_path}" "${base_model}"
# done

source /home/fine/uv/transformers/bin/activate && \
CUDA_VISIBLE_DEVICES=0 \
python ./infer-sft.py \
    --model_path "/data/download-model/${base_model}" \
    --lora_path "model/${base_model}/${job}-1" \
    --lora_path "model/${base_model}/${job}-2" \
    --job_name "$job"