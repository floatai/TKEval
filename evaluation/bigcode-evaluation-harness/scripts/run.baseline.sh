
set -x

pretrained_model_path=$1
max_length_generation=$2
model_mark=$3
big_model=$4

rm -rf ~/.cache/huggingface/datasets/openai_humaneval/

sed -i 's#DATASET_PATH =.*#DATASET_PATH = "/mnt/data/user/tc_agi/yh/datasets/openai_humaneval/"#g' bigcode_eval/tasks/humaneval.py

if [ ${big_model} -eq "0" ];then
    python -u main.py \
        --model ${pretrained_model_path} \
        --tasks humaneval \
        --max_length_generation ${max_length_generation} \
        --temperature 0.2 \
        --do_sample True \
        --n_samples 1 \
        --batch_size 1 \
        --allow_code_execution \
        --save_generations \
        --precision bf16 \
        --max_memory_per_gpu auto \
        --save_generations_path /data/checkpoints/tk_probe/results/baseline/${model_mark}.humaneval.baseline.jsonl
else
    accelerate launch main.py \
        --model ${pretrained_model_path} \
        --tasks humaneval \
        --max_length_generation ${max_length_generation} \
        --temperature 0.2 \
        --do_sample True \
        --n_samples 1 \
        --batch_size 1 \
        --allow_code_execution \
        --save_generations \
        --precision bf16 \
        --max_memory_per_gpu auto \
        --save_generations_path /data/checkpoints/tk_probe/results/baseline/${model_mark}.humaneval.baseline.jsonl
fi