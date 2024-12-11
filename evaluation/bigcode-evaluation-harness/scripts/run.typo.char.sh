
set -x

pretrained_model_path=$1
max_length_generation=$2
model_mark=$3
model_name=$4
scramble_type=$5
ngram=$6
big_model=$7

rm -rf ~/.cache/huggingface/datasets/openai_humaneval/

sed -i 's#DATASET_PATH =.*#DATASET_PATH = "/mnt/data/user/tc_agi/yh/datasets_mix/token_prob_0612/data.'${scramble_type}'/'${model_mark}'/ngram_'${ngram}'/openai_humaneval"#g' bigcode_eval/tasks/humaneval.py

if [ ${big_model} -eq "0" ];then
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
        --save_generations_path /data/checkpoints/tk_probe/results/${model_name}.${scramble_type}.ngram_${ngram}.jsonl
else
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
        --save_generations_path /data/checkpoints/tk_probe/results/${model_name}.${scramble_type}.ngram_${ngram}.jsonl
fi