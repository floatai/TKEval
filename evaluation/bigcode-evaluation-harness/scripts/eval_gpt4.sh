
set -x

for file in `ls data`
do
    python scripts/process_gpt4_for_eval.py data/${file} data_processed/${file}
done

accelerate launch  main.py \
    --tasks humaneval \
    --allow_code_execution \
    --load_generations_path data_processed/humaneval_gpt4-turbo_output.jsonl \
    --model gpt4_turbo > log/humaneval.gpt4_turbo.log 2>&1

for i in "perm" "noise"
do
    for j in "2" "3" "5"
    do
        accelerate launch  main.py \
            --tasks humaneval \
            --allow_code_execution \
            --load_generations_path data_processed/humaneval-${i}_${j}-gram_gpt4-turbo_output.jsonl \
            --model gpt4_turbo > log/humaneval.gpt4_turbo.${i}.${j}_gram.log 2>&1
    done
done