download_dir=$1

cd ${download_dir}/TKEval

echo ">> start processing mmlu"
for type in permute noise
do
    for ngram in 2 3 5
    do
        cd typographical_variation/data.typo.char.${type}/ngram_${ngram}/mmlu_no_train
        sed -i 's#archive = .*#archive = "'`pwd`'/data.tar"#g' mmlu_no_train.py
        cd -
    done
done

for type in permute noise
do
    for model in mistral llama3
    do
        for ngram in 2 3 5
        do
            cd typographical_variation/data.typo.token.${type}/${model}/ngram_${ngram}/mmlu_no_train
            sed -i 's#archive = .*#archive = "'`pwd`'/data.tar"#g' mmlu_no_train.py
            cd -
        done
    done
done

echo ">> end processing mmlu"