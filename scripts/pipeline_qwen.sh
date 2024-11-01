# Modify the base url to your vllm server
BASE_URL=http://localhost:8000/v1

for LANG1 in cpp java python; do
    for LANG2 in cpp java python; do
        if [[ ${LANG1} == ${LANG2} ]]; then
            continue
        fi
        python pipeline.py \
            --model-name-or-path Qwen/Qwen2.5-72B-Instruct \
            --tokenizer Qwen/Qwen2.5-72B-Instruct \
            --temperature 0.2 \
            --top-p 0.95 \
            --max-tokens 8192 \
            --base-url ${BASE_URL} \
            --api-key RANDOM \
            --input-path data/LongTrans/test/${LANG1}_w_tests.json \
            --output-path data/PAST/Qwen2.5-72B-Instruct/test/${LANG1}-${LANG2}.json \
            --lang1 ${LANG1} \
            --lang2 ${LANG2} \
            --num-proc 128
    done
done