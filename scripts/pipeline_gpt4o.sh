# Add your gpt api-key here
API_KEY=sk-xxx

for LANG1 in cpp java python; do
    for LANG2 in cpp java python; do
        if [[ ${LANG1} == ${LANG2} ]]; then
            continue
        fi

        python pipeline.py \
            --model-name-or-path gpt-4o-2024-08-06 \
            --temperature 0.2 \
            --top-p 0.95 \
            --max-tokens 8192 \
            --base-url https://api.openai.com/v1 \
            --config configs/gpt4o_api.yaml \
            --input-path data/LongTrans/test/${LANG1}_w_tests_small.json \
            --output-path data/PAST/gpt-4o/${LANG1}-${LANG2}.json \
            --lang1 ${LANG1} \
            --lang2 ${LANG2} \
            --num-proc 128
    done
done