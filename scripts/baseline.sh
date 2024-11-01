#!/bin/bash

MODEL=$1
PORT=$2
SPLIT=$3
if [ -z $3 ]; then
    SPLIT=test
fi

MODEL_NAME=$(echo $MODEL | awk -F'/' '{print $2}')

for LANG1 in cpp java python; do
    for LANG2 in cpp java python; do
        if [[ ${LANG1} == ${LANG2} ]]; then
            continue
        fi
        python baseline.py \
            --lang1 ${LANG1} \
            --lang2 ${LANG2} \
            --base-url http://localhost:${PORT}/v1 \
            --api-key RANDOM \
            --input-path data/LongTrans/${SPLIT}/${LANG1}.json \
            --output-path data/Baseline/${MODEL_NAME}/${SPLIT}/${LANG1}-${LANG2}.json \
            --model-name-or-path ${MODEL} \
            --max-model-len 8192 \
            --max-new-tokens 4096 \
            --sort-by-length \
            --batch-size 64 \
            --num-proc 128
    done
done