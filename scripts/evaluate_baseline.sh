#!/bin/bash

DIR_NAME=$1
SPLIT=$2

# 检查目录是否存在
if [ ! -d "$DIR_NAME" ]; then
  echo "$DIR_NAME not exists."
  exit 1
fi

if [ -z "$2" ]; then
    SPLIT=test
fi

# 遍历目录下的所有 .json 文件
for FILE in "$DIR_NAME"/*.json; do
  # 获取文件名（不包括路径）
  BASENAME=$(basename "$FILE" .json)

  # 检查文件名是否包含 "result"
  if [[ ! "$BASENAME" == *"result"* ]]; then
    # 执行 run_evaluation.py
    echo "Executing $FILE"
    python run_evaluation.py --problem-path deepmind/code_contests --sample-file "$FILE" --split "$SPLIT" --n-workers 128
  fi
done