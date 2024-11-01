# PAST


This is the source code and artifacts for paper "Enhancing Large Language Models in Long Code Translation through Instrumentation and Program State Alignment".


## Install

### Packages
```bash
pip install -r requirements
```

### Execution Environments
```bash
cd mxeval
bash prepare_env_ubuntu.sh
```

### Model Weights
Download `Qwen2.5-72B-Instruct` from [huggingface](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct).
```python
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
```

### Dataset
```bash
cd data
unzip LongTrans.zip
```

## Baseline

First, start a vllm server. You need at least 2 A100-80G to run a 72B model.
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/vllm_server.sh Qwen/Qwen2.5-72B-Instruct
```

Second, run the baseline.
```bash
bash scripts/baseline.sh Qwen/Qwen2.5-72B-Instruct 8000 test
```

Finally, evaluate the results based on execution.
```bash
bash scripts/evaluate_baseline.sh data/Baseline/Qwen2.5-72B-Instruct/test
```

## PAST

### Qwen2.5-72B
Also, you need to start a vllm server.
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/vllm_server.sh Qwen/Qwen2.5-72B-Instruct
```
Then
```bash
bash scripts/pipeline_qwen.sh
```


### GPT-4o
First, configure your `api-ky` in the `scripts/pipeline_gpt4o.sh`
```bash
# Add your gpt api-key here
API_KEY=sk-xxx
```
Then, we can run with OpenAI's API.
```bash
bash scripts/pipeline_gpt4o.sh
```