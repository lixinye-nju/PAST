import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Unset http_proxy and https_proxy 
# in case the request is forward to the proxy server
# and fail to loop back.
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

import re
import fire
import datasets 
import numpy as np
import openai
import json
import backoff
import asyncio


from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset
from typing import Any, Literal, List, Dict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import write_jsonl, completion2code, num_tokens_from_messages, num_tokens_from_string
from mxeval.evaluation import check_compilable


PROMPT_TEMPLATE = (
    "```{}\n" + 
    "{}\n" + 
    "```\n" +
    "Translate above {} code to {}.\n" + 
    "The result should be wrapped in code fences.\n" + 
    "DO NOT include other contents.\n"
)


@backoff.on_exception(backoff.expo, (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError), max_tries=16)
async def dispatch_openai_chat_requests(client:AsyncOpenAI, 
                                        messages_list: list[list[dict[str, Any]]], 
                                        **kwargs):
    async_responses = [
        client.chat.completions.create(
            messages=x,
            timeout=3600*24,
            **kwargs
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)



def main(input_path: str,
         output_path: str,
         model_name_or_path: str,
         lang1: str=Literal["cpp", "python", "java", "python2"],
         lang2: str=Literal["cpp", "python", "java", "python2"],
         batch_size: int=16,
         temperature: float=0.2,
         top_p: float=0.95,
         num_samples: int=1,
         max_new_tokens: int=4096,
         max_model_len: int=8192,
         num_proc: int=128,
         sort_by_length: bool=False,
         base_url: str=None,
         api_key: str=None,
         trust_remote_code: bool=True
):


    if "localhost" in base_url:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)

    if model_name_or_path not in ["gpt-4o-mini", "gpt-4o-2024-08-06",
                                  "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    else:
        tokenizer = None

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    print(f"Loading data from {input_path}.")
    src_data = load_dataset("json", data_files=[input_path], split="train")
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if os.path.isfile(output_path):
        print(f"Loading cache from {output_path}.")
        dst_data = load_dataset("json", data_files=[output_path], split="train")
        seen_ids = set(dst_data["task_id"])
        src_data = src_data.filter(lambda x: x["id"] not in seen_ids, num_proc=num_proc)
    

    
    def count_message_tokens(messages: List[Dict[str, str]]):
        if model_name_or_path in ["gpt-4o-mini", "gpt-4o-2024-08-06",
                                  "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]:
            return num_tokens_from_messages(messages)
        else:        
            compiled_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(compiled_prompt).input_ids
            return len(input_ids)
    
    def count_resp_tokens(response: str):
        if model_name_or_path in ["gpt-4o-mini", "gpt-4o-2024-08-06",
                                  "claude-3-haiku-20240307", "claude-3-5-sonnet-20240620"]:
            return num_tokens_from_string(response)
        else:
            return len(tokenizer(response).input_ids)

    def prepare_messages(example):
        code = example["code"]
        prompt = PROMPT_TEMPLATE.format(lang1, code, lang1, lang2)
        if model_name_or_path in ["google/codegemma-7b-it"]:
            message = [
                {"role": "user", "content": prompt}
            ]
        elif model_name_or_path in ["codellama/CodeLlama-7b-Instruct-hf", 
                                    "codellama/CodeLlama-13b-Instruct-hf", 
                                    "codellama/CodeLlama-34b-Instruct-hf"]:
            # Alignment of codellama is poor.
            prompt += "FORMAT:\n"
            prompt += f"```{lang2}\n"
            
            prefix = {"python": "import sys\n",
                      "cpp": "#include <iostream>\n",
                      "java": "import java.util.*;\n"}
            prompt += prefix[lang2]
            prompt += f"{'#' if lang2 == 'python' else '//'} YOUR_CODE_HERE\n"
            prompt += "```\n"
            
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        elif model_name_or_path in ["saves/round0/checkpoint-2000"]:
            prompt = f"INPUT: {lang1.upper()}\n"
            prompt += f"{code}\n"
            prompt += f"OUTPUT: {lang2.upper()}\n"
            message = [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ]
        else:
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        prompt_tokens = count_message_tokens(message)
        example["message"] = message
        example["prompt_tokens"] = prompt_tokens
        return example
    
    src_data = src_data.map(prepare_messages, num_proc=num_proc)
    src_data = src_data.filter(lambda x: x["prompt_tokens"] < (max_model_len - max_new_tokens), 
                               num_proc=num_proc)

    if sort_by_length:
        src_data = src_data.sort("prompt_tokens")

    
    def simple_collate(batch: List):
        keys = batch[0].keys()
        collated = {
            key: [sample[key] for sample in batch]
            for key in keys
        }
        return collated    

    dataloader = DataLoader(src_data, 
                            batch_size=batch_size, 
                            drop_last=False,
                            shuffle=False,
                            collate_fn=simple_collate)

    for batch in tqdm(dataloader):
        ids = batch["id"]
        code = batch["code"]
        lang = batch["language"]
        names = batch["name"]
        messages_list = batch["message"]
        prompt_tokens = batch["prompt_tokens"]
        
        assert len(lang) > 0 and np.all([x == lang[0] for x in lang]) and lang[0] == lang1, \
            f"Expected all elements in lang to be {lang1}, but found {lang}"
            

        try:
            responses = asyncio.run(
                dispatch_openai_chat_requests(client,
                                              messages_list, 
                                              model=model_name_or_path,
                                              max_tokens=max_new_tokens,
                                              n=num_samples,
                                              temperature=temperature,
                                              top_p=top_p)
            )
        except openai.BadRequestError as e:
            print(f"Encountered BadRequestException: {e}, skipped batch.")
            continue
        
            
        contents = [
            {
                "task_id": ids[i],
                "name": names[i],
                "model": model_name_or_path,
                "completion_id": responses[i].id,
                "prompt_tokens": prompt_tokens[i],
                "completion_tokens": count_resp_tokens(responses[i].choices[j].message.content),
                "finish_reason": responses[i].choices[j].finish_reason,
                "lang1": lang1,
                "lang2": lang2,
                "code1": code[i],
                "code2": responses[i].choices[j].message.content,
                "code2_santized": completion2code(responses[i].choices[j].message.content),
            }
            for j in range(num_samples)
            for i in range(len(code))
        ]
        
        # contents = [
        #     x for x in contents
        #     if check_compilable(x["code2_santized"], x["lang2"])[0]
        # ]
        
        write_jsonl(output_path, contents, append=True)



if __name__ == "__main__":
    fire.Fire(main)
