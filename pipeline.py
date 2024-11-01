import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Unset http_proxy and https_proxy 
# in case the request is forward to the proxy server
# and fail to loop back.
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import sys
import fire
import copy
import datasets 
import numpy as np
import openai
import json
import backoff
import asyncio
import queue
import threading
import traceback
import difflib

from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset, Dataset
from typing import Any, Literal, List, Dict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from jinja2 import Template
from multiprocessing import Queue, Process
from concurrent.futures import ProcessPoolExecutor, as_completed


from utils import (write_jsonl, completion2code, extract_code_fences,
                   FlexibleArgumentParser, count_text_tokens, count_message_tokens,
                   get_sha256_hash, generate_io_diff, num_tokens_from_string)
from mxeval.evaluation import check_compilable, execute_code
from mxeval.execution import strcmp


BREAKPOINT_PROMPT_PATH="prompt_templates/optimized_breakpoint.jinja"
TRANSLATE_PROMPT_PATH="prompt_templates/translate.jinja"
REPAIR_RPOMPT_PATH="prompt_templates/repair.jinja"
LOCALIZE_PROMPT_PATH="prompt_templates/localize.jinja"
TRANSLATE_SNIPPET_PROMPT_PATH="prompt_templates/translate_snippet.jinja"
ECHOBACK_PROMPT_PATH="prompt_templates/echoback.jinja"


class Session:
    
    def __init__(
        self,
        client: openai.Client,
        model_name_or_path: str,
        sampling_params: Dict[str, Any] = {}
    ):
        self.client = client
        self.model = model_name_or_path
        self.sampling_params = sampling_params
        
        self.messages = []
        self.reset()
        
    def reset(self):
        self.messages = [
            {"role": "system", "content": "You are a helpful AI programming assistant.\n"}
        ]
        
    @backoff.on_exception(backoff.expo, (openai.APITimeoutError, 
                                         openai.APIConnectionError, 
                                         openai.RateLimitError), max_tries=4)
    def create_chat_completion(self, messages):
        return self.client.chat.completions.create(messages=self.messages,
                                                   model=self.model,
                                                   timeout=3600,
                                                   **self.sampling_params)
        
    def query_llm(self):
        try:
            response = self.create_chat_completion(self.messages)
        except openai.BadRequestError as e:
            print(f"Encountered BadRequestException: {e}, skipped.")
            return None
        except json.decoder.JSONDecodeError as e:
            print(f"Encountered JSONDecodeError: {e}, skipped.")
            return None
        except openai.APIStatusError as e:
            print(f"Encountered APIStatusError: {e}")
            sys.exit(0)
            return None
        
        if response.choices[0].finish_reason == "stop":
            return response.choices[0].message.content
        else:
            print(f"Encounter finished_reason={response.choices[0].finish_reason}")
            return None
        
    def query(self, text: str):
        self.messages += [{"role": "user", "content": text}]
        response = self.query_llm()
        if response is not None:
            self.messages += [{"role": "assistant", "content": response}]
        return response



class BreakpointRepair:
    
    def __init__(
        self,
        client: openai.Client,
        model_name_or_path: str,
        lang1: str,
        lang2: str,
        sampling_params: Dict[str, Any] = {},
        tokenizer_name: str=None,
        max_input_len: int=4096,
        add_breakpoints_rounds: int=4,
        repair_rounds: int=1
    ):
        self.client = client
        self.model = model_name_or_path
        self.lang1 = lang1
        self.lang2 = lang2
        self.sampling_params = sampling_params
        self.max_input_len = 4096
        self.add_breakpoints_rounds = add_breakpoints_rounds
        self.repair_rounds = repair_rounds
        
        if tokenizer_name is None:
            self.tokenizer_name = model_name_or_path
        else:
            self.tokenizer_name = tokenizer_name

        self.output_columns = []
        
        with open(BREAKPOINT_PROMPT_PATH, "r") as f:
            self.breakpoint_prompt = Template(f.read())
            
        with open(TRANSLATE_PROMPT_PATH, "r") as f:
            self.translate_prompt = Template(f.read())
        
        with open(REPAIR_RPOMPT_PATH, "r") as f:
            self.repair_prompt = Template(f.read())
    
        with open(LOCALIZE_PROMPT_PATH, "r") as f:
            self.localize_prompt = Template(f.read())
        
        with open(TRANSLATE_SNIPPET_PROMPT_PATH, "r") as f:
            self.translate_snippet_prompt = Template(f.read())
    
        with open(ECHOBACK_PROMPT_PATH, "r") as f:
            self.echoback_prompt = Template(f.read())
            
        if self.model not in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            self.tokenizer = None
    
    # Main logic
    def process(
        self,
        example: Dict[str, Any]
    ):
        results = self._initialize_results(example)
        if not self._check_compilable_and_token_length(results, example):
            return results
        
        step1_done = False
        for _ in range(self.add_breakpoints_rounds):
            if self._add_breakpoints(results, example):
                step1_done = True
                break
        if not step1_done:
            return results
        
        step1_compile = False
        for _ in range(self.add_breakpoints_rounds):
            if self._execute_code1_public_tests(results, example):
                step1_compile = True
                break
        if not step1_compile:
            return results
        
        if not self._translate_code_with_breakpoints(results, example):
            return results
        
        success = (
            self._execute_translated_code(results, example) or
            self._repair_translated_code(results, example) or
            self._localize_and_revise_code(results, example) 
        )
        

    
        if success:
            self._execute_code1_all_tests(results, example)
            self._execute_code2_all_tests(results, example)

        old_results = copy.deepcopy(results)
        results = {k: v for k, v in old_results.items() if k in self.output_columns}
            
        return results

    def _initialize_results(self, example: Dict[str, Any]) -> Dict[str, Any]:
        id_ = example["id"]
        code1 = example["code"]
        name = example["name"]
        results = dict(
            task_id=id_,
            name=name,
            model=self.model,
            lang1=self.lang1,
            lang2=self.lang2,
            completion_id=get_sha256_hash(code1),
            pass_public_tests=False,
            pass_all_tests=False,
            finish_reason="",
            code1=code1,
            code1_w_breakpoints="",
            code2_w_breakpoints="",
            code2_repaired="",
            code2_revised="",
            code2_santized="",
            sessions={}
        )
        self.output_columns = list(results.keys())
        return results

    def _check_compilable_and_token_length(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        code1 = example["code"]
        lang = example["language"]
        if not check_compilable(code=code1, language=lang):
            results["finish_reason"] = "Step 0 (Check compilable): code is not compilable."
            return False
        
        if self.model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            if num_tokens_from_string(code1) > self.max_input_len:
                results["finish_reason"] = "Step 0 (Check token length): code exceeds max input length."
                return False
        else:
            if count_text_tokens(code1, self.tokenizer) > self.max_input_len:
                results["finish_reason"] = "Step 0 (Check token length): code exceeds max input length."
                return False
        return True

    def _add_breakpoints(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        session_breakpoint = Session(self.client, self.model, self.sampling_params)
        query1 = self.breakpoint_prompt.render({
            "lang1": example["language"], 
            "code": example["code"]
        })
        response1 = session_breakpoint.query(query1)
        results["sessions"]["breakpoint"] = session_breakpoint.messages
        if response1 is None:
            results["finish_reason"] = "Step 1 (Add breakpoints): failed to query LLM."
            return False
        try:
            code1_w_breakpoints = extract_code_fences(response1)[-1]
        except:
            results["finish_reason"] = "Step 1 (Add breakpoints): cannot find code block."
            return False
        
        results["code1_w_breakpoints"] = code1_w_breakpoints
        return True

    def _execute_code1_public_tests(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        code1_w_breakpoints = results["code1_w_breakpoints"]
        public_test_inputs = example["public_tests"]["input"]
        
        if len(public_test_inputs) == 0:
            public_test_inputs = example["private_tests"]["input"][:3]
        
        code1_public_exec_results = execute_code(code=code1_w_breakpoints,
                                                 language=self.lang1,
                                                 test_inputs=public_test_inputs)
        
        if not code1_public_exec_results["compiled"]:
            results["finish_reason"] = "Step 1 (Add breakpoints): failed to compile."
            return False
        results["code1_public_outputs"] = code1_public_exec_results["outputs"]
        return True
    
    def _execute_code1_all_tests(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        code1_w_breakpoints = results["code1_w_breakpoints"]
        # all_test_inputs = (example["public_tests"]["input"] +
        #                    example["private_tests"]["input"] + 
        #                    example["generated_tests"]["input"])
        all_test_inputs = example["private_tests"]["input"]
        code1_all_exec_results = execute_code(code=code1_w_breakpoints,
                                              language=self.lang1,
                                              test_inputs=all_test_inputs)
        
        results["code1_all_outputs"] = code1_all_exec_results["outputs"]
        if not code1_all_exec_results["compiled"]:
            return False
        return True

    def _translate_code_with_breakpoints(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        session_translate = Session(self.client, self.model, self.sampling_params)
        query2 = self.translate_prompt.render({
            "lang1": self.lang1, 
            "lang2": self.lang2,
            "code": results["code1_w_breakpoints"]
        })
        response2 = session_translate.query(query2)
        results["sessions"]["translate"] = session_translate.messages
        if response2 is None:
            results["finish_reason"] = "Step 2 (Translate): failed to query LLM."    
            return False
        try:
            results["code2_w_breakpoints"] = extract_code_fences(response2)[-1]
        except:
            results["finish_reason"] = "Step 2 (Translate): cannot find code block."
            return False
        
        results["code2_santized"] = results["code2_w_breakpoints"]
        return True

    def _execute_translated_code(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        code2_w_breakpoints = results["code2_w_breakpoints"]
        passed = self._execute_code2_public_tests(results=results, example=example, code2=code2_w_breakpoints)
        if passed:
            results["finish_reason"] = "Step 2 (Translate): success."
            return True
        return False

    def _repair_translated_code(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        session_repair = Session(self.client, self.model, self.sampling_params)
        query3 = self.repair_prompt.render({
            "lang1": self.lang1,
            "lang2": self.lang2,
            "code1": results["code1_w_breakpoints"],
            "code2": results["code2_w_breakpoints"],
            "error_message": results["code2_public_message"],
            "diff": generate_io_diff(test_inputs=example["public_tests"]["input"],
                                     expected_outputs=results["code1_public_outputs"],
                                     actual_outputs=results["code2_public_outputs"])
        })
        
        response3 = session_repair.query(query3)
        results["sessions"]["repair"] = session_repair.messages
        if response3 is None:
            results["finish_reason"] = "Step 3 (Repair): failed to query LLM."
            return False
        
        try:
            results["code2_repaired"] = extract_code_fences(response3)[-1]
        except:
            results["finish_reason"] = "Step 3 (Repair): cannot find code block."
            return False
        
        results["code2_santized"] = results["code2_repaired"]
        return self._execute_repaired_code(results, example)

    def _execute_repaired_code(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        code2_repaired = results["code2_repaired"]
        passed = self._execute_code2_public_tests(results=results, example=example, code2=code2_repaired)
        if passed:
            results["finish_reason"] = "Step 3 (Repair): success."
            return True
        return False

    def _localize_and_revise_code(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        session_localize = Session(self.client, self.model, self.sampling_params)
        session_echoback = Session(self.client, self.model, self.sampling_params)
        session_translate_snippet = Session(self.client, self.model, self.sampling_params)
        
        query4 = self.localize_prompt.render({
            "lang1": self.lang1,
            "lang2": self.lang2,
            "code1": results["code1_w_breakpoints"],
            "code2": results["code2_repaired"],
            "error_message": results["code2_public_message"],
            "diff": generate_io_diff(test_inputs=example["public_tests"]["input"],
                                     expected_outputs=results["code1_public_outputs"],
                                     actual_outputs=results["code2_public_outputs"])
        })
        response4 = session_localize.query(query4)
        results["sessions"]["localize"] = session_localize.messages
        if response4 is None:
            results["finish_reason"] = "Step 4 (Localize): failed to query LLM."
            return False
        
        try:
            code1_localize_snippet = extract_code_fences(response4)[-1]
        except:
            results["finish_reason"] = "Step 4 (Localize): cannot find code block."
            return False
        
        query5 = self.translate_snippet_prompt.render({
            "lang1": self.lang1,
            "lang2": self.lang2,
            "code": code1_localize_snippet
        })
        response5 = session_translate_snippet.query(query5)
        results["sessions"]["translate-snippet"] = session_translate_snippet.messages
        if response5 is None:
            results["finish_reason"] = "Step 5 (Translate Snippet): failed to query LLM."
            return False
        
        try:
            code2_revised_snippet = extract_code_fences(response5)[-1]
        except:
            results["finish_reason"] = "Step 5 (Translate Snippet): cannot find code block."
            return False
        
        query6 = self.echoback_prompt.render({
            "lang2": self.lang2,
            "complete_code": results["code2_repaired"],
            "revised_code": code2_revised_snippet
        })
        response6 = session_echoback.query(query6)
        results["sessions"]["echoback"] = session_echoback.messages
        if response6 is None:
            results["finish_reason"] = "Step 6 (Echo-back): failed to query LLM."
            return False
        
        try:
            results["code2_revised"] = extract_code_fences(response6)[-1]
        except:
            results["finish_reason"] = "Step 6 (Echo-back): cannot find code block."
        results["code2_santized"] = results["code2_revised"]
        return self._execute_revised_code(results, example)

    def _execute_revised_code(self, results: Dict[str, Any], example: Dict[str, Any]) -> bool:
        code2_revised = results["code2_revised"]
        passed = self._execute_code2_public_tests(results=results, example=example, code2=code2_revised)

        results["finish_reason"] = f"Step 6 (Echo-back): END"
        return passed
    
    def _execute_code2_public_tests(self,
                                    results: Dict[str, Any],
                                    example: Dict[str, Any],
                                    code2: str) -> bool:
        public_test_inputs = example["public_tests"]["input"]
        if len(public_test_inputs) == 0:
            public_test_inputs = example["private_tests"]["input"][:3]
        
        code2_public_exec_results = execute_code(code=code2,
                                                 language=self.lang2,
                                                 test_inputs=public_test_inputs,
                                                 test_outputs=results["code1_public_outputs"])
        results["pass_public_tests"] = code2_public_exec_results["passed"]
        results["code2_public_outputs"] = code2_public_exec_results["outputs"]
        results["code2_public_message"] = code2_public_exec_results["message"]
        return results["pass_public_tests"]
    
    def _execute_code2_all_tests(self, 
                                 results: Dict[str, Any], 
                                 example: Dict[str, Any]) -> bool:
        # all_test_inputs = (example["public_tests"]["input"] +
        #                    example["private_tests"]["input"] + 
        #                    example["generated_tests"]["input"])
        all_test_inputs = example["private_tests"]["input"]

        code2_all_exec_results = execute_code(code=results["code2_santized"],
                                              language=self.lang2,
                                              test_inputs=all_test_inputs,
                                              test_outputs=results["code1_all_outputs"])
        results["pass_all_tests"] = code2_all_exec_results["passed"]
        return results["pass_all_tests"]



def parse_args():
    parser = FlexibleArgumentParser()
    parser.add_argument("--input-path", type=str, required=True,
                        help="Path of input json file.")
    parser.add_argument("--output-path", type=str, required=True,
                        help="Path of output json file.")
    parser.add_argument("--model-name-or-path", type=str, required=True,
                        help="Model name or local path of a LLM.")
    parser.add_argument("--lang1", type=str, choices=["cpp", "java", "python"],
                        help="Source language in translation.")
    parser.add_argument("--lang2", type=str, choices=["cpp", "java", "python"],
                        help="Target language in translation")
    
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Name of tokenizer. Default to `model-name-or-path`.")
    
    
    # Sampling params
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature in decoding stage. Set to 0 for greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="Threshold parameter in top-p sampling.")
    parser.add_argument("--max-tokens", type=int, default=4096, 
                        help="Maximum tokens in completion.")
    
    
    parser.add_argument("--num-proc", type=int, default=64,
                        help="Number of workers.")
    
    # OpenAI client params
    parser.add_argument("--base-url", type=str, 
                        help="URL of openai or vllm client.")
    parser.add_argument("--api-key", type=str, default="", 
                        help="API key.")

    args = parser.parse_args()
    return args
    

def writer(writer_queue: Queue, output_path):
    while True:
        try:
            data = writer_queue.get(timeout=1)
            if data == "END":
                break
            if data is not None:
                write_jsonl(output_path, [data], append=True)
        except queue.Empty:
            continue


if __name__ == "__main__":
    args = parse_args()
    
    # Unset http_proxy and https_proxy 
    # in case the request is forward to the proxy server
    # and fail to loop back.
    if "localhost" in args.base_url:
        os.environ.pop("http_proxy", None)
        os.environ.pop("https_proxy", None)
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )
    
    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens
    }

    pipeline = BreakpointRepair(client=client,
                                model_name_or_path=args.model_name_or_path,
                                tokenizer_name=args.tokenizer,
                                lang1=args.lang1,
                                lang2=args.lang2,
                                sampling_params=sampling_params)
    
    src_data = load_dataset("json", data_files=[args.input_path], split="train")
    print(f"Loading {len(src_data)} samples from {args.input_path}")
    
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if os.path.isfile(args.output_path):
        
        # simple workaround
        with open(args.output_path, "r") as f:
            lines = f.readlines()
        
        objs = [json.loads(line) for line in lines if len(line.strip()) > 0]
        dst_data = Dataset.from_list(objs)
        
        # dst_data = load_dataset("json", data_files=[args.output_path], split="train")
        print(f"Loading {len(dst_data)} cache examples from {args.output_path}.")
        seed_ids = set(dst_data["task_id"])
        src_data = src_data.filter(lambda x: x["id"] not in seed_ids, num_proc=args.num_proc)
    
    src_data = src_data.shuffle()
    # for sample in tqdm(src_data, desc="Translating: "):
    #     result = pipeline.process(sample)
    #     if result is not None:
    #         write_jsonl(args.output_path, [result], append=True)
            


    # executing and saving results concurrently
    writer_queue = Queue()
    writer_process = Process(target=writer, args=(writer_queue, args.output_path))
    writer_process.start()
    
    def map_func(example, queue: Queue):
        result = pipeline.process(example)
        queue.put(result)
        return example
    
    src_data.map(lambda x: map_func(x, writer_queue), num_proc=args.num_proc, desc="Translating: ")
    
    writer_queue.put("END")
    writer_process.join()
    

