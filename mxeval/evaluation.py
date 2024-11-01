# Original Copyright 2021 OpenAI under MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from io import UnsupportedOperation
import itertools
import os
import time
import fire
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from typing import Dict, Iterable, List, Union, Literal


import numpy as np
from tqdm import tqdm
from utils import stream_jsonl, write_jsonl

from datasets import load_dataset, Features, Value, Sequence
# Amazon modification
# import check correctness for all languages

from mxeval.execution import (
    check_correctness_python,
    check_correctness_cpp,
    check_correctness_java,
    check_compilable_python,
    check_compilable_cpp,
    check_compilable_java,
    execute_cpp,
    execute_java,
    execute_python
)

# see execution.py for details
# Check the generated samples against test suites.
check_correctness_function_map = {
    "python": check_correctness_python,
    "java": check_correctness_java,
    # "javascript": check_correctness_javascript,
    # "typescript": check_correctness_typescript,
    # "kotlin": check_correctness_kotlin,
    # "ruby": check_correctness_ruby,
    # "php": check_correctness_php,
    "cpp": check_correctness_cpp,
    # "csharp": check_correctness_csharp,
    # "go": check_correctness_go,
    # "perl": check_correctness_perl,
    # "scala": check_correctness_scala,
    # "swift": check_correctness_swift,
}

check_compilable_function_map = {
    "python": check_compilable_python,
    "java": check_compilable_java,
    "cpp": check_compilable_cpp
}


execute_code_function_map = {
    "python": execute_python,
    "java": execute_java,
    "cpp": execute_cpp
}

def execute_code(code: str, 
                 language: str,
                 test_inputs: List[str],
                 test_outputs: List[str] = None):
    return execute_code_function_map[language](code, 
                                               test_inputs=test_inputs,
                                               test_outputs=test_outputs)


def check_compilable(code: str, 
                     language: str):
    return check_compilable_function_map[language](code)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def get_execute_function(lang):
    lang = lang.lower()
    assert lang in check_correctness_function_map, f"Language {lang} is not among the supported languages: {check_correctness_function_map.keys()}"
    return check_correctness_function_map[lang]


def evaluate_functional_correctness(
    sample_file: str,
    problem_path: str,
    split: str = Literal["train", "valid", "test"],
    k: List[int] = [1, 10, 100],
    n_workers: int = os.cpu_count() - 1,
    timeout: float = 10.0,
    use_cache: bool = True,
    verbose: bool = False,
    early_stop: bool = True,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl"
    """

    # if type(problem_file) is not dict:
    #     problems = read_problems(problem_file)
    # else:
    #     print("Skip reading problems -- using problem_file (dict) as problems")
    #     problems = problem_file

    if (
        problem_path.endswith(".json") or
        problem_path.endswith(".jsonl")
    ):
        print(f"Loading problems from json file `{problem_path}`...")
        problem_dataset = load_dataset("json", data_files=[problem_path], split="train")
        problems = {example["task_id"]: example for example in problem_dataset}
    else:
        print(f"Loading problems from huggingface repo {problem_path} - {split} split.")
        problem_dataset = load_dataset(problem_path, split=split)
        problems = {example["name"]: example for example in problem_dataset}


    seed = int(time.time() * 1000000) % 1000000
    np.random.seed(seed=seed)  # microsecond


    def execute_func(sample):
        name = sample["name"]
        translated_code = sample["code2_santized"]
        language = sample["lang2"]
        result = check_correctness_function_map[language](problem=problems[name], 
                                                          completion=translated_code, 
                                                          timeout=timeout, 
                                                          verbose=verbose,
                                                          early_stop=early_stop)
        return {**sample, **result}

    
    samples = load_dataset("json", data_files=[sample_file], split="train")
    print(f"Reading {len(samples)} samples from {sample_file}")
    
    out_file = sample_file + "_results.json"
    if os.path.exists(out_file):
        done_samples = load_dataset("json", data_files=[out_file], split="train")
        print(f"Loading {len(done_samples)} from cache {out_file}")
        done_task_ids = set(done_samples["completion_id"])
        samples = samples.filter(lambda x: x["completion_id"] not in done_task_ids)
        print(f"{len(samples)} samples remained to process")
    
    print("Running test suites...")
    

    # 使用 concurrent.futures.ThreadPoolExecutor 来并行处理样本，
    # 并使用 queue.Queue 来存储结果
    result_queue = queue.Queue()
    progress_bar = tqdm(total=len(samples), desc="Executing sample: ")
    
    def process_sample(sample):
        result = execute_func(sample)
        result_queue.put(result)
        return result
        
    def write_result_to_file():
        while True:
            try:
                result = result_queue.get(timeout=1)
                if result is None:
                    break
                write_jsonl(out_file, [result], append=True)
                progress_bar.update()
            except queue.Empty:
                continue
            
    import threading
    writer_thread = threading.Thread(target=write_result_to_file)
    writer_thread.start()

    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_sample, sample) for sample in samples]
        for future in as_completed(futures):
            result = future.result()
            
            
    # Signal the writer thread to finish
    result_queue.put(None)
    writer_thread.join()
    
    
    return


