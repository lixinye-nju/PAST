import os
import sys
import json
import time
import random
import hashlib
import gzip
import re
import argparse
import logging 
import yaml
import difflib
import tiktoken

from typing import Iterable, List, Dict, Any, Optional, Union
from transformers import AutoTokenizer

class HashGenerator:
    
    def __init__(self):
        self.generated_hash = set()
    
    def get_sha1_hash(self, input_str: Optional[str] = ""):
        while True:
            timestamp = str(time.time()).encode('utf-8')
            random_num = str(random.randint(0, 1000000)).encode('utf-8')
            hash_object = hashlib.sha1(timestamp + random_num + input_str.encode('utf-8'))
            hex_dig = hash_object.hexdigest()
            
            if hex_dig not in self.generated_hash:
                self.generated_hash.add(hex_dig)
                return hex_dig
            
#
# src: https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L1114
#
class FlexibleArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that allows both underscore and dash in names."""

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        if '--config' in args:
            args = FlexibleArgumentParser._pull_args_from_config(args)

        # Convert underscores to dashes and vice versa in argument names
        processed_args = []
        for arg in args:
            if arg.startswith('--'):
                if '=' in arg:
                    key, value = arg.split('=', 1)
                    key = '--' + key[len('--'):].replace('_', '-')
                    processed_args.append(f'{key}={value}')
                else:
                    processed_args.append('--' +
                                          arg[len('--'):].replace('_', '-'))
            else:
                processed_args.append(arg)

        return super().parse_args(processed_args, namespace)

    @staticmethod
    def _pull_args_from_config(args: List[str]) -> List[str]:
        """Method to pull arguments specified in the config file
        into the command-line args variable.

        The arguments in config file will be inserted between
        the argument list.

        example:
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        ```python
        $: vllm {serve,chat,complete} "facebook/opt-12B" \
            --config config.yaml -tp 2
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--config', 'config.yaml',
            '-tp', '2'
        ]
        $: args = [
            "serve,chat,complete",
            "facebook/opt-12B",
            '--port', '12323',
            '--tensor-parallel-size', '4',
            '-tp', '2'
            ]
        ```

        Please note how the config args are inserted after the sub command.
        this way the order of priorities is maintained when these are args
        parsed by super().
        """
        assert args.count(
            '--config') <= 1, "More than one config file specified!"

        index = args.index('--config')
        if index == len(args) - 1:
            raise ValueError("No config file specified! \
                             Please check your command-line arguments.")

        file_path = args[index + 1]

        config_args = FlexibleArgumentParser._load_config_file(file_path)

        # 0th index is for {serve,chat,complete}
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        args = config_args + args[1:index] + args[index + 2:]

        return args

    @staticmethod
    def _load_config_file(file_path: str) -> List[str]:
        """Loads a yaml file and returns the key value pairs as a
        flattened list with argparse like pattern
        ```yaml
            port: 12323
            tensor-parallel-size: 4
        ```
        returns:
            processed_args: list[str] = [
                '--port': '12323',
                '--tensor-parallel-size': '4'
            ]

        """

        extension: str = file_path.split('.')[-1]
        if extension not in ('yaml', 'yml'):
            raise ValueError(
                "Config file must be of a yaml/yml type.\
                              %s supplied", extension)

        # only expecting a flat dictionary of atomic types
        processed_args: List[str] = []

        config: Dict[str, Union[int, str]] = {}
        try:
            with open(file_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logging.error(
                "Unable to read the config file at %s. \
                Make sure path is correct", file_path)
            raise ex

        for key, value in config.items():
            processed_args.append('--' + key)
            processed_args.append(str(value))

        return processed_args



def get_sha256_hash(input_str: Optional[str] = "") -> str:
    timestamp = str(time.time()).encode('utf-8')
    salt = str(random.randint(0, 1000000)).encode('utf-8')
    hash_object = hashlib.sha256(timestamp + salt + input_str.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return hex_dig


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)



def write_jsonl(filename: str, data: Iterable[dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)

    with open(filename, mode) as fp:
        for x in data:
            fp.write((json.dumps(x) + "\n").encode('utf-8'))



def unpack_batch(batch: Dict[str, List[Any]]):
    batch_size = len(next(iter(batch.values())))  # 获取任意一个特征列表的长度
    batch_list = [
        {key: value[i] for key, value in batch.items()}
        for i in range(batch_size)
    ]
    return batch_list


def completion2code(completion: str):
    # Pattern to match code blocks with optional language specification
    pattern = r"```(?:[\w]*?\n)?([\s\S]*?)```"
    
    # Find all code blocks in the completion
    matched = re.findall(pattern, completion)
    
    if not matched:
        return completion
    
    # Return the first code block found
    return matched[0].strip()


def extract_code_fences(text):
    # 正则表达式匹配 Markdown 代码块
    pattern = r"```(?:[\w]*?\n)?([\s\S]*?)```"
    # 使用 re.DOTALL 标志，使得 . 可以匹配换行符
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def count_message_tokens(messages: List[Dict[str, str]],
                         tokenizer: AutoTokenizer):
    compiled_prompt = tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True)
    input_ids = tokenizer(compiled_prompt).input_ids
    return len(input_ids)
    

def count_text_tokens(text: str, tokenizer: AutoTokenizer):
    return len(tokenizer(text).input_ids)


def strcmp(actual_output: str, 
           expected_output: str,
           strict_compare: bool,
           remove_breakpoints: bool=True):
    if remove_breakpoints:
        # remove the content between <breakpoint> and </breakpoint>
        
        actual_output_lines = actual_output.splitlines()
        expected_output_lines = expected_output.splitlines()
        # Incase the output is too long and regex engine gets stuck
        if len(actual_output_lines) > 256 or len(expected_output_lines) > 256:
            actual_output_lines = actual_output_lines[:128] + actual_output_lines[-128:]
            expected_output_lines = expected_output_lines[:128] + expected_output_lines[-128:]
            actual_output = "\n".join(actual_output_lines)
            expected_output = "\n".join(expected_output_lines)
            
            
        pattern = re.compile(r'<breakpoint>(.*?)</breakpoint>', re.DOTALL)
        actual_output = pattern.sub('', actual_output)
        expected_output = pattern.sub('', expected_output)
    if strict_compare:
        return actual_output.strip() == expected_output.strip()
    else:
        actual_output = re.sub(r"\s+", "\n", actual_output.strip())
        expected_output = re.sub(r"\s+", "\n", expected_output.strip())
        return actual_output == expected_output




def generate_io_diff(test_inputs: List[str],
                     expected_outputs: List[str],
                     actual_outputs: List[str],
                     strict_compare: bool=False,
                     remove_breakpoints: bool=True):
    # 找出最后一个错误的 output 的索引
    idx = -1
    length = min([len(actual_outputs), len(expected_outputs), len(test_inputs)])
    for i in range(length):
        if not strcmp(actual_output=actual_outputs[i], 
                      expected_output=expected_outputs[i],
                      strict_compare=strict_compare,
                      remove_breakpoints=remove_breakpoints):
            idx = i
            break
    
    if idx == -1:
        # return "All outputs are correct."
        return ""
    
    test_input = test_inputs[idx]
    expected_output = expected_outputs[idx]
    actual_output = actual_outputs[idx]
    
    # 判断 expected_output 和 actual_output 是否超过 32 行
    expected_lines = expected_output.splitlines()
    actual_lines = actual_output.splitlines()
    
    if len(expected_lines) <= 32 and len(actual_lines) <= 32:
        # 直接打印输入、预期输出和实际输出
        report = (f"Input\n{test_input}\n"
                  f"Expected output\n{expected_output}\n"
                  f"Actual output\n{actual_output}\n")
    else:
        # avoid too much computation
        expected_lines = expected_lines[:256]
        actual_lines = actual_lines[:256]
        
        # 使用 difflib 生成美观的 diff
        diff = difflib.unified_diff(expected_lines, actual_lines, 
                                    fromfile='Expected output', 
                                    tofile='Actual output')
        diff = list(diff)
        if len(diff) > 32:
            diff = diff[:32] + ["......"]
        diff_text = '\n'.join(diff)
        report = (f"Input\n{test_input}\n"
                  f"Diff\n{diff_text}\n")
    
    return report


def num_tokens_from_string(string: str, model="gpt-4o-mini-2024-07-18") -> int:
    """Returns the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens



def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens