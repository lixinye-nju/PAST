# Original Copyright 2021 OpenAI under MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# check_correctness_* functions are AWS additions

import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import random
import shutil
import signal
import string
import subprocess
import tempfile
import time
import errno
import numpy as np
import copy
import re
from typing import Dict, Optional, List
import threading
lock = threading.Lock()


def merge_dicts(
    dict_list: List[Dict[str, List[str]]]
):
    if len(dict_list) == 0:
        return {}
    result = copy.deepcopy(dict_list[0])
    for dict2merge in dict_list[1:]:
        for key, value in dict2merge.items():
            if key in result:
                result[key] += value
            else:
                result[key] = value
    return result

def remove_breakpoints(text):
    # 优化后的正则表达式
    pattern = re.compile(r'<breakpoint>(.*?)</breakpoint>', re.DOTALL)
    return pattern.sub('', text)


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


def execute_java(code: str, 
                 test_inputs: List[str], 
                 test_outputs: List[str] = None,
                 exec_timeout: float = 100,
                 compile_timeout: float=100,
                 early_stop: bool=True,
                 strict_compare: bool=False,
                 verbose=False):
    language = "java"
    current_dir = os.path.dirname(os.path.realpath(__file__))
    entire_string = code
    base_path = setup_base_path(current_dir, f"{language}_exec_eval", "")
    try:
        os.makedirs(base_path, exist_ok=False)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    pattern = r'\bpublic\s+class\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
    match = re.search(pattern, entire_string)
    class_name = match.group(1) if match else "Main"

    path = os.path.join(base_path, f"{class_name}.{language}")
    with open(path, "w") as f:
        f.write(entire_string)
        
    elapsed = -1.0
    exec_result = []
    passed = False
    actual_outputs = []
    
    try:
        exec_result_compile = subprocess.run(
            [f"javac", os.path.basename(path)],
            timeout=int(compile_timeout),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_path
        )
        compiled = exec_result_compile.returncode == 0
        message = f"Compilation Error:\n{exec_result_compile.stderr}" if not compiled else "Compilation Success"
        if verbose:
            print("exec_result_compile", exec_result_compile)
    except Exception as e:
        compiled = False
        passed = False
        exec_result = []
        message = f"Compilation Error:\n{str(e)}"
        elapsed = -1.0
        


    if compiled:
        if test_outputs is not None:
            test_inputs = test_inputs[:len(test_outputs)]
            
        for idx, input_data in enumerate(test_inputs):
            try:
                start = time.time()
                exec_result_run = subprocess.run(
                    [f"java", "-cp", ".", class_name],
                    input=input_data,
                    timeout=int(exec_timeout),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=base_path
                )
                
                elapsed = 1000.0 * (time.time() - start)
                if verbose:
                    print("exec result run", exec_result_run)
                    
            
                # Execution finished
                if exec_result_run.returncode == 0:
                    actual_output = exec_result_run.stdout.strip()
                    actual_outputs.append(actual_output),
                    if test_outputs is not None:
                        if strcmp(actual_output, test_outputs[idx], strict_compare):
                            # message = (f"Accepted\nInput:\n{input_data}\n" + 
                            #            f"Output\n{actual_output}\n")
                            message = "Accepted"
                            exec_result.append(1)
                        else:
                            # message = (f"Wrong Answer\nInput:\n{input_data}\n" + 
                            #         f"Expected\n{test_outputs[idx]}\nGot\n{actual_output}")
                            message = "Wrong Answer"
                            exec_result.append(0)
                            if early_stop:
                                break
                # Runtime Error
                else:
                    # message = (f"Runtime Error\nInput:\n{input_data}\n" + 
                    #            f"Output:\n{exec_result_run.stdout.strip()}\n" + 
                    #         f"Error Message:\n{exec_result_run.stderr}")
                    message = f"Runtime Error: {exec_result_run.stderr}"
                    exec_result.append(-1)
                    if early_stop:
                        break
                    
            except Exception as e:
                # message = (f"Runtime Error\nInput:\n{input_data}\n" + 
                #         f"Error Message:\n{str(e)}")
                message = f"Runtime Error: {str(e)}"
                exec_result.append(-1)
                if early_stop:
                    break
    
        passed = (
            test_outputs is not None and
            compiled and
            bool(np.all(np.array(exec_result) == 1))
        )
    try:
        shutil.rmtree(base_path)
    except Exception as e:
        pass

    return dict(
        time_elapsed=elapsed,
        exec_result=exec_result,
        compiled=compiled,
        passed=passed,
        message=message,
        inputs=test_inputs,
        outputs=actual_outputs
    )
            

def execute_helper(code: str,
                   test_inputs: List[str],
                   test_outputs: List[str] = None,
                   language: str = None,
                   extension: str = None, 
                   compile_command_lambda=None,
                   subprocess_command_lambda=None,
                   extra_cleanup=None,
                   cwd=None,
                   exec_timeout: float = 100,
                   compile_timeout: float = 100,
                   early_stop: bool=True,
                   strict_compare: bool=False,
                   verbose=False):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    entire_string = code

    language_dirname = f"{language}_exec_eval"

    base_path = setup_base_path(current_dir, language_dirname, extension)
    path = base_path + f"{extension}"

    if cwd is not None:
        cwd = os.path.dirname(base_path)
    with open(path, "w") as f:
        f.write(entire_string)
        
    elapsed = -1.0
    exec_result = []
    passed = False
    actual_outputs = []
    
    assert compile_command_lambda is not None
    try:
        exec_result_compile = subprocess.run(
            compile_command_lambda(base_path),
            timeout=int(compile_timeout),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
        )
        compiled = exec_result_compile.returncode == 0
        message = f"Compilation Error: {exec_result_compile.stderr}" if not compiled else "Compilation Success"
        if verbose:
            print("exec_result_compile", exec_result_compile)
    except Exception as e:
        compiled = False
        passed = False
        exec_result = []
        message = f"Compilation Error: {str(e)}"
        elapsed = -1.0
        
    if compiled:
        if test_outputs is not None:
            test_inputs = test_inputs[:len(test_outputs)]
            
        for idx, input_data in enumerate(test_inputs):
            try:
                start = time.time()
                exec_result_run = subprocess.run(
                    subprocess_command_lambda(base_path),
                    input=input_data,
                    timeout=int(exec_timeout),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                )
                
                elapsed = 1000.0 * (time.time() - start)
                if verbose:
                    print("exec result run", exec_result_run)
                    
            
                # Execution finished
                if exec_result_run.returncode == 0:
                    actual_output = exec_result_run.stdout.strip()
                    actual_outputs.append(actual_output),
                    if test_outputs is not None:
                        if strcmp(actual_output, test_outputs[idx], strict_compare):
                            # message = (f"Accepted\nInput:\n{input_data}\n" + 
                            #            f"Output\n{test_outputs[idx]}\n")
                            message = "Accepted"
                            exec_result.append(1)
                        else:
                            # message = (f"Wrong Answer\nInput:\n{input_data}\n" + 
                            #            f"Expected\n{test_outputs[idx]}\nGot{actual_output}")
                            message = "Wrong Answer"
                            exec_result.append(0)
                            if early_stop:
                                break
                # Runtime Error
                else:
                    # message = (f"Runtime Error\nInput:\n{input_data}\n" + 
                    #         f"Error Message:\n{exec_result_run.stderr}")
                    message = f"Runtime Error: {exec_result_run.stderr}"
                    exec_result.append(-1)
                    if early_stop:
                        break
                    
            except Exception as e:
                # message = (f"Runtime Error\nInput:\n{input_data}\n" + 
                #         f"Error Message:\n{str(e)}")
                message = f"Runtime Error: {str(e)}"
                exec_result.append(-1)
                if early_stop:
                    break
                
        passed = (
            test_outputs is not None and
            compiled and
            bool(np.all(np.array(exec_result) == 1))
        )
        
    
    # clean up
    try:
        os.remove(path)
    except Exception as e:
        pass
    
    try:
        if extra_cleanup is not None:
            extra_remove_path = extra_cleanup(base_path)
            assert isinstance(extra_remove_path, str)
            os.remove(extra_remove_path)
    except Exception as e:
        pass
    
    
    return dict(
        time_elapsed=elapsed,
        exec_result=exec_result,
        compiled=compiled,
        passed=passed,
        message=message,
        inputs=test_inputs,
        outputs=actual_outputs
    )
    

def execute_cpp(code: str, 
                test_inputs: List[str], 
                test_outputs: List[str] = None,
                exec_timeout: float = 100,
                compile_timeout: float=100,
                early_stop: bool=True,
                strict_compare: bool=False,
                verbose=False):
    return execute_helper(
        code=code,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        language="cpp",
        extension=".cpp",
        compile_command_lambda=lambda x: [
            "g++",
            f"{os.path.basename(x)}.cpp",
            "-o",
            f"{os.path.basename(x)}_cpp",
        ],
        compile_timeout=compile_timeout,
        exec_timeout=exec_timeout,
        subprocess_command_lambda=lambda x: [f"./{os.path.basename(x)}_cpp"],
        extra_cleanup=lambda x: f"{x}_cpp",
        cwd=True,
        early_stop=early_stop,
        strict_compare=strict_compare,
        verbose=verbose
    )
    

def execute_python(code: str, 
                   test_inputs: List[str], 
                   test_outputs: List[str] = None,
                   exec_timeout: float = 100,
                   compile_timeout: float=100,
                   early_stop: bool=True,
                   strict_compare: bool=False,
                   verbose=False):
    
    return execute_helper(
        code=code,
        test_inputs=test_inputs,
        test_outputs=test_outputs,
        language="python",
        extension=".py",
        compile_command_lambda=lambda x: [
            "python",
            "-m", 
            "py_compile",
            f"{os.path.basename(x)}.py"
        ],
        compile_timeout=compile_timeout,
        exec_timeout=exec_timeout,
        subprocess_command_lambda=lambda x: [
            "python",
            f"{os.path.basename(x)}.py"
        ],
        cwd=True,
        early_stop=early_stop,
        strict_compare=strict_compare,
        verbose=verbose
    )


def check_compilable_java(
    code: str, 
    compile_timeout: float=100,
    verbose: bool=False,
    language="java"
):
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    entire_string = code
    base_path = setup_base_path(current_dir, f"{language}_exec_eval", "")
    try:
        os.makedirs(base_path, exist_ok=False)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    pattern = r'\bpublic\s+class\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
    match = re.search(pattern, entire_string)
    class_name = match.group(1) if match else "Main"

    path = os.path.join(base_path, f"{class_name}.{language}")

    with open(path, "w") as f:
        f.write(entire_string)

    elapsed = -1.0
    exec_result = []
    passed = False
    try:
        exec_result_compile = subprocess.run(
            [f"javac", os.path.basename(path)],
            timeout=int(compile_timeout),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_path
        )
        compiled = exec_result_compile.returncode == 0
        message = exec_result_compile.stderr
        if verbose:
            print("exec_result_compile", exec_result_compile)
    except Exception as e:
        compiled = False
        message = str(e)

    try:
        shutil.rmtree(base_path)
    except Exception as e:
        if verbose:
            print(f"Error cleaning up directory {base_path}: {e}")


    return compiled, message


def check_correctness_java(
    problem: Dict,
    completion: str,
    timeout: float,
    verbose=False,
    language="java",
    compile_timeout: float = 100,
    early_stop: bool = True,
    strict_compare: bool = False
):
    """
    Run all evaluation under java_exec_eval + randomized directory to avoid collision.
    Using subprocess with concurrent.futures for multi-thread evaluation.
    Make sure to clean up resources even if the test cases fail.
    """

    current_dir = os.path.dirname(os.path.realpath(__file__))
    entire_string = completion
    base_path = setup_base_path(current_dir, f"{language}_exec_eval", "")
    try:
        os.makedirs(base_path, exist_ok=False)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        
    pattern = r'\bpublic\s+class\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
    match = re.search(pattern, entire_string)
    class_name = match.group(1) if match else "Main"

    path = os.path.join(base_path, f"{class_name}.{language}")

    with open(path, "w") as f:
        f.write(entire_string)

    elapsed = -1.0
    exec_result = []
    passed = False
    try:
        exec_result_compile = subprocess.run(
            [f"javac", os.path.basename(path)],
            timeout=int(compile_timeout),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=base_path
        )
        compiled = exec_result_compile.returncode == 0
        if verbose:
            print("exec_result_compile", exec_result_compile)
    except Exception as e:
        compiled = False
        passed = False
        exec_result = []
        message = str(e)
        elapsed = -1.0
        

    
    tests = merge_dicts([problem["public_tests"], 
                         problem["private_tests"], 
                         problem["generated_tests"]])
    
    if compiled:
        
        inputs, outputs = tests["input"], tests["output"]
        for input_data, expected_output in zip(inputs, outputs):        
            try:
                start = time.time()
                exec_result_run = subprocess.run(
                    [f"java", "-cp", ".", class_name],
                    input=input_data,
                    timeout=int(timeout),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=base_path
                )
                elapsed = 1000.0 * (time.time() - start)
                if verbose:
                    print("exec result run", exec_result_run)
            
                
                actual_output = exec_result_run.stdout.strip()
                if exec_result_run.returncode == 0:
                    if strcmp(actual_output, expected_output, strict_compare):
                        message = f"Passed\n actual_output:\n{actual_output}\nexpected_output:\n{expected_output}"
                        exec_result.append(1)
                    else:
                        message = f"Output mismatch. Input:\n{input_data}\nExpected:\n{expected_output.strip()}\nGot:\n{actual_output}\n"
                        exec_result.append(0)
                        if early_stop:
                            break

                else:
                    message = f"Input:\n{input_data}\n\n" + exec_result_run.stderr
                    exec_result.append(-1)
                    if early_stop:
                        break

            except Exception as e:
                message = str(e)
                exec_result.append(-1)
                if early_stop:
                    break
                
            if verbose:
                print(message)

        passed = compiled and bool(np.all(np.array(exec_result) == 1))
        if verbose:
            print(f"Test Result: Passed={passed}, Message={message}, Elapsed={elapsed}")

    else:
        # branch for compile failed.
        passed = False
        message = exec_result_compile.stderr
        elapsed = -1.0

    try:
        shutil.rmtree(base_path)
    except Exception as e:
        if verbose:
            print(f"Error cleaning up directory {base_path}: {e}")

    return dict(
        time_elapsed=elapsed,
        name=problem["name"],
        compiled=compiled,
        result=exec_result,
        passed=passed,
        message=message
    )


def check_correctness_cpp(
    problem: Dict,
    completion: str,
    timeout: float,
    verbose=False,
    early_stop=True,
    strict_compare=False
):
    return check_correctness_helper(
        problem=problem,
        completion=completion,
        timeout=timeout,
        verbose=verbose,
        language="cpp",
        extension=".cpp",
        compile_command_lambda=lambda x: [
            "g++",
            f"{os.path.basename(x)}.cpp",
            "-o",
            f"{os.path.basename(x)}_cpp",
        ],
        compile_timeout=100,
        subprocess_command_lambda=lambda x: [f"./{os.path.basename(x)}_cpp"],
        extra_cleanup=lambda x: f"{x}_cpp",
        cwd=True,
        early_stop=early_stop,
        strict_compare=strict_compare
    )

def check_correctness_python(
    problem: Dict,
    completion: str,
    timeout: float, 
    verbose: bool=False,
    early_stop: bool=True,
    strict_compare: bool=False
):
    return check_correctness_helper(
        problem=problem,
        completion=completion,
        timeout=timeout,
        verbose=verbose,
        language="python",
        extension=".py",
        compile_command_lambda=lambda x: [
            "python",
            "-m", 
            "py_compile",
            f"{os.path.basename(x)}.py"
        ],
        compile_timeout=100,
        subprocess_command_lambda=lambda x: [
            "python",
            f"{os.path.basename(x)}.py"
        ],
        early_stop=early_stop,
        strict_compare=strict_compare,
        cwd=True
    )


def setup_base_path(
    current_dir,
    language_dirname,
    extension
):
    with lock:
        if not os.path.isdir(os.path.join(current_dir, language_dirname)):
            os.makedirs(os.path.join(current_dir, language_dirname))

    num_attempts, path = 0, None
    while True:
        num_attempts += 1
        if num_attempts > 10:
            assert False, "Unable to avoid filename collision"
        basename = "".join(
            random.choices(string.ascii_lowercase + string.ascii_uppercase, k=10)
        )

        base_path = os.path.join(current_dir, language_dirname, f"{basename}")
        path = base_path + f"{extension}"

        if extension == "":
            if not os.path.isdir(path):
                to_return = path
                break
        if not os.path.isfile(path):
            to_return = base_path
            break

    return to_return


def check_compilable_cpp(
    code: str,
    compile_timeout: float=100,
    verbose=False
):
    return check_compilable_helper(
        code=code,
        compile_timeout=compile_timeout,
        verbose=verbose,
        language="cpp",
        extension=".cpp",
        compile_command_lambda=lambda x: [
            "g++",
            f"{os.path.basename(x)}.cpp",
            "-o",
            f"{os.path.basename(x)}_cpp",
        ],
        extra_cleanup=lambda x: f"{x}_cpp",
        cwd=True
    )
    

def check_compilable_python(
    code: str,
    compile_timeout: float=100,
    verbose=False
):
    return check_compilable_helper(
        code=code,
        compile_timeout=compile_timeout,
        verbose=verbose,
        language="python",
        extension=".py",
        compile_command_lambda=lambda x: [
            "python",
            "-m",
            "py_compile",
            f"{os.path.basename(x)}.py"
        ],
        cwd=True
    )


def check_compilable_helper(
    code: str,
    compile_timeout: float=100,
    verbose: bool=False,
    language=None,
    extension=None,
    compile_command_lambda=None,
    extra_cleanup=None,
    cwd=None
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    entire_string = code

    language_dirname = f"{language}_exec_eval"

    base_path = setup_base_path(current_dir, language_dirname, extension)
    path = base_path + f"{extension}"

    if cwd is not None:
        cwd = os.path.dirname(base_path)
    with open(path, "w") as f:
        f.write(entire_string)
        
    if compile_command_lambda is not None:
        try:
            compile_result = subprocess.run(
                compile_command_lambda(base_path),
                timeout=int(compile_timeout),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
            )
            compiled = compile_result.returncode == 2 if language == "typescript" else compile_result.returncode == 0
            message = compile_result.stderr
        except Exception as e:
            compiled, message = False, str(e)
    else:
        compiled = True
    
    # clean up
    try:
        os.remove(path)
    except Exception as e:
        if verbose:
            print(f"Error trying to clean up file: {e}")
    try:
        if extra_cleanup is not None:
            extra_remove_path = extra_cleanup(base_path)
            assert isinstance(extra_remove_path, str)
            os.remove(extra_remove_path)
    except Exception as e:
        if verbose:
            print(f"Error trying to clean up file: {e}")

    return compiled, message



def check_correctness_helper(
    problem: Dict,
    completion: str,
    timeout: float,
    verbose=False,
    language=None,
    extension=None,
    subprocess_command_lambda=None,
    compile_timeout=100,
    compile_command_lambda=None,
    extra_cleanup=None,
    cwd=None,
    early_stop: bool=True,
    strict_compare: bool=False
):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    entire_string = completion

    language_dirname = f"{language}_exec_eval"

    base_path = setup_base_path(current_dir, language_dirname, extension)
    path = base_path + f"{extension}"

    if cwd is not None:
        cwd = os.path.dirname(base_path)
    with open(path, "w") as f:
        f.write(entire_string)
        
        
    elapsed = -1.0
    exec_result = []
    passed = True
    
    if compile_command_lambda is not None:
        try:
            compile_result = subprocess.run(
                compile_command_lambda(base_path),
                timeout=int(compile_timeout),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
            )
            compiled = compile_result.returncode == 2 if language == "typescript" else compile_result.returncode == 0
            message = compile_result.stderr
        except Exception as e:
            compiled, message = False, str(e)
    else:
        compiled = True

        
    tests = merge_dicts([problem["public_tests"], 
                         problem["private_tests"], 
                         problem["generated_tests"]])
    
    if compiled:
        inputs, outputs = tests["input"], tests["output"]
        for input_data, expected_output in zip(inputs, outputs):
            try:
                start = time.time()
                exec_result_run = subprocess.run(
                    subprocess_command_lambda(base_path),
                    input=input_data,
                    timeout=int(timeout),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd,
                )
                elapsed = 1000.0 * (time.time() - start)
                if verbose:
                    print("exec result run", exec_result_run)

                actual_output = exec_result_run.stdout.strip()
                if exec_result_run.returncode == 0:
                    if strcmp(actual_output, expected_output, strict_compare):
                        message = f"Passed\nInput:\n{input_data}\Expected:\n{expected_output.strip()}\nexpected_output:\n{actual_output}"
                        exec_result.append(1)
                    else:
                        message = f"Output mismatch.\nInput:\n{input_data}\nExpected:\n{expected_output.strip()}\nGot:\n{actual_output}\n"
                        exec_result.append(0)
                        if early_stop:
                            break
                else:
                    message = f"Input:\n{input_data}\n\n" + exec_result_run.stderr
                    exec_result.append(-1)
                    if early_stop:
                        break

            except Exception as e:
                message = str(e)
                exec_result.append(-1)
                if early_stop:
                    break
            
        passed = (
            compiled and len(exec_result) > 0 and 
            bool(np.all(np.array(exec_result) == 1))
        )
        if verbose:
            print(f"Test Result: Passed={passed}, Message={message}, Elapsed={elapsed}")

    else:
        passed, elapsed = False, -1.0

    if early_stop and verbose:
        print(f"Test Result: Passed={passed}, Message={message}, Elapsed={elapsed}")


    # clean up
    try:
        os.remove(path)
    except Exception as e:
        if verbose:
            print(f"Error trying to clean up file: {e}")
    try:
        if extra_cleanup is not None:
            extra_remove_path = extra_cleanup(base_path)
            assert isinstance(extra_remove_path, str)
            os.remove(extra_remove_path)
    except Exception as e:
        if verbose:
            print(f"Error trying to clean up file: {e}")

    # get result
    return dict(
        time_elapsed=elapsed,
        name=problem["name"],
        passed=passed,
        result=exec_result,
        compiled=compiled,
        message=message
    )




if __name__ == "__main__":
    # simple tests
    
    # '1575_A. Another Sorting Problem'
    # from datasets import load_dataset
    # ds = load_dataset("deepmind/code_contests", split="test")
    # problem = ds[0]
    
    # import json
    # with open("mxeval/test_problem.json", "r") as f:
    #     first_line = f.readline()
    #     try:
    #         data_dict = json.loads(first_line)
    #     except json.JSONDecodeError as e:
    #         print(f"Error parsing JSON: {e}")
    #         data_dict = None
            
    # problem = data_dict
    # check_list = [9, 112, 113]
    # failed_list = []
    # for i, (solution, language) in enumerate(zip(problem["solutions"]["solution"], problem["solutions"]["language"])):
    #     if i not in check_list:
    #         continue
    #     if language == 2:
    #         result = check_correctness_cpp(problem, solution, 10,
    #                                        verbose=True)
    #     elif language == 3:
    #         result = check_correctness_python(problem, solution, 10,
    #                                           verbose=True)
    #     elif language == 4:
    #         result = check_correctness_java(problem, solution, 10,
    #                                         verbose=True)
    #     else:
    #         continue
        

    #     print(result)
    #     if not result["passed"]:
    #         print(f"Solution {i} failed.")
    #         print(solution)
    #         failed_list.append(i)
            
    # print(f"failed solutions: {failed_list}")
    # print(f"{len(failed_list)} / {len(problem['solutions']['solution'])}")
    
    from datasets import load_dataset
    ds = load_dataset("deepmind/code_contests", split="test")
    problems = {example["name"]: example for example in ds}
    
    import json
    with open("mxeval/test_cpp2python.json", "r") as fp:
        solution = json.load(fp)
        
    print(solution["code2_santized"])
    # result = check_correctness_python(problems[solution['name']], solution['code2_santized'], 10, verbose=True)
    results = execute_python(solution["code2_santized"],
                           test_inputs=problems[solution['name']]['public_tests']['input'],
                           test_outputs=problems[solution['name']]['public_tests']['output'])
    print(results)
