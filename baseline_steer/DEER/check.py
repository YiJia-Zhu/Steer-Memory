import json
from transformers import AutoTokenizer

import re
import importlib.util
import os
import argparse

import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.data_loader import load_data
from utils.grader import *
from utils.answer_extractor import extract_gold, extract_pred
import pickle
from math import comb
import pdb


def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--generation_path", default="test", type=str)

    parser.add_argument("--prompt_type", default="qwen-base", type=str)

    args = parser.parse_args()
    


    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format

def read_jsonl(file_path):

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:

            json_obj = json.loads(line.strip())
            data.append(json_obj)
    return data



def infer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    examples = load_data(args.data_name, args.split, args.data_dir)
    file_outputs = read_jsonl(args.generation_path)
    
    
    print("llm generate done")
    print(len(file_outputs))
    
    pass_at_k_list = []
    k = args.k
    
    correct_cnt = 0
    example_by_idx = {ex["idx"]: ex for ex in examples} if examples else {}
    outputs_have_idx = all("idx" in fo for fo in file_outputs)

    evaluated_outputs = []
    skipped_without_gold = 0
    missing_example = 0

    for i in tqdm(range(len(file_outputs)), "check correct..."):
        output = file_outputs[i]

        # Prefer matching by idx when available to avoid order mismatches.
        example = None
        if outputs_have_idx and example_by_idx:
            example = example_by_idx.get(output.get("idx"))
            if example is None:
                missing_example += 1
        elif i < len(examples):
            example = examples[i]

        gold_field = None
        if example is not None:
            gold_field = example.get("answer", "")
        if gold_field in [None, ""]:
            gold_field = output.get("gold_answer")

        if gold_field in [None, ""]:
            skipped_without_gold += 1
            continue

        gt_ans = extract_gold(args.data_name, str(gold_field))
        if gt_ans is None:
            skipped_without_gold += 1
            continue

        generated_responses = output.get('generated_responses', [])
        generated_answers = [extract_pred(args.data_name, generated_response) for generated_response in generated_responses]
        is_correct_list = [
            check_is_correct(generated_answer, gt_ans) if generated_answer is not None and gt_ans is not None else False
            for generated_answer in generated_answers
        ]
        is_correct = any(is_correct_list)
        if is_correct:
            correct_cnt += 1

        output['generated_answers'] = generated_answers
        output['gold_answer'] = gt_ans
        output['is_correct'] = is_correct
        output['answers_correctness'] = is_correct_list
        evaluated_outputs.append(output)
        
        if len(is_correct_list) > 1:
            correct_answers = sum(is_correct_list)
            n = len(generated_answers)
            if correct_answers > 0:
                if n - correct_answers < k:
                    pass_at_k = 1
                else:
                    pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                pass_at_k_list.append(pass_at_k)
            else:
                pass_at_k_list.append(0)
                
    
    total_eval = len(evaluated_outputs)
    total_outputs = len(file_outputs)
    if missing_example or skipped_without_gold:
        print(f"Missing matching examples for {missing_example} outputs; skipped {skipped_without_gold} without usable gold answers.")
    print(f"correct cnt / total cnt: {correct_cnt}/{total_eval}")
    if total_eval > 0:
        print(f"Acc: {correct_cnt / total_eval:.4f}")
    else:
        print("Acc: 0.0000 (no examples evaluated)")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        if total_eval > 0:
            print(f"Pass@1: {correct_cnt}/{total_eval} = {correct_cnt / total_eval:.4f}")
        else:
            print("Pass@1: 0/0 = 0.0000")



    response_length = []
    token_num = []
    wait_num = []
    alt_num = []

    test_num = len(evaluated_outputs)
    correct_num = 0
    for data in evaluated_outputs:
        if not data.get('generated_responses'):
            continue
        response_length.append(len(data['generated_responses'][0].split()))
        tokens_response_len = len(tokenizer(data['generated_responses'][0])['input_ids'])
        token_num.append(tokens_response_len)
        

    if test_num > 0:
        avg_response_length = sum(response_length) / test_num if response_length else 0
        avg_token_num = sum(token_num) / test_num if token_num else 0

        print("length:", avg_response_length)
        print('token_num:', avg_token_num)
    else:
        print("No responses to compute length/token statistics.")


if __name__ == "__main__":
    args = parse_args()
    infer(args)
