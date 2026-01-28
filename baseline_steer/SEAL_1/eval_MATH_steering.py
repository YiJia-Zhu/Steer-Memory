import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import  AutoTokenizer
from modeling_utils.modeling_qwen2 import Qwen2ForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from peft import PeftModel, PeftConfig

import sys
import os
import gc
from tqdm import trange

from get_math_results import main as eval_main
from data_loader import load_dataset_seal
from answer_extractor import extract_answer_by_dataset, compare_answers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def evaluate_and_save_classification(predictions, dataset_lower: str, save_dir: str, token_counts):
    """
    Evaluate multiple-choice / non-math datasets using answer_extractor.
    Saves a math_eval.jsonl-compatible file for downstream hidden_analysis.
    """
    results = []
    correct = 0
    for pred in predictions:
        gens = pred["model_generation"]
        all_pred = [extract_answer_by_dataset(dataset_lower, g) for g in gens]
        all_eval = [compare_answers(p, pred["answer"], dataset_lower) for p in all_pred]
        mv_pred = Counter(all_pred).most_common(1)[0][0] if all_pred else None
        mv_index = all_pred.index(mv_pred) if mv_pred in all_pred else 0
        mv_eval = all_eval[mv_index] if all_eval else False
        correct += int(mv_eval)
        pred_copy = dict(pred)
        pred_copy.update({
            "all_pred": all_pred,
            "all_eval": all_eval,
            "mv_pred": mv_pred,
            "mv_eval": mv_eval,
            "mv_index": mv_index,
        })
        results.append(pred_copy)

    acc = correct / len(predictions) if predictions else 0.0
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts) if token_counts else 0

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "math_eval.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    metrics = {"acc": acc}
    if token_counts:
        metrics["avg_tokens"] = avg_tokens
        metrics["total_tokens"] = total_tokens
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    print(f"\nEval -> acc: {acc:.3f}, avg_tokens: {avg_tokens:.2f}, total_tokens: {total_tokens}")


def extract_box(pred_str):
    ans = pred_str.split("boxed")[-1]
    if len(ans) == 0:
        return ""
    elif ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()

    return a


def extract_last_number(pred_str):
    o = re.sub(r"(\d),(\d)", r"\1\2", pred_str)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", o)
    if numbers:
        ans = numbers[-1]
    else:
        ans = None
    return ans


def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    
    # Try to load using new data loader for supported datasets
    supported_datasets = ["aime_2024", "aime25", "amc23", "arc-c", "math500", "openbookqa"]
    if args.dataset.lower() in supported_datasets:
        # Use new data loader with datasets directory (local to SEAL_1)
        data_root = "datasets"  # Relative path, will look for datasets/ in current directory
        offset = getattr(args, 'offset', 0)  # Default to 0 if not provided
        test_data = load_dataset_seal(args.dataset, data_root, max_examples=None, offset=offset)
    elif args.dataset == "MATH500":
        data = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for example in data:
            gt = extract_box(example["solution"])
            test_data.append({
                "question": example["problem"],
                "answer": example["solution"],
                "gt":gt,
            })
    elif args.dataset == "GSM":
        data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer":example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError(f"Dataset not supported: {args.dataset}")
    if args.start:
        test_data = test_data[args.start:]
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Choose appropriate prompt based on dataset type
    dataset_lower = args.dataset.lower()
    if dataset_lower in ["arc-c", "openbookqa"]:
        # Multiple choice questions
        prefix = "Answer the following multiple choice question. Choose the correct option (A, B, C, D, or E) and explain your reasoning.\n"
    else:
        # Math questions
        prefix = "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
    
    prompts = []
    for i, example in enumerate(test_data):
        prompt = prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        if args.use_chat_format:
            user_content = prefix + "Question: " + example["question"].strip()
            if "deepseek" in args.model_name_or_path or "qwen" in args.model_name_or_path.lower():
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])


    if "qwen" in args.model_name_or_path.lower():
        model = Qwen2ForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    else:
        raise ValueError("Model not supported")
    
    if args.steering:
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        steer_vec = steer_vec.to(model.device)
        model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  steer_coef=args.steering_coef, tokenizer=tokenizer)

    outputs = []
    token_counts = []
    for i in trange(0, len(prompts), args.batch_size):
        if args.steering:
            model.start_new_round()
        batch = prompts[i:i+args.batch_size]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model.generate(**tokenized_batch, do_sample=False, max_new_tokens=args.max_tokens,use_cache=True)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        # Count tokens generated for each example in the batch
        for o in output:
            tokens_generated = len(o) - prompt_len
            token_counts.append(tokens_generated)
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)

    outputs = [[trim_output(o)] for o in outputs]


    predictions = [{
        "prompt": prompt,
        "problem": example["question"],
        "answer": example["gt"],
        "solution":  example["answer"],
        "model_generation": output,
        "tokens_used": tokens,
    } for example, output, prompt, tokens in zip(test_data, outputs, prompts, token_counts)]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")
    
    # Print token statistics
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts) if token_counts else 0
    print(f"\nToken Statistics:")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Average tokens per example: {avg_tokens:.2f}")
    print(f"  Min tokens: {min(token_counts) if token_counts else 0}")
    print(f"  Max tokens: {max(token_counts) if token_counts else 0}")

    # Evaluate and save metrics
    if dataset_lower in {"math500", "math-500", "aime_2024", "aime2024", "aime25", "aime_25", "aime_2025", "amc23", "amc_23", "amc_2023"}:
        eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
    else:
        evaluate_and_save_classification(predictions, dataset_lower, args.save_dir, token_counts)
    
    # Clean up model and free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of examples to skip from the beginning (for train/eval split)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MATH",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--steering",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--steering_vector",
        type=str,
        default=None
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0
    )


    args = parser.parse_args()

    if args.steering:
        vector_name_split = args.steering_vector.split("/")[-3:]
        vector_name_split[-1] = vector_name_split[-1].split(".")[0]
        name = "_".join(vector_name_split)
        args.save_dir = os.path.join(args.save_dir, name, f"coef_{args.steering_coef}")
    else:
        args.save_dir = os.path.join(args.save_dir, "base")
    
    if args.remove_bos:
        args.save_dir = args.save_dir + "_remove_bos"

    if args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")
        
    print(args.save_dir)
    main(args)
    # Evaluation is handled inside main for all datasets


        
