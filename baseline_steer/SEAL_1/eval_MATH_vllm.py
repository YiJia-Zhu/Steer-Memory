
import argparse
import os
import re
import json
import random
import torch
import torch.nn.functional as F
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial
from typing import List, Sequence


import sys
import os
import gc

from get_math_results import main as eval_main
from data_loader import load_dataset_seal
from answer_extractor import extract_answer_by_dataset, compare_answers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

exact_match = evaluate.load("exact_match")


def average_recent(values: List[float], window: int) -> float:
    """Compute average of last N values; returns 0 if not enough values."""
    if window <= 0 or len(values) < window:
        return 0.0
    return sum(values[-window:]) / window


def ends_with_sequence(tokens: List[int], seq: Sequence[int]) -> bool:
    """Check if tokens end with given sequence."""
    if not seq or len(tokens) < len(seq):
        return False
    return tokens[-len(seq):] == list(seq)


def build_prompt_prefix(dataset_lower: str, method: str) -> str:
    """
    Choose prompt prefix by dataset type and method.
    - vanilla: original SEAL prompt (math / multiple choice)
    - cod: Chain-of-Draft inspired concise reasoning
    """
    if method == "cod":
        if dataset_lower in ["arc-c", "openbookqa"]:
            return (
                "Solve the following problem step by step.\n"
                "Draft each reasoning step in <=5 words, keep it concise.\n"
                "After the reasoning, choose the correct option (A, B, C, D, or E).\n"
                "Put only the chosen letter on the last line in the format:\n"
                "#### <A/B/C/D/E>\n"
            )
        return (
            "Think step by step, but keep each draft line under 5 words. "
            "Return the final numeric answer within \\boxed{} and avoid extra text."
        )

    if dataset_lower in ["arc-c", "openbookqa"]:
        return (
            "Solve the following problem step by step.\n"
            "You MUST explain your reasoning before giving the final choice.\n"
            "After the reasoning, choose the correct option (A, B, C, D, or E).\n"
            "Put only the chosen letter on the last line in the format:\n"
            "#### <A/B/C/D/E>\n"
        )
    return "Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}."


def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits


def calc_answer_confidence(model, tokenizer, base_ids, answer_prompt_ids, max_steps=20, stop_ids=None):
    """
    粗略评估当前思考前缀后接上答案提示时的平均最大概率，用于 DEER 早停。
    """
    device = base_ids.device
    input_ids = torch.cat([base_ids, answer_prompt_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    scores = []
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = outputs.past_key_values

        for _ in range(max_steps):
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            max_prob, next_token = torch.max(probs, dim=-1)
            token_id = next_token.item()
            scores.append(max_prob.item())

            if stop_ids and token_id in stop_ids:
                break

            outputs = model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values

    return sum(scores) / len(scores) if scores else 0.0


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
        # Majority vote over extracted preds (fallback to first)
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

def deer_generate_single(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    threshold: float,
    patience: int,
    think_ratio: float,
    max_think_tokens: int,
    min_think_tokens: int,
    answer_tokens: int,
    stop_sequences: List[List[int]],
):
    """
    近似原始 DEER 的思考-判定-回答流程，保持 SEAL 现有 prompt 不变：
    - 思考阶段贪心生成，遇到 stop token 或长度上限时停；否则在思考后做“答案前瞻”概率检查，
      如果高于阈值则提前进入答题，否则追加“Wait”继续思考。
    - 答题阶段用剩余预算生成答案。
    返回 (生成文本, token 总数, stop_reason)。
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"][0].tolist()

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_key_values = None

    generated_tokens: List[int] = []
    # Use a large thinking budget by default (closer to original DEER); fall back to max_tokens when unset/negative.
    effective_max_think = max_tokens if max_think_tokens is None or max_think_tokens <= 0 else max_think_tokens
    think_budget = max(
        min_think_tokens + 1,  # ensure we always allow at least the minimum thinking tokens
        min(max_tokens - 1, int(max_tokens * think_ratio), effective_max_think),
    )
    # Treat "Wait" as a soft boundary token: stop the current generation chunk but keep thinking.
    continue_ids = tokenizer("Wait", add_special_tokens=False)["input_ids"]
    continue_stop_ids = set(continue_ids)

    # Hard stop tokens that should end the thinking phase.
    think_stop_ids = set()
    for seq in stop_sequences:
        think_stop_ids.update(seq)
    # Combine for chunk-level stopping (either boundary or real stop).
    segment_stop_ids = think_stop_ids | continue_stop_ids

    answer_prompt_ids = tokenizer("\n</think>\n\nFinal answer: ", add_special_tokens=False)["input_ids"]
    answer_prompt_tensor = torch.tensor([answer_prompt_ids], device=device)

    stop_reason = None
    prob_history: List[float] = []

    orig_eos = getattr(model.config, "eos_token_id", None)
    orig_gen_eos = getattr(getattr(model, "generation_config", model.config), "eos_token_id", None)
    # 禁用思考阶段的默认 EOS，避免过早停在 <|im_end|>
    model.config.eos_token_id = None
    if hasattr(model, "generation_config"):
        model.generation_config.eos_token_id = None

    # Generate in small chunks to allow early-exit checks to trigger before the full think budget is consumed.
    max_step = max(16, min(128, think_budget))

    with torch.no_grad():
        while True:
            remaining_think = think_budget - len(generated_tokens)
            if remaining_think <= 0:
                stop_reason = "max_think"
                break

            step_budget = min(max_step, remaining_think)
            gen_outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=step_budget,
                do_sample=False,
                eos_token_id=list(segment_stop_ids) if segment_stop_ids else None,
                return_dict_in_generate=True,
                output_scores=False,
                use_cache=True,
            )

            new_tokens = gen_outputs.sequences[0].tolist()[len(input_ids[0]):]
            # HuggingFace includes the stop token when generation halts early; strip it and record separately.
            stop_token = None
            if new_tokens and new_tokens[-1] in segment_stop_ids:
                stop_token = new_tokens[-1]
                new_tokens = new_tokens[:-1]

            generated_tokens.extend(new_tokens)
            input_ids = torch.tensor([prompt_ids + generated_tokens], device=device)
            attention_mask = torch.ones_like(input_ids)

            if stop_token is not None and stop_token in think_stop_ids:
                stop_reason = "stop_token"
                break

            # Enforce a minimum thinking length before early-exit checks
            if len(generated_tokens) >= min_think_tokens:
                base_ids = torch.tensor([prompt_ids + generated_tokens], device=device)
                pred_prob = calc_answer_confidence(
                    model,
                    tokenizer,
                    base_ids,
                    answer_prompt_tensor,
                    max_steps=20,
                    stop_ids=think_stop_ids,
                )
                prob_history.append(pred_prob)
                avg_prob = average_recent(prob_history, patience) if patience and patience > 0 else pred_prob
                if avg_prob >= threshold:
                    stop_reason = "probability"
                    break

            remaining_think = think_budget - len(generated_tokens)
            if remaining_think <= 0:
                stop_reason = "max_think"
                break

            # Even if the model did not emit a continue token, append one to keep the loop segmented.
            if continue_ids:
                generated_tokens.extend(continue_ids)
                input_ids = torch.tensor([prompt_ids + generated_tokens], device=device)
                attention_mask = torch.ones_like(input_ids)
            else:
                stop_reason = "no_continue_token"
                break

    # 恢复模型原始的 eos 设置，供答题阶段使用
    model.config.eos_token_id = orig_eos
    if hasattr(model, "generation_config"):
        model.generation_config.eos_token_id = orig_gen_eos

        thinking_ids = torch.tensor([generated_tokens], device=device)
        final_input_ids = torch.cat([inputs["input_ids"], thinking_ids, answer_prompt_tensor], dim=1)

        remaining_budget = max(max_tokens - len(generated_tokens), 1)
        if answer_tokens is None or answer_tokens <= 0:
            answer_budget = remaining_budget
        else:
            answer_budget = max(1, min(answer_tokens, remaining_budget))

        final_outputs = model.generate(
            input_ids=final_input_ids,
            max_new_tokens=answer_budget,
            do_sample=False,
            use_cache=True,
        )

        full_output_ids = final_outputs[0].tolist()[len(prompt_ids):]
        final_text = tokenizer.decode(full_output_ids, skip_special_tokens=True)

        total_tokens = len(generated_tokens) + (len(final_outputs[0]) - len(final_input_ids[0]))

    return final_text, total_tokens, stop_reason


def main(args):
    random.seed(42)

    # Normalize method for internal branching (seal treated as vanilla prompt/steering)
    method_kind = "vanilla" if args.method == "seal" else args.method

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
    elif args.dataset == "MATH_train":
        data_path = "data/MATH/train.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                gt = extract_box(example["solution"])
                test_data.append({
                    "question": example["problem"],
                    "answer": example["solution"],
                    "gt":gt,
                })
    elif args.dataset in ["GSM", "GSM_train"]:
        if args.dataset == "GSM_train":
            data_path = "data/gsm/train.jsonl"
        else:
            data_path = "data/gsm/test.jsonl"
        with open(data_path) as fin:
            for line in fin:
                example = json.loads(line)
                answer = example["answer"].split("####")[1].strip()
                answer =  re.sub(r"(\d),(\d)", r"\1\2", answer)
                test_data.append({
                    "question": example["question"],
                    "answer": example["answer"].split("####")[0].strip(),
                    "gt": answer
                })
    else:
        raise ValueError(f"Dataset not supported: {args.dataset}")
    if args.max_examples and len(test_data) > args.max_examples:
        test_data = test_data[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Choose appropriate prompt based on dataset type and method
    dataset_lower = args.dataset.lower()
    prefix = build_prompt_prefix(dataset_lower, method_kind)
    prompt_prefix = prefix if prefix.endswith("\n") else prefix + "\n"
    
    prompts = []
    for i, example in enumerate(test_data):
        prompt = prompt_prefix + "Question: " + example["question"].strip() + "\nAnswer: "
        if args.use_chat_format:
            user_content = prompt_prefix + "Question: " + example["question"].strip()
            if "gemma" in args.model_name_or_path or "deepseek" in args.model_name_or_path or "qwen" in args.model_name_or_path.lower():
                # Align with Steer-Memory-114 style: always include a system header to avoid bare <|User|> prompts
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_content},
                ]
            else:
                messages = [{"role": "system", "content": prompt_prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    token_counts = []
    meta_infos = []
    model_generations = []

    if method_kind == "deer":
        deer_max_think_tokens = args.deer_max_think_tokens if args.deer_max_think_tokens is not None else args.max_tokens
        # HuggingFace generation with early-exit
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        stop_sequences = [
            tokenizer("</think>", add_special_tokens=False)["input_ids"],
            tokenizer("####", add_special_tokens=False)["input_ids"],
        ]
        if tokenizer.eos_token_id is not None:
            stop_sequences.append([tokenizer.eos_token_id])
        stop_sequences = [seq for seq in stop_sequences if seq]

        for prompt in prompts:
            text, token_num, stop_reason = deer_generate_single(
                model,
                tokenizer,
                prompt,
                max_tokens=args.max_tokens,
                threshold=args.deer_threshold,
                patience=args.deer_patience,
                think_ratio=args.deer_think_ratio,
                max_think_tokens=deer_max_think_tokens,
                min_think_tokens=args.deer_min_think_tokens,
                answer_tokens=args.deer_answer_tokens,
                stop_sequences=stop_sequences,
            )
            model_generations.append([trim_output(text)])
            token_counts.append(token_num)
            meta_infos.append({"stop_reason": stop_reason})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        # vLLM path (vanilla / CoD prompt variants)
        model = LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            swap_space=16,
            gpu_memory_utilization=0.8,
            enable_lora=args.peft is not None,
            tensor_parallel_size=1,
            max_lora_rank=128,
            max_model_len=args.max_tokens + 2000,
        )

        if not args.logit_adjustment:
            sampling_params = SamplingParams(n=1, temperature=0, max_tokens=args.max_tokens)
        else:
            vocab = tokenizer.get_vocab()
            logit_adjustment_tokens = torch.LongTensor(
                [vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]
            ).to("cuda")
            logit_adjustment_process = partial(
                logit_adjustment,
                adjust_ids=logit_adjustment_tokens,
                values=args.logit_adjustment_value,
                max_len=args.logit_adjustment_max_len,
            )
            sampling_params = SamplingParams(
                n=1,
                temperature=0,
                max_tokens=args.max_tokens,
                logits_processors=[logit_adjustment_process],
            )

        if args.peft is not None:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
        else:
            outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

        # Extract token counts from vLLM outputs (before processing)
        for output in outputs:
            if hasattr(output, 'outputs') and output.outputs:
                token_ids = getattr(output.outputs[0], 'token_ids', [])
                token_counts.append(len(token_ids) if token_ids else 0)
            else:
                token_counts.append(0)

        result = []
        for output in outputs:
            attempts = []
            for ith_output in output.outputs:
                attempts.append(ith_output.text)
            result.append(attempts)

        model_generations = [[trim_output(o) for o in output] for output in result]
        meta_infos = [None for _ in model_generations]
        del model

    predictions = []
    for example, output, prompt, tokens, meta in zip(test_data, model_generations, prompts, token_counts, meta_infos):
        pred_entry = {
            "prompt": prompt,
            "problem": example["question"],
            "answer": example["gt"],
            "solution":  example["answer"],
            "model_generation": output,
            "tokens_used": tokens,
            "method": args.method,
        }
        if meta:
            pred_entry.update(meta)
        predictions.append(pred_entry)

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")
    
    # Calculate and save token statistics
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts) if token_counts else 0
    print(f"\nToken Statistics:")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Average tokens per example: {avg_tokens:.2f}")
    print(f"  Min tokens: {min(token_counts) if token_counts else 0}")
    print(f"  Max tokens: {max(token_counts) if token_counts else 0}")

    # Evaluate and save metrics
    if dataset_lower in {"math500", "math-500", "aime_2024", "aime2024", "aime25", "aime_25", "aime_2025", "amc23", "amc_23", "amc_2023"}:
        # Use original math grader
        eval_main(os.path.join(args.save_dir, "predictions.jsonl"), save=True, k=None, output_dir=args.save_dir)
    else:
        evaluate_and_save_classification(predictions, dataset_lower, args.save_dir, token_counts)
    
    # Clean up models and free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max_examples",
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
        "--peft",
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
        "--method",
        type=str,
        default="seal",
        choices=["vanilla", "seal", "cod", "deer"],
        help="Prompt/decoding variant: seal/vanilla (default), cod (chain-of-draft-style prompt), deer (early exit)",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--deer_threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for DEER early exit (rolling average of max token prob).",
    )
    parser.add_argument(
        "--deer_patience",
        type=int,
        default=4,
        help="Number of recent tokens used to compute rolling confidence for DEER.",
    )
    parser.add_argument(
        "--deer_think_ratio",
        type=float,
        default=0.9,
        help="Portion of max_tokens allocated to the thinking phase in DEER.",
    )
    parser.add_argument(
        "--deer_max_think_tokens",
        type=int,
        default=None,
        help="Upper bound on thinking tokens for DEER to avoid extremely long loops. If unset, use max_tokens.",
    )
    parser.add_argument(
        "--deer_min_think_tokens",
        type=int,
        default=16,
        help="Minimum thinking tokens before DEER early-exit can trigger.",
    )
    parser.add_argument(
        "--deer_answer_tokens",
        type=int,
        default=None,
        help="Answer token budget after early exit in DEER. If unset or <=0, use all remaining tokens.",
    )


    args = parser.parse_args()

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)

    main(args)
    # math datasets are evaluated inside main; no extra call needed here
    



        
