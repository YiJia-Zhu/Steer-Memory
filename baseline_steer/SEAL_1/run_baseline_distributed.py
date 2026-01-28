#!/usr/bin/env python3
"""
SEAL Baseline - Distributed Multi-GPU Runner
Automatically distributes 24 configurations across available GPUs
Supports graceful shutdown on Ctrl+C
"""

import os
import sys
import argparse
import subprocess
import signal
import time
from datetime import datetime
from multiprocessing import Process, Queue, Event
from queue import Empty
import json

# Dataset configurations: train_examples (for vector extraction) and eval_examples (for testing)
# Following Steer-Memory-114 configuration: first N for mining, next M for evaluation (no overlap)
DATASET_CONFIG = {
    'math500': {'train': 100, 'eval': 400},      # 0-99 for training, 100-499 for eval
    'aime_2024': {'train': 10, 'eval': 20},      # 0-9 for training, 10-29 for eval
    'aime25': {'train': 10, 'eval': 20},         # 0-9 for training, 10-29 for eval
    'amc23': {'train': 10, 'eval': 30},          # 0-9 for training, 10-39 for eval
    'arc-c': {'train': 100, 'eval': 199},        # 0-99 for training, 100-298 for eval (total 299)
    'openbookqa': {'train': 100, 'eval': 400},   # 0-99 for training, 100-499 for eval (total 500)
}

# Dataset-specific max token overrides
DATASET_MAX_TOKENS = {
    'arc-c': 1024,
    'openbookqa': 1024,
}

# Global event for graceful shutdown
shutdown_event = Event()
active_processes = []

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n" + "="*60)
    print("🛑 Ctrl+C detected! Shutting down gracefully...")
    print("="*60)
    shutdown_event.set()
    
    # Terminate all active processes
    for proc in active_processes:
        if proc.is_alive():
            print(f"Terminating process {proc.pid}...")
            proc.terminate()
    
    # Wait for processes to finish
    for proc in active_processes:
        proc.join(timeout=5)
        if proc.is_alive():
            print(f"Force killing process {proc.pid}...")
            proc.kill()
    
    print("✓ All processes stopped")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_model_short_name(model_name):
    """Convert full model name to short name"""
    mapping = {
        "DeepSeek-R1-Distill-Qwen-1.5B": "DS-R1-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B": "DS-R1-7B",
        "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
        "Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    }
    return mapping.get(model_name, model_name)


def supports_steering(model_name):
    """Check if model supports steering"""
    return "qwen" in model_name.lower()


def run_single_config(gpu_id, model_path, model_short, dataset, args, task_id, total_tasks):
    """Run complete pipeline for a single configuration on specified GPU"""
    
    # Set GPU visibility
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    config_name = f"[{args.method}] {model_short} + {dataset}"
    print(f"\n{'='*60}")
    print(f"[GPU {gpu_id}] Task {task_id}/{total_tasks}: {config_name}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    # Create output directory
    save_dir = os.path.join(args.output_dir, model_short, dataset)
    os.makedirs(save_dir, exist_ok=True)
    
    # Log file for this configuration
    log_file = os.path.join(save_dir, "run.log")
    
    dataset_max_tokens = DATASET_MAX_TOKENS.get(dataset, args.max_tokens)

    try:
        # ============================================
        # STEP 1: Baseline Evaluation (for vector extraction)
        # ============================================
        print(f"[GPU {gpu_id}] [{config_name}] Step 1/3: Running baseline evaluation...")
        
        # Get dataset-specific configuration
        dataset_cfg = DATASET_CONFIG.get(dataset, {'train': None, 'eval': None})
        
        cmd = [
            "python", "eval_MATH_vllm.py",
            "--model_name_or_path", model_path,
            "--dataset", dataset,
            "--max_tokens", str(dataset_max_tokens),
            "--save_dir", save_dir,
            "--use_chat_format",
            "--remove_bos",
            "--method", args.method,
        ]

        if args.method == "deer":
            cmd.extend([
                "--deer_threshold", str(args.deer_threshold),
                "--deer_patience", str(args.deer_patience),
                "--deer_think_ratio", str(args.deer_think_ratio),
                "--deer_min_think_tokens", str(args.deer_min_think_tokens),
            ])
            if args.deer_max_think_tokens is not None:
                cmd.extend(["--deer_max_think_tokens", str(args.deer_max_think_tokens)])
            if args.deer_answer_tokens is not None:
                cmd.extend(["--deer_answer_tokens", str(args.deer_answer_tokens)])
        
        # Use dataset split appropriately:
        # - For steering methods (vanilla/seal), keep train split for vector mining.
        # - For non-steering methods (cod/deer), evaluate on the eval split to match SEAL final metrics.
        if args.method in ("cod", "deer") and dataset_cfg['train'] and dataset_cfg['eval']:
            cmd.extend(["--max_examples", str(dataset_cfg['eval'])])
            cmd.extend(["--offset", str(dataset_cfg['train'])])
        elif args.max_examples:
            # Quick test mode: use user-specified limit
            cmd.extend(["--max_examples", str(args.max_examples)])
            cmd.extend(["--offset", "0"])
        elif dataset_cfg['train']:
            # Normal steering pipeline: use dataset-specific train split
            cmd.extend(["--max_examples", str(dataset_cfg['train'])])
            cmd.extend(["--offset", "0"])
        
        with open(log_file, 'w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        
        # Check if baseline succeeded
        if not os.path.exists(os.path.join(save_dir, "predictions.jsonl")):
            raise Exception("Baseline evaluation failed: no predictions file")
        
        print(f"[GPU {gpu_id}] [{config_name}] ✓ Baseline evaluation completed")
        
        # ============================================
        # STEP 2: Vector Generation (if supported)
        # ============================================
        if args.method in ("vanilla", "seal") and supports_steering(model_path):
            print(f"[GPU {gpu_id}] [{config_name}] Step 2/3: Generating steering vectors...")
            
            # 2.1: Generate hidden states for correct samples
            cmd = [
                "python", "hidden_analysis.py",
                "--model_path", model_path,
                "--data_dir", save_dir,
                "--data_path", os.path.join(save_dir, "predictions.jsonl"),
                "--type", "correct",
                "--max_seq_len", str(dataset_max_tokens),
            ]
            
            with open(log_file, 'a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            
            # 2.2: Generate hidden states for incorrect samples
            cmd[cmd.index("correct")] = "incorrect"
            
            with open(log_file, 'a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            
            # 2.3: Generate steering vector
            cmd = [
                "python", "vector_generation.py",
                "--data_dir", save_dir,
                "--prefixs", "correct", "incorrect",
                "--layers", str(args.steering_layer),
                "--save_prefix", "seal",
                "--overwrite",
            ]
            
            with open(log_file, 'a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            
            vector_path = os.path.join(save_dir, f"vector_seal/layer_{args.steering_layer}_transition_reflection_steervec.pt")
            
            if not os.path.exists(vector_path):
                raise Exception("Vector generation failed: no vector file")
            
            print(f"[GPU {gpu_id}] [{config_name}] ✓ Vector generation completed")
            
            # ============================================
            # STEP 3: Steering Evaluation (on separate eval set)
            # ============================================
            print(f"[GPU {gpu_id}] [{config_name}] Step 3/3: Running steering evaluation...")
            
            steering_dir = os.path.join(save_dir, "steering_eval")
            
            cmd = [
                "python", "eval_MATH_steering.py",
                "--model_name_or_path", model_path,
                "--dataset", dataset,
                "--max_tokens", str(dataset_max_tokens),
                "--batch_size", "1",
                "--save_dir", steering_dir,
                "--steering",
                "--steering_vector", vector_path,
                "--steering_layer", str(args.steering_layer),
                "--steering_coef", str(args.steering_coef),
                "--use_chat_format",
                "--remove_bos",
            ]
            
            # Use eval samples for final testing (skip train samples)
            if args.max_examples:
                # Quick test mode: use user-specified limit with same offset
                cmd.extend(["--max_examples", str(args.max_examples)])
                cmd.extend(["--offset", str(args.max_examples)])  # Skip train samples
            elif dataset_cfg['eval']:
                # Normal mode: use dataset-specific eval split
                cmd.extend(["--max_examples", str(dataset_cfg['eval'])])
                cmd.extend(["--offset", str(dataset_cfg['train'])])  # Skip train samples
            
            with open(log_file, 'a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            
            print(f"[GPU {gpu_id}] [{config_name}] ✓ Steering evaluation completed")
        else:
            print(f"[GPU {gpu_id}] [{config_name}] Skipping vector generation and steering (method={args.method} or model not supported)")
        
        print(f"\n[GPU {gpu_id}] [{config_name}] ✅ ALL STEPS COMPLETED")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return True, config_name
        
    except subprocess.CalledProcessError as e:
        print(f"\n[GPU {gpu_id}] [{config_name}] ❌ FAILED: Command returned non-zero exit status")
        print(f"Check log: {log_file}\n")
        return False, config_name
    except Exception as e:
        print(f"\n[GPU {gpu_id}] [{config_name}] ❌ FAILED: {str(e)}")
        print(f"Check log: {log_file}\n")
        return False, config_name


def worker(gpu_id, task_queue, result_queue, args):
    """Worker process that runs tasks on a specific GPU"""
    while not shutdown_event.is_set():
        try:
            # Get task from queue with timeout
            task = task_queue.get(timeout=1)
            if task is None:  # Poison pill
                break
            
            task_id, total_tasks, model_path, model_short, dataset = task
            
            # Run the task
            success, config_name = run_single_config(
                gpu_id, model_path, model_short, dataset, args, task_id, total_tasks
            )
            
            # Report result
            result_queue.put((success, config_name, gpu_id))
            
        except Empty:
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] Worker error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="SEAL Baseline - Distributed Multi-GPU Runner")
    
    # GPU configuration
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    )
    
    # Model and dataset configuration
    parser.add_argument(
        "--model_root",
        type=str,
        default="/storage/szn_data/model_weights",
        help="Root directory containing model weights"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen2.5-3B-Instruct,Qwen2.5-7B-Instruct,DeepSeek-R1-Distill-Qwen-1.5B,DeepSeek-R1-Distill-Qwen-7B",
        help="Comma-separated list of model names"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default="aime_2024,aime25,amc23,arc-c,math500,openbookqa",
        help="Comma-separated list of dataset names"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="seal",
        choices=["vanilla", "seal", "cod", "deer"],
        help="Which baseline variant to run"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum number of tokens for generation"
    )
    parser.add_argument(
        "--deer_threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for DEER early exit"
    )
    parser.add_argument(
        "--deer_patience",
        type=int,
        default=4,
        help="Rolling window size for DEER confidence"
    )
    parser.add_argument(
        "--deer_think_ratio",
        type=float,
        default=0.9,
        help="Portion of max_tokens allocated to thinking in DEER"
    )
    parser.add_argument(
        "--deer_max_think_tokens",
        type=int,
        default=None,
        help="Upper bound on DEER thinking tokens"
    )
    parser.add_argument(
        "--deer_min_think_tokens",
        type=int,
        default=16,
        help="Minimum thinking tokens before DEER can early exit"
    )
    parser.add_argument(
        "--deer_answer_tokens",
        type=int,
        default=None,
        help="Answer token budget for DEER"
    )
    
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (None for all)"
    )
    
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=20,
        help="Layer to apply steering"
    )
    
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=1.0,
        help="Steering coefficient"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/baseline_full",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    
    # Parse models and datasets
    models = [x.strip() for x in args.models.split(',')]
    datasets = [x.strip() for x in args.datasets.split(',')]
    
    print("="*60)
    print("SEAL Baseline - Distributed Multi-GPU Runner")
    print("="*60)
    print(f"Available GPUs: {gpu_ids} ({num_gpus} GPUs)")
    print(f"Models: {models} ({len(models)} models)")
    print(f"Datasets: {datasets} ({len(datasets)} datasets)")
    print(f"Method: {args.method}")
    print(f"Total configurations: {len(models) * len(datasets)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max examples: {args.max_examples if args.max_examples else 'All'}")
    print("="*60)
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create task queue
    task_queue = Queue()
    result_queue = Queue()
    
    # Generate all tasks
    task_id = 0
    total_tasks = len(models) * len(datasets)
    
    for model_name in models:
        model_path = os.path.join(args.model_root, model_name)
        model_short = get_model_short_name(model_name)
        
        for dataset in datasets:
            task_id += 1
            task_queue.put((task_id, total_tasks, model_path, model_short, dataset))
    
    # Add poison pills to stop workers
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # Start worker processes
    print(f"Starting {num_gpus} worker processes...")
    print()
    
    workers = []
    for gpu_id in gpu_ids:
        p = Process(target=worker, args=(gpu_id, task_queue, result_queue, args))
        p.start()
        workers.append(p)
        active_processes.append(p)
    
    # Monitor progress
    completed = 0
    failed = 0
    success_configs = []
    failed_configs = []
    
    start_time = time.time()
    
    while completed + failed < total_tasks:
        if shutdown_event.is_set():
            break
        
        try:
            success, config_name, gpu_id = result_queue.get(timeout=1)
            if success:
                completed += 1
                success_configs.append(config_name)
            else:
                failed += 1
                failed_configs.append(config_name)
            
            # Print progress
            elapsed = time.time() - start_time
            avg_time = elapsed / (completed + failed) if (completed + failed) > 0 else 0
            remaining = total_tasks - (completed + failed)
            eta = avg_time * remaining
            
            print(f"\n{'='*60}")
            print(f"Progress: {completed + failed}/{total_tasks} "
                  f"(✓ {completed}, ✗ {failed})")
            print(f"Elapsed: {elapsed/60:.1f} min, ETA: {eta/60:.1f} min")
            print(f"{'='*60}\n")
            
        except Empty:
            continue
    
    # Wait for all workers to finish
    print("\nWaiting for all workers to finish...")
    for p in workers:
        p.join()
    
    # Print final summary
    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total configurations: {total_tasks}")
    print(f"Completed successfully: {completed}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average time per config: {total_time/total_tasks:.1f} seconds")
    print("="*60)
    
    # Save summary
    summary_file = os.path.join(args.output_dir, "run_summary.json")
    summary = {
        "total_tasks": total_tasks,
        "completed": completed,
        "failed": failed,
        "total_time_seconds": total_time,
        "avg_time_per_task": total_time / total_tasks if total_tasks > 0 else 0,
        "success_configs": success_configs,
        "failed_configs": failed_configs,
        "gpus_used": gpu_ids,
        "method": args.method,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    if failed > 0:
        print("\n⚠️  Some configurations failed. Check individual log files for details.")
        print("Failed configurations:")
        for config in failed_configs:
            print(f"  - {config}")
        sys.exit(1)
    else:
        print("\n✅ All configurations completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
