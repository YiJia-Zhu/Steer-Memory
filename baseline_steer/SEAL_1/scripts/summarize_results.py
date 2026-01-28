#!/usr/bin/env python3
"""
SEAL Baseline - Results Summarizer
Collects all baseline and steering evaluation results and generates CSV summaries
"""

import os
import sys
import json
import argparse
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_metrics(metrics_path):
    """Load metrics from a JSON file"""
    if not os.path.exists(metrics_path):
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {metrics_path}: {e}")
        return None


def collect_results(output_dir, models, datasets, methods):
    """Collect all baseline and steering evaluation results"""
    results = []
    
    for method in methods:
        method_label = method if method else "vanilla"
        base_dir = os.path.join(output_dir, method) if method else output_dir

        for model in models:
            for dataset in datasets:
                current_label = method_label
                # Path to dataset results
                dataset_dir = os.path.join(base_dir, model, dataset)
                
                # Fallback to legacy layout (no method folder)
                if not os.path.exists(dataset_dir) and method:
                    legacy_dir = os.path.join(output_dir, model, dataset)
                    if os.path.exists(legacy_dir):
                        dataset_dir = legacy_dir
                        current_label = "vanilla"
            
                if not os.path.exists(dataset_dir):
                    print(f"Warning: {dataset_dir} does not exist, skipping...")
                    continue
                
                # Collect baseline metrics
                baseline_metrics_path = os.path.join(dataset_dir, "metrics.json")
                baseline_metrics = load_metrics(baseline_metrics_path)
                
                # Collect steering evaluation metrics
                steering_dir = os.path.join(dataset_dir, "steering_eval")
                steering_metrics = None
                
                if os.path.exists(steering_dir):
                    # Find the steering evaluation results
                    # Pattern: {dataset}_vector_seal_layer_{layer}_transition_reflection_steervec/coef_{coef}_remove_bos/metrics.json
                    for subdir in os.listdir(steering_dir):
                        subdir_path = os.path.join(steering_dir, subdir)
                        if os.path.isdir(subdir_path):
                            for coef_dir in os.listdir(subdir_path):
                                coef_dir_path = os.path.join(subdir_path, coef_dir)
                                if os.path.isdir(coef_dir_path):
                                    steering_metrics_path = os.path.join(coef_dir_path, "metrics.json")
                                    steering_metrics = load_metrics(steering_metrics_path)
                                    break
                            if steering_metrics:
                                break
                
                # Create result entry
                result = {
                    'method': current_label,
                    'model': model,
                    'dataset': dataset,
                    'baseline_acc': baseline_metrics.get('acc', None) if baseline_metrics else None,
                    'baseline_avg_tokens': baseline_metrics.get('avg_tokens', None) if baseline_metrics else None,
                    'baseline_total_tokens': baseline_metrics.get('total_tokens', None) if baseline_metrics else None,
                    'steering_acc': steering_metrics.get('acc', None) if steering_metrics else None,
                    'steering_avg_tokens': steering_metrics.get('avg_tokens', None) if steering_metrics else None,
                    'steering_total_tokens': steering_metrics.get('total_tokens', None) if steering_metrics else None,
                }
                
                # Calculate improvement
                if result['baseline_acc'] is not None and result['steering_acc'] is not None:
                    result['acc_improvement'] = result['steering_acc'] - result['baseline_acc']
                    result['acc_improvement_pct'] = (result['acc_improvement'] / result['baseline_acc'] * 100) if result['baseline_acc'] > 0 else None
                else:
                    result['acc_improvement'] = None
                    result['acc_improvement_pct'] = None
                
                results.append(result)
    
    return results


def collect_all_runs_metrics(root_dir):
    """
    Collect metrics across multiple runs under a root directory.
    Expects layout: {root}/{run}/{method}/{model}/{dataset}/metrics.json
    Skips any metrics inside steering_eval folders.
    """
    results = []
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: {root_dir} does not exist")
        return results

    for metrics_path in root_path.rglob("metrics.json"):
        # Avoid counting steering evaluation metrics twice
        if any(part == "steering_eval" for part in metrics_path.parts):
            continue

        try:
            rel_parts = metrics_path.relative_to(root_path).parts
        except ValueError:
            continue

        if len(rel_parts) < 4:
            continue

        run_name, method, model, dataset = rel_parts[0:4]
        metrics = load_metrics(metrics_path)
        if not metrics:
            continue

        results.append({
            "run": run_name,
            "method": method,
            "model": model,
            "dataset": dataset,
            "acc": metrics.get("acc"),
            "avg_tokens": metrics.get("avg_tokens"),
            "total_tokens": metrics.get("total_tokens"),
            "metrics_path": str(metrics_path),
        })

    return sorted(results, key=lambda x: (x["run"], x["method"], x["model"], x["dataset"]))


def write_csv(filename, headers, rows):
    """Write data to CSV file"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_key(results, group_key, agg_keys):
    """Aggregate results by a grouping key"""
    groups = defaultdict(lambda: defaultdict(list))
    
    # Group results
    for result in results:
        group_value = result[group_key]
        for agg_key in agg_keys:
            value = result.get(agg_key)
            if value is not None:
                groups[group_value][agg_key].append(value)
    
    # Calculate averages
    aggregated = []
    for group_value, values_dict in groups.items():
        agg_result = {group_key: group_value}
        for agg_key, values in values_dict.items():
            if values:
                agg_result[agg_key] = sum(values) / len(values)
            else:
                agg_result[agg_key] = None
        aggregated.append(agg_result)
    
    return aggregated


def aggregate_by_keys(results, group_keys, agg_keys):
    """Aggregate results by multiple grouping keys."""
    groups = defaultdict(lambda: defaultdict(list))

    for result in results:
        key = tuple(result.get(k) for k in group_keys)
        for agg_key in agg_keys:
            value = result.get(agg_key)
            if value is not None:
                groups[key][agg_key].append(value)

    aggregated = []
    for key, values_dict in groups.items():
        entry = {k: v for k, v in zip(group_keys, key)}
        for agg_key, values in values_dict.items():
            entry[agg_key] = sum(values) / len(values) if values else None
        aggregated.append(entry)

    return aggregated


def format_table(headers, rows):
    """Format data as a text table"""
    # Calculate column widths
    col_widths = {h: len(h) for h in headers}
    for row in rows:
        for header in headers:
            value = row.get(header, '')
            if value is None:
                value = 'N/A'
            elif isinstance(value, float):
                value = f"{value:.4f}"
            else:
                value = str(value)
            col_widths[header] = max(col_widths[header], len(value))
    
    # Create table
    lines = []
    
    # Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Rows
    for row in rows:
        row_values = []
        for header in headers:
            value = row.get(header, '')
            if value is None:
                value = 'N/A'
            elif isinstance(value, float):
                value = f"{value:.4f}"
            else:
                value = str(value)
            row_values.append(value.ljust(col_widths[header]))
        lines.append(" | ".join(row_values))
    
    return "\n".join(lines)


def generate_summary(results, output_dir):
    """Generate summary CSV files"""
    if not results:
        print("No results to summarize")
        return
    
    # Save full results
    full_csv_path = os.path.join(output_dir, "results_summary.csv")
    full_headers = ['method', 'model', 'dataset', 'baseline_acc', 'steering_acc', 'acc_improvement', 
                    'acc_improvement_pct', 'baseline_avg_tokens', 'steering_avg_tokens',
                    'baseline_total_tokens', 'steering_total_tokens']
    write_csv(full_csv_path, full_headers, results)
    print(f"\n✅ Full results saved to: {full_csv_path}")
    
    # Create summary by model
    agg_keys = ['baseline_acc', 'steering_acc', 'acc_improvement', 'baseline_avg_tokens', 'steering_avg_tokens']
    model_summary = aggregate_by_key(results, 'model', agg_keys)
    
    model_summary_path = os.path.join(output_dir, "results_by_model.csv")
    model_headers = ['model'] + agg_keys
    write_csv(model_summary_path, model_headers, model_summary)
    print(f"✅ Model summary saved to: {model_summary_path}")
    
    # Create summary by dataset
    dataset_summary = aggregate_by_key(results, 'dataset', agg_keys)
    
    dataset_summary_path = os.path.join(output_dir, "results_by_dataset.csv")
    dataset_headers = ['dataset'] + agg_keys
    write_csv(dataset_summary_path, dataset_headers, dataset_summary)
    print(f"✅ Dataset summary saved to: {dataset_summary_path}")

    # Create method-aware summaries
    method_model_summary = aggregate_by_keys(results, ['method', 'model'], agg_keys)
    method_model_path = os.path.join(output_dir, "results_by_method_model.csv")
    method_model_headers = ['method', 'model'] + agg_keys
    write_csv(method_model_path, method_model_headers, method_model_summary)
    print(f"✅ Method+Model summary saved to: {method_model_path}")

    method_dataset_summary = aggregate_by_keys(results, ['method', 'dataset'], agg_keys)
    method_dataset_path = os.path.join(output_dir, "results_by_method_dataset.csv")
    method_dataset_headers = ['method', 'dataset'] + agg_keys
    write_csv(method_dataset_path, method_dataset_headers, method_dataset_summary)
    print(f"✅ Method+Dataset summary saved to: {method_dataset_path}")
    
    # Print summary tables
    print("\n" + "="*100)
    print("RESULTS SUMMARY")
    print("="*100)
    print("\nFull Results:")
    print(format_table(full_headers, results))
    
    print("\n" + "-"*100)
    print("Summary by Model:")
    print(format_table(model_headers, model_summary))
    
    print("\n" + "-"*100)
    print("Summary by Dataset:")
    print(format_table(dataset_headers, dataset_summary))
    print("="*100)
    
    # Calculate overall statistics
    baseline_accs = [r['baseline_acc'] for r in results if r['baseline_acc'] is not None]
    steering_accs = [r['steering_acc'] for r in results if r['steering_acc'] is not None]
    improvements = [r['acc_improvement'] for r in results if r['acc_improvement'] is not None]
    
    overall_stats = {
        'total_configs': len(results),
        'completed_baseline': len(baseline_accs),
        'completed_steering': len(steering_accs),
        'avg_baseline_acc': sum(baseline_accs) / len(baseline_accs) if baseline_accs else None,
        'avg_steering_acc': sum(steering_accs) / len(steering_accs) if steering_accs else None,
        'avg_improvement': sum(improvements) / len(improvements) if improvements else None,
    }
    
    print("\n" + "="*100)
    print("OVERALL STATISTICS")
    print("="*100)
    for key, value in overall_stats.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
    print("="*100 + "\n")
    
    # Save overall statistics
    stats_path = os.path.join(output_dir, "overall_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"✅ Overall statistics saved to: {stats_path}\n")


def generate_combined_summary(root_dir, output_path=None):
    """Generate a single CSV that aggregates all runs under a root directory."""
    results = collect_all_runs_metrics(root_dir)
    if not results:
        print(f"No metrics.json files found under {root_dir}")
        return 1

    output_csv = output_path or os.path.join(root_dir, "all_runs_metrics.csv")
    headers = ["run", "method", "model", "dataset", "acc", "avg_tokens", "total_tokens", "metrics_path"]
    write_csv(output_csv, headers, results)

    run_count = len(set(r["run"] for r in results))
    print("\n" + "="*100)
    print("COMBINED RUNS SUMMARY")
    print("="*100)
    print(f"Root: {root_dir}")
    print(f"Runs combined: {run_count}")
    print(f"Total rows: {len(results)}")
    print(f"✅ Combined CSV saved to: {output_csv}\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="SEAL Baseline - Results Summarizer")
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory containing results (single run mode)"
    )

    parser.add_argument(
        "--scan_root",
        type=str,
        default=None,
        help="Root directory containing multiple runs to aggregate (e.g., results)"
    )

    parser.add_argument(
        "--combined_output",
        type=str,
        default=None,
        help="Optional CSV path when using --scan_root (default: <scan_root>/all_runs_metrics.csv)"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen2.5-3B,Qwen2.5-7B,DS-R1-1.5B,DS-R1-7B",
        help="Comma-separated list of model short names"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default="aime_2024,aime25,amc23,arc-c,math500,openbookqa",
        help="Comma-separated list of dataset names"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="seal,cod,deer",
        help="Comma-separated list of methods (e.g., vanilla,cod,deer)"
    )
    
    args = parser.parse_args()

    # Combined multi-run mode
    if args.scan_root:
        exit_code = generate_combined_summary(args.scan_root, args.combined_output)
        sys.exit(exit_code)

    if not args.output_dir:
        parser.error("Either --output_dir (single run) or --scan_root (aggregate mode) must be provided.")
    
    # Parse models and datasets
    models = [x.strip() for x in args.models.split(',')]
    datasets = [x.strip() for x in args.datasets.split(',')]
    methods = [x.strip() for x in args.methods.split(',')]
    
    print("="*100)
    print("SEAL Baseline - Results Summarizer")
    print("="*100)
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Methods: {methods}")
    print(f"Total configurations: {len(models) * len(datasets) * len(methods)}")
    print("="*100 + "\n")
    
    # Collect results
    print("Collecting results...")
    results = collect_results(args.output_dir, models, datasets, methods)
    print(f"✅ Collected {len(results)} results\n")
    
    # Generate summary
    generate_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
