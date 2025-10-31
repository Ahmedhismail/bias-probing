"""
Results I/O utilities for saving and loading experiment results to/from CSV files.
"""
import os
import csv
import json
import torch
from pathlib import Path
from typing import Any


def save_results_to_csv(
    results: list[tuple],
    alpha_values: torch.Tensor,
    out_dir: str,
    model_name: str,
    dataset: str,
    inject_layer: int,
) -> Path:
    """
    Save experiment results to a CSV file.
    
    Args:
        results: List of (read_layer, bias_results, random_results, orth_results, metric_name) tuples
        alpha_values: Tensor of alpha values used
        out_dir: Output directory
        model_name: Name of the model
        dataset: Name of the dataset
        inject_layer: Injection layer number
        
    Returns:
        Path to the saved CSV file
    """
    # Create results directory
    results_dir = Path(out_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean model name for filename
    clean_model_name = model_name.replace("/", "_")
    csv_path = results_dir / f"{clean_model_name}_{dataset}_inj{inject_layer}.csv"
    
    with open(csv_path, 'w', newline='') as f:
        # Determine if we need special handling for rank_changes
        has_rank_changes = any(metric_name == "rank_changes" for _, _, _, _, metric_name in results)
        
        if has_rank_changes and all(metric_name == "rank_changes" for _, _, _, _, metric_name in results):
            # All results are rank_changes - use special format
            writer = csv.DictWriter(f, fieldnames=[
                'inject_layer', 'read_layer', 'alpha', 'vector_type', 'metric_name', 'choice', 'rank'
            ])
            writer.writeheader()
            
            for read_layer, bias_results, random_results, orth_results, metric_name in results:
                choice1_ranks_bias, choice2_ranks_bias = bias_results
                choice1_ranks_random, choice2_ranks_random = random_results
                choice1_ranks_orth, choice2_ranks_orth = orth_results
                
                for i, alpha in enumerate(alpha_values.tolist()):
                    # Bias vector
                    writer.writerow({
                        'inject_layer': inject_layer,
                        'read_layer': read_layer,
                        'alpha': alpha,
                        'vector_type': 'bias',
                        'metric_name': metric_name,
                        'choice': 'A',
                        'rank': choice1_ranks_bias[i]
                    })
                    writer.writerow({
                        'inject_layer': inject_layer,
                        'read_layer': read_layer,
                        'alpha': alpha,
                        'vector_type': 'bias',
                        'metric_name': metric_name,
                        'choice': 'B',
                        'rank': choice2_ranks_bias[i]
                    })
                    # Random vector
                    writer.writerow({
                        'inject_layer': inject_layer,
                        'read_layer': read_layer,
                        'alpha': alpha,
                        'vector_type': 'random',
                        'metric_name': metric_name,
                        'choice': 'A',
                        'rank': choice1_ranks_random[i]
                    })
                    writer.writerow({
                        'inject_layer': inject_layer,
                        'read_layer': read_layer,
                        'alpha': alpha,
                        'vector_type': 'random',
                        'metric_name': metric_name,
                        'choice': 'B',
                        'rank': choice2_ranks_random[i]
                    })
                    # Orth vector
                    writer.writerow({
                        'inject_layer': inject_layer,
                        'read_layer': read_layer,
                        'alpha': alpha,
                        'vector_type': 'orth',
                        'metric_name': metric_name,
                        'choice': 'A',
                        'rank': choice1_ranks_orth[i]
                    })
                    writer.writerow({
                        'inject_layer': inject_layer,
                        'read_layer': read_layer,
                        'alpha': alpha,
                        'vector_type': 'orth',
                        'metric_name': metric_name,
                        'choice': 'B',
                        'rank': choice2_ranks_orth[i]
                    })
        else:
            # Standard format for scalar metrics
            writer = csv.DictWriter(f, fieldnames=[
                'inject_layer', 'read_layer', 'alpha', 'vector_type', 'metric_name', 'value'
            ])
            writer.writeheader()
            
            for read_layer, bias_results, random_results, orth_results, metric_name in results:
                # Convert results to lists if they're tensors
                if isinstance(bias_results, torch.Tensor):
                    bias_results = bias_results.tolist()
                if isinstance(random_results, torch.Tensor):
                    random_results = random_results.tolist()
                if isinstance(orth_results, torch.Tensor):
                    orth_results = orth_results.tolist()
                
                # Handle rank_changes specially
                if metric_name == "rank_changes":
                    choice1_ranks_bias, choice2_ranks_bias = bias_results
                    choice1_ranks_random, choice2_ranks_random = random_results
                    choice1_ranks_orth, choice2_ranks_orth = orth_results
                    
                    for i, alpha in enumerate(alpha_values.tolist()):
                        # Write as separate choice A and B rows (flat structure)
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'bias',
                            'metric_name': f"{metric_name}_choiceA",
                            'value': choice1_ranks_bias[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'bias',
                            'metric_name': f"{metric_name}_choiceB",
                            'value': choice2_ranks_bias[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'random',
                            'metric_name': f"{metric_name}_choiceA",
                            'value': choice1_ranks_random[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'random',
                            'metric_name': f"{metric_name}_choiceB",
                            'value': choice2_ranks_random[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'orth',
                            'metric_name': f"{metric_name}_choiceA",
                            'value': choice1_ranks_orth[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'orth',
                            'metric_name': f"{metric_name}_choiceB",
                            'value': choice2_ranks_orth[i]
                        })
                else:
                    # Standard scalar metric
                    for i, alpha in enumerate(alpha_values.tolist()):
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'bias',
                            'metric_name': metric_name,
                            'value': bias_results[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'random',
                            'metric_name': metric_name,
                            'value': random_results[i]
                        })
                        writer.writerow({
                            'inject_layer': inject_layer,
                            'read_layer': read_layer,
                            'alpha': alpha,
                            'vector_type': 'orth',
                            'metric_name': metric_name,
                            'value': orth_results[i]
                        })
    
    return csv_path


def save_metadata(out_dir: str, config: Any) -> Path:
    """
    Save experiment configuration metadata to JSON.
    
    Args:
        out_dir: Output directory
        config: ExperimentConfig object
        
    Returns:
        Path to the saved metadata file
    """
    results_dir = Path(out_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = results_dir / "metadata.json"
    
    # Convert config to dict (handle special types)
    metadata = {
        'model_name': config.model_name,
        'dataset': config.dataset,
        'inject_site': config.inject_site,
        'read_site': config.read_site,
        'alpha_start': config.alpha_start,
        'alpha_stop': config.alpha_stop,
        'alpha_step': config.alpha_step,
        'inject_layers': config.inject_layers,
        'read_layers': config.read_layers,
        'seed': config.seed,
        'batch_size': config.batch_size,
        'prepend_bos': config.prepend_bos,
        'steer_all_tokens': config.steer_all_tokens,
        'metric': config.metric,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path


def load_results_from_csv(csv_path: Path) -> dict:
    """
    Load experiment results from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Dictionary with structure: {
            (inject_layer, read_layer, metric_name): {
                'bias': [...],
                'random': [...],
                'orth': [...]
            }
        }
    """
    results = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            inject_layer = int(row['inject_layer'])
            read_layer = int(row['read_layer'])
            metric_name = row['metric_name']
            vector_type = row['vector_type']
            
            key = (inject_layer, read_layer, metric_name)
            
            if key not in results:
                results[key] = {'bias': [], 'random': [], 'orth': []}
            
            if 'choice' in row:
                # Rank changes format
                results[key][vector_type].append((row['choice'], float(row['rank'])))
            else:
                # Standard format
                results[key][vector_type].append(float(row['value']))
    
    return results

