from __future__ import annotations
import torch
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.data import load_train_dataset, load_test_dataset
from BRC_Experiment.Modularized.model import (
    load_model, 
    get_choice_token_ids, 
    validate_hook_points, 
    print_model_info
)
from BRC_Experiment.Modularized.plotting import plot_and_save_brc_curves
from BRC_Experiment.Modularized.vectors import build_vectors
from BRC_Experiment.Modularized.steering import sweep_alpha
from BRC_Experiment.Modularized.metrics import logit_diffs, prob_diffs, compute_perplexity, odds_ratios, rank_changes, kl_divergences
from BRC_Experiment.Modularized.utils import build_alpha_range, configure_determinism, get_device, build_hook_name
from BRC_Experiment.Modularized.observability import create_progress_tracker
from BRC_Experiment.Modularized.results_io import save_results_to_csv, save_metadata


class Experiment:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config 
        self.progress_tracker = create_progress_tracker(enabled=self.config.show_progress)
        configure_determinism(self.config.seed) # Set seed for reproducibility
        self.device = get_device() # Get device for model
        
        # Load model with progress tracking
        self.model = load_model(self.config.model_name, self.device, self.progress_tracker)
        
        # Print model information for user awareness
        print_model_info(self.model, self.progress_tracker)
        
        # Load both train and test datasets
        self.train_prompt_pairs = load_train_dataset(self.config.dataset)
        self.test_prompts = load_test_dataset(self.config.dataset)
        
        self.progress_tracker.log(f"Loaded {len(self.train_prompt_pairs)} training pairs and {len(self.test_prompts)} test prompts", "success")

        # Get choice token IDs with validation (will raise ValueError if tokenization incompatible)
        self.progress_tracker.log("Validating tokenization...", "info")
        self.choice1_id, self.choice2_id = get_choice_token_ids(self.model)

        # Build alpha range
        self.alpha_values = build_alpha_range(self.config.alpha_start, self.config.alpha_stop, self.config.alpha_step) 

        # Get sequence of inject and read layers
        n_layers = self.model.cfg.n_layers # Get total number of layers in model
        self.inject_layers = (list(self.config.inject_layers) if self.config.inject_layers is not None else list(range(n_layers))) # If inject_layers is not specified, use all layers
        self.read_layers = (list(self.config.read_layers) if self.config.read_layers is not None else list(range(n_layers))) # If read_layers is not specified, use all layers
        
        # Validate hook points exist in model before running experiment
        validate_hook_points(
            self.model, 
            self.config.inject_site, 
            self.config.read_site,
            self.inject_layers,
            self.read_layers
        )  
    
    def _get_metrics_to_run(self):
        """Determine which metrics to run based on config."""
        all_metrics = {
            "logit_diffs": lambda logits: logit_diffs(logits, self.choice1_id, self.choice2_id),
            "prob_diffs": lambda logits: prob_diffs(logits, self.choice1_id, self.choice2_id),
            "odds_ratios": lambda logits: odds_ratios(logits, self.choice1_id, self.choice2_id),
            "compute_perplexity": lambda logits: compute_perplexity(logits, self.choice1_id),
            "rank_changes": lambda logits: rank_changes(logits, self.choice1_id, self.choice2_id),
            "kl_divergences": lambda logits: kl_divergences(logits, self.alpha_values),
        }
        if self.config.metric is None:
            return all_metrics
        else:
            if self.config.metric not in all_metrics:
                raise ValueError(f"Unknown metric: {self.config.metric}")
            return {self.config.metric: all_metrics[self.config.metric]}


    def run_experiment(self) -> None:
        # ====== PHASE 1: Setup and determine metrics to run ======
        metrics_to_run = self._get_metrics_to_run()
        progress_tracker = self.progress_tracker
        vector_names = ["bias", "random", "orth"]
        
        # Save experiment metadata
        if self.config.save_csv:
            save_metadata(self.config.out_dir, self.config)
            progress_tracker.log(f"Saved experiment metadata to {self.config.out_dir}/results/metadata.json", "success")
        
        # ====== PHASE 2: Iterate through layer combinations and compute metrics ======
        # Collect all metric data for global y-limits computation
        all_metric_data = {}  # metric_name -> list of all values
        all_results = []
        progress_tracker.log(f"Batch size: {self.config.batch_size}", "info")
        progress_tracker.log(f"Starting experiment: {len(self.inject_layers)} inject layers × {len(self.read_layers)} read layers × {len(self.test_prompts)} prompts", "info")
        
        for inj_layer in progress_tracker.track_injection_layers(self.inject_layers):
            # Track results for this inject layer (for CSV saving)
            inject_layer_results = []

            vectors = build_vectors(
                self.model,
                inj_layer,
                self.train_prompt_pairs,
                self.config.prepend_bos,
                self.device,
                inject_site=self.config.inject_site,
                model_name=self.config.model_name,
                dataset=self.config.dataset,
                batch_size=self.config.batch_size,
            ) # Build vectors for each inject_layer

            # Filter read layers that are greater than injection layer
            valid_read_layers = [rl for rl in self.read_layers if rl > inj_layer]
            
            for read_layer in progress_tracker.track_read_layers(valid_read_layers):
    
                inject_hook = build_hook_name(inj_layer, self.config.inject_site)
                read_hook = build_hook_name(read_layer, self.config.read_site)

                # ====== PHASE 2a: Compute steered logits (batched by default) ======
                accumulated_logits = {k: [] for k in vector_names}
                batch_size = min(self.config.batch_size, len(self.test_prompts))
                batches = [self.test_prompts[i:i+batch_size] for i in range(0, len(self.test_prompts), batch_size)]
                
                for batch in progress_tracker.track_test_prompts(batches):
                    for vector_name in vector_names:
                        # sweep_alpha now handles batching automatically: [n_prompts, n_alphas, vocab_size]
                        batch_logits = sweep_alpha(
                            self.model, vectors[vector_name], self.alpha_values, batch,
                            inj_layer, read_layer, inject_hook, read_hook,
                            self.config.prepend_bos, self.device, self.config.steer_all_tokens
                        )
                        accumulated_logits[vector_name].extend(batch_logits)

                # Average across prompts: [n_alphas, vocab_size]
                logits_by_vec = {k: torch.stack(accumulated_logits[k]).mean(dim=0) for k in vector_names}

                bias_logits   = logits_by_vec["bias"]
                random_logits = logits_by_vec["random"]
                orth_logits   = logits_by_vec["orth"]

                # ====== PHASE 2b: Compute each metric and collect data for global limits ======
                for metric_name, metric_func in metrics_to_run.items():
                    bias_results = metric_func(bias_logits)
                    random_results = metric_func(random_logits)
                    orth_results = metric_func(orth_logits)
                    
                    # Initialize metric data collection if first time seeing this metric
                    if metric_name not in all_metric_data:
                        all_metric_data[metric_name] = []
                    
                    # Collect data for global y-limits (handle special cases)
                    if metric_name == "rank_changes":
                        choice1_ranks, choice2_ranks = bias_results
                        all_metric_data[metric_name].extend(choice1_ranks + choice2_ranks)
                    elif metric_name == "kl_divergences":
                        # For KL divergence, only bias vector is meaningful
                        all_metric_data[metric_name].extend(bias_results)
                    else:
                        # For scalar metrics, collect all values from all vectors
                        all_metric_data[metric_name].extend([*bias_results, *random_results, *orth_results])
                    
                    # Store results for CSV saving (this inject layer only)
                    inject_layer_results.append((read_layer, bias_results, random_results, orth_results, metric_name))
                    
                    # Store all results for plotting
                    all_results.append((inj_layer, read_layer, bias_results, random_results, orth_results, metric_name))
            
            # ====== Save CSV for this inject layer ======
            if self.config.save_csv and inject_layer_results:
                csv_path = save_results_to_csv(
                    inject_layer_results,
                    self.alpha_values,
                    self.config.out_dir,
                    self.config.model_name,
                    self.config.dataset,
                    inj_layer
                )
                progress_tracker.log(f"Saved results for inject layer {inj_layer} to {csv_path.name}", "success")
        
        # ====== PHASE 3: Compute global y-limits for all metrics ======
        global_y_limits = {}
        for metric_name, data in all_metric_data.items():
            if not data:
                continue
                
            if metric_name == "rank_changes":
                # Special handling for ranks (inverted axis)
                min_rank, max_rank = min(data), max(data)
                rank_range = max_rank - min_rank
                y_padding = min(rank_range * 0.1, min_rank - 1) if min_rank > 1 else 0
                global_y_limits[metric_name] = (max_rank + y_padding, max(1, min_rank - y_padding))
            else:
                # Standard global limits for scalar metrics
                metric_min, metric_max = min(data), max(data)
                if metric_max == metric_min:
                    pad = 0.1 if metric_max == 0 else abs(metric_max) * 0.1
                    global_y_limits[metric_name] = (metric_min - pad, metric_max + pad)
                else:
                    yr = metric_max - metric_min
                    y_min = metric_min - 0.1 * yr
                    y_max = metric_max + 0.1 * yr
                    global_y_limits[metric_name] = (y_min, y_max)
        
        # ====== PHASE 4: Plot all results with global y-limits ======
        for result in progress_tracker.track_plotting(all_results, desc="Generating plots"):
            inj_layer, read_layer, bias_results, random_results, orth_results, metric_name = result
            y_limits = global_y_limits.get(metric_name, (0, 1))  # Fallback if no data
            
            plot_and_save_brc_curves(
                bias_diffs=bias_results,
                random_diffs=random_results,
                orth_diffs=orth_results,
                alpha_values=self.alpha_values,
                inj_layer=inj_layer,
                read_layer=read_layer,
                inject_site=self.config.inject_site,
                read_site=self.config.read_site,
                out_dir=self.config.out_dir,
                y_limits=y_limits,
                metric_name=metric_name,
                use_log_scale=self.config.use_log_scale,
                dataset_name=self.config.dataset,
                model_name=self.config.model_name,
                log_scale_both=self.config.log_scale_both,
            )
