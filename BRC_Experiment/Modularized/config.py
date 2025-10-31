from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class ExperimentConfig:
    model_name: str = "gpt2-small"
    prepend_bos: bool = True

    inject_site: str = "hook_resid_mid"
    read_site: str = "hook_resid_post"

    alpha_start: float = -10.0
    alpha_stop: float = 10.0
    alpha_step: float = 0.5

    # If None, use all layers (0..n_layers-1)
    inject_layers: Optional[Sequence[int]] = None
    read_layers: Optional[Sequence[int]] = None

    seed: int = 42
    out_dir: str = "graphs"
    
    # Metric to use (if None, runs all available metrics)
    metric: Optional[str] = None
    
    # Whether to steer all token positions or just the last token
    steer_all_tokens: bool = True
    
    # Whether to use log scale for plotting (useful for very small values)
    use_log_scale: bool = False
    
    # Whether to use log scale on both axes (useful for exponential growth)
    log_scale_both: bool = False
    
    # Dataset to use: any dataset name from data/ directory (e.g., "deference", "reassurance", "satisficing", "sycophancy")
    dataset: str = "deference"

    # Whether to show progress bars during execution
    show_progress: bool = True
    
    # Batch size for GPU operations (higher = faster but more memory)
    batch_size: int = 16


