"""Observability utilities for experiment progress tracking."""

from typing import Iterator, TypeVar, Sequence, Optional
from tqdm import tqdm
import contextlib
import sys

T = TypeVar('T')


class ExperimentProgressTracker:
    """Provides progress tracking for BRC experiments with cleaner output."""
    
    def __init__(self, show_progress: bool = True) -> None:
        """Initialize the progress tracker.
        
        Args:
            show_progress: Whether to show progress bars. Set to False for silent mode.
        """
        self.show_progress = show_progress
        self._active_pbar = None
    
    def log(self, message: str, level: str = "info") -> None:
        """Print a message that plays nicely with progress bars.
        
        Args:
            message: The message to print
            level: Message level (info, success, warning, error)
        """
        if not self.show_progress:
            print(message)
            return
        
        # Use tqdm.write to avoid conflicts with progress bars
        prefix = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗"
        }.get(level, "•")
        
        tqdm.write(f"{prefix} {message}")
    
    def track_injection_layers(self, inject_layers: Sequence[T], desc: str = "Processing injection layers") -> Iterator[T]:
        """Track progress through injection layers (outermost loop)."""
        if not self.show_progress:
            yield from inject_layers
            return
            
        with tqdm(inject_layers, desc=desc, unit="layer", leave=True, file=sys.stdout) as pbar:
            self._active_pbar = pbar
            for layer in pbar:
                pbar.set_postfix(layer=layer, refresh=True)
                yield layer
            self._active_pbar = None
    
    def track_read_layers(self, read_layers: Sequence[T], desc: str = "  ├─ Read layers") -> Iterator[T]:
        """Track progress through read layers (inner loop)."""
        if not self.show_progress:
            yield from read_layers
            return
            
        with tqdm(read_layers, desc=desc, unit="layer", leave=False, file=sys.stdout) as pbar:
            for layer in pbar:
                pbar.set_postfix(layer=layer, refresh=True)
                yield layer
    
    def track_test_prompts(self, test_prompts: Sequence[T], desc: str = "    └─ Evaluating") -> Iterator[T]:
        """Track progress through test prompts/batches."""
        if not self.show_progress:
            yield from test_prompts
            return

        # Auto-detect batches (lists) vs individual prompts (strings)
        unit_name = "batch" if (test_prompts and isinstance(test_prompts[0], list)) else "prompt"
        
        with tqdm(test_prompts, desc=desc, unit=unit_name, leave=False, file=sys.stdout) as pbar:
            for item in pbar:
                yield item
    
    def track_plotting(self, results: Sequence[T], desc: str = "Generating plots") -> Iterator[T]:
        """Track progress through plotting phase."""
        if not self.show_progress:
            yield from results
            return
            
        with tqdm(results, desc=desc, unit="plot", leave=True, file=sys.stdout) as pbar:
            for result in pbar:
                yield result
    
    @contextlib.contextmanager
    def track_model_loading(self, model_name: str):
        """Context manager for tracking model loading progress."""
        if not self.show_progress:
            yield
            return
            
        with tqdm(total=100, desc=f"Loading {model_name}", unit="%", leave=True, file=sys.stdout, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{postfix}]') as pbar:
            pbar.set_postfix_str("Initializing...")
            yield ModelLoadingProgress(pbar)
            pbar.update(100 - pbar.n)  # Complete the bar
            pbar.set_postfix_str("Complete")


class ModelLoadingProgress:
    """Helper class to update model loading progress."""
    
    def __init__(self, pbar: tqdm):
        self.pbar = pbar
        
    def update(self, amount: int = 10, stage: Optional[str] = None):
        """Update progress by the given amount."""
        self.pbar.update(min(amount, 100 - self.pbar.n))
        if stage:
            self.pbar.set_postfix_str(stage)


def create_progress_tracker(enabled: bool = True) -> ExperimentProgressTracker:
    """Factory function to create a progress tracker.
    
    Args:
        enabled: Whether progress tracking should be enabled.
        
    Returns:
        ExperimentProgressTracker instance.
    """
    return ExperimentProgressTracker(show_progress=enabled)
