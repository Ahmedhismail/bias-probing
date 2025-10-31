"""Vector extraction and construction for bias response curves."""

from typing import List
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm

from BRC_Experiment.Modularized.utils import unit_vector
from BRC_Experiment.Modularized.cache import VectorCache


@torch.no_grad()
def residual_at_last_token(model: HookedTransformer, prompt: str, layer: int, site: str, prepend_bos: bool, device: torch.device) -> torch.Tensor:
    """Extract the residual activation at the last token position from a specific layer and site."""
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def grab(activation: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        cache["resid"] = activation.detach()
        return activation

    _ = model.run_with_hooks(tokens, return_type=None, stop_at_layer=layer + 1, fwd_hooks=[(f"blocks.{layer}.{site}", grab)])
    return cache["resid"][0, last_idx, :].clone().to(device)


def build_vectors(
    model: HookedTransformer,
    inj_layer: int,
    prompt_pairs: List[tuple[str, str]],
    prepend_bos: bool,
    device: torch.device,
    inject_site: str,
    model_name: str = "",
    dataset: str = "",
    batch_size: int = 16,
) -> dict[str, torch.Tensor]:
    """
    Build bias, random, and orthogonal steering vectors.
    
    Uses cache-aside pattern: always check cache first, compute and cache if missing,
    then return the cached result.
    
    Args:
        model: The transformer model
        inj_layer: Layer to extract activations from
        prompt_pairs: List of (positive, negative) prompt pairs
        prepend_bos: Whether to prepend BOS token
        device: Device to run computations on
        inject_site: Hook site name (e.g., 'hook_resid_mid')
        model_name: Name of the model (for caching)
        dataset: Name of the dataset (for caching)
    
    Returns:
        Dictionary with 'bias', 'random', and 'orth' unit vectors
    """
    # Initialize cache
    cache = VectorCache()
    
    # Always try to load from cache first
    if model_name and dataset:
        cached_vectors = cache.load(model_name, dataset, inj_layer, inject_site, device)
        if cached_vectors is not None:
            tqdm.write(f"    ✓ Using cached vectors (layer {inj_layer})")
            return cached_vectors
    
    # Compute vectors if not cached - BATCHED
    tqdm.write(f"    • Computing vectors for layer {inj_layer} (batch_size={batch_size}, {len(prompt_pairs)*2} prompts)")
    
    # Extract all prompts in batches
    all_prompts = [p for pair in prompt_pairs for p in pair]  # Flatten pairs
    all_activations = []
    
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i:i+batch_size]
        tokens_list = [model.to_tokens(p, prepend_bos=prepend_bos) for p in batch]
        max_len = max(t.shape[1] for t in tokens_list)
        
        padded_tokens = []
        last_indices = []
        for tokens in tokens_list:
            seq_len = tokens.shape[1]
            last_indices.append(seq_len - 1)
            if seq_len < max_len:
                padding = torch.zeros((1, max_len - seq_len), dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=1)
            padded_tokens.append(tokens)
        
        batch_tokens = torch.cat(padded_tokens, dim=0).to(device)
        cache_dict: dict[str, torch.Tensor] = {}
        
        def grab(activation: torch.Tensor, hook) -> torch.Tensor:
            cache_dict["resid"] = activation.detach()
            return activation
        
        _ = model.run_with_hooks(batch_tokens, return_type=None, stop_at_layer=inj_layer + 1, 
                                fwd_hooks=[(f"blocks.{inj_layer}.{inject_site}", grab)])
        
        for j, last_idx in enumerate(last_indices):
            all_activations.append(cache_dict["resid"][j, last_idx, :].clone())
    
    all_activations = torch.stack(all_activations)
    
    # Split back into positive and negative
    positive_acts = all_activations[0::2]  # Even indices
    negative_acts = all_activations[1::2]  # Odd indices
    
    bias_vec = (positive_acts - negative_acts).mean(dim=0) # Average difference between all pairs

    bias_vec = unit_vector(bias_vec)
    rand_vec = unit_vector(torch.randn_like(bias_vec))
    orth_seed = torch.randn_like(bias_vec)
    orth_vec = unit_vector(orth_seed - (orth_seed @ bias_vec) * bias_vec) # Orthogonal to bias vector
    
    vectors = {"bias": bias_vec, "random": rand_vec, "orth": orth_vec}
    
    # Always save to cache after computing
    if model_name and dataset:
        cache.save(vectors, model_name, dataset, inj_layer, inject_site)
    
    return vectors
