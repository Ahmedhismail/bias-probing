from typing import Tuple, Optional, List
import os
import torch
from transformer_lens import HookedTransformer
from dotenv import load_dotenv
from tqdm import tqdm


def load_model(model_name: str, device: torch.device, progress_tracker=None) -> HookedTransformer:
    """Load a model with optional progress tracking."""
    load_dotenv()  
    if progress_tracker is not None:
        with progress_tracker.track_model_loading(model_name) as progress:
            progress.update(20, "Downloading model")
            model = HookedTransformer.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
            progress.update(30, "Moving to device")
            model = model.to(device)
            progress.update(20, "Setting eval mode")
            model = model.eval()
            progress.update(30, "Ready")
            return model
    else:
        # Original simple loading without progress
        load_dotenv()
        model = HookedTransformer.from_pretrained(model_name, token=os.getenv("HF_TOKEN")).to(device).eval()
        return model


def _extract_single_token_id(model: HookedTransformer, text: str, token_name: str) -> int:
    """
    Extract token ID for a text that MUST tokenize to a single token.
    Tries multiple variations (with/without leading space) and validates.
    
    Args:
        model: The model with tokenizer
        text: The text to tokenize (e.g., "A", " A")
        token_name: Descriptive name for error messages (e.g., "Choice A")
    
    Returns:
        Token ID as integer
        
    Raises:
        ValueError: If text doesn't tokenize to exactly one token with any variation
    """
    variations = [text, f" {text.lstrip()}", text.lstrip()]
    
    for variant in variations:
        tokens = model.to_tokens(variant, prepend_bos=False)
        
        # Validate shape: should be [batch_size=1, seq_len]
        assert tokens.ndim == 2, f"Expected 2D tensor, got shape {tokens.shape}"
        assert tokens.shape[0] == 1, f"Expected batch_size=1, got {tokens.shape[0]}"
        
        seq_len = tokens.shape[1]
        
        if seq_len == 1:
            token_id = int(tokens[0, 0])
            tqdm.write(f"    ✓ Token '{token_name}' ('{variant}') → ID {token_id}")
            return token_id
    
    # If we get here, none of the variations worked
    token_counts = [model.to_tokens(v, prepend_bos=False).shape[1] for v in variations]
    raise ValueError(
        f"Token '{token_name}' does not tokenize to a single token in model '{model.cfg.model_name}'.\n"
        f"Tried variations: {variations}\n"
        f"Token counts: {token_counts}\n"
        f"This model's tokenizer may not be compatible with the current dataset format.\n"
        f"Consider using a different model or modifying the dataset format."
    )


def get_choice_token_ids(model: HookedTransformer) -> Tuple[int, int]:
    """
    Return token ids for choice letters (A/B).
    Used for multiple-choice format datasets (reassurance, deference, etc.).
    
    Validates that both choices tokenize to single tokens.
    
    Returns:
        Tuple of (choice1_id, choice2_id) corresponding to A and B
    """
    choice1_id = _extract_single_token_id(model, "A", "Choice A")
    choice2_id = _extract_single_token_id(model, "B", "Choice B")
    return choice1_id, choice2_id


def validate_hook_points(model: HookedTransformer, inject_site: str, read_site: str, 
                        inject_layers: List[int], read_layers: List[int]) -> None:
    """
    Validate that all required hook points exist in the model.
    
    Args:
        model: The model to validate
        inject_site: Hook site name for injection (e.g., 'hook_resid_mid')
        read_site: Hook site name for reading (e.g., 'hook_resid_post')
        inject_layers: List of injection layer indices
        read_layers: List of read layer indices
        
    Raises:
        ValueError: If any hook point doesn't exist in the model
    """
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"VALIDATING HOOK POINTS for model: {model.cfg.model_name}")
    tqdm.write(f"{'='*60}")
    
    # Get all available hooks in the model
    available_hooks = set(model.hook_dict.keys())
    
    # Build all required hook names
    required_hooks = set()
    for layer in inject_layers:
        required_hooks.add(f"blocks.{layer}.{inject_site}")
    for layer in read_layers:
        required_hooks.add(f"blocks.{layer}.{read_site}")
    
    # Check for missing hooks
    missing_hooks = required_hooks - available_hooks
    
    if missing_hooks:
        # Provide helpful error with available alternatives
        available_sites = set()
        for hook_name in available_hooks:
            if hook_name.startswith("blocks.") and hook_name.count(".") == 2:
                _, _, site = hook_name.split(".", 2)
                available_sites.add(site)
        
        raise ValueError(
            f"Hook point validation FAILED!\n\n"
            f"Missing hooks: {sorted(missing_hooks)}\n\n"
            f"Available hook sites in this model: {sorted(available_sites)}\n\n"
            f"You specified:\n"
            f"  - inject_site: '{inject_site}'\n"
            f"  - read_site: '{read_site}'\n\n"
            f"These hook sites may not exist in '{model.cfg.model_name}'.\n"
            f"Please use --inject-site and --read-site CLI flags with valid hook names from the list above."
        )
    
    tqdm.write(f"✓ All hook points validated successfully!")
    tqdm.write(f"  Inject site '{inject_site}' exists for layers: {inject_layers}")
    tqdm.write(f"  Read site '{read_site}' exists for layers: {read_layers}")
    tqdm.write(f"{'='*60}\n")


def get_available_hook_sites(model: HookedTransformer) -> List[str]:
    """
    Get list of available hook site names in the model (excluding layer-specific parts).
    
    Returns:
        Sorted list of unique hook site names (e.g., ['hook_resid_pre', 'hook_resid_post', ...])
    """
    hook_sites = set()
    for hook_name in model.hook_dict.keys():
        # Extract site name from patterns like "blocks.0.hook_resid_post"
        if hook_name.startswith("blocks.") and hook_name.count(".") == 2:
            _, _, site = hook_name.split(".", 2)
            hook_sites.add(site)
    
    return sorted(hook_sites)


def print_model_info(model: HookedTransformer, progress_tracker=None) -> None:
    """Print useful information about the loaded model for debugging."""
    log_func = progress_tracker.log if progress_tracker else print
    
    if progress_tracker:
        log_func(f"Model: {model.cfg.model_name} ({model.cfg.n_layers} layers, {model.cfg.d_model}d, {model.cfg.d_vocab} vocab)", "success")
    else:
        print(f"\n{'='*60}")
        print(f"MODEL INFORMATION: {model.cfg.model_name}")
        print(f"{'='*60}")
        print(f"  Architecture: {model.cfg.model_name}")
        print(f"  Total layers: {model.cfg.n_layers}")
        print(f"  Hidden size: {model.cfg.d_model}")
        print(f"  Vocab size: {model.cfg.d_vocab}")
        print(f"  Context length: {model.cfg.n_ctx}")
        print(f"\n  Available hook sites:")
        for site in get_available_hook_sites(model):
            print(f"    - {site}")
        print(f"{'='*60}\n")


