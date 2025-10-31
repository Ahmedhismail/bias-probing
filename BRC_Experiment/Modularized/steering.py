"""Model steering operations for bias response curves."""

from typing import Iterable, List
import torch
from transformer_lens import HookedTransformer

#TODO: passing in the same parameters to different functions, probably first consolidate with passing experiment config object and make that more robust


@torch.no_grad()
def get_steered_logits(
    model: HookedTransformer,
    prompt: str,
    steer_vec: torch.Tensor,
    alpha: float,
    inject_hook_name: str,
    read_hook_name: str,
    inject_layer: int,
    read_layer: int,
    prepend_bos: bool,
    device: torch.device,
    steer_all_tokens: bool = True,
) -> torch.Tensor:
    """
    Returns: Logits for the last token of the prompt after steering.
    """

    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    last_idx = tokens.shape[1] - 1
    cache: dict[str, torch.Tensor] = {}

    def do_steer(act: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        vec = steer_vec.to(act.device)
        if steer_all_tokens:
            # Steer all token positions
            act[:, :, :] = act[:, :, :] + (alpha * vec)
        else:
            # Steer only the last token position (original behavior)
            act[:, last_idx, :] = act[:, last_idx, :] + (alpha * vec)
        return act

    def do_read(act: torch.Tensor, hook) -> torch.Tensor:  # type: ignore[no-redef]
        cache["resid"] = act.detach().clone()
        return act

    _ = model.run_with_hooks(
        tokens,
        return_type=None,
        stop_at_layer=max(inject_layer, read_layer) + 1,
        fwd_hooks=[(inject_hook_name, do_steer), (read_hook_name, do_read)],
    )

    resid = model.ln_final(cache["resid"][:, last_idx : last_idx + 1, :])
    logits = model.unembed(resid)[0, 0, :]
    return logits


@torch.no_grad()
def sweep_alpha(
    model: HookedTransformer,
    vector: torch.Tensor,
    alpha_values: Iterable[float],
    prompts: str | list[str],  # Now handles both single prompt AND batches!
    inj_layer: int,
    read_layer: int,
    inject_hook_name: str,
    read_hook_name: str,
    prepend_bos: bool,
    device: torch.device,
    steer_all_tokens: bool = True,
) -> torch.Tensor:
    """
    Sweep alpha values with batching for efficiency.
    
    Args:
        prompts: Single prompt (str) or list of prompts for batching
    
    Returns:
        - If single prompt: [n_alphas, vocab_size]
        - If batch: [n_prompts, n_alphas, vocab_size]
    """
    alpha_list = list(alpha_values)
    is_single = isinstance(prompts, str)
    prompt_list = [prompts] if is_single else prompts
    
    # Tokenize and pad
    all_tokens = [model.to_tokens(p, prepend_bos=prepend_bos).to(device) for p in prompt_list]
    max_len = max(t.shape[1] for t in all_tokens)
    last_indices = [t.shape[1] - 1 for t in all_tokens]
    
    padded = [torch.cat([t, torch.zeros((1, max_len - t.shape[1]), dtype=t.dtype, device=device)], dim=1) 
              if t.shape[1] < max_len else t for t in all_tokens]
    prompt_tokens = torch.cat(padded, dim=0)  # [n_prompts, seq_len]
    
    # Replicate for alphas: [n_prompts * n_alphas, seq_len]
    batch_tokens = prompt_tokens.repeat_interleave(len(alpha_list), dim=0)
    alpha_tensor = torch.tensor(alpha_list, device=device, dtype=vector.dtype).repeat(len(prompt_list))
    
    cache: dict[str, torch.Tensor] = {}
    
    def do_steer(act: torch.Tensor, hook) -> torch.Tensor:
        vec = vector.to(act.device)
        for i, alpha in enumerate(alpha_tensor):
            if steer_all_tokens:
                act[i, :, :] += alpha * vec
            else:
                act[i, last_indices[i // len(alpha_list)], :] += alpha * vec
        return act
    
    def do_read(act: torch.Tensor, hook) -> torch.Tensor:
        cache["resid"] = act.detach().clone()
        return act
    
    model.run_with_hooks(
        batch_tokens, return_type=None,
        stop_at_layer=max(inj_layer, read_layer) + 1,
        fwd_hooks=[(inject_hook_name, do_steer), (read_hook_name, do_read)]
    )
    
    # Extract logits per prompt
    all_logits = []
    for i, last_idx in enumerate(last_indices):
        start, end = i * len(alpha_list), (i + 1) * len(alpha_list)
        resid = cache["resid"][start:end, last_idx:last_idx+1, :]
        logits = model.unembed(model.ln_final(resid))[:, 0, :]
        all_logits.append(logits)
    
    result = torch.stack(all_logits, dim=0)
    return result[0] if is_single else result