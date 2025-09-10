"""Model steering operations for bias response curves."""

from typing import Iterable, List, Union
import torch
from transformer_lens import HookedTransformer

#TODO: passing in the same parameters to different functions, probably first consolidate with passing experiment config object and make that more robust


@torch.no_grad()
def get_steered_logits(
    model: HookedTransformer,
    prompt: Union[str, List[str]],
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
    Returns: Logits for the last token of each prompt after steering.
    If single prompt: shape [vocab_size]
    If multiple prompts: shape [n_prompts, vocab_size]
    """

    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos).to(device)
    # Handle both single prompt (shape [1, seq_len]) and multiple prompts (shape [n_prompts, seq_len])
    is_single_prompt = isinstance(prompt, str)
    if is_single_prompt:
        last_idx = tokens.shape[1] - 1
    else:
        last_idx = tokens.shape[1] - 1  # Same for all prompts in batch

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
    # For multiple prompts, resid shape is [n_prompts, 1, hidden_dim]
    # For single prompt, resid shape is [1, 1, hidden_dim]
    all_logits = model.unembed(resid)  # Shape: [n_prompts, 1, vocab_size] or [1, 1, vocab_size]

    # Squeeze out the sequence dimension and return appropriate shape
    if is_single_prompt:
        return all_logits[0, 0, :]  # Shape: [vocab_size]
    else:
        return all_logits[:, 0, :]  # Shape: [n_prompts, vocab_size]


@torch.no_grad()
def sweep_alpha(
    model: HookedTransformer,
    vector: torch.Tensor,
    alpha_values: Iterable[float],
    prompt: Union[str, List[str]],
    inj_layer: int,
    read_layer: int,
    inject_hook_name: str,
    read_hook_name: str,
    prepend_bos: bool,
    device: torch.device,
    steer_all_tokens: bool = True,
) -> torch.Tensor:
    """
    Returns a tensor with last-token logits for each alpha value.
    Single prompt: shape [n_alphas, vocab_size]
    Multiple prompts: shape [n_alphas, n_prompts, vocab_size]
    """
    out = []
    for alpha in alpha_values:
        alpha = float(alpha)
        logits = get_steered_logits(
            model=model,
            prompt=prompt,
            steer_vec=vector,
            alpha=alpha,
            inject_hook_name=inject_hook_name,
            read_hook_name=read_hook_name,
            inject_layer=inj_layer,
            read_layer=read_layer,
            prepend_bos=prepend_bos,
            device=device,
            steer_all_tokens=steer_all_tokens,
        )
        out.append(logits.to(device))

    stacked = torch.stack(out, dim=0)
    return stacked  # [n_alphas, vocab_size] or [n_alphas, n_prompts, vocab_size]