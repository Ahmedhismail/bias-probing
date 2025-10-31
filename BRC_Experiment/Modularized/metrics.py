"""
Metric computation functions for bias response curves.

This module provides comprehensive metrics for analyzing bias response curves,
including both local effects (on specific token pairs) and global effects 
(on the entire distribution).

Local Effects Metrics:
- logit_diffs: Raw margin between tokens (linear in α, clean signal)
- prob_diffs: Normalized probability differences (bounded [-1,1])
- odds_ratios: Human-interpretable relative likelihood (e^Δ)
- rank_changes: Position movement of tokens in sorted vocabulary

Global Effects Metrics:
- compute_perplexity: Model fluency/naturalness preservation
- kl_divergences: Overall distribution shift measurement
- top_k_analysis: Qualitative inspection of distributional changes
"""

from typing import List, Tuple, Dict, Any, Optional
import torch


# ==================== LOCAL EFFECTS METRICS ====================

@torch.no_grad()
def logit_diffs(logits_batch: torch.Tensor, choice1_id: int, choice2_id: int) -> List[float]:
    """
    Compute logit differences (Choice1 - Choice2) from a list of logit tensors.
    
    Args:
        logits_batch: tensor of shape [batch_size, vocab_size] containing average logits for all alpha values
        choice1_id: Token ID for first choice 
        choice2_id: Token ID for second choice
    
    Returns:
        List of logit differences (Δ = z_tok1 - z_tok2)
    """
    
    # Compute differences in a single operation
    diffs_batch = logits_batch[:, choice1_id] - logits_batch[:, choice2_id]  # [batch_size]
    return diffs_batch.tolist()


@torch.no_grad()
def prob_diffs(logits_batch: torch.Tensor, choice1_id: int, choice2_id: int) -> List[float]:
    """
    Compute probability differences (Choice1 - Choice2) from logits using softmax.

    Args:
        logits_batch: tensor of shape [batch_size, vocab_size] containing logits for all alpha values
        choice1_id: Token ID for first choice
        choice2_id: Token ID for second choice

    Returns:
        List of probability differences (P(Choice1) - P(Choice2)), bounded in [-1,1]
    """
    # Compute softmax probabilities in a single operation
    probs_batch = torch.softmax(logits_batch, dim=-1)  # [batch_size, vocab_size]

    # Compute probability differences in a single operation
    prob_diffs_batch = probs_batch[:, choice1_id] - probs_batch[:, choice2_id]  # [batch_size]

    return prob_diffs_batch.tolist()


@torch.no_grad()
def odds_ratios(logits_batch: torch.Tensor, choice1_id: int, choice2_id: int) -> List[float]:
    """
    Compute odds ratios (e^Δ) from logit differences.

    Human-interpretable metric showing relative likelihood. E.g., value of 7.0
    means "Token A is 7× more likely than token B."

    Args:
        logits_batch: tensor of shape [batch_size, vocab_size] containing logits for all alpha values
        choice1_id: Token ID for first choice
        choice2_id: Token ID for second choice

    formula: e^(logit_diff) = e^(z_tok1 - z_tok2)

    Returns:
        List of odds ratios (e^(logit_diff))
    """
    
    # Compute logit differences and exponentiate in a single operation
    logit_diffs_batch = logits_batch[:, choice1_id] - logits_batch[:, choice2_id]  # [batch_size]
    odds_ratios_batch = torch.exp(logit_diffs_batch)  # [batch_size]
    
    return odds_ratios_batch.tolist()


@torch.no_grad()
def rank_changes(
    logits_batch: torch.Tensor,
    choice1_id: int,
    choice2_id: int,
) -> Tuple[List[int], List[int]]:
    """
    Compute rank changes for choice1 and choice2 tokens across steering strengths.
    Shows how token positions move in the sorted vocabulary as steering strength varies.

    Args:
        logits_batch: tensor of shape [batch_size, vocab_size] containing logits for all alpha values
        choice1_id: Token ID for first choice
        choice2_id: Token ID for second choice

    Returns:
        Tuple of (choice1_ranks, choice2_ranks) where each is a List[int] of ranks
    """
    # Get ranks for both choice tokens in a single operation
    # argsort gives indices sorted by value (ascending), so we need descending for ranks
    sorted_indices = torch.argsort(logits_batch, dim=-1, descending=True)  # [batch_size, vocab_size]

    # Create rank tensors: for each position, what rank does each token have?
    # We need to find where choice1_id and choice2_id appear in the sorted indices
    vocab_size = logits_batch.shape[-1]
    batch_size = logits_batch.shape[0]

    # Create a tensor where ranks[b, token_id] = rank of token_id in batch b
    ranks = torch.zeros(batch_size, vocab_size, dtype=torch.long, device=logits_batch.device)
    batch_indices = torch.arange(batch_size, device=logits_batch.device).unsqueeze(1)  # [batch_size, 1]
    rank_indices = torch.arange(vocab_size, device=logits_batch.device).unsqueeze(0)   # [1, vocab_size]

    # Fill in the ranks: ranks[batch, token_id] = rank_position
    ranks[batch_indices, sorted_indices] = rank_indices

    # Extract ranks for our target tokens
    choice1_ranks = ranks[:, choice1_id].tolist()  # [batch_size]
    choice2_ranks = ranks[:, choice2_id].tolist()  # [batch_size]

    return choice1_ranks, choice2_ranks


# ==================== GLOBAL EFFECTS METRICS ====================

@torch.no_grad()
def compute_perplexity(logits_batch: torch.Tensor, target_token_id: int) -> List[float]:
    """
    Compute perplexity for each set of logits given a target token.

    Args:
        logits_batch: tensor of shape [batch_size, vocab_size] containing logits for all alpha values
        target_token_id: ID of the target token (e.g., he or she)

    Returns:
        List of perplexity values
    """
    # Compute softmax probabilities in a single operation
    probs_batch = torch.softmax(logits_batch, dim=-1)  # [batch_size, vocab_size]

    # Extract target token probabilities and clamp to avoid log(0)
    target_probs = probs_batch[:, target_token_id].clamp(min=1e-10)  # [batch_size]

    # Compute negative log likelihood and perplexity in a single operation
    nll_batch = -torch.log(target_probs)  # [batch_size]
    perplexity_batch = torch.exp(nll_batch)  # [batch_size]

    return perplexity_batch.tolist()


@torch.no_grad()
def kl_divergences(
    logits_batch: torch.Tensor,
    alpha_values: List[float]
) -> List[float]:
    """
    Compute KL divergence between baseline (α=0) and steered distributions.

    Measures overall shift of the whole distribution, not just target tokens.
    High KL = steering perturbs many logits (broad/noisy effect).
    Low KL = steering is specific to bias direction.

    Args:
        logits_batch: tensor of shape [batch_size, vocab_size] containing logits for all alpha values
        alpha_values: List of alpha values corresponding to rows in logits_batch

    Returns:
        List of KL divergence values D_KL(P_baseline || P_steered) (ordered from baseline to highest alpha)
    """
    # Find baseline index (alpha closest to 0)
    baseline_idx = min(range(len(alpha_values)), key=lambda i: abs(alpha_values[i]))
    baseline_logits = logits_batch[baseline_idx]  # [vocab_size]

    # Create baseline batch (repeat baseline for each alpha)
    baseline_batch = baseline_logits.unsqueeze(0).expand_as(logits_batch)  # [batch_size, vocab_size]
    
    # Compute softmax probabilities in batched operations
    baseline_probs_batch = torch.softmax(baseline_batch, dim=-1)  # [batch_size, vocab_size]
    steered_probs_batch = torch.softmax(logits_batch, dim=-1)  # [batch_size, vocab_size]
    
    # Compute KL divergence in a single batched operation
    # KL divergence: D_KL(P||Q) = sum P(x) * log(P(x) / Q(x))
    # Add small epsilon to both numerator and denominator to avoid log(0) and division by 0
    eps = 1e-10
    baseline_probs_safe = torch.clamp(baseline_probs_batch, min=eps)
    steered_probs_safe = torch.clamp(steered_probs_batch, min=eps)
    
    kl_divs_batch = torch.sum(
        baseline_probs_safe * torch.log(baseline_probs_safe / steered_probs_safe), 
        dim=-1
    )  
    
    return kl_divs_batch.tolist()