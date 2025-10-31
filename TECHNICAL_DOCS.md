# Technical Documentation - Bias Probing Project

**Internal Reference Guide**

---

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Experiment Loop Breakdown](#experiment-loop-breakdown)
3. [Module Reference](#module-reference)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Key Concepts](#key-concepts)
6. [Example Execution Trace](#example-execution-trace)

---

## High-Level Overview

### What This Project Does
Tests how language models respond to **activation steering** - injecting directional vectors into the model's internal representations to influence behavior (e.g., making it more supportive vs. unsupportive).

### Core Workflow
```
1. Load model and dataset
2. Extract steering vectors from training pairs (supportive vs unsupportive prompts)
3. Inject vectors at various layers with different strengths (alpha values)
4. Measure effects using multiple metrics (logit differences, probabilities, etc.)
5. Plot response curves showing how steering strength affects model behavior
```

---

## Experiment Loop Breakdown

### Phase 1: Initialization (happens once)

```python
# Entry point: cli.py → Experiment.__init__()

1. Configure determinism (set random seeds)
2. Load model via TransformerLens
   ├─ Print model info (layers, vocab, available hooks)
   └─ Validate model is compatible
3. Load datasets
   ├─ Training pairs: (positive_prompt, negative_prompt) × 160
   └─ Test prompts: base questions × 40
4. Validate tokenization
   ├─ Check "A" → single token ID
   └─ Check "B" → single token ID
5. Build alpha range (steering strengths to test)
   └─ e.g., [-2.0, -1.0, 0.0, 1.0, 2.0]
6. Determine layer combinations
   ├─ inject_layers: where to inject steering vectors
   └─ read_layers: where to read out results (must be > inject_layer)
7. Validate hook points
   └─ Ensure inject_site and read_site exist in model
```

### Phase 2: Vector Construction (per inject layer)

```python
# For each inject_layer in inject_layers:

def build_vectors(model, inject_layer, train_pairs):
    # vectors.py → build_vectors()
    
    # 1. Check cache first
    if cached_vectors_exist():
        return load_from_cache()
    
    # 2. Extract activations from training pairs
    activations = []
    for (positive_prompt, negative_prompt) in train_pairs:
        act_pos = extract_activation(positive_prompt, inject_layer)
        act_neg = extract_activation(negative_prompt, inject_layer)
        activations.append(act_pos - act_neg)
    
    # 3. Compute mean difference vector
    bias_vector = mean(activations)
    bias_vector = normalize(bias_vector)  # Unit vector
    
    # 4. Generate control vectors
    random_vector = normalize(random_direction())
    orthogonal_vector = normalize(project_out_bias(random_direction()))
    
    # 5. Cache and return
    cache_vectors(bias_vector, random_vector, orthogonal_vector)
    return {
        "bias": bias_vector,      # The actual steering direction
        "random": random_vector,  # Control: random direction
        "orth": orthogonal_vector # Control: perpendicular to bias
    }
```

### Phase 3: Steering & Measurement (nested loops)

```python
# For each inject_layer:
#   For each read_layer (where read_layer > inject_layer):
#     For each test_prompt:
#       For each vector_type (bias, random, orth):
#         For each alpha value:

def sweep_alpha(model, vector, alpha_values, prompt, ...):
    # steering.py → sweep_alpha()
    
    logits_for_all_alphas = []
    
    for alpha in alpha_values:  # e.g., [-2, -1, 0, 1, 2]
        
        # 1. Tokenize prompt
        tokens = model.to_tokens(prompt)
        
        # 2. Run model with hooks
        def inject_hook(activation, hook):
            # Add: activation += alpha * vector
            activation[:, :, :] += alpha * vector
            return activation
        
        def read_hook(activation, hook):
            # Capture the activation at read layer
            cached_activation = activation.detach().clone()
            return activation
        
        model.run_with_hooks(
            tokens,
            fwd_hooks=[
                (inject_hook_name, inject_hook),
                (read_hook_name, read_hook)
            ]
        )
        
        # 3. Get logits for last token
        logits = model.unembed(model.ln_final(cached_activation[:, -1, :]))
        logits_for_all_alphas.append(logits)
    
    return stack(logits_for_all_alphas)  # Shape: [n_alphas, vocab_size]

# After collecting logits for all prompts, average them:
averaged_logits = mean(logits_across_prompts)  # [n_alphas, vocab_size]
```

### Phase 4: Metrics Computation

```python
# metrics.py - various functions

def compute_metrics(logits_batch, choice1_id, choice2_id):
    # logits_batch shape: [n_alphas, vocab_size]
    
    # METRIC 1: Logit Differences
    logit_diffs = logits[:, choice1_id] - logits[:, choice2_id]
    # Returns: [α=-2: diff1, α=-1: diff2, α=0: diff3, ...]
    
    # METRIC 2: Probability Differences  
    probs = softmax(logits)
    prob_diffs = probs[:, choice1_id] - probs[:, choice2_id]
    
    # METRIC 3: Odds Ratios
    odds_ratios = exp(logit_diffs)
    
    # METRIC 4: Rank Changes
    sorted_indices = argsort(logits, descending=True)
    rank_choice1 = find_rank(sorted_indices, choice1_id)
    rank_choice2 = find_rank(sorted_indices, choice2_id)
    
    # METRIC 5: Perplexity
    perplexity = exp(-log(probs[:, target_token]))
    
    # METRIC 6: KL Divergence
    baseline_probs = softmax(logits[alpha=0])
    kl_divs = sum(baseline_probs * log(baseline_probs / probs))
```

### Phase 5: Plotting

```python
# plotting.py → plot_and_save_brc_curves()

def plot_brc_curves(bias_results, random_results, orth_results, alpha_values):
    
    # 1. Create figure
    fig, ax = plt.subplots()
    
    # 2. Plot three lines
    ax.plot(alpha_values, bias_results, label="bias", color="blue")
    ax.plot(alpha_values, random_results, label="random", color="orange")
    ax.plot(alpha_values, orth_results, label="orth", color="green")
    
    # 3. Add reference line at y=0
    ax.axhline(0, linestyle="--", color="black")
    
    # 4. Label and save
    ax.set_xlabel("Steering coefficient α")
    ax.set_ylabel("Metric value")
    plt.savefig(output_path)
```

---

## Module Reference

### 1. `config.py` - Configuration

```python
@dataclass
class ExperimentConfig:
    """Stores all experiment hyperparameters."""
    
    # Model settings
    model_name: str = "gpt2-small"
    prepend_bos: bool = True
    
    # Hook locations
    inject_site: str = "hook_resid_mid"  # Where to inject vector
    read_site: str = "hook_resid_post"   # Where to read results
    
    # Alpha sweep parameters
    alpha_start: float = -10.0
    alpha_stop: float = 10.0
    alpha_step: float = 0.5
    
    # Layer specifications
    inject_layers: Optional[List[int]] = None  # None = all layers
    read_layers: Optional[List[int]] = None
    
    # Experiment settings
    seed: int = 42
    out_dir: str = "graphs"
    metric: Optional[str] = None  # None = run all metrics
    steer_all_tokens: bool = True
    use_log_scale: bool = False
    dataset: str = "reassurance"
    show_progress: bool = True
```

### 2. `model.py` - Model Loading & Validation

#### Key Functions:

**`load_model(model_name, device, progress_tracker) → HookedTransformer`**
- Loads model via TransformerLens
- Moves to GPU/CPU
- Sets to eval mode

**`_extract_single_token_id(model, text, token_name) → int`**
- **Purpose**: Validate text tokenizes to exactly 1 token
- **Strategy**: Tries variations (e.g., "A", " A")
- **Raises**: ValueError if multi-token
- **Returns**: Token ID as integer

**`get_choice_token_ids(model) → Tuple[int, int]`**
- Gets token IDs for "A" and "B"
- Used for multiple-choice datasets
- Validates single-token assumption

**`get_pronoun_token_ids(model) → Tuple[int, int]`**
- Gets token IDs for "he" and "she"
- Used for Winogender gender bias dataset

**`validate_hook_points(model, inject_site, read_site, inject_layers, read_layers) → None`**
- **Purpose**: Verify hook points exist before running
- **Checks**: All `blocks.{layer}.{site}` combinations
- **Raises**: ValueError with helpful message if missing
- **Prints**: Available hook sites for debugging

**`print_model_info(model) → None`**
- Displays model architecture details
- Lists all available hook sites
- Useful for debugging new models

**`get_available_hook_sites(model) → List[str]`**
- Returns list of hook site names
- Example: `["hook_resid_pre", "hook_resid_mid", "hook_resid_post", ...]`

### 3. `data.py` - Dataset Loading

#### Key Functions:

**`load_train_dataset(dataset_name, data_base_path) → List[Tuple[str, str]]`**
- **Purpose**: Load training pairs for building steering vectors
- **Format**: `[(positive_prompt, negative_prompt), ...]`
- **Example**:
  ```python
  [
    ("Can you help me?\n\nI choose (A", "Can you help me?\n\nI choose (B"),
    ...
  ]
  ```
- **Location**: `data/{dataset_name}/{dataset_name}_train.json`

**`load_test_dataset(dataset_name, data_base_path) → List[str]`**
- **Purpose**: Load test prompts for evaluation
- **Format**: `[base_prompt, ...]`
- **Example**: `["Do you think I'll be okay?\n\nI choose (", ...]`
- **Location**: `data/{dataset_name}/{dataset_name}_test.json`

**Dataset JSON Schema:**
```json
{
  "question": "Full question text\n\nChoices:\n (A) option1\n (B) option2",
  "answer_matching_behavior": "(A)",
  "answer_not_matching_behavior": "(B)"
}
```

### 4. `vectors.py` - Steering Vector Construction

#### Key Functions:

**`residual_at_last_token(model, prompt, layer, site, prepend_bos, device) → Tensor`**
- **Purpose**: Extract activation at specific layer/position
- **Strategy**: 
  1. Tokenize prompt
  2. Run forward pass with hook
  3. Capture activation at last token
- **Returns**: Tensor of shape `[d_model]`

**`build_vectors(model, inj_layer, prompt_pairs, prepend_bos, device, inject_site, model_name, dataset) → Dict[str, Tensor]`**
- **Purpose**: Build steering vectors from training data
- **Caching**: Uses `VectorCache` to avoid recomputation
- **Process**:
  1. Check cache
  2. For each pair, compute `activation_diff = act_pos - act_neg`
  3. Average all differences → `bias_vector`
  4. Generate random and orthogonal controls
  5. Normalize all to unit vectors
  6. Save to cache
- **Returns**: `{"bias": vec1, "random": vec2, "orth": vec3}`

### 5. `steering.py` - Activation Steering

#### Key Functions:

**`get_steered_logits(model, prompt, steer_vec, alpha, inject_hook_name, read_hook_name, inject_layer, read_layer, prepend_bos, device, steer_all_tokens) → Tensor`**
- **Purpose**: Get model logits with steering applied
- **Process**:
  1. Tokenize prompt
  2. Define injection hook: `activation += alpha * steer_vec`
  3. Define read hook: capture activation
  4. Run model with both hooks
  5. Apply layer norm and unembedding
- **Returns**: Logits for last token, shape `[vocab_size]`

**`sweep_alpha(model, vector, alpha_values, prompt, inj_layer, read_layer, inject_hook_name, read_hook_name, prepend_bos, device, steer_all_tokens) → Tensor`**
- **Purpose**: Get logits for all alpha values
- **Process**: Calls `get_steered_logits` for each alpha
- **Returns**: Tensor of shape `[n_alphas, vocab_size]`

### 6. `metrics.py` - Metrics Computation

All metrics take `logits_batch` of shape `[n_alphas, vocab_size]`:

**`logit_diffs(logits_batch, choice1_id, choice2_id) → List[float]`**
- **Formula**: `logit(A) - logit(B)`
- **Interpretation**: Raw preference margin
- **Range**: Unbounded

**`prob_diffs(logits_batch, choice1_id, choice2_id) → List[float]`**
- **Formula**: `P(A) - P(B)` where `P = softmax(logits)`
- **Interpretation**: Probability difference
- **Range**: [-1, 1]

**`odds_ratios(logits_batch, choice1_id, choice2_id) → List[float]`**
- **Formula**: `exp(logit(A) - logit(B))`
- **Interpretation**: "A is X times more likely than B"
- **Range**: [0, ∞), with 1.0 = parity

**`rank_changes(logits_batch, choice1_id, choice2_id) → Tuple[List[int], List[int]]`**
- **Purpose**: Track token rank positions
- **Returns**: (choice1_ranks, choice2_ranks) across alphas
- **Range**: [1, vocab_size]

**`compute_perplexity(logits_batch, target_token_id) → List[float]`**
- **Formula**: `exp(-log(P(target)))`
- **Interpretation**: Model confidence (lower = more confident)
- **Range**: [1, ∞)

**`kl_divergences(logits_batch, alpha_values) → List[float]`**
- **Formula**: `KL(P_baseline || P_steered)`
- **Interpretation**: Distribution shift magnitude
- **Baseline**: Distribution at α=0

### 7. `plotting.py` - Visualization

#### Key Functions:

**`plot_and_save_brc_curves(bias_diffs, random_diffs, orth_diffs, alpha_values, inj_layer, read_layer, inject_site, read_site, out_dir, y_limits, metric_name, use_log_scale, dataset_name, model_name, log_scale_both) → None`**
- **Purpose**: Create and save BRC plot
- **Features**:
  - Three lines: bias (blue), random (orange), orth (green)
  - Reference line at y=0
  - Global y-limits for consistency
  - Optional log scale
  - Metric-specific formatting
- **Output**: `{out_dir}/{dataset}/{model}/{metric}/injL{inj}/brc_{metric}_injL{inj}_{inject_site}_readL{read}_{read_site}.png`

**`plot_rank_changes_and_save(...)`**
- Special plotting for rank metrics
- Inverted y-axis (rank 1 at top)
- Dual lines for both choices
- Crossing point annotation

**`plot_kl_divergences_and_save(...)`**
- Only plots bias vector (controls not meaningful)
- Reference line at KL=0

### 8. `cache.py` - Vector Caching

```python
class VectorCache:
    """File-based cache for steering vectors."""
    
    def __init__(self, cache_dir="cache/vectors"):
        # Hierarchical structure: cache/vectors/{model}/{dataset}/layer{N}_{site}.pt
    
    def load(model_name, dataset, layer, inject_site, device) → Optional[Dict]:
        # Returns {"bias": vec, "random": vec, "orth": vec} if exists
    
    def save(vectors, model_name, dataset, layer, inject_site) → None:
        # Saves vectors to disk (on CPU for portability)
    
    def exists(model_name, dataset, layer, inject_site) → bool:
        # Check if cached vectors exist
    
    def clear_cache(...) → int:
        # Delete cached vectors (optionally filtered by model/dataset)
```

### 9. `experiment.py` - Main Orchestration

```python
class Experiment:
    """Main experiment orchestrator."""
    
    def __init__(self, config: ExperimentConfig):
        # Phase 1: Setup
        # - Load model
        # - Load datasets
        # - Validate tokenization
        # - Validate hooks
        # - Build alpha range
        # - Determine layers
    
    def _get_metrics_to_run(self) → Dict[str, Callable]:
        # Returns mapping of metric_name → metric_function
        # If config.metric is None, returns all metrics
    
    def run_experiment(self) → None:
        # Phase 2-5: Main loop
        # For each inject_layer:
        #   Build vectors
        #   For each read_layer:
        #     For each test_prompt:
        #       For each vector_type:
        #         Sweep alpha
        #     Compute metrics
        #     Collect for global limits
        # Compute global y-limits
        # Generate all plots
```

### 10. `cli.py` - Command Line Interface

```python
def build_parser() → ArgumentParser:
    # Defines all CLI arguments
    # --model-name, --dataset, --metric, --inject-layers, etc.

def main(argv: Optional[List[str]]) → None:
    # 1. Parse arguments
    # 2. Build ExperimentConfig
    # 3. Create and run Experiment
```

### 11. `utils.py` - Utility Functions

**`get_device() → torch.device`**
- Returns CUDA if available, else CPU

**`configure_determinism(seed: int) → None`**
- Sets all random seeds
- Configures CUDA for deterministic behavior

**`build_alpha_range(start, stop, step) → np.ndarray`**
- Creates array of alpha values

**`unit_vector(x: Tensor) → Tensor`**
- Normalizes vector to unit length

**`build_hook_name(layer: int, site: str) → str`**
- Creates hook name: `blocks.{layer}.{site}`

**`parse_layer_spec(spec: str) → List[int]`**
- Parses layer specifications:
  - `"all"` → None
  - `"0,2,5"` → [0, 2, 5]
  - `"3-8"` → [3, 4, 5, 6, 7]

### 12. `observability.py` - Progress Tracking

```python
class ExperimentProgressTracker:
    """Nested progress bars for experiment loops."""
    
    def track_injection_layers(layers) → Iterator:
        # Top-level progress bar
    
    def track_read_layers(layers) → Iterator:
        # Second-level progress bar
    
    def track_vector_types(types) → Iterator:
        # Third-level progress bar
    
    def track_test_prompts(prompts) → Iterator:
        # Fourth-level progress bar
    
    def track_plotting(results) → Iterator:
        # Final plotting progress
    
    def track_model_loading(model_name) → ContextManager:
        # Model download progress
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         EXPERIMENT FLOW                          │
└─────────────────────────────────────────────────────────────────┘

INPUT: Config (model, dataset, layers, alpha range, metric)
  │
  ├─> Load Model (model.py)
  │     ├─> Download from HuggingFace
  │     ├─> Validate tokenization (A, B → single tokens)
  │     └─> Validate hook points exist
  │
  ├─> Load Datasets (data.py)
  │     ├─> Training pairs (160) for building vectors
  │     └─> Test prompts (40) for evaluation
  │
  └─> FOR each inject_layer:
        │
        ├─> Build Vectors (vectors.py)
        │     ├─> Check cache
        │     ├─> Extract activations from training pairs
        │     ├─> Average: bias_vector = mean(pos - neg)
        │     ├─> Generate random & orthogonal controls
        │     └─> Save to cache
        │
        └─> FOR each read_layer > inject_layer:
              │
              ├─> FOR each test_prompt (40):
              │     └─> FOR each vector (bias, random, orth):
              │           └─> FOR each alpha in [-2, -1, 0, 1, 2]:
              │                 ├─> Inject: activation += alpha * vector
              │                 ├─> Read: capture output activation
              │                 └─> Store: logits[alpha, vocab_size]
              │
              ├─> Average logits across all test prompts
              │     Result: [n_alphas, vocab_size] per vector
              │
              ├─> Compute Metrics (metrics.py)
              │     ├─> logit_diffs: logit[A] - logit[B]
              │     ├─> prob_diffs: P(A) - P(B)
              │     ├─> odds_ratios: exp(logit_diff)
              │     ├─> rank_changes: rank(A), rank(B)
              │     ├─> perplexity: exp(-log P(target))
              │     └─> kl_divergences: KL(baseline || steered)
              │
              └─> Collect results for global y-limits

After all layer combinations:
  │
  ├─> Compute Global Y-Limits (per metric)
  │     Purpose: Consistent y-axis across all plots
  │
  └─> Generate Plots (plotting.py)
        FOR each (inject_layer, read_layer, metric):
          ├─> Plot 3 lines (bias, random, orth)
          ├─> Apply global y-limits
          └─> Save: graphs/{dataset}/{model}/{metric}/injL{i}/plot.png

OUTPUT: Directory of plots showing steering response curves
```

---

## Key Concepts

### 1. Activation Steering

**What it is**: Adding a directional vector to a model's internal representations to influence its behavior.

**Mathematical form**:
```
activation_new = activation_original + (alpha * steering_vector)
```

Where:
- `steering_vector`: Direction extracted from training pairs
- `alpha`: Strength coefficient (negative = opposite direction)

### 2. Hook Points

**What they are**: Named locations in the model where you can intercept and modify activations.

**Common sites** (GPT-2):
- `hook_resid_pre`: Before attention + MLP
- `hook_resid_mid`: After attention, before MLP
- `hook_resid_post`: After attention + MLP
- `hook_attn_out`: Attention output
- `hook_mlp_out`: MLP output

**Why two hook points**:
- `inject_site`: Where we add the steering vector
- `read_site`: Where we measure the effect (must be downstream)

### 3. Steering Vector Construction

**Goal**: Find direction in activation space that represents "supportive vs. unsupportive"

**Process**:
1. Run model on training pairs
2. For each pair, compute difference: `act_supportive - act_unsupportive`
3. Average all differences
4. Normalize to unit vector

**Why average?**: Captures general concept across many examples

**Control vectors**:
- **Random**: Arbitrary direction (should have no meaningful effect)
- **Orthogonal**: Perpendicular to bias (should have no effect on target behavior)

### 4. Alpha Sweep

**Purpose**: Test different steering strengths to see response curve

**Typical values**: [-10, -9, -8, ..., -1, 0, 1, ..., 8, 9, 10]

**Interpretation**:
- `α = 0`: No steering (baseline)
- `α > 0`: Push toward supportive direction
- `α < 0`: Push toward unsupportive direction

### 5. Metrics

**Local effects** (on specific tokens):
- Logit diffs: Raw model preference
- Prob diffs: Normalized preference
- Odds ratios: Human-interpretable likelihood
- Rank changes: Position in vocabulary

**Global effects** (on whole distribution):
- Perplexity: Model confidence
- KL divergence: Distribution shift magnitude

### 6. Caching Strategy

**Why cache?**: Vector computation is expensive (many forward passes)

**What's cached**: The three steering vectors per (model, dataset, layer, site)

**Cache structure**:
```
cache/vectors/
  gpt2_small/
    reassurance/
      layer0_hook_resid_mid.pt
      layer1_hook_resid_mid.pt
      ...
```

**Invalidation**: Automatic if model/dataset/layer changes

---

## Example Execution Trace

### Minimal Command:
```bash
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --dataset reassurance \
  --metric logit_diffs \
  --inject-layers 2 \
  --read-layers 3 \
  --alpha-start -2 \
  --alpha-stop 2 \
  --alpha-step 1
```

### Execution Trace:

```
[cli.py:main]
  ↓ Parse arguments
  ↓ Create ExperimentConfig
  ↓ Call Experiment(config)

[experiment.py:__init__]
  ↓ configure_determinism(seed=42)
  ↓ device = get_device() → cuda
  ↓ model = load_model("gpt2-small", cuda)
    [model.py:load_model]
      ↓ HookedTransformer.from_pretrained("gpt2-small")
      ↓ model.to(cuda)
      ↓ model.eval()
      ↓ return model
  ↓ print_model_info(model)
    OUTPUT: "Architecture: gpt2, Layers: 12, Vocab: 50257..."
  
  ↓ train_pairs = load_train_dataset("reassurance")
    [data.py:load_train_dataset]
      ↓ Load data/reassurance/reassurance_train.json
      ↓ Parse 160 examples
      ↓ Create pairs: ("...I choose (A", "...I choose (B")
      ↓ return 160 pairs
  
  ↓ test_prompts = load_test_dataset("reassurance")
    [data.py:load_test_dataset]
      ↓ Load data/reassurance/reassurance_test.json
      ↓ Create 40 base prompts: "...I choose ("
      ↓ return 40 prompts
  
  ↓ print(f"Loaded {len(train_pairs)} training pairs...")
    OUTPUT: "Loaded 160 training pairs and 40 test prompts"
  
  ↓ choice1_id, choice2_id = get_choice_token_ids(model)
    [model.py:get_choice_token_ids]
      ↓ _extract_single_token_id(model, "A", "Choice A")
        ↓ Try "A" → tokens shape [1, 1] ✓
        ↓ return token_id = 32
      ↓ _extract_single_token_id(model, "B", "Choice B")
        ↓ Try "B" → tokens shape [1, 1] ✓
        ↓ return token_id = 33
      ↓ return (32, 33)
    OUTPUT: "✓ Token 'Choice A' ('A') → single token ID 32"
    OUTPUT: "✓ Token 'Choice B' ('B') → single token ID 33"
  
  ↓ alpha_values = build_alpha_range(-2, 2, 1)
    ↓ return [-2.0, -1.0, 0.0, 1.0, 2.0]  # 5 values
  
  ↓ inject_layers = [2]
  ↓ read_layers = [3]
  
  ↓ validate_hook_points(model, "hook_resid_mid", "hook_resid_post", [2], [3])
    [model.py:validate_hook_points]
      ↓ Check: "blocks.2.hook_resid_mid" in model.hook_dict ✓
      ↓ Check: "blocks.3.hook_resid_post" in model.hook_dict ✓
      ↓ print("✓ All hook points validated successfully!")
    OUTPUT: "✓ Inject site 'hook_resid_mid' exists for layers: [2]"
    OUTPUT: "✓ Read site 'hook_resid_post' exists for layers: [3]"

[experiment.py:run_experiment]
  ↓ metrics_to_run = _get_metrics_to_run()
    ↓ return {"logit_diffs": lambda logits: logit_diffs(logits, 32, 33)}
  
  ↓ FOR inject_layer=2:
      
      ↓ vectors = build_vectors(model, layer=2, train_pairs, ...)
        [vectors.py:build_vectors]
          ↓ cache = VectorCache()
          ↓ cached = cache.load("gpt2-small", "reassurance", 2, "hook_resid_mid")
            [cache.py:load]
              ↓ Check: cache/vectors/gpt2_small/reassurance/layer2_hook_resid_mid.pt
              ↓ File doesn't exist
              ↓ return None
          
          ↓ Computing vectors... (not cached)
          ↓ FOR each of 160 training pairs:
              ↓ act_pos = residual_at_last_token(model, positive_prompt, 2, "hook_resid_mid")
              ↓ act_neg = residual_at_last_token(model, negative_prompt, 2, "hook_resid_mid")
              ↓ diff = act_pos - act_neg  # shape [768]
              ↓ accumulate diffs
          
          ↓ bias_vec = mean(all_diffs)  # [768]
          ↓ bias_vec = unit_vector(bias_vec)
          
          ↓ rand_vec = unit_vector(torch.randn(768))
          
          ↓ orth_seed = torch.randn(768)
          ↓ orth_vec = unit_vector(orth_seed - (orth_seed @ bias_vec) * bias_vec)
          
          ↓ vectors = {"bias": bias_vec, "random": rand_vec, "orth": orth_vec}
          
          ↓ cache.save(vectors, "gpt2-small", "reassurance", 2, "hook_resid_mid")
          ↓ return vectors
        
        OUTPUT: "Cached vectors saved to gpt2_small/reassurance/layer2_hook_resid_mid.pt"
      
      ↓ FOR read_layer=3:
          
          ↓ accumulated_logits = {"bias": [], "random": [], "orth": []}
          
          ↓ FOR test_prompt in 40 test_prompts:  # "Do you think I'll be okay?..."
              
              ↓ FOR vector_name in ["bias", "random", "orth"]:
                  
                  ↓ logits = sweep_alpha(model, vectors[vector_name], [-2,-1,0,1,2], test_prompt, ...)
                    [steering.py:sweep_alpha]
                      ↓ out = []
                      ↓ FOR alpha in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                          ↓ logits = get_steered_logits(model, test_prompt, vector, alpha, ...)
                            [steering.py:get_steered_logits]
                              ↓ tokens = model.to_tokens(test_prompt)  # [1, seq_len]
                              ↓ last_idx = seq_len - 1
                              
                              ↓ def inject_hook(act, hook):
                              ↓     act[:, :, :] += alpha * vector  # Add steering
                              ↓     return act
                              
                              ↓ def read_hook(act, hook):
                              ↓     cache["resid"] = act.detach().clone()
                              ↓     return act
                              
                              ↓ model.run_with_hooks(tokens, fwd_hooks=[
                              ↓     ("blocks.2.hook_resid_mid", inject_hook),
                              ↓     ("blocks.3.hook_resid_post", read_hook)
                              ↓ ])
                              
                              ↓ resid = model.ln_final(cache["resid"][:, last_idx, :])
                              ↓ logits = model.unembed(resid)[0, 0, :]  # [vocab_size]
                              ↓ return logits
                          
                          ↓ out.append(logits)
                      
                      ↓ return torch.stack(out)  # [5, 50257]
                  
                  ↓ accumulated_logits[vector_name].append(logits)
          
          ↓ Average across 40 test prompts:
          ↓ bias_logits = mean(accumulated_logits["bias"])    # [5, 50257]
          ↓ random_logits = mean(accumulated_logits["random"]) # [5, 50257]
          ↓ orth_logits = mean(accumulated_logits["orth"])    # [5, 50257]
          
          ↓ Compute metric:
          ↓ bias_results = logit_diffs(bias_logits, choice1_id=32, choice2_id=33)
            [metrics.py:logit_diffs]
              ↓ diffs = logits[:, 32] - logits[:, 33]  # [5]
              ↓ return diffs.tolist()
              # e.g., [-0.5, -0.2, 0.0, 0.3, 0.6]
          
          ↓ random_results = logit_diffs(random_logits, 32, 33)
            # e.g., [0.1, 0.05, 0.0, -0.05, -0.1]
          
          ↓ orth_results = logit_diffs(orth_logits, 32, 33)
            # e.g., [-0.02, -0.01, 0.0, 0.01, 0.02]
          
          ↓ Collect results for global limits
          ↓ all_metric_data["logit_diffs"].extend([
          ↓     -0.5, -0.2, 0.0, 0.3, 0.6,  # bias
          ↓     0.1, 0.05, 0.0, -0.05, -0.1, # random
          ↓     -0.02, -0.01, 0.0, 0.01, 0.02 # orth
          ↓ ])
  
  ↓ Compute global y-limits:
  ↓ y_min = min(all_metric_data["logit_diffs"]) - 10% padding
  ↓ y_max = max(all_metric_data["logit_diffs"]) + 10% padding
  ↓ global_y_limits["logit_diffs"] = (y_min, y_max)
  
  ↓ Generate plots:
  ↓ FOR (inj_layer=2, read_layer=3, bias_results, random_results, orth_results, "logit_diffs"):
      
      ↓ plot_and_save_brc_curves(
      ↓     bias_diffs=[-0.5, -0.2, 0.0, 0.3, 0.6],
      ↓     random_diffs=[0.1, 0.05, 0.0, -0.05, -0.1],
      ↓     orth_diffs=[-0.02, -0.01, 0.0, 0.01, 0.02],
      ↓     alpha_values=[-2, -1, 0, 1, 2],
      ↓     y_limits=(y_min, y_max),
      ↓     ...
      ↓ )
        [plotting.py:plot_and_save_brc_curves]
          ↓ Create figure
          ↓ Plot three lines:
          ↓   - Blue line (bias): goes from -0.5 to 0.6 as α increases
          ↓   - Orange line (random): wiggles around 0
          ↓   - Green line (orth): stays near 0
          ↓ Add horizontal line at y=0
          ↓ Set labels, title
          ↓ Save to: test_minimal/reassurance/gpt2-small/logit_diffs/injL2/brc_logit_diffs_injL2_hook_resid_mid_readL3_hook_resid_post.png

OUTPUT FILE CREATED:
  test_minimal/reassurance/gpt2-small/logit_diffs/injL2/
    └─ brc_logit_diffs_injL2_hook_resid_mid_readL3_hook_resid_post.png

EXPERIMENT COMPLETE!
```

### What the Plot Shows:

**X-axis**: Steering strength α ∈ [-2, -1, 0, 1, 2]

**Y-axis**: Logit difference (logit(A) - logit(B))

**Lines**:
- **Bias (blue)**: Strong positive slope → steering works! As α increases, model prefers choice A more
- **Random (orange)**: Flat/wiggly → no systematic effect (good!)
- **Orth (green)**: Near zero → perpendicular direction has no effect (good!)

**Interpretation**: Steering in the bias direction successfully shifts model preference from choice B (unsupportive) toward choice A (supportive).

---

## Quick Reference

### Common CLI Commands:

```bash
# Minimal test (1 layer pair, 1 metric)
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --dataset reassurance \
  --metric logit_diffs \
  --inject-layers 2 \
  --read-layers 3 \
  --alpha-start -2 --alpha-stop 2 --alpha-step 1

# Full sweep (all metrics, multiple layers)
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --dataset reassurance \
  --inject-layers 0-6 \
  --read-layers 1-12 \
  --alpha-start -10 --alpha-stop 10 --alpha-step 0.5

# Different dataset
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --dataset deference \
  --metric prob_diffs

# Different model (make sure it's compatible!)
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-medium \
  --dataset reassurance
```

### File Locations:

```
Project Root/
├─ BRC_Experiment/Modularized/    # Source code
├─ data/{dataset}/                 # Training & test data
├─ cache/vectors/                  # Cached steering vectors
├─ graphs/ (or test_minimal/)      # Output plots
└─ TECHNICAL_DOCS.md               # This file
```

### Debugging Tips:

1. **Check model info**: Look at printed output for available hooks
2. **Verify tokenization**: Check if A, B tokenize to single tokens
3. **Examine cache**: Look in `cache/vectors/` to see what's saved
4. **Inspect plots**: Check if bias line has stronger effect than controls
5. **Try small first**: Test with 1-2 layers before full sweep

---

**END OF DOCUMENTATION**

