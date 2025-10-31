# Cognitive Behavioral Modeling via Activation Steering (CBMAS)

CBMAS is a modular, production-style refactor of a Bias-Repelling Control (BRC) experiment built on TransformerLens. It supports multiple datasets (Winogender for gender bias, Reassurance for supportive/unsupportive responses), constructs steering vectors, sweeps steering strengths (alpha), and plots differences using configurable metrics (logit differences, probability differences, or perplexity).

## Features
- Clean module boundaries: config, data, model, steering, plotting, experiment, CLI
- **Multiple datasets**: Winogender (gender bias), Reassurance (supportive responses), Deference, Satisficing, Sycophancy
- **Multiple metrics**: `logit_diffs`, `prob_diffs`, `odds_ratios`, `rank_changes`, `compute_perplexity`, `kl_divergences`
- **Model validation**: Automatic checks for tokenization and hook point compatibility
- **Vector caching**: Smart caching system to avoid recomputation
- Deterministic runs when possible (cuBLAS/CUDNN settings and seeds)
- CLI to run experiments with configurable hyperparameters
- Quick-start example in `main.py` for fast debugging
- Organized output structure: `graphs/{dataset}/{model}/{metric}/...`

## Installation
1. Python 3.10+ recommended
2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
If you need CUDA-enabled PyTorch, adjust the torch/torchvision/torchaudio wheels as per your system.

## Model Compatibility

### Validated Models
The following models have been tested and work correctly:
- ✅ `gpt2-small` (GPT-2 124M)
- ✅ `gpt2-medium` (GPT-2 355M)
- ✅ `gpt2-large` (GPT-2 774M)
- ✅ `gpt2-xl` (GPT-2 1.5B)

### Automatic Validation
When you run an experiment, the system automatically validates:

1. **Tokenization**: Checks that choice tokens ("A", "B") tokenize to single tokens
   - If multi-token, experiment stops with clear error message
   - Tries multiple variations (with/without leading space)

2. **Hook Points**: Verifies that specified hook sites exist in the model
   - Prints available hook sites if validation fails
   - Example sites: `hook_resid_pre`, `hook_resid_mid`, `hook_resid_post`

3. **Model Information**: Displays architecture details on startup
   - Number of layers, hidden size, vocabulary size
   - All available hook points

### Testing New Models

To test a different model (e.g., GPT-Neo, Pythia, LLaMA):

```bash
# The validation system will tell you if it's compatible
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt-neo-125M \
  --dataset reassurance \
  --inject-layers 2 \
  --read-layers 3 \
  --alpha-start -2 --alpha-stop 2 --alpha-step 1
```

**If tokenization fails**: The model's tokenizer doesn't tokenize "A"/"B" as single tokens. You may need to:
- Modify the dataset format
- Use different choice markers

**If hook validation fails**: The model uses different hook point names. You'll see:
- List of available hook sites
- Use `--inject-site` and `--read-site` flags with correct names

### Hook Point Reference

Common TransformerLens hook sites (availability varies by model):
- `hook_resid_pre`: Before attention block
- `hook_resid_mid`: After attention, before MLP (default inject)
- `hook_resid_post`: After attention + MLP (default read)
- `hook_attn_out`: Attention output
- `hook_mlp_out`: MLP output
- `hook_q_input`, `hook_k_input`, `hook_v_input`: Attention inputs

## Project Structure
```
BRC_Experiment/
  Modularized/
    __init__.py
    config.py          # ExperimentConfig dataclass
    utils.py           # device, determinism, alpha grid, layer parsing
    data.py            # Winogender loader and prompt pairs
    model.py           # HookedTransformer loader and pronoun token ids
    steering.py        # residual capture, vector building, steering sweep
    plotting.py        # plot_and_save_brc_curves
    experiment.py      # Experiment orchestration
    cli.py             # CLI entrypoint
    main.py            # Quick-start runner or delegate to CLI
```

## Quick Start (no install)
We ship `main.py` with a built-in import shim, so you can run directly without installing the package:
```bash
python BRC_Experiment/Modularized/main.py
```
This runs a small, fast example (layers 0→1, alphas -1,0) and saves outputs to `graphs_debug/`.

## Testing Different Metrics

Test each metric with a small 3-layer setup:

**Logit differences:**
```bash
python -c "
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
cfg = ExperimentConfig(inject_layers=[0,1], read_layers=[2,3,4], alpha_start=-5, alpha_stop=5, alpha_step=2.5, out_dir='test_logit', metric='logit_diffs')
Experiment(cfg).run_experiment()
"
```

**Probability differences:**
```bash
python -c "
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
cfg = ExperimentConfig(inject_layers=[0,1], read_layers=[2,3,4], alpha_start=-5, alpha_stop=5, alpha_step=2.5, out_dir='test_prob', metric='prob_diffs')
Experiment(cfg).run_experiment()
"
```

**Perplexity:**
```bash
python -c "
from BRC_Experiment.Modularized.config import ExperimentConfig
from BRC_Experiment.Modularized.experiment import Experiment
cfg = ExperimentConfig(inject_layers=[0,1], read_layers=[2,3,4], alpha_start=-5, alpha_stop=5, alpha_step=2.5, out_dir='test_perplexity', metric='compute_perplexity')
Experiment(cfg).run_experiment()
"
```

## CLI Usage

**Basic Usage:**
```bash
# Show help
python -m BRC_Experiment.Modularized.cli --help

# Run with defaults
python -m BRC_Experiment.Modularized.cli

# Alternative without package install
PYTHONPATH=$(pwd) python -m BRC_Experiment.Modularized.cli
```

**Common Options:**
```bash
python -m BRC_Experiment.Modularized.cli \
  --model-name gpt2-small \
  --prefix "The doctor said that " \
  --metric logit_diffs \
  --alpha-start -10 --alpha-stop 10 --alpha-step 0.5 \
  --inject-layers 0-4 \
  --read-layers 1-6 \
  --out-dir graphs
```

**Available Datasets:**
- `reassurance` (default): Supportive vs unsupportive responses
- `deference`: Authority bias (deferring to experts/authority figures)
- `satisficing`: Good-enough bias (accepting satisfactory vs optimal solutions)
- `sycophancy`: Agreement bias (agreeing with user's stated views)
- `winogender`: Gender bias analysis (he vs she pronouns)

**Available Metrics:**
- `logit_diffs`: Raw logit differences (Choice1 - Choice2)
- `prob_diffs`: Probability differences P(Choice1) - P(Choice2) (auto-scaled to %)
- `odds_ratios`: Human-interpretable likelihood ratios (e^Δ)
- `rank_changes`: Token rank positions in vocabulary
- `compute_perplexity`: Model confidence/fluency measure
- `kl_divergences`: Distribution shift from baseline
- _(omit --metric to run all metrics at once)_

**Layer Specifications:**
- `all` (default): All layers 0 to n_layers-1
- Comma-separated: `0,2,5`
- Range: `3-8` (start inclusive, end exclusive)

## Notes
- The dataset `oskarvanderwal/winogender` is pulled automatically via `datasets`. Internet access is required the first time.
- Figures are saved as PNG: `graphs/{dataset}/{model}/{metric}/injL{inj}/brc_{metric}_injL{inj}_{inject_site}_readL{read}_{read_site}.png`.
- Determinism is best-effort due to CUDA/BLAS constraints.
- **Vector caching**: Steering vectors are cached in `cache/vectors/{model}/{dataset}/` to avoid recomputation

## Technical Documentation

For detailed information about the codebase:
- **[TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)**: Complete internal reference guide
  - Step-by-step experiment loop breakdown
  - Full module and function reference
  - Data flow diagrams
  - Example execution trace with real values
  - Quick lookup for all classes and functions

## Testing
A pytest suite can be created to mock heavy dependencies. Example categories:
- utils: seeds, device, alpha grid, layer parsing
- data: dataset loader mocked
- model: TransformerLens loader mocked
- steering: fake model for sweeps
- plotting: file creation
- experiment: orchestration with monkeypatched components
- cli: argument plumbing

## License
CC-BY 4.0
