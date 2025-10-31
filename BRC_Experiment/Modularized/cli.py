import argparse
from typing import Optional
import time


from BRC_Experiment.Modularized.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run BRC experiment with configurable hyperparameters.")
    p.add_argument("--model-name", type=str, default="gpt2-small", help="Transformer-Lens model name")
    p.add_argument("--prepend-bos", action="store_true", help="Prepend BOS token during tokenization")
    p.add_argument("--no-prepend-bos", dest="prepend_bos", action="store_false", help="Do not prepend BOS token")
    p.set_defaults(prepend_bos=True)

    p.add_argument("--inject-site", type=str, default="hook_resid_mid", help="Injection hook site name")
    p.add_argument("--read-site", type=str, default="hook_resid_post", help="Readout hook site name")

    p.add_argument("--alpha-start", type=float, default=-10.0)
    p.add_argument("--alpha-stop", type=float, default=10.0)
    p.add_argument("--alpha-step", type=float, default=0.5)

    p.add_argument("--inject-layers", type=str, default="all", help='Layer spec like "all", "0,2,4" or "3-8"')
    p.add_argument("--read-layers", type=str, default="all", help='Layer spec like "all", "1,3,5" or "2-10"')

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default="graphs")
    p.add_argument("--metric", type=str, default=None, 
                   choices=["logit_diffs", "prob_diffs", "compute_perplexity", "odds_ratios", "rank_changes", "kl_divergences"],
                   help="Metric to compute (if not specified, runs all metrics): logit_diffs, prob_diffs, odds_ratios, compute_perplexity, rank_changes, or kl_divergences")
    p.add_argument("--steer-last-token-only", action="store_true", 
                   help="Steer only the last token position instead of all tokens (default: steer all tokens)")
    p.add_argument("--use-log-scale", action="store_true", 
                   help="Force log scale for plotting (automatically enabled for prob_diffs metric)")
    p.add_argument("--dataset", type=str, default="reassurance",
                   choices=["reassurance", "deference", "satisficing", "sycophancy"],
                   help="Dataset to use: reassurance (supportive/unsupportive), deference (authority bias), satisficing (good-enough bias), or sycophancy (authority agreement bias)")
    p.add_argument("--no-progress", action="store_true", 
                   help="Disable progress bars for silent execution")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for GPU operations (default: 16, higher = faster but more VRAM)")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # lightweight layer-spec parser to avoid importing utils on --help
    def _parse_layer_spec(spec: str | None):
        if spec is None:
            return None
        s = spec.strip().lower()
        if s in ("", "all"):
            return None
        if "," in s:
            return [int(x.strip()) for x in s.split(",") if x.strip()]
        if "-" in s:
            a, b = s.split("-", 1)
            return list(range(int(a), int(b)+1))
        return [int(s)]

    config = ExperimentConfig(
        model_name=args.model_name,
        prepend_bos=args.prepend_bos,
        inject_site=args.inject_site,
        read_site=args.read_site,
        alpha_start=args.alpha_start,
        alpha_stop=args.alpha_stop,
        alpha_step=args.alpha_step,
        inject_layers=_parse_layer_spec(args.inject_layers),
        read_layers=_parse_layer_spec(args.read_layers),
        seed=args.seed,
        out_dir=args.out_dir,
        metric=args.metric,
        steer_all_tokens=not args.steer_last_token_only,
        use_log_scale=args.use_log_scale,
        dataset=args.dataset,
        show_progress=not args.no_progress,
        batch_size=args.batch_size,
    )
    
    from BRC_Experiment.Modularized.experiment import Experiment    
    Experiment(config).run_experiment()


if __name__ == "__main__":
    main()


