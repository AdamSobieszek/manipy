# main.py
"""
Run a small comparison suite on MNIST by default.

Example:
  python main.py --steps 400 --log-dir ./runs/exp1
  python main.py --steps 400 --log-dir ./runs/exp2 --subset 10000
"""
import os, argparse, json
from tester import OptimizerComparisonTester, OptimizerConfig, build_default_mnist_tester

def build_candidate_suite(lr: float):
    # A few illustrative TraversalAdamW configurations
    return [
        OptimizerConfig("trav_none_all", "v1.TraversalAdamW",
                        dict(lr=lr, traversal_mode="none", moment1_source="trav", moment2_source="trav")),
        OptimizerConfig("trav_row_all", "v1.TraversalAdamW",
                        dict(lr=lr, traversal_mode="row", moment1_source="trav", moment2_source="trav")),
        OptimizerConfig("trav_full_mixed", "v1.TraversalAdamW",
                        dict(lr=lr, traversal_mode="full", moment1_source="grad", moment2_source="trav")),
        OptimizerConfig("trav_batch_all_clamped", "v1.TraversalAdamW",
                        dict(lr=lr, traversal_mode="batch", moment1_source="trav", moment2_source="trav",
                             traversal_gain_clamp=10.0)),
    ]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=str, default="./runs")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--subset", type=int, default=0, help="subset train size for quick runs")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lr", type=float, default=1e-3, help="LR for candidate optimizers")
    ap.add_argument("--baseline-lr", type=float, default=1e-3)
    ap.add_argument("--baseline-wd", type=float, default=1e-2)
    args = ap.parse_args()

    tester, dataset_name = build_default_mnist_tester(batch_size=args.batch_size, subset=(args.subset or None))
    os.makedirs(args.log_dir, exist_ok=True)

    suite = build_candidate_suite(lr=args.lr)
    print(f"Dataset={dataset_name} | steps={args.steps} | logging to {args.log_dir}")
    print("Candidate configs:")
    for c in suite:
        print(" -", c.as_readable())

    results = tester.run_suite(
        candidate_cfgs=suite,
        steps=args.steps,
        log_dir=args.log_dir,
        seed=args.seed,
        baseline_kwargs={"lr": args.baseline_lr, "weight_decay": args.baseline_wd},
    )

    index_path = os.path.join(args.log_dir, "suite_results.json")
    print(f"\nWrote suite index: {index_path}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()