"""Example: benchmark GRN inference on synthetic data via BEELINE.

Runs the full loop:  synthetic GRN -> BEELINE inputs -> inference -> evaluation.

Run from the repo root:
    python examples/beeline_benchmark.py [--beeline-repo /path/to/Beeline]

Without a Beeline checkout (or without Docker), the script still exports the
BEELINE inputs and config, then falls back to iqcell's pure-python scorer
(:func:`iqcell.beeline.score_ranking`) so you can see AUPRC/AUROC without any
external dependency. Point ``--beeline-repo`` at a cloned + initialized Beeline
repo (see docs/beeline.md) to drive the real Docker-backed pipeline.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from iqcell.simulation import GRNSpec, HillParams, SyntheticGRNGenerator, NoiseConfig
from iqcell.beeline import (
    export_beeline_inputs,
    write_config,
    BeelineRunner,
    read_ranked_edges,
    parse_evaluation,
    score_ranking,
)


def build_spec() -> GRNSpec:
    """A small hematopoietic-style feed-forward GRN (as a DAG)."""
    genes = ["Sox2", "Gata2", "Gata1", "Pu1", "Klf1"]
    spec = GRNSpec(genes)
    spec.set_rule("Gata2", activators=["Sox2"])
    spec.set_rule("Gata1", activators=["Gata2"], repressors=["Pu1"])
    spec.set_rule("Pu1", activators=["Gata2"])
    spec.set_rule(
        "Klf1",
        activators=["Gata1"],
        repressors=["Pu1"],
        hill={"Gata1": HillParams(K=0.4, n=6.0)},
    )
    spec.validate()
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--beeline-repo",
        default=os.environ.get("BEELINE_REPO"),
        help="Path to a cloned + initialized Murali-group/Beeline checkout. "
        "If omitted, only the pure-python fallback scorer is used.",
    )
    parser.add_argument(
        "--work-dir",
        default=os.path.join(os.path.dirname(__file__), "data", "beeline"),
        help="Directory for BEELINE inputs/outputs and the generated config.",
    )
    parser.add_argument("--n-cells", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    input_dir = os.path.join(work_dir, "inputs")
    output_dir = os.path.join(work_dir, "outputs")
    config_path = os.path.join(work_dir, "config.yaml")
    os.makedirs(input_dir, exist_ok=True)

    # 1. Simulate a known ground-truth GRN.
    spec = build_spec()
    gen = SyntheticGRNGenerator(
        spec,
        n_cells=args.n_cells,
        reg_skip=8,
        noise=NoiseConfig(biological=0.05, dropout=0.2),
        seed=args.seed,
    )
    ground_truth = list(spec.edges())
    print("Ground-truth signed edges:")
    for reg, tgt, sign in ground_truth:
        print(f"  {reg} --{'>' if sign == 1 else '|'} {tgt}")

    # 2. Export BEELINE inputs + config.
    paths = export_beeline_inputs(gen, input_dir, dataset_id="synthetic", run_id="run1")
    write_config(config_path, input_dir=input_dir, output_dir=output_dir)
    print(f"\nExported BEELINE inputs under {input_dir}")
    for key, path in paths.items():
        print(f"  {key}: {os.path.relpath(path, work_dir)}")
    print(f"Config: {os.path.relpath(config_path, work_dir)}")

    # 3. Drive the real BEELINE pipeline if a repo is available.
    ran_beeline = False
    if args.beeline_repo:
        runner = BeelineRunner(args.beeline_repo)
        status = runner.check_available()
        print(f"\nBEELINE availability: {status.message}")
        if status.available:
            print("Running BLRunner.py (inference)...")
            runner.run(config_path)
            print("Running BLEvaluator.py (AUPRC/AUROC/EPR)...")
            runner.evaluate(config_path, auc=True, epr=True)
            metrics = parse_evaluation(output_dir)
            print("\nBEELINE evaluation metrics:")
            for name, value in sorted(metrics.items()):
                print(f"  {name}: {value}")
            ran_beeline = True
        else:
            print("Falling back to the pure-python scorer (see docs/beeline.md).")

    # 4. Pure-python fallback: score each algorithm's rankedEdges.csv, if any,
    #    otherwise score a perfect ranking to demonstrate the scorer.
    if not ran_beeline:
        print("\nPure-python scoring (no BEELINE run):")
        run_out = os.path.join(output_dir, "synthetic", "run1")
        scored_any = False
        if os.path.isdir(run_out):
            for algo in sorted(os.listdir(run_out)):
                ranked = os.path.join(run_out, algo, "rankedEdges.csv")
                if os.path.isfile(ranked):
                    df = read_ranked_edges(ranked)
                    metrics = score_ranking(df, ground_truth)
                    print(f"  {algo}: {metrics}")
                    scored_any = True
        if not scored_any:
            # Demonstrate the scorer on an oracle ranking of the true edges.
            oracle = pd.DataFrame(
                [(reg, tgt, 1.0) for reg, tgt, _ in ground_truth],
                columns=["Gene1", "Gene2", "EdgeWeight"],
            )
            print(f"  oracle ranking: {score_ranking(oracle, ground_truth)}")


if __name__ == "__main__":
    main()
