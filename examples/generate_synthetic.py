"""Example: generate fake scRNA-seq data from a known ground-truth GRN.

Run from the repo root:  python examples/generate_synthetic.py

Builds a small feed-forward GRN, simulates continuous expression along
pseudotime with scRNA-seq-like noise, and emits:
  - an AnnData object (compatible with the iqcell binarization/hierarchy pipeline)
  - torch tensors for a target gene (compatible with the SINN And/Or model)
  - CSV files (examples/data/expression.csv, examples/data/pseudotime.csv)
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from iqcell.simulation import GRNSpec, HillParams, SyntheticGRNGenerator, NoiseConfig


def build_spec() -> GRNSpec:
    genes = ["Sox2", "Gata2", "Gata1", "Pu1", "Klf1"]
    spec = GRNSpec(genes)
    # Sox2 is a root driver gene (no regulators).
    spec.set_rule("Gata2", activators=["Sox2"])
    # Classic Gata1 <-> Pu.1 style antagonism (as a DAG here):
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
    spec = build_spec()
    gen = SyntheticGRNGenerator(
        spec,
        n_cells=600,
        reg_skip=8,
        noise=NoiseConfig(biological=0.05, dropout=0.2),
        seed=0,
    )

    adata = gen.to_anndata(binarize=True)
    print("AnnData:", adata)
    print("Ground-truth signed adjacency (rows=regulator, cols=target):")
    print(pd.DataFrame(adata.uns["grn_adjacency"], index=spec.genes, columns=spec.genes))

    # Tensors for the SINN And/Or logic engine, target = Klf1
    x_act, x_rep, y, t = gen.to_tensors("Klf1", binarize=True)
    print(f"\nKlf1 tensors: x_act={tuple(x_act.shape)} x_rep={tuple(x_rep.shape)} "
          f"y={tuple(y.shape)} t={tuple(t.shape)}")

    # Write CSVs (continuous noisy expression + pseudotime)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    expr_df = pd.DataFrame(
        gen.expression, columns=spec.genes, index=[f"cell_{i}" for i in range(gen.n_cells)]
    )
    expr_df.to_csv(os.path.join(data_dir, "expression.csv"))
    pd.DataFrame(
        {"pseudotime": gen.pseudotime},
        index=[f"cell_{i}" for i in range(gen.n_cells)],
    ).to_csv(os.path.join(data_dir, "pseudotime.csv"))
    print(f"\nWrote {data_dir}/expression.csv and pseudotime.csv")


if __name__ == "__main__":
    main()
