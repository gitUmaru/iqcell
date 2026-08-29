"""Synthetic scRNA-seq GRN data generation for IQCELL.

Build fake single-cell expression data from an explicit, known ground-truth
gene regulatory network so inference modules can be validated against truth.

Example:
    >>> from iqcell.simulation import GRNSpec, SyntheticGRNGenerator, NoiseConfig
    >>> spec = GRNSpec(["A", "B", "C"])
    >>> spec.set_rule("B", activators=["A"])
    >>> spec.set_rule("C", activators=["B"], repressors=["A"])
    >>> gen = SyntheticGRNGenerator(spec, n_cells=400, reg_skip=5, seed=0)
    >>> adata = gen.to_anndata()
    >>> x_act, x_rep, y, t = gen.to_tensors("C")
"""
from .grn_spec import GRNSpec, GeneRule, HillParams
from .generator import SyntheticGRNGenerator, NoiseConfig, RootSignal

__all__ = [
    "GRNSpec",
    "GeneRule",
    "HillParams",
    "SyntheticGRNGenerator",
    "NoiseConfig",
    "RootSignal",
]
