"""Tests for the synthetic scRNA-seq GRN data generator."""
import numpy as np
import pytest

from iqcell.simulation import (
    GRNSpec,
    HillParams,
    SyntheticGRNGenerator,
    NoiseConfig,
    RootSignal,
)


def make_spec():
    # A -> B -> C, and A -| C (feed-forward loop with repression)
    spec = GRNSpec(["A", "B", "C"])
    spec.set_rule("B", activators=["A"])
    spec.set_rule("C", activators=["B"], repressors=["A"])
    return spec


# ---- GRNSpec ----------------------------------------------------------------

def test_spec_validate_ok():
    make_spec().validate()  # should not raise


def test_spec_roots():
    spec = make_spec()
    assert spec.roots() == ["A"]


def test_spec_edges_and_adjacency():
    spec = make_spec()
    edges = set(spec.edges())
    assert ("A", "B", 1) in edges
    assert ("B", "C", 1) in edges
    assert ("A", "C", -1) in edges

    A = spec.adjacency()
    ia, ib, ic = spec.index_of("A"), spec.index_of("B"), spec.index_of("C")
    assert A[ia, ib] == 1
    assert A[ib, ic] == 1
    assert A[ia, ic] == -1


def test_spec_topological_order():
    spec = make_spec()
    order = spec.topological_order()
    assert order.index("A") < order.index("B") < order.index("C")


def test_spec_rejects_cycle():
    spec = GRNSpec(["A", "B"])
    spec.set_rule("A", activators=["B"])
    spec.set_rule("B", activators=["A"])
    with pytest.raises(ValueError, match="cycle"):
        spec.validate()


def test_spec_rejects_unknown_regulator():
    spec = GRNSpec(["A", "B"])
    spec.set_rule("B", activators=["Z"])
    with pytest.raises(ValueError, match="unknown regulator"):
        spec.validate()


def test_spec_rejects_self_regulation():
    spec = GRNSpec(["A"])
    spec.set_rule("A", activators=["A"])
    with pytest.raises(ValueError, match="Self-regulation"):
        spec.validate()


def test_spec_rejects_no_roots():
    spec = GRNSpec(["A", "B", "C"])
    spec.set_rule("A", activators=["B"])
    spec.set_rule("B", activators=["C"])
    spec.set_rule("C", activators=["A"])
    with pytest.raises(ValueError):
        spec.validate()


def test_hillparams_validation():
    with pytest.raises(ValueError):
        HillParams(K=0)
    with pytest.raises(ValueError):
        HillParams(n=-1)


# ---- Generator --------------------------------------------------------------

def test_reproducibility():
    spec = make_spec()
    g1 = SyntheticGRNGenerator(spec, n_cells=200, seed=42).simulate()
    g2 = SyntheticGRNGenerator(spec, n_cells=200, seed=42).simulate()
    np.testing.assert_array_equal(g1.expression, g2.expression)
    np.testing.assert_array_equal(g1.clean, g2.clean)


def test_shapes():
    spec = make_spec()
    gen = SyntheticGRNGenerator(spec, n_cells=300, seed=0).simulate()
    assert gen.clean.shape == (300, 3)
    assert gen.expression.shape == (300, 3)
    assert gen.pseudotime.shape == (300,)


def test_clean_expression_in_range():
    spec = make_spec()
    gen = SyntheticGRNGenerator(spec, n_cells=300, seed=0).simulate()
    # clean expression should stay within [0, 1] given default basal/vmax
    assert gen.clean.min() >= -1e-9
    assert gen.clean.max() <= 1.0 + 1e-9


def test_no_noise_clean_equals_expression():
    spec = make_spec()
    gen = SyntheticGRNGenerator(
        spec, n_cells=200, noise=NoiseConfig(biological=0, dropout=0), seed=1
    ).simulate()
    np.testing.assert_array_equal(gen.clean, gen.expression)


def test_dropout_zeros_values():
    spec = make_spec()
    gen = SyntheticGRNGenerator(
        spec,
        n_cells=500,
        noise=NoiseConfig(biological=0, dropout=0.5),
        seed=3,
    ).simulate()
    # a lot of nonzero clean values should be zeroed by dropout
    zeroed = (gen.clean > 0.1) & (gen.expression == 0.0)
    assert zeroed.sum() > 0


def test_activation_target_responds_to_regulator():
    # B is a pure activation target of A: where A is high (lagged), B should be high
    spec = make_spec()
    gen = SyntheticGRNGenerator(
        spec, n_cells=400, reg_skip=0, noise=NoiseConfig(biological=0, dropout=0),
        seed=5,
    ).simulate()
    a = gen.clean[:, spec.index_of("A")]
    b = gen.clean[:, spec.index_of("B")]
    # positive correlation between A and its downstream activation target B
    corr = np.corrcoef(a, b)[0, 1]
    assert corr > 0.5


def test_reg_skip_bounds():
    spec = make_spec()
    with pytest.raises(ValueError):
        SyntheticGRNGenerator(spec, n_cells=10, reg_skip=10)


# ---- Outputs ----------------------------------------------------------------

def test_to_anndata():
    spec = make_spec()
    gen = SyntheticGRNGenerator(spec, n_cells=250, seed=0)
    adata = gen.to_anndata(binarize=True)
    assert adata.X.shape == (250, 3)
    assert list(adata.var_names) == ["A", "B", "C"]
    assert "pseudotime" in adata.obs
    assert adata.raw is not None
    assert adata.raw.X.shape == (250, 3)
    # X is binary
    assert set(np.unique(adata.X)).issubset({0.0, 1.0})
    # raw is continuous (not all binary)
    assert not set(np.unique(adata.raw.X)).issubset({0.0, 1.0})
    # ground truth stored
    assert adata.uns["grn_genes"] == ["A", "B", "C"]
    assert adata.uns["grn_adjacency"].shape == (3, 3)


def test_to_anndata_matches_kmeans_interface():
    # KMeans binarizer reads data.raw.X; ensure it exists and is continuous
    spec = make_spec()
    adata = SyntheticGRNGenerator(spec, n_cells=100, seed=0).to_anndata()
    assert adata.raw.X is not None
    assert adata.raw.X.shape == adata.X.shape


def test_to_tensors():
    import torch

    spec = make_spec()
    gen = SyntheticGRNGenerator(spec, n_cells=200, seed=0)
    x_act, x_rep, y, t = gen.to_tensors("C", binarize=True)
    assert isinstance(x_act, torch.Tensor)
    assert x_act.shape == (200, 1)  # C has 1 activator (B)
    assert x_rep.shape == (200, 1)  # C has 1 repressor (A)
    assert y.shape == (200, 1)
    assert t.shape == (200, 1)
    # binary states
    assert set(torch.unique(y).tolist()).issubset({0.0, 1.0})


def test_to_tensors_root_gene_has_no_regulators():
    spec = make_spec()
    gen = SyntheticGRNGenerator(spec, n_cells=100, seed=0)
    x_act, x_rep, y, t = gen.to_tensors("A")
    assert x_act.shape == (100, 0)
    assert x_rep.shape == (100, 0)


def test_to_tensors_feeds_expression_dataset():
    # Output must be consumable by iqcell.utils.dataset.ExpressionData_And_Or
    from iqcell.utils.dataset import ExpressionData_And_Or

    spec = make_spec()
    gen = SyntheticGRNGenerator(spec, n_cells=120, seed=0)
    x_act, x_rep, y, t = gen.to_tensors("C")
    ds = ExpressionData_And_Or(
        x_act.numpy(), x_rep.numpy(), y.numpy().ravel(), t.numpy().ravel()
    )
    assert len(ds) == 120
    xa, xr, yy, tt = ds[0]
    assert xa.shape[0] == 1 and xr.shape[0] == 1


def test_ground_truth():
    spec = make_spec()
    gt = SyntheticGRNGenerator(spec, n_cells=50, seed=0).ground_truth()
    assert gt["genes"] == ["A", "B", "C"]
    assert ("A", "B", 1) in gt["edges"]
    assert gt["adjacency"].shape == (3, 3)


# ---- RootSignal (deterministic driver timing) ------------------------------

def test_root_signal_timing():
    # Two activators peak early, target is their AND; a late repressor shuts it off.
    spec = GRNSpec(["Act1", "Act2", "Rep", "Target"])
    spec.set_rule("Target", activators=["Act1", "Act2"], repressors=["Rep"])
    gen = SyntheticGRNGenerator(
        spec,
        n_cells=600,
        reg_skip=0,
        noise=NoiseConfig(biological=0, dropout=0),
        root_signals={
            "Act1": RootSignal(centers=[0.35], width=0.12),
            "Act2": RootSignal(centers=[0.4], width=0.12),
            "Rep": RootSignal(centers=[0.75], width=0.1),
        },
        seed=0,
    ).simulate()
    tgt = gen.clean[:, spec.index_of("Target")]
    t = gen.pseudotime
    early = tgt[(t > 0.3) & (t < 0.5)].mean()   # activators on, repressor off
    late = tgt[(t > 0.7) & (t < 0.85)].mean()   # repressor on
    assert early > 0.4, f"target should be ON early, got {early}"
    assert late < 0.1, f"target should be OFF late (repressed), got {late}"


def test_root_signal_and_logic_needs_both_activators():
    # If only one activator is present, the AND target should stay low.
    spec = GRNSpec(["Act1", "Act2", "Target"])
    spec.set_rule("Target", activators=["Act1", "Act2"])
    gen = SyntheticGRNGenerator(
        spec,
        n_cells=400,
        reg_skip=0,
        noise=NoiseConfig(biological=0, dropout=0),
        root_signals={
            "Act1": RootSignal(centers=[0.3], width=0.08),
            "Act2": RootSignal(centers=[0.7], width=0.08),  # non-overlapping
        },
        seed=0,
    ).simulate()
    tgt = gen.clean[:, spec.index_of("Target")]
    # activators never co-occur -> target never strongly on
    assert tgt.max() < 0.5


def test_root_signal_oscillator_multiple_pulses():
    spec = GRNSpec(["Osc"])
    gen = SyntheticGRNGenerator(
        spec,
        n_cells=600,
        noise=NoiseConfig(biological=0, dropout=0),
        root_signals={"Osc": RootSignal(centers=[0.2, 0.5, 0.8], width=0.05)},
        seed=0,
    ).simulate()
    osc = gen.clean[:, spec.index_of("Osc")]
    # count peaks: threshold crossings upward
    high = osc > 0.5
    rises = np.sum((~high[:-1]) & (high[1:]))
    assert rises == 3, f"expected 3 pulses, found {rises}"


def test_root_signal_rejects_non_root():
    spec = make_spec()  # B is regulated by A
    with pytest.raises(ValueError, match="non-root"):
        SyntheticGRNGenerator(spec, n_cells=100, root_signals={"B": RootSignal([0.5])})


def test_root_signal_rejects_unknown_gene():
    spec = make_spec()
    with pytest.raises(KeyError):
        SyntheticGRNGenerator(spec, n_cells=100, root_signals={"Z": RootSignal([0.5])})
