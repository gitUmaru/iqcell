"""Tests for the BEELINE integration (iqcell.beeline)."""
import os
import random

import pandas as pd
import pytest

from iqcell.simulation import GRNSpec, SyntheticGRNGenerator
from iqcell.beeline import (
    export_beeline_inputs,
    write_config,
    BeelineRunner,
    read_ranked_edges,
    parse_evaluation,
    score_ranking,
)


def make_generator(n_cells=120, seed=0):
    # A -> B -> C, and A -| C (feed-forward loop with repression)
    spec = GRNSpec(["A", "B", "C"])
    spec.set_rule("B", activators=["A"])
    spec.set_rule("C", activators=["B"], repressors=["A"])
    return SyntheticGRNGenerator(spec, n_cells=n_cells, seed=seed)


# --- (a) export writes the 3 files with correct shapes ---------------------


def test_export_writes_three_files_with_correct_layout(tmp_path):
    gen = make_generator()
    out_dir = str(tmp_path / "inputs")
    paths = export_beeline_inputs(gen, out_dir, dataset_id="synthetic", run_id="run1")

    for key in ("expression", "pseudotime", "network", "run_dir"):
        assert key in paths
    assert os.path.isfile(paths["expression"])
    assert os.path.isfile(paths["pseudotime"])
    assert os.path.isfile(paths["network"])

    # Files land in the expected BEELINE subtree.
    assert paths["expression"].endswith(
        os.path.join("synthetic", "run1", "ExpressionData.csv")
    )
    assert paths["network"].endswith(os.path.join("synthetic", "refNetwork.csv"))

    n_cells = len(gen.pseudotime)
    genes = list(gen.spec.genes)

    # ExpressionData.csv is genes x cells: gene-name index, cell-id columns.
    expr = pd.read_csv(paths["expression"], index_col=0)
    assert expr.shape == (len(genes), n_cells)
    assert list(expr.index) == genes
    assert list(expr.columns) == [f"cell_{i}" for i in range(n_cells)]

    # PseudoTime.csv indexed by cell id.
    ptime = pd.read_csv(paths["pseudotime"], index_col=0)
    assert ptime.shape[0] == n_cells
    assert list(ptime.index) == [f"cell_{i}" for i in range(n_cells)]

    # refNetwork Type maps signs to +/-.
    net = pd.read_csv(paths["network"])
    assert list(net.columns) == ["Gene1", "Gene2", "Type"]
    assert set(net["Type"]).issubset({"+", "-"})
    # A -| C should be represented as a repression edge.
    rep = net[(net["Gene1"] == "A") & (net["Gene2"] == "C")]
    assert len(rep) == 1 and rep["Type"].iloc[0] == "-"
    act = net[(net["Gene1"] == "A") & (net["Gene2"] == "B")]
    assert len(act) == 1 and act["Type"].iloc[0] == "+"


def test_export_does_not_resimulate_when_already_simulated(tmp_path):
    gen = make_generator()
    gen.simulate()
    first = gen.expression.copy()
    export_beeline_inputs(gen, str(tmp_path / "inputs"))
    # Same object, unchanged values (no re-simulation).
    assert gen.expression is not None
    assert (gen.expression == first).all()


def test_export_use_clean(tmp_path):
    gen = make_generator()
    paths = export_beeline_inputs(gen, str(tmp_path / "inputs"), use_clean=True)
    expr = pd.read_csv(paths["expression"], index_col=0)
    # clean matrix is genes x cells too.
    assert expr.shape == (gen.spec.n_genes, len(gen.pseudotime))


# --- (b) write_config round-trips key fields --------------------------------


def test_write_config_roundtrip(tmp_path):
    config_path = str(tmp_path / "config.yaml")
    write_config(
        config_path,
        input_dir="inputs",
        output_dir="outputs",
        dataset_id="synthetic",
        run_id="run1",
    )
    assert os.path.isfile(config_path)
    text = open(config_path, encoding="utf-8").read()

    try:
        import yaml

        cfg = yaml.safe_load(text)
        assert cfg["input_settings"]["input_dir"] == "inputs"
        assert cfg["output_settings"]["output_dir"] == "outputs"
        ds = cfg["input_settings"]["datasets"][0]
        assert ds["dataset_id"] == "synthetic"
        assert ds["groundTruthNetwork"] == "refNetwork.csv"
        assert ds["runs"][0]["run_id"] == "run1"
        algo_ids = {a["algorithm_id"] for a in cfg["input_settings"]["algorithms"]}
        assert {"PIDC", "GENIE3", "PEARSON"}.issubset(algo_ids)
        images = {a["algorithm_id"]: a["image"] for a in cfg["input_settings"]["algorithms"]}
        assert images["PEARSON"] == "local"
    except ImportError:
        for needle in (
            "input_dir",
            "inputs",
            "output_dir",
            "outputs",
            "synthetic",
            "run1",
            "PIDC",
            "GENIE3",
            "PEARSON",
            "local",
        ):
            assert needle in text


def test_write_config_custom_algorithms(tmp_path):
    config_path = str(tmp_path / "config.yaml")
    write_config(
        config_path,
        input_dir="inputs",
        output_dir="outputs",
        algorithms=[
            {"algorithm_id": "PEARSON", "image": "local", "should_run": True, "params": {}}
        ],
    )
    text = open(config_path, encoding="utf-8").read()
    assert "PEARSON" in text
    assert "GENIE3" not in text


# --- (c) BeelineRunner.check_available is graceful ---------------------------


def test_check_available_nonexistent_repo_no_exception():
    runner = BeelineRunner("/nonexistent/path/to/beeline")
    status = runner.check_available()  # must not raise
    assert status.available is False
    assert status.has_blrunner is False
    assert isinstance(status.message, str) and status.message


def test_run_and_evaluate_raise_when_unavailable():
    from iqcell.beeline import BeelineNotAvailableError

    runner = BeelineRunner("/nonexistent/path/to/beeline")
    with pytest.raises(BeelineNotAvailableError):
        runner.run("config.yaml")
    with pytest.raises(BeelineNotAvailableError):
        runner.evaluate("config.yaml")


# --- (d) score_ranking: perfect vs random -----------------------------------


def _all_directed_pairs(genes):
    return [(a, b) for a in genes for b in genes if a != b]


def test_score_ranking_perfect_and_random():
    genes = ["A", "B", "C", "D"]
    truth = [("A", "B", 1), ("B", "C", 1), ("A", "C", -1)]
    truth_pairs = {(g1, g2) for g1, g2, _ in truth}
    all_pairs = _all_directed_pairs(genes)

    # Perfect ranking: true edges get high weight, non-edges get low weight.
    perfect = [
        (g1, g2, 1.0 if (g1, g2) in truth_pairs else 0.0) for g1, g2 in all_pairs
    ]
    scores = score_ranking(perfect, truth)
    assert scores["auprc"] == pytest.approx(1.0, abs=1e-9)
    assert scores["auroc"] == pytest.approx(1.0, abs=1e-9)

    # Random ranking: AUROC should be near chance (0.5).
    rng = random.Random(42)
    random_edges = [(g1, g2, rng.random()) for g1, g2 in all_pairs]
    rscores = score_ranking(random_edges, truth)
    assert 0.0 <= rscores["auprc"] <= 1.0
    assert 0.15 <= rscores["auroc"] <= 0.85


def test_score_ranking_accepts_dataframe():
    truth = [("A", "B", 1), ("B", "C", 1)]
    df = pd.DataFrame(
        {
            "Gene1": ["A", "B", "A", "C"],
            "Gene2": ["B", "C", "C", "A"],
            "EdgeWeight": [0.9, 0.8, 0.1, 0.05],
        }
    )
    scores = score_ranking(df, truth)
    assert scores["auprc"] == pytest.approx(1.0, abs=1e-9)
    assert scores["auroc"] == pytest.approx(1.0, abs=1e-9)


# --- results readers ---------------------------------------------------------


def test_read_ranked_edges(tmp_path):
    p = tmp_path / "rankedEdges.csv"
    p.write_text("Gene1,Gene2,EdgeWeight\nA,B,0.9\nB,C,0.5\n")
    df = read_ranked_edges(str(p))
    assert list(df.columns) == ["Gene1", "Gene2", "EdgeWeight"]
    assert len(df) == 2


def test_parse_evaluation_empty(tmp_path):
    assert parse_evaluation(str(tmp_path)) == {}


def test_parse_evaluation_reads_metric(tmp_path):
    out = tmp_path / "synthetic" / "run1"
    out.mkdir(parents=True)
    (out / "AUPRC.csv").write_text(",PIDC\nsynthetic,0.75\n")
    metrics = parse_evaluation(str(tmp_path))
    assert "auprc" in metrics
    assert metrics["auprc"] == pytest.approx(0.75)
