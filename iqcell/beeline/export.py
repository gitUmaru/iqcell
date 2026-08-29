"""Export iqcell synthetic GRN data into BEELINE's expected input layout.

BEELINE (Murali-group/Beeline) expects, per run, a directory
``input_dir/dataset_id/run_id/`` containing:

- ``ExpressionData.csv`` — genes x cells (gene names as the row index, cell ids
  as columns).
- ``PseudoTime.csv`` — indexed by cell id, with a pseudotime column.

and a dataset-level ground-truth network CSV (``refNetwork.csv`` in BoolODE's
convention, ``GroundTruthNetwork.csv`` by BEELINE default) at
``input_dir/dataset_id/`` with columns ``Gene1,Gene2,Type`` where ``Type`` is
``+`` (activation) or ``-`` (repression).

These helpers turn a :class:`~iqcell.simulation.SyntheticGRNGenerator` into
those files so the synthetic ground-truth network can be benchmarked with the
real BEELINE pipeline.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd


def _ensure_simulated(generator) -> None:
    """Run ``generator.simulate()`` only if it has not produced expression yet.

    Avoids mutating a generator whose expression is already populated.
    """
    if getattr(generator, "expression", None) is None:
        generator.simulate()


def _cell_ids(generator) -> List[str]:
    n_cells = len(generator.pseudotime)
    return [f"cell_{i}" for i in range(n_cells)]


def export_beeline_inputs(
    generator,
    out_dir: str,
    dataset_id: str = "synthetic",
    run_id: str = "run1",
    use_clean: bool = False,
    network_filename: str = "refNetwork.csv",
) -> Dict[str, str]:
    """Write BEELINE-compatible inputs for a synthetic GRN generator.

    Parameters
    ----------
    generator:
        A :class:`~iqcell.simulation.SyntheticGRNGenerator`. Simulated in place
        only if it has not been simulated yet.
    out_dir:
        BEELINE ``input_dir``. The dataset/run subtree is created underneath it.
    dataset_id, run_id:
        Directory names for the dataset group and the individual run.
    use_clean:
        Export the noise-free ``generator.clean`` matrix instead of the noisy
        ``generator.expression`` (default is the realistic noisy matrix).
    network_filename:
        Filename for the ground-truth network CSV, written at
        ``out_dir/dataset_id/``.

    Returns
    -------
    dict
        Mapping with keys ``expression``, ``pseudotime``, ``network`` and
        ``run_dir`` pointing at the written paths/directory.
    """
    _ensure_simulated(generator)

    matrix = generator.clean if use_clean else generator.expression
    if matrix is None:  # pragma: no cover - simulate() guarantees non-None
        raise RuntimeError("Generator produced no expression matrix.")

    genes: List[str] = list(generator.spec.genes)
    cells = _cell_ids(generator)

    dataset_dir = os.path.join(out_dir, dataset_id)
    run_dir = os.path.join(dataset_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ExpressionData.csv: genes x cells. Internal matrix is cells x genes.
    expr_df = pd.DataFrame(matrix.T, index=genes, columns=cells)
    expr_path = os.path.join(run_dir, "ExpressionData.csv")
    expr_df.to_csv(expr_path)

    # PseudoTime.csv: indexed by cell id, one pseudotime column.
    ptime_df = pd.DataFrame(
        {"PseudoTime": generator.pseudotime}, index=cells
    )
    ptime_path = os.path.join(run_dir, "PseudoTime.csv")
    ptime_df.to_csv(ptime_path)

    # refNetwork.csv: Gene1,Gene2,Type (+/-) at the dataset level.
    rows = [
        {"Gene1": reg, "Gene2": tgt, "Type": "+" if sign > 0 else "-"}
        for reg, tgt, sign in generator.spec.edges()
    ]
    net_df = pd.DataFrame(rows, columns=["Gene1", "Gene2", "Type"])
    net_path = os.path.join(dataset_dir, network_filename)
    net_df.to_csv(net_path, index=False)

    return {
        "expression": expr_path,
        "pseudotime": ptime_path,
        "network": net_path,
        "run_dir": run_dir,
    }


def _default_algorithms() -> List[Dict[str, object]]:
    return [
        {"algorithm_id": "PIDC", "image": "grnbeeline/pidc:base", "should_run": True, "params": {}},
        {"algorithm_id": "GENIE3", "image": "grnbeeline/arboreto:base", "should_run": True, "params": {}},
        {"algorithm_id": "PEARSON", "image": "local", "should_run": True, "params": {}},
    ]


def _emit_yaml(config: dict) -> str:
    """Hand-emit a minimal, valid YAML config so pyyaml is not required."""
    inp = config["input_settings"]
    out = config["output_settings"]
    lines: List[str] = ["input_settings:"]
    lines.append(f"    input_dir: \"{inp['input_dir']}\"")
    lines.append("    datasets:")
    for ds in inp["datasets"]:
        lines.append(f"        - dataset_id: \"{ds['dataset_id']}\"")
        lines.append(f"          groundTruthNetwork: \"{ds['groundTruthNetwork']}\"")
        lines.append("          runs:")
        for run in ds["runs"]:
            lines.append(f"            - run_id: \"{run['run_id']}\"")
    lines.append("    algorithms:")
    for algo in inp["algorithms"]:
        lines.append(f"        - algorithm_id: \"{algo['algorithm_id']}\"")
        lines.append(f"          image: \"{algo['image']}\"")
        lines.append(f"          should_run: {'True' if algo['should_run'] else 'False'}")
        params = algo.get("params") or {}
        if params:
            lines.append("          params:")
            for k, v in params.items():
                lines.append(f"              {k}: {v}")
        else:
            lines.append("          params: {}")
    lines.append("output_settings:")
    lines.append(f"    output_dir: \"{out['output_dir']}\"")
    return "\n".join(lines) + "\n"


def write_config(
    config_path: str,
    input_dir: str,
    output_dir: str,
    dataset_id: str = "synthetic",
    run_id: str = "run1",
    network_filename: str = "refNetwork.csv",
    algorithms: Optional[List[Dict[str, object]]] = None,
) -> str:
    """Write a BEELINE-compatible YAML config file.

    Uses pyyaml when importable; otherwise hand-emits valid YAML so pyyaml is
    not a hard dependency. Returns the path written.
    """
    if algorithms is None:
        algorithms = _default_algorithms()

    config = {
        "input_settings": {
            "input_dir": input_dir,
            "datasets": [
                {
                    "dataset_id": dataset_id,
                    "groundTruthNetwork": network_filename,
                    "runs": [{"run_id": run_id}],
                }
            ],
            "algorithms": algorithms,
        },
        "output_settings": {"output_dir": output_dir},
    }

    parent = os.path.dirname(os.path.abspath(config_path))
    os.makedirs(parent, exist_ok=True)

    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    except ImportError:
        text = _emit_yaml(config)

    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return config_path
