"""BEELINE integration for IQCELL.

Wrap the real Murali-group/Beeline benchmarking pipeline: export synthetic GRN
data into BEELINE's input layout, generate a config, drive a cloned Beeline
repo via subprocess, and read/score the ranked-edge results. A pure-python
scorer (:func:`score_ranking`) lets you evaluate rankings without a working
BEELINE/Docker install.

Example:
    >>> from iqcell.simulation import GRNSpec, SyntheticGRNGenerator
    >>> from iqcell.beeline import export_beeline_inputs, write_config
    >>> spec = GRNSpec(["A", "B", "C"])
    >>> spec.set_rule("B", activators=["A"])
    >>> spec.set_rule("C", activators=["B"], repressors=["A"])
    >>> gen = SyntheticGRNGenerator(spec, n_cells=200, seed=0)
    >>> paths = export_beeline_inputs(gen, "inputs")
    >>> write_config("config.yaml", input_dir="inputs", output_dir="outputs")
    'config.yaml'
"""
from .export import export_beeline_inputs, write_config
from .runner import BeelineRunner, BeelineNotAvailableError, BeelineStatus
from .results import read_ranked_edges, parse_evaluation, score_ranking

__all__ = [
    "export_beeline_inputs",
    "write_config",
    "BeelineRunner",
    "BeelineNotAvailableError",
    "BeelineStatus",
    "read_ranked_edges",
    "parse_evaluation",
    "score_ranking",
]
