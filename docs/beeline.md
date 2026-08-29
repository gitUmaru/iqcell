# BEELINE integration

`iqcell.beeline` wraps the real [Murali-group/BEELINE](https://github.com/Murali-group/Beeline)
benchmarking pipeline so you can score GRN-inference algorithms on IQCELL's
synthetic ground-truth networks.

## Overview

The bridge does four things:

1. Exports a `SyntheticGRNGenerator` into BEELINE's expected input layout.
2. Generates a BEELINE-compatible YAML config.
3. Drives a cloned BEELINE repo's `BLRunner.py` / `BLEvaluator.py` via subprocess.
4. Reads the ranked-edge outputs and evaluation metrics back into Python.

A pure-python scorer (`score_ranking`) computes AUPRC/AUROC **without** a
BEELINE or Docker install, so you can get results immediately and only set up
the full pipeline when you need the real inference algorithms.

### Public API at a glance

All names below are importable from `iqcell.beeline`:

| Name | Kind | Purpose |
|------|------|---------|
| `export_beeline_inputs` | function | Write `ExpressionData.csv`, `PseudoTime.csv`, and `refNetwork.csv` |
| `write_config` | function | Write the BEELINE-compatible YAML config |
| `BeelineRunner` | class | Drive a cloned BEELINE repo via subprocess |
| `BeelineStatus` | dataclass | Result of `BeelineRunner.check_available()` |
| `BeelineNotAvailableError` | exception | Raised when BEELINE/Docker is not usable for a run |
| `read_ranked_edges` | function | Read a `rankedEdges.csv` into a DataFrame |
| `parse_evaluation` | function | Collect AUPRC/AUROC/EPR from evaluation CSVs |
| `score_ranking` | function | Pure-python AUPRC/AUROC scorer (no BEELINE needed) |

## TL;DR

```bash
# Fast path — export + pure-python scoring, no external deps:
python examples/beeline_benchmark.py

# Full path — clone + initialize BEELINE (needs Docker), then run for real:
scripts/bootstrap_beeline.sh
python examples/beeline_benchmark.py --beeline-repo ./.beeline
```

## What BEELINE expects

BEELINE reads inputs from `input_dir/<dataset_id>/<run_id>/` and writes outputs
to `output_dir/<dataset_id>/<run_id>/<algorithm_id>/rankedEdges.csv`.

The pipeline uses these files:

| File | Location | Format |
|------|----------|--------|
| `ExpressionData.csv` | `<dataset_id>/<run_id>/` | genes × cells; gene names as row index, cell ids as columns |
| `PseudoTime.csv` | `<dataset_id>/<run_id>/` | index = cell ids, one `PseudoTime` column |
| `refNetwork.csv` | `<dataset_id>/` | ground truth: `Gene1,Gene2,Type` (`Type` is `+`/`-`) |
| `config.yaml` | anywhere | pipeline configuration (see below) |

`export_beeline_inputs` writes all three data files; `write_config` writes the
YAML.

## Step 1 — export synthetic data

Export the generator's data and write a config file:

```python
from iqcell.simulation import GRNSpec, SyntheticGRNGenerator, NoiseConfig
from iqcell.beeline import export_beeline_inputs, write_config

spec = GRNSpec(["Sox2", "Gata2", "Gata1", "Pu1", "Klf1"])
spec.set_rule("Gata2", activators=["Sox2"])
spec.set_rule("Gata1", activators=["Gata2"], repressors=["Pu1"])
spec.set_rule("Pu1", activators=["Gata2"])
spec.set_rule("Klf1", activators=["Gata1"], repressors=["Pu1"])

gen = SyntheticGRNGenerator(spec, n_cells=500, noise=NoiseConfig(dropout=0.2), seed=0)

paths = export_beeline_inputs(gen, "work/inputs", dataset_id="synthetic", run_id="run1")
write_config("work/config.yaml", input_dir="work/inputs", output_dir="work/outputs")
```

Notes on `export_beeline_inputs`:

- It uses the noisy `gen.expression` by default. Pass `use_clean=True` to
  export the noise-free signal instead.
- It calls `simulate()` for you if the generator has not been run yet.

### Choosing algorithms

By default `write_config` enables three algorithms:

- **PIDC** (`grnbeeline/pidc:base`)
- **GENIE3** (`grnbeeline/arboreto:base`)
- **PEARSON** (`local`, no Docker)

Override the defaults with the `algorithms=` argument:

```python
write_config(
    "work/config.yaml",
    input_dir="work/inputs",
    output_dir="work/outputs",
    algorithms=[
        {"algorithm_id": "PIDC", "image": "grnbeeline/pidc:base", "should_run": True, "params": {}},
        {"algorithm_id": "PPCOR", "image": "grnbeeline/ppcor:base", "should_run": True, "params": {"pVal": 0.01}},
    ],
)
```

## Step 2 — set up BEELINE (one time)

BEELINE runs each algorithm in a Docker container, so **Docker must be
installed and running**.

1. Clone the repo and pull the algorithm images with the bootstrap script:

   ```bash
   scripts/bootstrap_beeline.sh            # clones into ./.beeline
   scripts/bootstrap_beeline.sh /opt/beeline   # or a custom location
   ```

2. Set up BEELINE's conda environment, which its Python entry points also
   need. Following the upstream README:

   ```bash
   # inside the checkout
   bash utils/setupAnacondaVENV.sh
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate BEELINE
   ```

## Step 3 — run inference + evaluation

Run the algorithms and read back the metrics:

```python
from iqcell.beeline import BeelineRunner, parse_evaluation

runner = BeelineRunner(beeline_repo="./.beeline")  # python_exe="python" by default

status = runner.check_available()   # never raises, never invokes Docker
print(status)  # BeelineStatus(available=..., has_blrunner=..., has_docker=..., message=...)

if status.available:
    runner.run("work/config.yaml")                      # python BLRunner.py -c ...
    runner.evaluate("work/config.yaml", auc=True, epr=True)  # python BLEvaluator.py -c ... -a -e
    metrics = parse_evaluation("work/outputs")
    print(metrics)
```

If BEELINE's entry points need a specific interpreter (e.g. the activated
conda env is not the default `python`), pass `python_exe="/path/to/python"` to
`BeelineRunner`.

## Step 4 — scoring without BEELINE

You do not need BEELINE or Docker to score a ranked edge list against the known
ground truth. `score_ranking` computes AUPRC and AUROC over all directed gene
pairs (self-loops excluded; edge sign is ignored for presence):

```python
from iqcell.beeline import read_ranked_edges, score_ranking

ground_truth = list(spec.edges())                 # (regulator, target, sign)
ranked = read_ranked_edges("work/outputs/synthetic/run1/PIDC/rankedEdges.csv")
print(score_ranking(ranked, ground_truth))        # {'auprc': ..., 'auroc': ...}
```

`score_ranking` also accepts a plain iterable of `(Gene1, Gene2, weight)`
tuples, and uses scikit-learn if available, otherwise a pure-python fallback.

## End-to-end example

`examples/beeline_benchmark.py` runs the whole loop:

- Without `--beeline-repo`, it exports inputs and demonstrates the fallback
  scorer.
- With a bootstrapped checkout, it runs the real Docker-backed pipeline:

```bash
python examples/beeline_benchmark.py --beeline-repo ./.beeline --n-cells 500
```

## Troubleshooting

- **`check_available()` reports `has_docker=False`** — install Docker and ensure
  `docker` is on `PATH`.
- **`available=False` with `has_blrunner=False`** — the path is not a BEELINE
  checkout; re-run `scripts/bootstrap_beeline.sh` or point at the right repo.
- **`BeelineNotAvailableError` on `run()`/`evaluate()`** — same causes as above;
  the message includes setup guidance.
- **Nonzero exit from BEELINE** — the raised error includes BEELINE's captured
  stderr. Common causes: images not pulled (`utils/initialize.sh`), the
  `BEELINE` conda env not active, or the Docker daemon not running.
