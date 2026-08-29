# MEMORY — IQCELL fake scRNA-seq GRN data generator

Working log for building a synthetic/fake data generator for scRNA-seq Gene
Regulatory Network (GRN) inference in the IQCELL package.

## Repo overview

IQCELL infers, simulates, and studies executable logical GRNs from single-cell
RNA-seq data of developing cells (Heydari et al. 2022, PLoS Comput Biol).

Package layout (`iqcell/`):
- `binarization/` — discretize continuous expression to {0,1}
  - `base_class.py` — `Binarizer(ABC)`: `discretize(data)`, `_fit(data)`, `threshold`, `_trained`
  - `kmeans.py` — `KMeans(Binarizer)`: per-gene 2-cluster threshold = mean of centroids
  - `mean.py` — `Mean(Binarizer)`: per-gene threshold = column mean
- `gene_hierarchy/` — order genes along pseudotime by transition points
  - `base_class.py` — `GeneHierarchy(ABC)`: props `pseudo_time`, `binarized_data`,
    `transition_points`; abstract `calculate_hierarchy(adata)`, `_compute_transition_points()`
  - `hierarchy.py` — `IqCellGeneHierarchy`: smooths binarized data with
    `uniform_filter1d` over pseudotime, transition point = first index where
    smoothed signal >= 0.5
- `utils/`
  - `dataset.py` — torch Datasets:
    - `AnnDataDataset(anndata)` -> returns rows of `adata.X`
    - `ExpressionData_And_Or(x_act, x_rep, y, t)` -> (activators, repressors, target, time)
  - `sampler.py` — `RandomSampler`, `IqSampler` (torch Samplers)
- `logic_engine/`, `preprocesing/`, `interaction_network/` — EMPTY stubs (`__init__.py` only)

All `__init__.py` are empty (modules not wired for `import iqcell` yet).
NOTE: internal imports use `from base_class import ...` (not relative) — will
break as a package import; run from within the module dir currently.

## Data conventions (observed)

- Expression matrix: AnnData `adata.X` shape `(n_cells/obs, n_genes/var)`.
  `adata.raw.X` holds pre-binarization continuous values (used by KMeans `_fit`).
- Cells ordered along pseudotime (rows are time-ordered); `pseudo_time` is a
  per-cell 1D array. `examples/data/expression.csv` + `pseudotime.csv` are EMPTY (0 bytes).
- Binarized values are {0,1} floats.
- Gene names in `adata.var_names`.

## SINN model I/O (from examples/1. notebook_SINN_working_verson.ipynb)

The And/Or logic model (`And_Or_Layer`) consumes, per target gene:
- `x_act`: (n_time, n_activators) binary activator expression
- `x_rep`: (n_time, n_repressors) binary repressor expression
- `y`: (n_time,) binary target expression
- `t`: (n_time,) time / pseudotime

Existing synthetic generator in the notebook: `create_synthetic_data(A, R, GRN_A,
GRN_R, T, reg_skip, ...)`:
- `A`, `R`: lists of (center, duration) -> square-pulse expression per regulator
- logic: target = prod(activators[GRN_A]) * prod(1 - repressors[GRN_R])  (AND of
  activators AND NOT of repressors)
- noise: `noise_level_biological` (random dropout of regulator ON states),
  `noise_level_dropout` (random dropout of target + regulators)
- `reg_skip`: time lag between regulators and target (np.roll)
- returns torch tensors (x_act, x_rep, y)

This is the reference logic for the fake data generator. It is notebook-only /
pulse-based; there is no reusable package module yet.

## Goal

Build a reusable fake data generator that produces synthetic scRNA-seq-like data
with a KNOWN ground-truth GRN, so inference/binarization/hierarchy modules can be
validated against truth.

## Design decisions (confirmed with user)
- Output: BOTH AnnData object (X binary + raw.X continuous + var_names + obs
  pseudotime) AND torch tensors (x_act, x_rep, y, t) matching
  `ExpressionData_And_Or`.
- Dynamics: continuous Hill/sigmoid activation driven by regulator states ->
  continuous expression -> optional binarization.
- Noise: dropout (zero-inflation) + biological/stochastic noise + regulatory
  time lag (reg_skip). NO library-size/count noise for now.
- Topology: EXPLICIT spec only. User supplies per-gene activators & repressors
  (the known ground-truth GRN). No random topology generation.

## Planned module: iqcell/simulation/
- `iqcell/simulation/__init__.py` — exports generator
- `iqcell/simulation/grn_spec.py` — `GRNSpec`: genes, per-target activators/
  repressors, Hill params (K, n), basal/max rates, sign. Validation.
- `iqcell/simulation/generator.py` — `SyntheticGRNGenerator`:
    - resolves regulator ordering (roots -> downstream) from explicit spec
    - simulates continuous expression along pseudotime via Hill functions
      combined multiplicatively (AND activators * NOT repressors), with
      regulatory time lag
    - applies biological noise + dropout
    - `.to_anndata()` -> AnnData (X binary, raw.X continuous, var_names, obs['pseudotime'])
    - `.to_tensors(target)` -> (x_act, x_rep, y, t) for a chosen target gene
- tests in `tests/`

## Progress log
- [done] Reviewed repo structure, modules, notebook synthetic generator.
- [done] Created MEMORY.md.
- [done] Confirmed design decisions with user.
- [done] Implemented `iqcell/simulation/` package:
    - `grn_spec.py` — `HillParams`, `GeneRule`, `GRNSpec` (explicit topology,
      DAG validation via topological sort, signed adjacency, edge list).
    - `generator.py` — `NoiseConfig`, `SyntheticGRNGenerator` (Hill dynamics,
      reg_skip lag, biological noise + expression-scaled dropout,
      `.to_anndata()`, `.to_tensors(target)`, `.ground_truth()`).
    - `__init__.py` — public exports.
- [done] Tests `tests/test_simulation.py` — 22 passing.
- [done] Example `examples/generate_synthetic.py` — hematopoiesis-style GRN;
  writes AnnData + tensors + populates the empty examples/data/*.csv.
- [done] Integration-verified synthetic AnnData through real
  `iqcell.binarization.Mean` (X binary, raw.X continuous).
- [done] Notebook `examples/synthetic_grn_demo.ipynb` (22 cells) — defines the
  GRN, draws the network graph (activation/repression, root driver), plots
  clean vs noisy expression, expression + binarized heatmaps, and shows
  AnnData + tensor outputs. Executed end-to-end (0 errors, 4 figures).
  Built with nbformat; matplotlib/networkx/nbformat/ipykernel/nbconvert added
  to `.venv/`.
- [done] Added `RootSignal` (deterministic Gaussian-pulse root timing) +
  `root_signals=` arg to `SyntheticGRNGenerator` so driver genes can be timed
  explicitly (early activators, late repressor, multi-pulse oscillator). Only
  valid for root genes; validated. Exported from `iqcell.simulation`.
  5 new tests (27 total passing).
- [done] Reworked the notebook per user request:
    - Circular graph layout (deterministic ring; legend centered).
    - Biologically motivated erythroid GRN: GATA1 AND TAL1 -> KLF1 (both
      required, multiplicative Hill AND), SPI1 -| KLF1 as a LATE repressor
      (classic GATA1/PU.1 antagonism), CCNB1 isolated cell-cycle oscillator
      (3 pulses, no edges).
    - Verified logic in rendered data: KLF1 OFF early (0.0), ON mid-window
      (~0.86 when both activators high), OFF late (0.0 under SPI1); CCNB1 = 3
      pulses. Re-executed, 0 errors, 4 figures.

## Environment / running notes
- Dev deps installed into `.venv/` (numpy, anndata, pandas, torch, pytest).
  sklearn NOT installed — needed only by the KMeans binarizer.
- Run tests: `. .venv/bin/activate && python -m pytest tests/test_simulation.py -q`
- Run example: `python examples/generate_synthetic.py` (adds repo root to sys.path).

## Known pre-existing repo issues (NOT introduced here)
- Internal modules use non-relative imports (`from base_class import ...`), so
  `import iqcell.binarization.kmeans` fails unless the module dir is on sys.path.
  The new `iqcell/simulation/` uses relative imports and loads as `iqcell.simulation`.
- `iqcell/binarization/kmeans.py` defines class `KMeans` that shadows the imported
  `sklearn.cluster.KMeans`, so `KMeans(n_clusters=2)` in `_fit` is a latent bug.

## Possible next steps
- Per-gene noise/kinetic overrides; count-based (NB/Poisson) library-size noise.
- Scoring helper comparing inferred edges vs `ground_truth()` (AUROC/AUPR).
- Fix packaging import style so the whole pipeline imports as a package.
