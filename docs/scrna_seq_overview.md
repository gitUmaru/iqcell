# Single-Cell RNA Sequencing (scRNA-seq) and the Goals of IQCELL

## What is scRNA-seq?

Single-cell RNA sequencing (scRNA-seq) is a technique for measuring gene
expression — the set of RNA molecules (transcripts) actively produced by a
cell — one individual cell at a time.

Traditional ("bulk") RNA sequencing pools together thousands or millions of
cells and reports the *average* expression across all of them. That average
hides the fact that a tissue is a mixture of many different cell types and
cell states. scRNA-seq instead isolates each cell separately, captures its RNA,
and sequences it, producing a per-cell profile of which genes are on and how
strongly they are expressed.

### The typical workflow

1. **Cell isolation** — Tissue is dissociated into a suspension of single
   cells, which are then partitioned (e.g. into droplets or wells) so each cell
   is processed individually.
2. **Barcoding & capture** — Each cell's RNA is tagged with a unique molecular
   barcode so transcripts can be traced back to their cell of origin.
3. **Reverse transcription & amplification** — RNA is converted to cDNA and
   amplified.
4. **Sequencing** — The barcoded cDNA is sequenced on a high-throughput
   sequencer.
5. **Quantification** — Reads are mapped back to genes, producing a
   **gene × cell expression matrix**: rows are genes, columns are cells, and
   each entry is a count of how much of that gene was detected in that cell.

### Why it matters

- **Cellular heterogeneity** — Reveals distinct cell types and subpopulations
  within a tissue that bulk methods blur together.
- **Developmental trajectories** — By capturing cells at many stages of
  differentiation, scRNA-seq lets researchers reconstruct the "pseudotime"
  ordering of how cells progress from one state to another.
- **Gene regulation** — Provides the raw data needed to infer which genes
  regulate one another (gene regulatory networks, or GRNs).

### Common challenges

- **Sparsity / dropout** — Many genes read as zero even when expressed, simply
  because so little RNA is present per cell.
- **Noise** — Technical variation between cells can be large.
- **High dimensionality** — Tens of thousands of genes across thousands of
  cells require specialized statistical and computational methods.

## What This Project (IQCELL) Aims to Accomplish

IQCELL is an analysis platform developed in the Zandstra lab that uses
scRNA-seq data from developing cells to **infer, simulate, and study
executable logical gene regulatory networks (GRNs)**.

A gene regulatory network describes how genes turn each other on and off. By
learning these networks directly from single-cell data, IQCELL can model *why*
cells follow the developmental paths they do — and predict what happens when
that regulation is disrupted.

### Core objectives

- **Infer GRNs from scRNA-seq data** — Learn the regulatory relationships
  between genes directly from single-cell measurements of developing cells.
- **Build executable logical networks** — Represent gene interactions as
  logical rules that can actually be run/simulated, not just drawn as a diagram.
- **Simulate developmental trajectories** — Reproduce how gene expression
  evolves as cells differentiate over (pseudo)time.
- **Predict the effect of perturbations** — Model what happens to a
  developmental trajectory when specific genes are knocked out, over-expressed,
  or otherwise perturbed, mirroring genetic perturbation experiments.
- **Provide an integrative toolkit** — Offer modules for gene selection,
  binarization of expression, gene hierarchy construction, network inference,
  and logic-based simulation in one platform.

### Why this is useful

Dynamic simulations of the inferred networks have been shown to resemble
experimentally observed gene expression dynamics and to capture the effects of
genetic perturbation studies. This gives researchers a computational way to
form and test hypotheses about developmental biology — for example, predicting
which genes are the key drivers of a cell-fate decision — before committing to
costly wet-lab experiments.

## Reference

Heydari, T., Langley, M. A., Fisher, C. L., Aguilar-Hidalgo, D., Shukla, S.,
Yachie-Kinoshita, A., Hughes, M., McNagny, K. M., & Zandstra, P. W. (2022).
*IQCELL: A platform for predicting the effect of gene perturbations on
developmental trajectories using single-cell RNA-seq data.* PLoS Computational
Biology, 18(2), e1009907. https://doi.org/10.1371/journal.pcbi.1009907
