"""Synthetic scRNA-seq GRN data generator.

Given an explicit ground-truth :class:`GRNSpec`, simulate continuous
expression dynamics along a pseudotime axis using Hill-function regulation,
then add scRNA-seq-like noise (biological/stochastic noise + dropout) and a
regulatory time lag. Emits either an ``AnnData`` object (compatible with the
iqcell binarization / gene-hierarchy pipeline) or torch tensors (compatible
with ``iqcell.utils.dataset.ExpressionData_And_Or`` and the SINN And/Or model).

Model
-----
Cells are ordered along a pseudotime axis ``t in [0, 1]``. Root genes (no
regulators) are generated as smooth "pulse" driver signals. Downstream genes
are simulated in topological order: at each pseudotime point a target's
expression is

    drive(t) = prod_a act(x_a[t - lag]) * prod_r rep(x_r[t - lag])
    x_tgt(t) = basal + (vmax - basal) * drive(t)

where ``act`` / ``rep`` are Hill terms, and ``lag`` (``reg_skip`` cells)
introduces a causal delay between regulators and target.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grn_spec import GRNSpec, GeneRule


@dataclass
class RootSignal:
    """Deterministic pulse specification for a root/driver gene.

    Expression along pseudotime is a sum of Gaussian pulses, clipped to [0, 1].
    Use this to control *when* a driver gene is expressed (e.g. activators early,
    a repressor late, or an oscillating gene with several pulses).

    Attributes:
        centers: pseudotime location(s) of each pulse peak, in [0, 1].
        width: Gaussian std (in pseudotime units) shared by all pulses.
        amp: peak amplitude shared by all pulses.
    """

    centers: List[float]
    width: float = 0.08
    amp: float = 1.0

    def evaluate(self, t: np.ndarray) -> np.ndarray:
        signal = np.zeros_like(t)
        for c in self.centers:
            signal = signal + self.amp * np.exp(-0.5 * ((t - c) / self.width) ** 2)
        return np.clip(signal, 0.0, 1.0)


@dataclass
class NoiseConfig:
    """scRNA-seq-like noise parameters.

    Attributes:
        biological: std of Gaussian stochastic noise added to continuous
            expression (transcriptional bursting / cell-to-cell variability).
        dropout: probability that an expressed value is technically zeroed
            (zero-inflation / capture inefficiency).
        dropout_expr_scale: dropout probability is scaled by
            ``exp(-dropout_expr_scale * x)`` so highly expressed genes drop out
            less (set 0 for uniform dropout).
    """

    biological: float = 0.05
    dropout: float = 0.2
    dropout_expr_scale: float = 0.0


class SyntheticGRNGenerator:
    """Generate synthetic scRNA-seq expression from an explicit GRN.

    Args:
        spec: the ground-truth :class:`GRNSpec` (validated on construction).
        n_cells: number of cells (pseudotime points) to simulate.
        reg_skip: regulatory time lag in cells between regulators and target.
        noise: :class:`NoiseConfig` controlling biological noise + dropout.
        n_root_pulses: number of activation "waves" per root driver gene.
        root_signals: optional mapping of root gene name -> :class:`RootSignal`
            giving deterministic pulse timing. Root genes not listed fall back
            to randomly-placed pulses. Only valid for root (unregulated) genes.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        spec: GRNSpec,
        n_cells: int = 500,
        reg_skip: int = 5,
        noise: Optional[NoiseConfig] = None,
        n_root_pulses: int = 1,
        root_signals: Optional[Dict[str, RootSignal]] = None,
        seed: Optional[int] = None,
    ):
        spec.validate()
        if n_cells <= 0:
            raise ValueError("n_cells must be positive.")
        if reg_skip < 0 or reg_skip >= n_cells:
            raise ValueError("reg_skip must be in [0, n_cells).")

        root_signals = root_signals or {}
        roots = set(spec.roots())
        for gene in root_signals:
            if gene not in spec.rules:
                raise KeyError(f"root_signals references unknown gene: {gene!r}")
            if gene not in roots:
                raise ValueError(
                    f"root_signals given for {gene!r}, but it is a regulated "
                    "(non-root) gene; only root genes accept explicit signals."
                )

        self.spec = spec
        self.n_cells = n_cells
        self.reg_skip = reg_skip
        self.noise = noise or NoiseConfig()
        self.n_root_pulses = max(1, n_root_pulses)
        self.root_signals = root_signals
        self.rng = np.random.default_rng(seed)

        self.pseudotime = np.linspace(0.0, 1.0, n_cells)
        # continuous (clean) expression, cells x genes, filled by simulate()
        self.clean: Optional[np.ndarray] = None
        # noisy continuous expression (what a real assay would measure)
        self.expression: Optional[np.ndarray] = None

    # ---- Hill kinetics -------------------------------------------------
    @staticmethod
    def _activation(x: np.ndarray, K: float, n: float) -> np.ndarray:
        xn = np.power(np.clip(x, 0.0, None), n)
        return xn / (K ** n + xn)

    def _repression(self, x: np.ndarray, K: float, n: float) -> np.ndarray:
        return 1.0 - self._activation(x, K, n)

    # ---- root driver signals -------------------------------------------
    def _root_signal(self, gene_seed: int) -> np.ndarray:
        """Smooth pulse(s) along pseudotime for a root/driver gene in [0, 1]."""
        rng = np.random.default_rng(gene_seed)
        t = self.pseudotime
        signal = np.zeros_like(t)
        for _ in range(self.n_root_pulses):
            center = rng.uniform(0.15, 0.85)
            width = rng.uniform(0.08, 0.25)
            amp = rng.uniform(0.7, 1.0)
            signal += amp * np.exp(-0.5 * ((t - center) / width) ** 2)
        return np.clip(signal, 0.0, 1.0)

    # ---- simulation ----------------------------------------------------
    def _lagged(self, arr: np.ndarray) -> np.ndarray:
        """Shift a regulator trace forward in pseudotime by reg_skip cells."""
        if self.reg_skip == 0:
            return arr
        out = np.empty_like(arr)
        out[: self.reg_skip] = arr[0]  # hold the initial value during lag
        out[self.reg_skip :] = arr[: -self.reg_skip]
        return out

    def _drive(self, rule: GeneRule, expr: np.ndarray) -> np.ndarray:
        idx = self.spec.index_of
        drive = np.ones(self.n_cells, dtype=float)
        for a in rule.activators:
            hp = rule.hill_for(a)
            xa = self._lagged(expr[:, idx(a)])
            drive *= self._activation(xa, hp.K, hp.n)
        for r in rule.repressors:
            hp = rule.hill_for(r)
            xr = self._lagged(expr[:, idx(r)])
            drive *= self._repression(xr, hp.K, hp.n)
        return drive

    def simulate(self) -> "SyntheticGRNGenerator":
        """Run the forward simulation, populating clean + noisy expression."""
        n, g = self.n_cells, self.spec.n_genes
        clean = np.zeros((n, g), dtype=float)
        order = self.spec.topological_order()

        for gene in order:
            j = self.spec.index_of(gene)
            rule = self.spec.rules[gene]
            if rule.is_root:
                if gene in self.root_signals:
                    clean[:, j] = self.root_signals[gene].evaluate(self.pseudotime)
                else:
                    clean[:, j] = self._root_signal(int(self.rng.integers(0, 2**31)))
            else:
                drive = self._drive(rule, clean)
                clean[:, j] = rule.basal + (rule.vmax - rule.basal) * drive

        self.clean = clean
        self.expression = self._apply_noise(clean)
        return self

    def _apply_noise(self, clean: np.ndarray) -> np.ndarray:
        expr = clean.copy()
        # biological / stochastic noise (Gaussian), keep non-negative
        if self.noise.biological > 0:
            expr = expr + self.rng.normal(0.0, self.noise.biological, size=expr.shape)
            expr = np.clip(expr, 0.0, None)
        # dropout / zero-inflation
        if self.noise.dropout > 0:
            p = np.full(expr.shape, self.noise.dropout, dtype=float)
            if self.noise.dropout_expr_scale > 0:
                p = p * np.exp(-self.noise.dropout_expr_scale * expr)
            mask = self.rng.random(expr.shape) < p
            expr[mask] = 0.0
        return expr

    def _ensure_simulated(self) -> None:
        if self.expression is None:
            self.simulate()

    # ---- binarization (ground-truth labels) ----------------------------
    def _binarize(self, arr: np.ndarray) -> np.ndarray:
        """Threshold each gene at the midpoint of its basal..vmax range."""
        binary = np.zeros_like(arr)
        for gene in self.spec.genes:
            j = self.spec.index_of(gene)
            rule = self.spec.rules[gene]
            if rule.is_root:
                thr = 0.5
            else:
                thr = 0.5 * (rule.basal + rule.vmax)
            binary[:, j] = (arr[:, j] >= thr).astype(float)
        return binary

    # ---- outputs -------------------------------------------------------
    def to_anndata(self, binarize: bool = True):
        """Return an AnnData compatible with the iqcell pipeline.

        - ``adata.X``: binarized expression if ``binarize`` else continuous.
        - ``adata.raw``: continuous (noisy) expression, as expected by
          ``iqcell.binarization.KMeans._fit`` which reads ``data.raw.X``.
        - ``adata.var_names``: gene names.
        - ``adata.obs['pseudotime']``: per-cell pseudotime.
        - ``adata.uns['grn_adjacency']`` / ``['grn_genes']``: ground-truth GRN.
        """
        import anndata as ad
        import pandas as pd

        self._ensure_simulated()
        cont = self.expression
        X = self._binarize(cont) if binarize else cont.copy()

        obs = pd.DataFrame(
            {"pseudotime": self.pseudotime},
            index=[f"cell_{i}" for i in range(self.n_cells)],
        )
        var = pd.DataFrame(index=list(self.spec.genes))

        raw_adata = ad.AnnData(X=cont.copy(), obs=obs.copy(), var=var.copy())
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.raw = raw_adata
        adata.uns["grn_adjacency"] = self.spec.adjacency()
        adata.uns["grn_genes"] = list(self.spec.genes)
        adata.layers["clean"] = self.clean
        return adata

    def to_tensors(self, target: str, binarize: bool = True):
        """Return (x_act, x_rep, y, t) torch tensors for one target gene.

        Matches ``iqcell.utils.dataset.ExpressionData_And_Or``:
        - x_act: (n_cells, n_activators) regulator expression
        - x_rep: (n_cells, n_repressors) regulator expression
        - y:     (n_cells,) target expression
        - t:     (n_cells,) pseudotime

        If ``binarize`` the values are the {0,1} ground-truth states; otherwise
        continuous noisy expression.
        """
        import torch

        self._ensure_simulated()
        if target not in self.spec.rules:
            raise KeyError(f"Unknown target gene: {target!r}")
        rule = self.spec.rules[target]

        source = self._binarize(self.expression) if binarize else self.expression
        idx = self.spec.index_of

        def cols(names: List[str]) -> np.ndarray:
            if not names:
                return np.zeros((self.n_cells, 0), dtype=np.float32)
            return np.stack([source[:, idx(nm)] for nm in names], axis=1).astype(
                np.float32
            )

        x_act = cols(rule.activators)
        x_rep = cols(rule.repressors)
        y = source[:, idx(target)].astype(np.float32)
        t = self.pseudotime.astype(np.float32)

        return (
            torch.from_numpy(x_act),
            torch.from_numpy(x_rep),
            torch.from_numpy(y).reshape(-1, 1),
            torch.from_numpy(t).reshape(-1, 1),
        )

    def ground_truth(self) -> Dict[str, object]:
        """Return the known GRN as a dict of edges + signed adjacency."""
        return {
            "genes": list(self.spec.genes),
            "edges": list(self.spec.edges()),
            "adjacency": self.spec.adjacency(),
        }
