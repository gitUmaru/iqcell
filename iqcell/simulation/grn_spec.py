"""Explicit ground-truth GRN specification for synthetic scRNA-seq data.

The user supplies, per target gene, the set of activators and repressors along
with Hill-function kinetics. This spec is the *known truth* that inference
methods (binarization -> hierarchy -> logic engine) can be scored against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class HillParams:
    """Hill-function kinetics for a single regulator -> target edge.

    activation fraction for an activator:  x^n / (K^n + x^n)
    repression fraction for a repressor:   K^n / (K^n + x^n)   (= 1 - activation)

    Attributes:
        K: half-max threshold (in the same units as regulator expression).
        n: Hill coefficient (steepness/cooperativity), n > 0.
    """

    K: float = 0.5
    n: float = 4.0

    def __post_init__(self) -> None:
        if self.K <= 0:
            raise ValueError(f"Hill K must be > 0, got {self.K}")
        if self.n <= 0:
            raise ValueError(f"Hill n must be > 0, got {self.n}")


@dataclass
class GeneRule:
    """Regulation rule for one target gene.

    target expression rate is driven by the product (AND) of activator Hill
    terms times the product of repressor (NOT) Hill terms:

        drive = prod_a activation(x_a) * prod_r repression(x_r)
        expression = basal + (vmax - basal) * drive

    A gene with no regulators is a *root* / driver gene whose expression is
    generated as an input signal (see the generator's ``root_signal.

    Attributes:
        activators: list of gene names that activate this target.
        repressors: list of gene names that repress this target.
        hill: per-regulator Hill params keyed by regulator gene name. Missing
            regulators fall back to ``default_hill``.
        basal: baseline expression when drive is 0.
        vmax: maximum expression when drive is 1.
        default_hill: fallback Hill params for regulators not in ``hill``.
    """

    activators: List[str] = field(default_factory=list)
    repressors: List[str] = field(default_factory=list)
    hill: Dict[str, HillParams] = field(default_factory=dict)
    basal: float = 0.0
    vmax: float = 1.0
    default_hill: HillParams = field(default_factory=HillParams)

    @property
    def regulators(self) -> List[str]:
        return list(self.activators) + list(self.repressors)

    @property
    def is_root(self) -> bool:
        return len(self.activators) == 0 and len(self.repressors) == 0

    def hill_for(self, regulator: str) -> HillParams:
        return self.hill.get(regulator, self.default_hill)


class GRNSpec:
    """Explicit gene regulatory network specification.

    Example:
        >>> spec = GRNSpec(["A", "B", "C"])
        >>> spec.set_rule("B", activators=["A"])
        >>> spec.set_rule("C", activators=["B"], repressors=["A"])
        >>> spec.validate()
    """

    def __init__(self, genes: List[str]):
        if len(genes) != len(set(genes)):
            raise ValueError("Gene names must be unique.")
        self.genes: List[str] = list(genes)
        self._index: Dict[str, int] = {g: i for i, g in enumerate(self.genes)}
        self.rules: Dict[str, GeneRule] = {g: GeneRule() for g in self.genes}

    def set_rule(
        self,
        target: str,
        activators: Optional[List[str]] = None,
        repressors: Optional[List[str]] = None,
        hill: Optional[Dict[str, HillParams]] = None,
        basal: float = 0.0,
        vmax: float = 1.0,
        default_hill: Optional[HillParams] = None,
    ) -> "GRNSpec":
        if target not in self._index:
            raise KeyError(f"Unknown target gene: {target!r}")
        self.rules[target] = GeneRule(
            activators=list(activators or []),
            repressors=list(repressors or []),
            hill=dict(hill or {}),
            basal=basal,
            vmax=vmax,
            default_hill=default_hill or HillParams(),
        )
        return self

    def index_of(self, gene: str) -> int:
        return self._index[gene]

    @property
    def n_genes(self) -> int:
        return len(self.genes)

    def roots(self) -> List[str]:
        return [g for g in self.genes if self.rules[g].is_root]

    def edges(self):
        """Yield ground-truth edges as (regulator, target, sign) tuples.

        sign is +1 for activation, -1 for repression.
        """
        for target, rule in self.rules.items():
            for a in rule.activators:
                yield (a, target, 1)
            for r in rule.repressors:
                yield (r, target, -1)

    def adjacency(self):
        """Signed adjacency matrix A where A[i, j] is the sign of edge i -> j.

        Row i = regulator, column j = target. 0 = no edge.
        """
        import numpy as np

        n = self.n_genes
        A = np.zeros((n, n), dtype=int)
        for reg, tgt, sign in self.edges():
            A[self._index[reg], self._index[tgt]] = sign
        return A

    def validate(self) -> None:
        """Check referential integrity and that the network is acyclic (a DAG).

        A DAG guarantees a well-defined feed-forward simulation order. Roots
        (no regulators) are the input driver genes.
        """
        # 1. all referenced regulators exist
        for target, rule in self.rules.items():
            for reg in rule.regulators:
                if reg not in self._index:
                    raise ValueError(
                        f"Target {target!r} references unknown regulator {reg!r}."
                    )
                if reg == target:
                    raise ValueError(f"Self-regulation not supported: {target!r}.")

        # 2. acyclicity via topological sort (Kahn's algorithm)
        self.topological_order()  # raises if cyclic

        # 3. at least one root
        if not self.roots():
            raise ValueError(
                "GRN has no root genes (every gene has a regulator); "
                "at least one driver gene with no regulators is required."
            )

    def topological_order(self) -> List[str]:
        """Return genes ordered so regulators precede their targets.

        Raises ValueError if the network contains a cycle.
        """
        # in-degree = number of regulators of each gene
        indeg = {g: len(self.rules[g].regulators) for g in self.genes}
        # regulator -> list of targets
        downstream: Dict[str, List[str]] = {g: [] for g in self.genes}
        for reg, tgt, _ in self.edges():
            downstream[reg].append(tgt)

        queue = [g for g in self.genes if indeg[g] == 0]
        order: List[str] = []
        while queue:
            g = queue.pop(0)
            order.append(g)
            for tgt in downstream[g]:
                indeg[tgt] -= 1
                if indeg[tgt] == 0:
                    queue.append(tgt)

        if len(order) != self.n_genes:
            raise ValueError("GRN contains a cycle; a DAG is required.")
        return order
