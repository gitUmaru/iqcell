"""Read and score BEELINE outputs.

Includes a self-contained scorer so ranked edges can be evaluated against a
known ground-truth network even without a working BEELINE/Docker install.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import pandas as pd

EdgeTriple = Tuple[str, str, float]


def read_ranked_edges(path: str) -> pd.DataFrame:
    """Read a BEELINE ``rankedEdges.csv`` into a DataFrame.

    Expected columns: ``Gene1``, ``Gene2``, ``EdgeWeight``. BEELINE writes these
    tab- or comma-separated depending on the algorithm; both are handled.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_evaluation(output_dir: str) -> Dict[str, float]:
    """Collect AUPRC/AUROC/EPR values from BEELINE evaluation CSVs.

    Searches ``output_dir`` recursively for files matching ``*AUPRC*.csv``,
    ``*AUROC*.csv`` and ``*EPR*.csv``. Returns ``{}`` when none are found.
    """
    metrics: Dict[str, float] = {}
    patterns = {"auprc": "*AUPRC*.csv", "auroc": "*AUROC*.csv", "epr": "*EPR*.csv"}
    for key, pat in patterns.items():
        matches = glob.glob(os.path.join(output_dir, "**", pat), recursive=True)
        for match in matches:
            try:
                df = pd.read_csv(match, index_col=0)
            except Exception:
                continue
            numeric = df.select_dtypes(include="number")
            if numeric.empty:
                continue
            # Use the mean across algorithms/datasets as a single summary value.
            metrics[key] = float(numeric.to_numpy().mean())
            break
    return metrics


def _normalize_ranked(
    ranked_edges: Union[pd.DataFrame, Iterable[EdgeTriple]],
) -> List[EdgeTriple]:
    if isinstance(ranked_edges, pd.DataFrame):
        cols = {c.lower(): c for c in ranked_edges.columns}
        g1 = cols.get("gene1", ranked_edges.columns[0])
        g2 = cols.get("gene2", ranked_edges.columns[1])
        w = cols.get("edgeweight", ranked_edges.columns[2])
        return [
            (str(r[g1]), str(r[g2]), float(r[w]))
            for _, r in ranked_edges.iterrows()
        ]
    return [(str(a), str(b), float(wt)) for a, b, wt in ranked_edges]


def _truth_pairs(
    ground_truth_edges: Iterable[Sequence],
) -> set:
    """Directed (g1, g2) pairs present in the ground truth (sign ignored)."""
    pairs = set()
    for edge in ground_truth_edges:
        g1, g2 = str(edge[0]), str(edge[1])
        pairs.add((g1, g2))
    return pairs


def score_ranking(
    ranked_edges: Union[pd.DataFrame, Iterable[EdgeTriple]],
    ground_truth_edges: Iterable[Sequence],
) -> Dict[str, float]:
    """Score a ranked edge list against ground truth over all directed pairs.

    Builds the universe of directed gene pairs (excluding self-loops) from the
    genes seen in either the ranking or the ground truth, labels each pair
    1/0 by presence in ``ground_truth_edges`` (sign ignored), and scores the
    predicted ``EdgeWeight`` as the ranking. Uses sklearn when available,
    otherwise a pure-python implementation.

    Returns ``{"auprc": float, "auroc": float}``.
    """
    ranked = _normalize_ranked(ranked_edges)
    truth = _truth_pairs(ground_truth_edges)

    genes = set()
    for g1, g2, _ in ranked:
        genes.update((g1, g2))
    for g1, g2 in truth:
        genes.update((g1, g2))

    weight_by_pair = {(g1, g2): w for g1, g2, w in ranked}

    y_true: List[int] = []
    y_score: List[float] = []
    gene_list = sorted(genes)
    for g1 in gene_list:
        for g2 in gene_list:
            if g1 == g2:
                continue  # exclude self-loops
            y_true.append(1 if (g1, g2) in truth else 0)
            y_score.append(weight_by_pair.get((g1, g2), 0.0))

    return {
        "auprc": _average_precision(y_true, y_score),
        "auroc": _roc_auc(y_true, y_score),
    }


def _average_precision(y_true: List[int], y_score: List[float]) -> float:
    try:
        from sklearn.metrics import average_precision_score

        if sum(y_true) == 0:
            return 0.0
        return float(average_precision_score(y_true, y_score))
    except ImportError:
        return _average_precision_py(y_true, y_score)


def _roc_auc(y_true: List[int], y_score: List[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))
    except ImportError:
        return _roc_auc_py(y_true, y_score)


def _sorted_desc(y_true: List[int], y_score: List[float]) -> List[int]:
    """Labels ordered by descending score (stable for ties)."""
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    return [y_true[i] for i in order]


def _average_precision_py(y_true: List[int], y_score: List[float]) -> float:
    total_pos = sum(y_true)
    if total_pos == 0:
        return 0.0
    labels = _sorted_desc(y_true, y_score)
    tp = 0
    ap = 0.0
    for k, label in enumerate(labels, start=1):
        if label == 1:
            tp += 1
            precision_at_k = tp / k
            ap += precision_at_k
    return ap / total_pos


def _roc_auc_py(y_true: List[int], y_score: List[float]) -> float:
    pos = [s for s, y in zip(y_score, y_true) if y == 1]
    neg = [s for s, y in zip(y_score, y_true) if y == 0]
    if not pos or not neg:
        return 0.5
    # Rank-based Mann-Whitney U estimate of AUROC (handles ties at 0.5).
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))
