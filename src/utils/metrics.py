"""
Evaluation metrics for character evolution retrieval tasks.

Implements the metrics described in Section 4.2 of the MSEF paper:
    - Recall@K: Retrieval accuracy within top-K results
    - Average Precision (AP): Mean precision across all recall levels
    - Mean Average Precision (mAP): Average AP across all queries

These metrics evaluate how well the model can retrieve evolved character forms
given a query character from a different historical period.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import average_precision_score
import torch.nn.functional as F


def compute_similarity_matrix(
    query_embeddings: torch.Tensor,
    database_embeddings: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """
    Compute pairwise similarity between query and database embeddings.

    Args:
        query_embeddings: Query embedding vectors (N_q, D)
        database_embeddings: Database embedding vectors (N_db, D)
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Similarity matrix (N_q, N_db)
    """
    if metric == "cosine":
        # Normalize embeddings
        query_norm = F.normalize(query_embeddings, p=2, dim=1)
        db_norm = F.normalize(database_embeddings, p=2, dim=1)
        # Compute cosine similarity
        similarity = torch.mm(query_norm, db_norm.t())
    elif metric == "euclidean":
        # Compute negative Euclidean distance (higher is more similar)
        # Expand dimensions for broadcasting
        query_expanded = query_embeddings.unsqueeze(1)  # (N_q, 1, D)
        db_expanded = database_embeddings.unsqueeze(0)  # (1, N_db, D)
        # Compute L2 distance and negate
        similarity = -torch.norm(query_expanded - db_expanded, p=2, dim=2)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarity


def recall_at_k(
    query_embeddings: torch.Tensor,
    database_embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    metric: str = "cosine",
) -> Dict[int, float]:
    """
    Compute Recall@K for character evolution retrieval.

    Recall@K measures the fraction of queries where the correct character
    appears in the top-K retrieved results.

    Args:
        query_embeddings: Query character embeddings (N, D)
        database_embeddings: Database character embeddings (M, D)
        labels: Ground truth matching labels (N, M)
                labels[i, j] = 1 if query i matches database j
        k_values: List of K values to compute recall for
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Dictionary mapping K to Recall@K value

    Example:
        >>> query = torch.randn(100, 352)  # 100 query characters
        >>> database = torch.randn(500, 352)  # 500 database characters
        >>> labels = torch.zeros(100, 500)
        >>> # Set ground truth matches
        >>> for i in range(100):
        ...     labels[i, i] = 1  # Each query matches itself
        >>> recalls = recall_at_k(query, database, labels, k_values=[1, 5, 10])
        >>> print(f"Recall@1: {recalls[1]:.3f}")
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(
        query_embeddings, database_embeddings, metric
    )

    # Get top-K indices for each query
    num_queries = query_embeddings.size(0)
    max_k = max(k_values)

    # Sort by similarity (descending)
    _, top_k_indices = torch.topk(similarity, k=min(max_k, similarity.size(1)), dim=1)

    # Compute recall for each K
    recalls = {}
    for k in k_values:
        requested_k = k
        if k > similarity.size(1):
            k = similarity.size(1)

        # Get top-k predictions
        top_k_preds = top_k_indices[:, :k]  # (N, k)

        # Check if any top-k prediction matches ground truth
        num_correct = 0
        for i in range(num_queries):
            # Get ground truth matches for query i
            gt_matches = labels[i].nonzero(as_tuple=True)[0]
            if len(gt_matches) == 0:
                continue

            # Check if any top-k prediction is in ground truth
            top_k_set = set(top_k_preds[i].cpu().numpy())
            gt_set = set(gt_matches.cpu().numpy())

            if len(top_k_set & gt_set) > 0:
                num_correct += 1

        recalls[requested_k] = num_correct / num_queries

    return recalls


def average_precision(
    query_embeddings: torch.Tensor,
    database_embeddings: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "cosine",
) -> float:
    """
    Compute Mean Average Precision (mAP) for retrieval.

    Average Precision (AP) is the area under the precision-recall curve.
    mAP is the mean AP across all queries.

    Args:
        query_embeddings: Query character embeddings (N, D)
        database_embeddings: Database character embeddings (M, D)
        labels: Ground truth matching labels (N, M)
                labels[i, j] = 1 if query i matches database j
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Mean Average Precision across all queries

    Example:
        >>> query = torch.randn(100, 352)
        >>> database = torch.randn(500, 352)
        >>> labels = torch.zeros(100, 500)
        >>> for i in range(100):
        ...     labels[i, i] = 1
        >>> map_score = average_precision(query, database, labels)
        >>> print(f"mAP: {map_score:.3f}")
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(
        query_embeddings, database_embeddings, metric
    )

    # Convert to numpy for sklearn
    similarity_np = similarity.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Compute AP for each query
    aps = []
    num_queries = query_embeddings.size(0)

    for i in range(num_queries):
        # Get ground truth for query i
        y_true = labels_np[i]

        # Skip if no positive labels
        if y_true.sum() == 0:
            aps.append(0.0)
            continue

        # Get similarity scores
        y_score = similarity_np[i]

        # Compute average precision
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

    # Return mean AP
    if len(aps) == 0:
        return 0.0

    return np.mean(aps)


def precision_at_k(
    query_embeddings: torch.Tensor,
    database_embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    metric: str = "cosine",
) -> Dict[int, float]:
    """
    Compute Precision@K for retrieval.

    Precision@K measures the fraction of top-K retrieved items that are correct.

    Args:
        query_embeddings: Query character embeddings (N, D)
        database_embeddings: Database character embeddings (M, D)
        labels: Ground truth matching labels (N, M)
        k_values: List of K values to compute precision for
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Dictionary mapping K to Precision@K value
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(
        query_embeddings, database_embeddings, metric
    )

    # Get top-K indices for each query
    num_queries = query_embeddings.size(0)
    max_k = max(k_values)

    # Sort by similarity (descending)
    _, top_k_indices = torch.topk(similarity, k=min(max_k, similarity.size(1)), dim=1)

    # Compute precision for each K
    precisions = {}
    for k in k_values:
        requested_k = k
        if k > similarity.size(1):
            k = similarity.size(1)

        # Get top-k predictions
        top_k_preds = top_k_indices[:, :k]  # (N, k)

        # Compute precision for each query
        total_precision = 0.0
        for i in range(num_queries):
            # Get ground truth matches for query i
            gt_matches = labels[i].nonzero(as_tuple=True)[0]
            if len(gt_matches) == 0:
                continue

            # Count how many top-k predictions are correct
            top_k_set = set(top_k_preds[i].cpu().numpy())
            gt_set = set(gt_matches.cpu().numpy())
            num_correct = len(top_k_set & gt_set)

            # Precision = correct / k
            total_precision += num_correct / k

        precisions[requested_k] = total_precision / num_queries

    return precisions


def ndcg_at_k(
    query_embeddings: torch.Tensor,
    database_embeddings: torch.Tensor,
    labels: torch.Tensor,
    k_values: List[int] = [5, 10],
    metric: str = "cosine",
) -> Dict[int, float]:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG measures ranking quality with position-based discounting.

    Args:
        query_embeddings: Query character embeddings (N, D)
        database_embeddings: Database character embeddings (M, D)
        labels: Ground truth matching labels (N, M)
        k_values: List of K values to compute NDCG for
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Dictionary mapping K to NDCG@K value
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(
        query_embeddings, database_embeddings, metric
    )

    # Get top-K indices for each query
    num_queries = query_embeddings.size(0)
    max_k = max(k_values)

    # Sort by similarity (descending)
    _, top_k_indices = torch.topk(similarity, k=min(max_k, similarity.size(1)), dim=1)

    # Compute NDCG for each K
    ndcgs = {}
    for k in k_values:
        requested_k = k
        if k > similarity.size(1):
            k = similarity.size(1)

        # Get top-k predictions
        top_k_preds = top_k_indices[:, :k]  # (N, k)

        # Compute NDCG for each query
        total_ndcg = 0.0
        for i in range(num_queries):
            # Get ground truth relevance scores
            relevance = labels[i, top_k_preds[i]].cpu().numpy()

            # Compute DCG
            dcg = relevance[0] + np.sum(relevance[1:] / np.log2(np.arange(3, k + 2)))

            # Compute ideal DCG (sort by relevance)
            ideal_relevance = np.sort(labels[i].cpu().numpy())[::-1][:k]
            idcg = ideal_relevance[0] + np.sum(
                ideal_relevance[1:] / np.log2(np.arange(3, k + 2))
            )

            # Compute NDCG
            if idcg > 0:
                total_ndcg += dcg / idcg

        ndcgs[requested_k] = total_ndcg / num_queries

    return ndcgs


def compute_all_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, Union[float, Dict[int, float]]]:
    """
    Compute all evaluation metrics on a dataset.

    This function runs the model on the entire dataloader and computes:
        - Recall@K for K in k_values
        - Mean Average Precision (mAP)
        - Precision@K for K in k_values
        - NDCG@K for K in k_values

    Args:
        model: Evolution model with predict() method
        dataloader: DataLoader for evaluation dataset
        device: Device to run evaluation on
        k_values: List of K values for metrics

    Returns:
        Dictionary containing all metrics

    Example:
        >>> model = MSEFModel(...)
        >>> test_loader = create_dataloaders(...)[2]
        >>> metrics = compute_all_metrics(model, test_loader)
        >>> print(f"Recall@1: {metrics['recall'][1]:.3f}")
        >>> print(f"mAP: {metrics['map']:.3f}")
    """
    model.eval()
    model = model.to(device)
    queries, gallery, identities = [], {}, []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["features_src"].to(device)
            y = batch["features_tgt"].to(device)
            a = batch["time_src"].to(device)
            b = batch["time_tgt"].to(device)
            query = model.flow_batch(model.encode(x, a), a, b)
            target = model.encode(y, b)
            queries.append(query.cpu())
            identities.extend(str(c) for c in batch["char_id"])
            for oid, char, embedding in zip(
                batch["target_id"], batch["char_id"], target.cpu()
            ):
                gallery[oid] = (str(char), embedding)
    if not queries or not gallery:
        raise ValueError("Empty retrieval population")
    query_embeddings = torch.cat(queries)
    database_embeddings = torch.stack([r[1] for r in gallery.values()])
    labels = torch.tensor(
        [[float(c == r[0]) for r in gallery.values()] for c in identities]
    )
    return {
        "recall": recall_at_k(query_embeddings, database_embeddings, labels, k_values),
        "map": average_precision(query_embeddings, database_embeddings, labels),
        "precision": precision_at_k(
            query_embeddings, database_embeddings, labels, k_values
        ),
        "ndcg": ndcg_at_k(query_embeddings, database_embeddings, labels, k_values),
    }


def format_metrics(metrics: Dict) -> str:
    """
    Format metrics dictionary into a readable string.

    Args:
        metrics: Dictionary from compute_all_metrics()

    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Evaluation Metrics")
    lines.append("=" * 50)

    # Recall@K
    if "recall" in metrics:
        lines.append("\nRecall@K:")
        for k, value in sorted(metrics["recall"].items()):
            lines.append(f"  Recall@{k:2d}: {value:.4f}")

    # Precision@K
    if "precision" in metrics:
        lines.append("\nPrecision@K:")
        for k, value in sorted(metrics["precision"].items()):
            lines.append(f"  Precision@{k:2d}: {value:.4f}")

    # NDCG@K
    if "ndcg" in metrics:
        lines.append("\nNDCG@K:")
        for k, value in sorted(metrics["ndcg"].items()):
            lines.append(f"  NDCG@{k:2d}: {value:.4f}")

    # mAP
    if "map" in metrics:
        lines.append(f"\nMean Average Precision: {metrics['map']:.4f}")

    lines.append("=" * 50)

    return "\n".join(lines)
