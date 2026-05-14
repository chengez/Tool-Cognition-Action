#!/usr/bin/env python3
"""probe.py

Train linear probes to classify either:
- "necessity" (necessary vs unnecessary, regardless of called status)
- "action" (called vs not called, regardless of necessity)

This script expects cluster tensors stored under:
    clusters/<model>/<data_name>/{necessary|unnecessary}_{called|Notcalled}/

and filenames like:
    {necessary|unnecessary}_{called|Notcalled}_L<layer>_K-<pos>.pt

Uses a configurable train/test split on all combined data per layer.
Saves learned weight vectors separately.
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import re
import numpy as np
import torch
from tqdm import tqdm


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy score."""
    return np.mean(y_true == y_pred)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute binary F1 score."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def compute_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: int
) -> Tuple[float, float, float]:
    """Compute precision/recall/F1 for a specified positive label."""
    tp = np.sum((y_true == positive_label) & (y_pred == positive_label))
    fp = np.sum((y_true != positive_label) & (y_pred == positive_label))
    fn = np.sum((y_true == positive_label) & (y_pred != positive_label))

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Matthews correlation coefficient for binary labels."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / np.sqrt(denom)


class SimpleLinearProbe(torch.nn.Module):
    """Simple linear probe using PyTorch."""
    def __init__(self, input_dim, device='cpu'):
        super().__init__()
        self.device = device
        self.linear = torch.nn.Linear(input_dim, 1).to(device)

    def forward(self, x):
        return self.linear(x)

    def predict(self, x):
        """Binary predictions."""
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            probs = self.forward(x_tensor)
            return (probs > 0.5).cpu().numpy().flatten()

    def predict_proba(self, x):
        """Probability predictions."""
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float().to(self.device)
            probs = torch.sigmoid(self.forward(x_tensor))
            return probs.cpu().numpy().flatten()


def train_linear_probe(
    X,
    y,
    num_epochs=100,
    learning_rate=0.01,
    device='cpu',
    weight_decay=0.01,
    pos_weight: float = None,
):
    """
    Train a linear probe using PyTorch with L2 regularization.

    Args:
        X: Input features, shape (n_samples, n_features)
        y: Binary labels, shape (n_samples,)
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to use for training
        weight_decay: L2 regularization strength

    Returns:
        Trained probe model and training accuracy
    """
    input_dim = X.shape[1]

    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).float().unsqueeze(1).to(device)

    model = SimpleLinearProbe(input_dim, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if pos_weight is not None:
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device)
        )
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        preds = (logits > 0).float()
        train_acc = (preds == y_tensor).float().mean().item()

    return model, train_acc


def parse_bool(value: str) -> bool:
    """Parse a boolean value from string (T/F or true/false)."""
    if isinstance(value, bool):
        return value
    if value.upper() in ('T', 'TRUE', '1', 'YES'):
        return True
    elif value.upper() in ('F', 'FALSE', '0', 'NO'):
        return False
    else:
        raise ValueError(f"Cannot parse boolean from: {value}")


def bool_to_label(value: bool, feature: str) -> str:
    """Convert boolean to the label used in filenames."""
    label_map = {
        'necessity': {True: 'necessary', False: 'unnecessary'},
        'called': {True: 'called', False: 'Notcalled'}
    }
    if feature not in label_map:
        raise ValueError(f"Unknown feature: {feature}")
    return label_map[feature][value]


def extract_k_position(filename: str) -> str:
    """
    Extract K position from filename.
    Example: "necessary_called_L10_K-10.pt" -> "K-10"
    """
    match = re.search(r'_(K-?\d+)\.pt$', filename)
    if match:
        return match.group(1)
    return None


def get_layer_from_filename(filename: str) -> int:
    """
    Extract layer index from filename.
    Example: "necessary_called_L10_K-10.pt" -> 10
    """
    match = re.search(r'_L(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None


def discover_k_positions(cluster_dir: str, model_basename: str, data_name: str) -> List[str]:
    """
    Discover all K positions available in the cluster directory.
    """
    # We inspect one canonical cluster directory (necessary_called) to list K positions.
    base_path = os.path.join(cluster_dir, model_basename, data_name, "necessary_called")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Directory not found: {base_path}")

    k_positions = set()
    for filename in os.listdir(base_path):
        if filename.endswith(".pt"):
            k_pos = extract_k_position(filename)
            if k_pos:
                k_positions.add(k_pos)

    if not k_positions:
        raise ValueError(f"No .pt files found in {base_path}")

    def sort_key(k_str):
        match = re.search(r'-?(\d+)', k_str)
        if match:
            return -int(match.group(1))
        return 0

    return sorted(k_positions, key=sort_key)


def discover_layers(cluster_dir: str, model_basename: str, data_name: str, k_position: str) -> List[int]:
    """
    Discover all layers available for a given K position.
    """
    base_path = os.path.join(cluster_dir, model_basename, data_name, "necessary_called")
    layers = set()

    for filename in os.listdir(base_path):
        if filename.endswith(f"{k_position}.pt"):
            layer = get_layer_from_filename(filename)
            if layer is not None:
                layers.add(layer)

    return sorted(layers)


def load_cluster_tensor(
    model_basename: str,
    data_name: str,
    layer: int,
    necessity: bool,
    called: bool,
    k_position: str
) -> torch.Tensor:
    """
    Load a cluster tensor from disk.
    """
    necessity_str = bool_to_label(necessity, 'necessity')
    called_str = bool_to_label(called, 'called')

    cluster_subdir = f"{necessity_str}_{called_str}"
    filename = f"{necessity_str}_{called_str}_L{layer}_{k_position}.pt"
    filepath = os.path.join("clusters", model_basename, data_name, cluster_subdir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find cluster file: {filepath}")

    return torch.load(filepath)


def get_classification_data(
    model_basename: str,
    data_name: str,
    layer: int,
    k_position: str,
    classification_type: str,
    balance_clusters: bool = False,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Load data and labels based on classification type.
    Also return misalignment mask for "hard" samples.

    For "necessity" (necessity):
    - Positive samples: all necessary (called + not_called)
    - Negative samples: all unnecessary (called + not_called)
    - Misaligned: necessary_notcalled (label 1, hard to classify)
              and unnecessary_called (label 0, hard to classify)

    For "action" (called):
    - Positive samples: all called (necessary + unnecessary)
    - Negative samples: all not_called (necessary + unnecessary)
    - Misaligned: unnecessary_called (label 1, hard to classify)
              and necessary_notcalled (label 0, hard to classify)

    Args:
        balance_clusters: Deprecated here. Balancing is now applied after
                         the train/test split (see train_probes_for_position).
        seed: Random seed for balancing operations

    Returns:
        Tuple of (combined_data, labels, misaligned_mask)
    """
    if classification_type == "necessity":
        # Load necessary samples (called and not_called)
        pos_called = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=True, called=True,
            k_position=k_position
        )
        pos_notcalled = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=True, called=False,
            k_position=k_position
        )
        
        # Load unnecessary samples (called and not_called)
        neg_called = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=False, called=True,
            k_position=k_position
        )
        neg_notcalled = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=False, called=False,
            k_position=k_position
        )
        
        n_pos_called_orig = len(pos_called)
        n_pos_notcalled_orig = len(pos_notcalled)
        n_neg_called_orig = len(neg_called)
        n_neg_notcalled_orig = len(neg_notcalled)
        
        
        # Combine samples
        pos_samples = torch.cat([pos_called, pos_notcalled], dim=0)
        neg_samples = torch.cat([neg_called, neg_notcalled], dim=0)
        
        # Create misalignment mask: necessary_notcalled and unnecessary_called
        n_pos_called = len(pos_called)
        n_pos_notcalled = len(pos_notcalled)
        n_neg_called = len(neg_called)
        n_neg_notcalled = len(neg_notcalled)
        
        misaligned = np.concatenate([
            np.zeros(n_pos_called, dtype=bool),           # necessary_called - aligned
            np.ones(n_pos_notcalled, dtype=bool),         # necessary_notcalled - misaligned (label=1)
            np.ones(n_neg_called, dtype=bool),            # unnecessary_called - misaligned (label=0)
            np.zeros(n_neg_notcalled, dtype=bool)         # unnecessary_notcalled - aligned
        ])

    elif classification_type == "action":
        # Load called samples (necessary and unnecessary)
        pos_necessary = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=True, called=True,
            k_position=k_position
        )
        pos_unnecessary = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=False, called=True,
            k_position=k_position
        )
        
        # Load not_called samples (necessary and unnecessary)
        neg_necessary = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=True, called=False,
            k_position=k_position
        )
        neg_unnecessary = load_cluster_tensor(
            model_basename, data_name, layer,
            necessity=False, called=False,
            k_position=k_position
        )
        
        n_pos_necessary_orig = len(pos_necessary)
        n_pos_unnecessary_orig = len(pos_unnecessary)
        n_neg_necessary_orig = len(neg_necessary)
        n_neg_unnecessary_orig = len(neg_unnecessary)
        
        
        # Combine samples
        pos_samples = torch.cat([pos_necessary, pos_unnecessary], dim=0)
        neg_samples = torch.cat([neg_necessary, neg_unnecessary], dim=0)
        
        # Create misalignment mask: unnecessary_called and necessary_notcalled
        n_pos_necessary = len(pos_necessary)
        n_pos_unnecessary = len(pos_unnecessary)
        n_neg_necessary = len(neg_necessary)
        n_neg_unnecessary = len(neg_unnecessary)
        
        misaligned = np.concatenate([
            np.zeros(n_pos_necessary, dtype=bool),        # called_necessary - aligned
            np.ones(n_pos_unnecessary, dtype=bool),       # called_unnecessary - misaligned (label=1)
            np.ones(n_neg_necessary, dtype=bool),         # notcalled_necessary - misaligned (label=0)
            np.zeros(n_neg_unnecessary, dtype=bool)       # notcalled_unnecessary - aligned
        ])

    else:
        raise ValueError(f"Unknown classification_type: {classification_type}")

    # Combine and create labels
    X_all = torch.cat([pos_samples, neg_samples], dim=0).cpu().numpy()
    y_all = np.concatenate([
        np.ones(len(pos_samples)),
        np.zeros(len(neg_samples))
    ])

    # Return cluster sample counts along with data
    cluster_info = {
        'n_pos_called_or_necessary': n_pos_called if classification_type == "necessity" else n_pos_necessary,
        'n_pos_notcalled_or_unnecessary': n_pos_notcalled if classification_type == "necessity" else n_pos_unnecessary,
        'n_neg_called_or_necessary': n_neg_called if classification_type == "necessity" else n_neg_necessary,
        'n_neg_notcalled_or_unnecessary': n_neg_notcalled if classification_type == "necessity" else n_neg_unnecessary,
    }

    return X_all, y_all, misaligned, cluster_info



def save_probe_weights(
    probe: 'SimpleLinearProbe',
    layer: int,
    k_position: str,
    classification_type: str,
    output_dir: str,
    data_name: str,
    model_basename: str = None
):
    """
    Save the learned weight vector from a probe.
    """
    weights = probe.linear.weight.data.cpu().numpy()  # Shape: [1, input_dim]
    bias = probe.linear.bias.data.cpu().numpy()  # Shape: [1]

    filename = f"probe_weights_{classification_type}_L{layer}_{k_position}.npz"
    if model_basename:
        output_subdir = os.path.join(output_dir, model_basename, data_name)
    else:
        output_subdir = os.path.join(output_dir, data_name)
    os.makedirs(output_subdir, exist_ok=True)
    output_path = os.path.join(output_subdir, filename)

    np.savez(output_path, weights=weights, bias=bias)


def train_probes_for_position(
    model_basename: str,
    data_name: str,
    k_position: str,
    layers: List[int],
    classification_type: str,
    device: str = 'cpu',
    weights_output_dir: str = None,
    log_output_dir: str = None,
    balance_clusters: bool = False,
    use_pos_weight: bool = False,
    seed: int = 42,
    train_ratio: float = 0.7
) -> Tuple[Dict[int, Tuple[SimpleLinearProbe, float, Tuple[np.ndarray, np.ndarray]]], Dict]:
    """
    Train linear probes for each layer at the given K position.
    Classification type determines what we're classifying:
    - "necessity": necessary vs unnecessary
    - "action": called vs not_called

    Uses configurable train-test split on all data.

    Args:
        balance_clusters: If True, balance the 4 clusters by sampling equal amounts
        use_pos_weight: If True (and balance_clusters is False), use
                        BCEWithLogitsLoss pos_weight = n_neg / n_pos on train split.
        seed: Random seed for reproducibility
        train_ratio: Fraction of data to use for training (default: 0.7 for 70% train / 30% test)

    Returns:
        Tuple of (probes_dict, eval_results)
        - probes_dict: Dict mapping layer to (probe, training_accuracy, (X_test, y_test))
        - eval_results: Dict with test accuracy and F1 results
    """
    probes = {}
    training_results = []

    for layer in tqdm(layers, desc=f"Training probes for {k_position}"):
        try:
            # Load data based on classification type (includes misaligned mask)
            X_all, y_all, misaligned_all, cluster_info = get_classification_data(
                model_basename, data_name, layer, k_position, classification_type,
                balance_clusters=balance_clusters, seed=seed
            )
            
            # Print sample counts
            if classification_type == "necessity":
                print(f"  Layer {layer} sample counts:")
                print(f"    Necessary Called:     {cluster_info['n_pos_called_or_necessary']:6d}")
                print(f"    Necessary Not Called: {cluster_info['n_pos_notcalled_or_unnecessary']:6d}")
                print(f"    Unnecessary Called:   {cluster_info['n_neg_called_or_necessary']:6d}")
                print(f"    Unnecessary Not Called: {cluster_info['n_neg_notcalled_or_unnecessary']:6d}")
            else:  # action
                print(f"  Layer {layer} sample counts:")
                print(f"    Called Necessary:     {cluster_info['n_pos_called_or_necessary']:6d}")
                print(f"    Called Unnecessary:   {cluster_info['n_pos_notcalled_or_unnecessary']:6d}")
                print(f"    Not Called Necessary: {cluster_info['n_neg_called_or_necessary']:6d}")
                print(f"    Not Called Unnecessary: {cluster_info['n_neg_notcalled_or_unnecessary']:6d}")

            # Perform train-test split with specified ratio
            n_samples = len(X_all)
            train_size = int(train_ratio * n_samples)

            # Random indices
            np.random.seed(seed)  # For reproducibility
            indices = np.random.permutation(n_samples)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            # Balance only between positive and negative in the training split (if requested).
            if balance_clusters:
                train_pos_idx = train_indices[y_all[train_indices] == 1]
                train_neg_idx = train_indices[y_all[train_indices] == 0]

                if len(train_pos_idx) == 0 or len(train_neg_idx) == 0:
                    raise ValueError(f"Cannot balance training set for layer {layer}: one class is empty")

                min_count = min(len(train_pos_idx), len(train_neg_idx))
                train_pos_keep = np.random.choice(train_pos_idx, min_count, replace=False)
                train_neg_keep = np.random.choice(train_neg_idx, min_count, replace=False)
                train_keep = np.concatenate([train_pos_keep, train_neg_keep])

                dropped_train = np.setdiff1d(train_indices, train_keep, assume_unique=False)
                # Test set = original test + dropped training samples
                test_indices = np.concatenate([test_indices, dropped_train])
                train_indices = train_keep

            X_train = X_all[train_indices]
            y_train = y_all[train_indices]
            X_test = X_all[test_indices]
            y_test = y_all[test_indices]
            misaligned_test = misaligned_all[test_indices]

            # Optional class-frequency weighting on the positive class in
            # BCEWithLogitsLoss: pos_weight = n_neg / n_pos.
            pos_weight = None
            if (not balance_clusters) and use_pos_weight:
                n_pos_train = np.sum(y_train == 1)
                n_neg_train = np.sum(y_train == 0)
                if n_pos_train > 0 and n_neg_train > 0:
                    pos_weight = float(n_neg_train) / float(n_pos_train)

            # Train linear probe on training set
            probe, train_acc = train_linear_probe(
                X_train, y_train, num_epochs=100, learning_rate=0.01,
                device=device, weight_decay=0.01, pos_weight=pos_weight
            )
            probes[layer] = (probe, train_acc, (X_test, y_test))
            
            # Compute test metrics
            with torch.no_grad():
                X_test_tensor = torch.from_numpy(X_test).float().to(probe.device)
                logits = probe(X_test_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            preds = (probs > 0.5).astype(int)
            test_acc = compute_accuracy(y_test, preds)
            test_f1 = compute_f1(y_test, preds)
            test_mcc = compute_mcc(y_test, preds)
            
            # Compute per-class accuracies and F1 scores
            mask_class0 = y_test == 0
            mask_class1 = y_test == 1
            
            class0_acc = np.mean(preds[mask_class0] == y_test[mask_class0]) if np.any(mask_class0) else np.nan
            class1_acc = np.mean(preds[mask_class1] == y_test[mask_class1]) if np.any(mask_class1) else np.nan

            if np.any(mask_class0):
                class0_precision, class0_recall, class0_f1 = compute_precision_recall_f1(y_test, preds, positive_label=0)
            else:
                class0_precision, class0_recall, class0_f1 = np.nan, np.nan, np.nan

            if np.any(mask_class1):
                class1_precision, class1_recall, class1_f1 = compute_precision_recall_f1(y_test, preds, positive_label=1)
            else:
                class1_precision, class1_recall, class1_f1 = np.nan, np.nan, np.nan
            
            # Compute misaligned metrics
            mask_misaligned = misaligned_test == 1
            if np.any(mask_misaligned):
                misaligned_acc = compute_accuracy(y_test[mask_misaligned], preds[mask_misaligned])
                misaligned_f1 = compute_f1(y_test[mask_misaligned], preds[mask_misaligned])
                misaligned_count = np.sum(mask_misaligned)
            else:
                misaligned_acc = np.nan
                misaligned_f1 = np.nan
                misaligned_count = 0
            
            training_results.append({
                'layer': layer,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'test_mcc': test_mcc,
                'class0_acc': class0_acc,
                'class1_acc': class1_acc,
                'class0_f1': class0_f1,
                'class1_f1': class1_f1,
                'class0_precision': class0_precision,
                'class0_recall': class0_recall,
                'class1_precision': class1_precision,
                'class1_recall': class1_recall,
                'misaligned_acc': misaligned_acc,
                'misaligned_f1': misaligned_f1,
                'misaligned_count': misaligned_count,
                'train_pos': int(np.sum(y_train == 1)),
                'train_neg': int(np.sum(y_train == 0)),
                'test_pos': int(np.sum(y_test == 1)),
                'test_neg': int(np.sum(y_test == 0)),
                'test_size': len(X_test)
            })

            # Save weights if output directory is provided
            if weights_output_dir:
                save_probe_weights(probe, layer, k_position, classification_type, weights_output_dir, data_name, model_basename)

            # Log training accuracy
            tqdm.write(f"      Layer {layer}: train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}, test_f1 = {test_f1:.4f}")

        except FileNotFoundError as e:
            print(f"    ⚠ Skipping layer {layer}: File not found")
            continue
        except Exception as e:
            print(f"    ⚠ Skipping layer {layer}: {e}")
            continue

    # Log results to file if output directory is provided
    if log_output_dir and training_results:
        log_filename = f"test_results_{classification_type}_{k_position}.txt"
        log_subdir = os.path.join(log_output_dir, model_basename, data_name)
        os.makedirs(log_subdir, exist_ok=True)
        log_path = os.path.join(log_subdir, log_filename)
        
        with open(log_path, 'w') as f:
            f.write(f"Test Results Log - {classification_type} ({k_position})\n")
            f.write("=" * 90 + "\n\n")
            
            # Training accuracies
            f.write("TRAINING ACCURACY (70% of data)\n")
            f.write("-" * 90 + "\n")
            for result in sorted(training_results, key=lambda x: x['layer']):
                f.write(f"Layer {result['layer']}: {result['train_acc']:.4f}\n")
            
            f.write("\n")
            
            # Overall test accuracy and F1
            f.write("TEST PERFORMANCE - OVERALL (30% of data)\n")
            f.write("-" * 90 + "\n")
            for result in sorted(training_results, key=lambda x: x['layer']):
                f.write(
                    f"Layer {result['layer']}: Accuracy={result['test_acc']:.4f}, "
                    f"F1={result['test_f1']:.4f}, MCC={result['test_mcc']:.4f}\n"
                )

            f.write("\n")

            # Train/test class counts
            f.write("CLASS COUNTS (TRAIN/TEST)\n")
            f.write("-" * 90 + "\n")
            for result in sorted(training_results, key=lambda x: x['layer']):
                f.write(
                    f"Layer {result['layer']}: "
                    f"Train(pos={result['train_pos']}, neg={result['train_neg']}), "
                    f"Test(pos={result['test_pos']}, neg={result['test_neg']})\n"
                )
            
            f.write("\n")
            
            # Per-class accuracies and F1
            f.write("TEST PERFORMANCE - PER CLASS\n")
            f.write("-" * 90 + "\n")
            for result in sorted(training_results, key=lambda x: x['layer']):
                f.write(f"Layer {result['layer']}:\n")
                if not np.isnan(result['class0_acc']):
                    f.write(
                        f"  Class 0: Accuracy={result['class0_acc']:.4f}, "
                        f"Precision={result['class0_precision']:.4f}, "
                        f"Recall={result['class0_recall']:.4f}, "
                        f"F1={result['class0_f1']:.4f}\n"
                    )
                if not np.isnan(result['class1_acc']):
                    f.write(
                        f"  Class 1: Accuracy={result['class1_acc']:.4f}, "
                        f"Precision={result['class1_precision']:.4f}, "
                        f"Recall={result['class1_recall']:.4f}, "
                        f"F1={result['class1_f1']:.4f}\n"
                    )
            
            f.write("\n")
            
            # Misaligned samples performance
            f.write("TEST PERFORMANCE - MISALIGNED SAMPLES (Hard Cases)\n")
            f.write("-" * 90 + "\n")
            for result in sorted(training_results, key=lambda x: x['layer']):
                if result['misaligned_count'] > 0:
                    f.write(f"Layer {result['layer']} (n={result['misaligned_count']}): ")
                    if np.isnan(result['misaligned_acc']):
                        f.write("No misaligned samples in test set\n")
                    else:
                        f.write(f"Accuracy={result['misaligned_acc']:.4f}, F1={result['misaligned_f1']:.4f}\n")
                else:
                    f.write(f"Layer {result['layer']}: No misaligned samples in test set\n")
        
        print(f"✓ Saved test results log to: {log_path}")

    return probes, training_results


def evaluate_on_test_set(
    model_basename: str,
    layers: List[int],
    probes: Dict[int, Tuple[SimpleLinearProbe, float, Tuple[np.ndarray, np.ndarray]]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Evaluate probes on their respective test sets.
    Returns:
        - test_accs: Array of overall test accuracies per layer
        - class0_accs: Array of class 0 accuracies per layer
        - class1_accs: Array of class 1 accuracies per layer
        - proportions: Array of proportion predicted as necessary per layer
        - probs_dict: Dict mapping layer to sigmoid probabilities on test set
    """
    test_accs = []
    class0_accs = []
    class1_accs = []
    proportions = []
    probs_dict = {}

    for layer in layers:
        probe, _, (X_test, y_test) = probes[layer]

        # Evaluate on test set
        X = X_test
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(probe.device)
            logits = probe(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        probs_dict[layer] = probs

        # Test accuracy
        preds = (probs > 0.5).astype(int)
        test_acc = np.mean(preds == y_test)
        test_accs.append(test_acc)

        # Per-class accuracy
        mask_class0 = y_test == 0
        mask_class1 = y_test == 1

        if np.any(mask_class0):
            class0_acc = np.mean(preds[mask_class0] == y_test[mask_class0])
            class0_accs.append(class0_acc)
        else:
            class0_accs.append(np.nan)

        if np.any(mask_class1):
            class1_acc = np.mean(preds[mask_class1] == y_test[mask_class1])
            class1_accs.append(class1_acc)
        else:
            class1_accs.append(np.nan)

        # Proportion predicted as positive (necessary)
        prop = np.mean(preds)
        proportions.append(prop)

    return np.array(test_accs), np.array(class0_accs), np.array(class1_accs), np.array(proportions), probs_dict


def main():
    parser = argparse.ArgumentParser(
        description="Train linear probes for classification with train-test split and weight saving"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., 'Qwen3-8B')"
    )
    parser.add_argument(
        "--data_name", type=str, required=False, default="math_arithmetic_union",
        help="Data name (e.g., 'MetaTool' or 'math_arithmetic_union')"
    )
    parser.add_argument(
        "--classification_type", type=str, choices=["necessity", "action"],
        required=True,
        help="Classification type: 'necessity' (necessary vs unnecessary) or 'action' (called vs not_called)"
    )
    parser.add_argument(
        "--balance_clusters", action="store_true",
        help="If set, balance the 4 clusters by sampling equal amounts from each"
    )
    parser.add_argument(
        "--use_pos_weight", action="store_true",
        help=(
            "If set (and --balance_clusters is not used), apply class-frequency weighting "
            "in BCEWithLogitsLoss with pos_weight = n_neg / n_pos"
        )
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help="Train/test split ratio (default: 0.7 for 70% train / 30% test)"
    )
    parser.add_argument(
        "--save_plots", action="store_true",
        help="If set, plot the figures and save the plots as PDF files"
    )
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine output directories based on classification type
    plots_output_dir = f"probe_results_{args.classification_type}"
    weights_output_dir = f"probe_weights_{args.classification_type}"
    logs_output_dir = f"probe_results_{args.classification_type}_log"
    
    # Add seed suffix to directories if balancing is enabled
    if args.balance_clusters:
        plots_output_dir = f"probe_results_{args.classification_type}_balanced_seed{args.seed}"
        weights_output_dir = f"probe_weights_{args.classification_type}_balanced_seed{args.seed}"
        logs_output_dir = f"probe_results_{args.classification_type}_balanced_seed{args.seed}_log"

    print(f"Model: {args.model}")
    print(f"Data: {args.data_name}")
    print(f"Classification type: {args.classification_type}")
    print(f"Balance clusters: {args.balance_clusters}")
    print(f"Use pos_weight: {args.use_pos_weight}")
    print(f"Train/test split ratio: {args.train_ratio:.1%}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Plots output: {plots_output_dir}")
    print(f"Weights output: {weights_output_dir}")
    print(f"Logs output: {logs_output_dir}")
    print()

    # Discover K positions
    k_positions = discover_k_positions("clusters", args.model, args.data_name)

    print(f"Discovered K positions: {k_positions}")
    print()

    # Process each K position
    for k_position in k_positions:
        print(f"\n{'='*60}")
        print(f"Processing K position: {k_position}")
        print(f"{'='*60}")

        # Discover layers
        layers = discover_layers("clusters", args.model, args.data_name, k_position)

        print(f"Discovered layers: {layers}")

        # Train probes with specified train-test split
        print(f"Training probes on {args.train_ratio:.0%} of data ({args.train_ratio:.0%} train / {1-args.train_ratio:.0%} test split)...")
        probes, training_results = train_probes_for_position(
            args.model, args.data_name, k_position, layers,
            args.classification_type,
            device=device,
            weights_output_dir=weights_output_dir,
            log_output_dir=logs_output_dir,
            balance_clusters=args.balance_clusters,
            use_pos_weight=args.use_pos_weight,
            seed=args.seed,
            train_ratio=args.train_ratio
        )

        print(f"✓ Trained {len(probes)} probes")

        # Evaluate on test sets
        print(f"Evaluating on test sets ({1-args.train_ratio:.0%} of data)...")
        test_accs, class0_accs, class1_accs, test_props, test_probs = evaluate_on_test_set(args.model, layers, probes)

        valid_count = np.sum(~np.isnan(test_accs))
        if valid_count > 0:
            print(f"✓ Got test results for {valid_count}/{len(layers)} layers")
            print(f"  Mean test accuracy: {np.nanmean(test_accs):.4f}")
        else:
            print(f"⚠ No valid test results")



if __name__ == "__main__":
    main()
