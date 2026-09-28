"""Skip-Gram with Negative Sampling for learning semantic graph embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import networkx as nx
import requests
import zipfile
import json
import os
from typing import List, Dict, Set, Tuple
import unittest
from collections import Counter


# ============================================================================
# Utilities
# ============================================================================

def download_file(url, out_path):
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {out_path}")


def prepare_visual_genome_text(zip_url, zip_path="region_descriptions.json.zip",
                                json_path="region_descriptions.json",
                                output_path="vg_text.txt"):
    if os.path.exists(output_path):
        print(f"File {output_path} already exists.")
        return output_path
    if not os.path.exists(zip_path):
        download_file(zip_url, zip_path)
    if not os.path.exists(json_path):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    print(f"Processing {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    phrases = [region['phrase'] for img in data for region in img['regions']]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" . ".join(phrases))
    print(f"Processed {len(phrases):,} phrases")
    return output_path


def filter_punctuation_from_network(network_data, punctuation_tokens={'.', ',', '<RARE>', "'", '"', '!', '?'}):

    original_graph = network_data['graph']
    original_nodes = network_data['nodes']
    original_distance_matrix = network_data['distance_matrix']

    filtered_nodes = [n for n in original_nodes if n not in punctuation_tokens]
    old_indices = [i for i, n in enumerate(original_nodes) if n not in punctuation_tokens]
    filtered_distance_matrix = original_distance_matrix[np.ix_(old_indices, old_indices)]

    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(filtered_nodes)
    for u, v in original_graph.edges():
        if u in filtered_nodes and v in filtered_nodes:
            filtered_graph.add_edge(u, v)

    print(f"\nPUNCTUATION FILTER: {len(original_nodes)} -> {len(filtered_nodes)} nodes")
    return {**network_data, 'graph': filtered_graph, 'nodes': filtered_nodes, 'distance_matrix': filtered_distance_matrix}


# ============================================================================
# Dataset
# ============================================================================

class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, graph, nodes, distance_matrix, num_negative=15, context_size=1,
                 token_counts=None, neg_sampling_power=0.75, subsample_t=None):
        super().__init__()
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.vocab_size = len(nodes)
        self.num_negative = num_negative
        self.distance_matrix = distance_matrix

        self.contexts = self._build_contexts(context_size)
        self.context_indices = self._build_context_indices()
        self.neg_sampling_probs = self._build_neg_sampling_probs(token_counts, neg_sampling_power)
        self.keep_probs = self._build_keep_probs(token_counts, subsample_t)
        self.pairs, self.weights = self._generate_weighted_pairs()
        self._local_rng = None
        self._print_stats()

    def _build_contexts(self, context_size):
        contexts = {}
        for node in self.nodes:
            if node not in self.graph:
                contexts[node] = set()
                continue
            paths = nx.single_source_shortest_path_length(self.graph, node, cutoff=context_size)
            context_nodes = {n for n, dist in paths.items() if dist > 0 and n in self.node_to_idx}
            contexts[node] = context_nodes
        return contexts

    def _build_context_indices(self):
        context_indices = []
        for node in self.nodes:
            ctx = self.contexts.get(node, set())
            idxs = {self.node_to_idx[n] for n in ctx if n in self.node_to_idx}
            idxs.add(self.node_to_idx[node])
            context_indices.append(idxs)
        return context_indices

    def _build_neg_sampling_probs(self, token_counts, power):
        if not token_counts:
            return None
        counts = np.array([token_counts.get(node, 1) for node in self.nodes], dtype=np.float64)
        probs = np.power(counts, power)
        total = probs.sum()
        if total <= 0:
            return None
        return (probs / total).astype(np.float64)

    def _build_keep_probs(self, token_counts, subsample_t):
        if not token_counts or not subsample_t:
            return None
        total = float(sum(token_counts.values()))
        if total <= 0:
            return None
        keep_probs = []
        for node in self.nodes:
            f = max(token_counts.get(node, 1), 1) / total
            p_keep = (np.sqrt(subsample_t / f) + (subsample_t / f))
            keep_probs.append(min(1.0, p_keep))
        return np.array(keep_probs, dtype=np.float32)

    def _generate_weighted_pairs(self):
        pairs = []
        raw_distances = []
        rng = np.random.RandomState(42)
        for center_word, context_words in self.contexts.items():
            center_idx = self.node_to_idx[center_word]
            for context_word in context_words:
                context_idx = self.node_to_idx[context_word]
                if self.keep_probs is not None:
                    if rng.rand() > self.keep_probs[center_idx]:
                        continue
                    if rng.rand() > self.keep_probs[context_idx]:
                        continue
                pairs.append((center_idx, context_idx))
                raw_distances.append(self.distance_matrix[center_idx, context_idx])

        if len(pairs) == 0:
            return [], np.array([], dtype=np.float32)

        raw_distances = np.array(raw_distances, dtype=np.float32)
        max_dist = raw_distances.max()
        weights = (max_dist + 1) - raw_distances
        weights = np.sqrt(weights)
        clip_threshold = np.percentile(weights, 95) * 3
        weights = np.clip(weights, 1e-6, clip_threshold)
        weights = weights / weights.sum()
        return pairs, weights.astype(np.float32)

    def __getitem__(self, idx):
        if self._local_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            base_seed = torch.initial_seed()
            seed = base_seed + (worker_info.id if worker_info else 0)
            self._local_rng = np.random.RandomState(seed % (2**32))

        center_idx, context_idx = self.pairs[idx]
        excluded = self.context_indices[center_idx]

        negatives = self._sample_negatives(excluded)
        return np.int64(center_idx), np.int64(context_idx), negatives.astype(np.int64)

    def _sample_negatives(self, excluded):
        target = self.num_negative
        if excluded and len(excluded) >= self.vocab_size - 1:
            return self._local_rng.randint(0, self.vocab_size, size=target, dtype=np.int64)

        negatives = []
        excluded_list = list(excluded) if excluded else []
        attempts = 0
        max_attempts = 10
        while len(negatives) < target and attempts < max_attempts:
            draw_size = max(target * 4, 50)
            if self.neg_sampling_probs is None:
                samples = self._local_rng.randint(0, self.vocab_size, size=draw_size, dtype=np.int64)
            else:
                samples = self._local_rng.choice(
                    self.vocab_size, size=draw_size, replace=True, p=self.neg_sampling_probs
                ).astype(np.int64)
            if excluded_list:
                samples = samples[~np.isin(samples, excluded_list)]
            if samples.size:
                negatives.extend(samples.tolist())
            attempts += 1

        if len(negatives) < target:
            fill = self._local_rng.randint(
                0, self.vocab_size, size=(target - len(negatives)), dtype=np.int64
            )
            negatives.extend(fill.tolist())

        return np.array(negatives[:target], dtype=np.int64)

    def get_sample_weights(self):
        return self.weights

    def __len__(self):
        return len(self.pairs)

    def _print_stats(self):
        print(f"\nSkipGramDataset: {self.vocab_size} vocab, {len(self.pairs)} pairs, {self.num_negative} negatives")


# ============================================================================
# Model
# ============================================================================

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout=0.1):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_embeddings()

    def _init_embeddings(self):
        # Slightly larger initialization for better semantic separation
        bound = 1.0
        nn.init.uniform_(self.center_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.context_embeddings.weight, -bound, bound)

    def forward(self, center, context, negatives, apply_dropout=True, label_smoothing=0.0):
        center_emb = self.center_embeddings(center)
        if apply_dropout:
            center_emb = self.dropout(center_emb)

        context_emb = self.context_embeddings(context)
        negative_emb = self.context_embeddings(negatives)

        # Positive score
        pos_score = torch.sum(center_emb * context_emb, dim=1)

        # Negative scores
        neg_score = torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)

        if label_smoothing and label_smoothing > 0.0:
            pos_targets = torch.full_like(pos_score, 1.0 - label_smoothing)
            neg_targets = torch.full_like(neg_score, label_smoothing)
            pos_loss = F.binary_cross_entropy_with_logits(pos_score, pos_targets, reduction='none')
            neg_loss = F.binary_cross_entropy_with_logits(neg_score, neg_targets, reduction='none').sum(dim=1)
            return pos_loss + neg_loss

        # Standard SGNS loss
        pos_loss = F.logsigmoid(pos_score)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        return -(pos_loss + neg_loss)

    def get_embeddings(self):
        return self.center_embeddings.weight.detach().cpu().numpy()


# ============================================================================
# Training
# ============================================================================

def train_embeddings(network_data, embedding_dim=128, batch_size=512, epochs=20,
                     learning_rate=0.01, num_negative=15, validation_fraction=0.05,
                     context_size=1, dropout=0.1, weight_decay=0.0, label_smoothing=0.0,
                     patience=3, device=None, save_plot=True, neg_sampling_power=0.75,
                     max_grad_norm=None, num_workers=0, pin_memory=None,
                     persistent_workers=False, subsample_t=None, samples_per_epoch=None):

    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    network_data = filter_punctuation_from_network(network_data)
    nodes = network_data['nodes']
    graph = network_data['graph']
    distance_matrix = network_data['distance_matrix']
    token_counts = network_data.get('token_counts')

    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    split_idx = int(len(all_edges) * (1 - validation_fraction))

    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(all_edges[:split_idx])

    val_graph = nx.Graph()
    val_graph.add_nodes_from(nodes)
    val_graph.add_edges_from(all_edges[split_idx:])

    print(f"Train edges: {split_idx}, Val edges: {len(all_edges) - split_idx}")

    train_dataset = SkipGramDataset(
        train_graph, nodes, distance_matrix, num_negative, context_size,
        token_counts=token_counts, neg_sampling_power=neg_sampling_power,
        subsample_t=subsample_t
    )
    val_dataset = SkipGramDataset(
        val_graph, nodes, distance_matrix, num_negative, context_size,
        token_counts=token_counts, neg_sampling_power=neg_sampling_power,
        subsample_t=subsample_t
    )

    if len(train_dataset) == 0:
        print("ERROR: No training pairs!")
        return None

    num_samples = samples_per_epoch or len(train_dataset)
    sampler = WeightedRandomSampler(train_dataset.get_sample_weights(), num_samples, replacement=True)
    if pin_memory is None:
        pin_memory = (device.startswith("cuda") if isinstance(device, str) else device.type == "cuda")
    use_persistent = persistent_workers and num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )

    model = SkipGramModel(len(nodes), embedding_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\nTraining: {len(nodes)} vocab, {embedding_dim} dim, lr={learning_rate}")

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for centers, contexts, negs in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            loss = model(centers, contexts, negs, label_smoothing=label_smoothing).mean()
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for centers, contexts, negs in val_loader:
                centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
                total_val_loss += model(
                    centers, contexts, negs, False, label_smoothing=label_smoothing
                ).mean().item()

        val_loss = total_val_loss / max(len(val_loader), 1)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
            torch.save({'model_state_dict': best_model_state, 'nodes': nodes,
                       'vocab_size': len(nodes), 'embedding_dim': embedding_dim}, "best_model.pth")
            print(f"  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'o-', label='Train')
        plt.plot(val_losses, 's-', label='Validation')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.savefig('training_loss.png', dpi=150)
        plt.close()

    return {'nodes': nodes, 'embeddings': model.get_embeddings(), 'model': model,
            'train_losses': train_losses, 'val_losses': val_losses}


# ============================================================================
# Grid search
# ============================================================================

def run_skipgram_grid_search(network_data, base_config, grid, out_dir="grid_runs", seed=42):
    """
    Run a deterministic hyperparameter grid search for Skip-Gram training.
    Returns a pandas DataFrame when pandas is available; otherwise returns a list of dicts.
    """
    import itertools
    import time
    import datetime
    import traceback
    import platform
    import csv
    import random
    import shutil
    import sys
    from contextlib import redirect_stdout, redirect_stderr

    if not isinstance(base_config, dict):
        raise ValueError("base_config must be a dict")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("grid must be a non-empty dict of parameter lists")

    for key, values in grid.items():
        if key not in base_config:
            raise ValueError(f"Grid key '{key}' not found in base_config")
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            raise ValueError(f"Grid values for '{key}' must be a non-empty list")

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    grid_keys = list(grid.keys())
    grid_values = [grid[key] for key in grid_keys]
    combinations = list(itertools.product(*grid_values))
    total_runs = len(combinations)
    width = max(4, len(str(total_runs)))

    abbrev = {
        "learning_rate": "lr",
        "context_size": "ctx",
        "num_negative": "neg",
        "label_smoothing": "ls",
    }
    results_rows = []
    run_records = []
    durations = []

    def _format_value(value):
        text = str(value)
        return text.replace(os.sep, "_")

    def _format_suffix(config):
        parts = []
        for key in grid_keys:
            short = abbrev.get(key, key)
            parts.append(f"{short}{_format_value(config[key])}")
        return "_".join(parts)

    def _format_duration(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))

    def _set_seed(seed_value):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    def _hardware_info():
        info = {
            "cpu": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        return info

    def _load_json(path):
        with open(path, "r") as f:
            return json.load(f)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
            for stream in self.streams:
                stream.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

        def isatty(self):
            return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def _build_row(run_id, config, metrics):
        return {
            "run_id": run_id,
            "embedding_dim": config.get("embedding_dim"),
            "context_size": config.get("context_size"),
            "num_negative": config.get("num_negative"),
            "learning_rate": config.get("learning_rate"),
            "label_smoothing": config.get("label_smoothing"),
            "best_val_loss": metrics.get("best_val_loss"),
            "best_epoch": metrics.get("best_epoch"),
            "train_loss_last": metrics.get("train_loss_last"),
            "duration_seconds": metrics.get("duration_seconds"),
            "saved_checkpoint_path": metrics.get("saved_checkpoint_path"),
        }

    pbar = tqdm(combinations, total=total_runs, desc="Grid search", unit="run")
    eval_keys = ["similarity_examples", "analogy_examples", "cluster_seeds"]
    for run_idx, combo in enumerate(pbar, start=1):
        grid_override = dict(zip(grid_keys, combo))
        run_config = dict(base_config)
        run_config.update(grid_override)
        train_config = {k: v for k, v in run_config.items() if k not in eval_keys}

        run_id = f"run_{run_idx:0{width}d}"
        run_suffix = _format_suffix(run_config)
        run_dir = os.path.join(out_dir, f"{run_id}_{run_suffix}")
        os.makedirs(run_dir, exist_ok=True)

        metrics_path = os.path.join(run_dir, "metrics.json")
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(metrics_path):
            config = _load_json(config_path) if os.path.exists(config_path) else {}
            metrics = _load_json(metrics_path)
            results_rows.append(_build_row(run_id, config, metrics))
            run_records.append({
                "run_id": run_id,
                "run_dir": run_dir,
                "config": config,
                "metrics": metrics,
                "skipped": True,
            })
            if metrics.get("duration_seconds"):
                durations.append(metrics["duration_seconds"])
            tqdm.write(f"[{run_idx}/{total_runs}] {run_id} already complete; skipping.")
            continue

        with open(config_path, "w") as f:
            json.dump(run_config, f, indent=2, sort_keys=True)

        avg = (sum(durations) / len(durations)) if durations else None
        if avg:
            eta = avg * (total_runs - run_idx + 1)
            tqdm.write(f"[{run_idx}/{total_runs}] {run_id} avg {avg:.1f}s/run ETA {_format_duration(eta)}")
        else:
            tqdm.write(f"[{run_idx}/{total_runs}] {run_id} starting (estimating runtime)")

        log_path = os.path.join(run_dir, "training_log.txt")
        start_time = time.time()
        try:
            _set_seed(seed)
            original_cwd = os.getcwd()
            try:
                os.chdir(run_dir)
                with open(log_path, "w", buffering=1) as log_f:
                    stdout_tee = _Tee(sys.stdout, log_f)
                    stderr_tee = _Tee(sys.stderr, log_f)
                    with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                        results = train_embeddings(network_data=network_data, **train_config)
            finally:
                os.chdir(original_cwd)

            if results is None:
                raise RuntimeError("train_embeddings returned None")

            duration = time.time() - start_time
            durations.append(duration)
            train_losses = results.get("train_losses", [])
            val_losses = results.get("val_losses", [])
            final_epoch = len(train_losses)
            best_epoch = int(np.argmin(val_losses) + 1) if val_losses else None
            best_val_loss = float(min(val_losses)) if val_losses else None
            train_loss_last = float(train_losses[-1]) if train_losses else None

            default_ckpt = os.path.join(run_dir, "best_model.pth")
            saved_ckpt = None
            if os.path.exists(default_ckpt):
                saved_ckpt = os.path.join(run_dir, f"best_model_{run_id}.pth")
                shutil.copy2(default_ckpt, saved_ckpt)

            nn_eval_path = None
            eval_kwargs = {k: run_config[k] for k in eval_keys if k in run_config}
            if eval_kwargs:
                nn_eval_path = os.path.join(run_dir, "nearest_neighbors.txt")
                with open(nn_eval_path, "w") as nn_f, redirect_stdout(nn_f), redirect_stderr(nn_f):
                    analyze_embeddings(results["nodes"], results["embeddings"], **eval_kwargs)

            metrics = {
                "run_id": run_id,
                "final_epoch": final_epoch,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_loss_last": train_loss_last,
                "duration_seconds": duration,
                "saved_checkpoint_path": saved_ckpt,
                "seed": seed,
                "hardware": _hardware_info(),
            }
            if nn_eval_path:
                metrics["nearest_neighbors_path"] = nn_eval_path
                metrics["nearest_neighbors_config"] = eval_kwargs

            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)

            results_rows.append(_build_row(run_id, run_config, metrics))
            run_records.append({
                "run_id": run_id,
                "run_dir": run_dir,
                "config": run_config,
                "metrics": metrics,
            })
            del results
            tqdm.write(f"[{run_idx}/{total_runs}] {run_id} done in {duration:.1f}s")
        except Exception:
            duration = time.time() - start_time
            error_path = os.path.join(run_dir, "error.txt")
            with open(error_path, "w") as f:
                f.write(traceback.format_exc())
            run_records.append({
                "run_id": run_id,
                "run_dir": run_dir,
                "config": run_config,
                "error": "Run failed. See error.txt for details.",
                "duration_seconds": duration,
            })
            tqdm.write(f"[{run_idx}/{total_runs}] {run_id} failed; see {error_path}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    columns = [
        "run_id",
        "embedding_dim",
        "context_size",
        "num_negative",
        "learning_rate",
        "label_smoothing",
        "best_val_loss",
        "best_epoch",
        "train_loss_last",
        "duration_seconds",
        "saved_checkpoint_path",
    ]

    results_csv = os.path.join(out_dir, "results.csv")
    results_json = os.path.join(out_dir, "results.json")
    try:
        import pandas as pd
        results_df = pd.DataFrame(results_rows, columns=columns)
        results_df.to_csv(results_csv, index=False)
        results_df.to_json(results_json, orient="records", indent=2)
    except Exception:
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in results_rows:
                writer.writerow(row)
        with open(results_json, "w") as f:
            json.dump(results_rows, f, indent=2)
        results_df = results_rows

    best_run = None
    best_loss = float("inf")
    for record in run_records:
        metrics = record.get("metrics")
        if not metrics:
            continue
        loss = metrics.get("best_val_loss")
        if loss is None:
            continue
        if loss < best_loss:
            best_loss = loss
            best_run = record

    best_run_path = os.path.join(out_dir, "best_run.json")
    if best_run:
        best_payload = {
            "run_id": best_run["run_id"],
            "run_dir": best_run["run_dir"],
            "config": best_run["config"],
            "metrics": best_run["metrics"],
        }
        with open(best_run_path, "w") as f:
            json.dump(best_payload, f, indent=2, sort_keys=True)
        print(f"Best run: {best_run['run_id']} (val_loss={best_loss:.6f})")
        print(f"Best checkpoint: {best_run['metrics'].get('saved_checkpoint_path')}")
    else:
        with open(best_run_path, "w") as f:
            json.dump({"error": "No successful runs completed."}, f, indent=2)
        print("No successful runs completed; best_run.json written with error info.")

    return results_df


# ============================================================================
# Analysis
# ============================================================================

def find_similar_words(word, nodes, embeddings, top_k=10):
    if word not in nodes:
        return []
    idx = nodes.index(word)
    target = embeddings[idx]
    norms = np.linalg.norm(embeddings, axis=1)
    target_norm = np.linalg.norm(target)
    sims = (embeddings @ target) / (norms * target_norm + 1e-10)
    top_idx = np.argsort(-sims)[1:top_k+1]
    return [(nodes[i], float(sims[i])) for i in top_idx]


def solve_analogy(word_a, word_b, word_c, nodes, embeddings, top_k=5):
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if not all(w in node_to_idx for w in [word_a, word_b, word_c]):
        return []
    target = embeddings[node_to_idx[word_b]] - embeddings[node_to_idx[word_a]] + embeddings[node_to_idx[word_c]]
    norms = np.linalg.norm(embeddings, axis=1)
    target_norm = np.linalg.norm(target)
    sims = (embeddings @ target) / (norms * target_norm + 1e-10)
    exclude = {node_to_idx[w] for w in [word_a, word_b, word_c]}
    return [(nodes[i], float(sims[i])) for i in np.argsort(-sims) if i not in exclude][:top_k]


def visualize_embeddings(nodes, embeddings, output_file="embeddings_tsne.png", sample_size=200, annotate=True):
    n = min(sample_size, len(nodes))
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n-1))
    proj = tsne.fit_transform(embeddings[:n])
    plt.figure(figsize=(14, 14))
    plt.scatter(proj[:, 0], proj[:, 1], s=40, alpha=0.6)
    if annotate:
        for i, word in enumerate(nodes[:n]):
            plt.annotate(word, (proj[i, 0], proj[i, 1]), fontsize=8)
    plt.title(f"t-SNE of {n} Word Embeddings")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE to {output_file}")
    plt.close()


def analyze_embeddings(nodes, embeddings, similarity_examples=None, analogy_examples=None, cluster_seeds=None):
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)

    print(f"\nVocabulary: {len(nodes):,}  Embedding dim: {embeddings.shape[1]}")

    sample = embeddings[:min(100, len(embeddings))]
    norms = np.linalg.norm(sample, axis=1, keepdims=True)
    normalized = sample / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T
    sim_vals = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]

    print(f"\nSimilarity stats (100 word sample):")
    print(f"  Mean: {sim_vals.mean():.4f}  Std: {sim_vals.std():.4f}")
    print(f"  Min: {sim_vals.min():.4f}  Max: {sim_vals.max():.4f}")

    if similarity_examples:
        print("\n" + "="*80)
        print("NEAREST NEIGHBORS")
        print("="*80)
        for word in similarity_examples:
            similar = find_similar_words(word, nodes, embeddings, top_k=8)
            print(f"\nMost similar to '{word}':")
            if not similar:
                print("  (not in vocabulary)")
            else:
                for token, score in similar:
                    print(f"  {token:15s}  similarity={score:.4f}")

    if analogy_examples:
        print("\n" + "="*80)
        print("WORD ANALOGIES (a:b :: c:?)")
        print("="*80)
        for a, b, c in analogy_examples:
            results = solve_analogy(a, b, c, nodes, embeddings, top_k=3)
            print(f"\n{a}:{b} :: {c}:?")
            if results:
                for token, score in results:
                    print(f"  {token:15s}  score={score:.4f}")
            else:
                print("  (words not in vocabulary)")

    if cluster_seeds:
        print("\n" + "="*80)
        print("SEMANTIC CLUSTERS")
        print("="*80)
        for seed in cluster_seeds:
            if seed in nodes:
                cluster = find_similar_words(seed, nodes, embeddings, top_k=5)
                print(f"\n'{seed}': {', '.join([w for w, _ in cluster])}")

    print("\n" + "="*80)


# ============================================================================
# Tests
# ============================================================================

class TestSkipGramDataset(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.nodes = ['a', 'b', 'c', 'd', 'e']
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from([('a', 'b'), ('b', 'c'), ('c', 'd'), ('d', 'e')])
        self.distance_matrix = np.array([[0,1,2,3,4],[1,0,1,2,3],[2,1,0,1,2],[3,2,1,0,1],[4,3,2,1,0]], dtype=float)

    def test_dataset_creation(self):
        ds = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=2)
        self.assertEqual(ds.vocab_size, 5)
        self.assertGreater(len(ds), 0)

    def test_contexts_built(self):
        ds = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=2)
        self.assertEqual(len(ds.contexts), 5)
        self.assertIn('b', ds.contexts['a'])

    def test_getitem(self):
        ds = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=3)
        center, context, negs = ds[0]
        self.assertEqual(len(negs), 3)


class TestSkipGramModel(unittest.TestCase):
    def setUp(self):
        self.model = SkipGramModel(100, 32)

    def test_forward(self):
        center = torch.randint(0, 100, (8,))
        context = torch.randint(0, 100, (8,))
        negs = torch.randint(0, 100, (8, 5))
        loss = self.model(center, context, negs)
        self.assertEqual(loss.shape, (8,))

    def test_get_embeddings(self):
        emb = self.model.get_embeddings()
        self.assertEqual(emb.shape, (100, 32))


class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.graph = nx.karate_club_graph()
        self.graph = nx.relabel_nodes(self.graph, {n: str(n) for n in self.graph.nodes()})
        self.nodes = list(self.graph.nodes())
        n = len(self.nodes)
        self.distance_matrix = np.random.rand(n, n)
        self.distance_matrix = (self.distance_matrix + self.distance_matrix.T) / 2
        np.fill_diagonal(self.distance_matrix, 0)

    def test_full_pipeline(self):
        ds = SkipGramDataset(self.graph, self.nodes, self.distance_matrix, num_negative=5)
        model = SkipGramModel(len(self.nodes), 16)
        center, context, negs = ds[0]
        loss = model(torch.tensor([center]), torch.tensor([context]), torch.tensor([negs]))
        self.assertEqual(loss.shape, (1,))


def run_tests():
    print("="*70 + "\nRUNNING TESTS\n" + "="*70)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print("\n" + ("ALL TESTS PASSED!" if result.wasSuccessful() else "SOME TESTS FAILED!"))
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()
