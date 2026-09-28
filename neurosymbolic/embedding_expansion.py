"""Evolutionary expansion of a learned semantic embedding space with new concepts."""

import torch
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set, Union

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision
import pickle

try:
    from .semantic_embeddings import SkipGramModel, find_similar_words
except Exception as e:
    raise ImportError("semantic_embeddings is required for embedding expansion.") from e

try:
    from .text_graph import process_text_network
except Exception:
    process_text_network = None

import unittest
import tempfile
import os


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_trained_model(model_path: str, vocab_size: int,
                       embedding_dim: int, dropout: float) -> Tuple[torch.nn.Module, np.ndarray]:
    """Load trained Skip-Gram model and extract embeddings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)

    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        embeddings_tensor = model.get_embeddings()
        embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor)
                     else embeddings_tensor).astype(np.float32)

    print(f"✓ Loaded model: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return model, embeddings


def load_trained_model_from_checkpoint(model_path: str, dropout: float = 0.1) -> Tuple[torch.nn.Module, np.ndarray, List[str]]:
    """Load checkpoint and return model, embeddings, and nodes list."""
    checkpoint = torch.load(model_path, map_location="cpu")
    nodes = checkpoint.get("nodes")
    if nodes is None:
        raise KeyError("Checkpoint missing 'nodes' for vocabulary ordering.")
    vocab_size = checkpoint.get("vocab_size", len(nodes))
    embedding_dim = checkpoint.get("embedding_dim")
    if embedding_dim is None:
        raise KeyError("Checkpoint missing 'embedding_dim'.")
    model, embeddings = load_trained_model(model_path, vocab_size, embedding_dim, dropout)
    return model, embeddings, nodes

def create_mappings(nodes: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, np.ndarray]]:
    """Create word-to-index and index-to-word mappings."""
    word_to_idx = {word: idx for idx, word in enumerate(nodes)}
    idx_to_word = {idx: word for idx, word in enumerate(nodes)}
    return word_to_idx, idx_to_word


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics needed for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'global_std': np.std(embeddings)
    }


def get_cifar100_vocabulary() -> List[str]:
    """Download CIFAR-100 and extract class names."""
    print("\nLoading CIFAR-100 vocabulary...")
    def find_cifar_root(start: str, max_levels: int = 3) -> Optional[str]:
        cur = os.path.abspath(start)
        for _ in range(max_levels + 1):
            if os.path.exists(os.path.join(cur, "cifar-100-python")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None

    env_root = os.environ.get("CIFAR100_ROOT")
    local_root = find_cifar_root(os.getcwd())
    root = env_root or local_root
    if root and os.path.exists(os.path.join(root, "cifar-100-python")):
        dataset = torchvision.datasets.CIFAR100(root=root, train=True, download=False)
    else:
        dataset = torchvision.datasets.CIFAR100(root="./cifar100_data", train=True, download=True)
    print(f"✓ CIFAR-100 vocabulary loaded: {len(dataset.classes)} classes")
    return dataset.classes


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "to", "for", "from", "with",
    "at", "by", "is", "are", "was", "were", "be", "been", "this", "that", "these",
    "those", "it", "its", "as", "into", "over", "under", "above", "below", "near",
    "behind", "front", "left", "right", "up", "down", "off", "out", "across"
}


def build_token_counts(text_file: str, vocab_set: Optional[Set[str]] = None) -> Counter:
    """Count token frequencies in a corpus file for negative sampling and weighting."""
    counts = Counter()
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            tokens = re.findall(r"\b[a-z]+\b", line.lower())
            if vocab_set is not None:
                tokens = [t for t in tokens if t in vocab_set]
            if tokens:
                counts.update(tokens)
    return counts


def build_neg_sampling_probs(vocab_list: List[str], token_counts: Counter,
                             power: float = 0.75) -> Optional[np.ndarray]:
    """Create a unigram^power distribution aligned with vocab_list."""
    if not token_counts:
        return None
    counts = np.array([token_counts.get(w, 1) for w in vocab_list], dtype=np.float64)
    probs = np.power(counts, power)
    total = probs.sum()
    if total <= 0:
        return None
    return (probs / total).astype(np.float64)


def build_idf(token_counts: Counter) -> Dict[str, float]:
    """Compute IDF-style weights from token counts."""
    total = float(sum(token_counts.values()))
    if total <= 0:
        return {}
    idf = {w: np.log((1.0 + total) / (1.0 + c)) for w, c in token_counts.items()}
    if idf:
        mean_idf = sum(idf.values()) / len(idf)
        if mean_idf > 0:
            idf = {w: v / mean_idf for w, v in idf.items()}
    return idf

def analyze_vocabulary_overlap(cifar_vocab: List[str], network_vocab: List[str]) -> List[str]:
    """Analyze overlap between CIFAR-100 and network vocabulary."""
    cifar_set, network_set = set(cifar_vocab), set(network_vocab)
    overlapping = sorted(list(cifar_set.intersection(network_set)))
    missing = sorted(list(cifar_set - network_set))

    print(f"\n{'='*70}")
    print("VOCABULARY OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"CIFAR-100 vocabulary: {len(cifar_set)} classes")
    print(f"Network vocabulary: {len(network_set)} words")
    print(f"Overlapping words: {len(overlapping)} ({len(overlapping)/len(cifar_set)*100:.1f}%)")
    print(f"Missing from network: {len(missing)}")
    if overlapping:
        print(f"\nFound: {', '.join(overlapping)}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
    print(f"{'='*70}\n")

    return missing


def build_anchor_map(target_words: List[str], vocab_set: Set[str]) -> Dict[str, List[str]]:
    """
    Build simple anchor lists for target words using subword heuristics.
    Anchors are existing vocab words that likely relate to the target word.
    """
    anchors = {}
    for word in target_words:
        parts = [p for p in re.split(r"[-_ ]+", word) if p]
        candidates = []
        for p in parts:
            if p in vocab_set:
                candidates.append(p)
            if p.endswith("s") and p[:-1] in vocab_set:
                candidates.append(p[:-1])
            if p.endswith("es") and p[:-2] in vocab_set:
                candidates.append(p[:-2])

        filtered = [c for c in candidates if c in vocab_set]
        if filtered:
            anchors[word] = list(dict.fromkeys(filtered))
    return anchors


def _load_cifar_fine_to_coarse() -> Dict[str, str]:
    """Load CIFAR-100 fine -> coarse label mapping from local dataset if available."""
    def find_cifar_root(start: str, max_levels: int = 3) -> Optional[str]:
        cur = os.path.abspath(start)
        for _ in range(max_levels + 1):
            if os.path.exists(os.path.join(cur, "cifar-100-python")):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return None

    env_root = os.environ.get("CIFAR100_ROOT")
    local_root = find_cifar_root(os.getcwd())
    root = env_root or local_root
    if not root:
        return {}

    meta_path = os.path.join(root, "cifar-100-python", "meta")
    train_path = os.path.join(root, "cifar-100-python", "train")
    if not (os.path.exists(meta_path) and os.path.exists(train_path)):
        return {}

    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    fine_names = [n.decode("utf-8") for n in meta[b"fine_label_names"]]
    coarse_names = [n.decode("utf-8") for n in meta[b"coarse_label_names"]]

    with open(train_path, "rb") as f:
        train = pickle.load(f, encoding="bytes")
    fine_labels = train[b"fine_labels"]
    coarse_labels = train[b"coarse_labels"]

    mapping_idx = {}
    for f_idx, c_idx in zip(fine_labels, coarse_labels):
        mapping_idx[f_idx] = c_idx

    return {fine_names[f_idx]: coarse_names[c_idx] for f_idx, c_idx in mapping_idx.items()}


COARSE_ANCHOR_MAP = {
    "aquatic mammals": ["water", "sea", "ocean", "river", "lake", "fish", "animal", "blue"],
    "fish": ["fish", "water", "sea", "ocean", "river", "lake", "animal", "blue"],
    "flowers": ["flower", "plant", "petal", "yellow", "red", "pink"],
    "food containers": ["bowl", "cup", "plate", "bottle", "container"],
    "fruit and vegetables": ["fruit", "food", "plant", "leaf", "green"],
    "household electrical devices": ["phone", "keyboard", "computer", "clock", "television", "screen"],
    "household furniture": ["table", "chair", "bed", "couch", "sofa"],
    "insects": ["insect", "bug", "wing", "flower", "yellow", "black", "animal"],
    "large carnivores": ["animal", "bear", "cat", "dog", "lion", "tiger"],
    "large man-made outdoor things": ["building", "road", "bridge", "street", "tower", "wall"],
    "large natural outdoor scenes": ["mountain", "grass", "tree", "field", "sky", "cloud"],
    "large omnivores and herbivores": ["animal", "horse", "cow", "elephant", "giraffe", "grass"],
    "medium-sized mammals": ["animal", "dog", "cat", "horse", "rabbit", "fox"],
    "non-insect invertebrates": ["animal", "water", "shell", "sea"],
    "people": ["person", "man", "woman", "people"],
    "reptiles": ["animal", "tail", "leg", "snake", "lizard"],
    "small mammals": ["animal", "mouse", "cat", "dog", "rabbit", "squirrel"],
    "trees": ["tree", "plant", "leaf", "branch"],
    "vehicles 1": ["vehicle", "car", "truck", "bus", "train"],
    "vehicles 2": ["vehicle", "car", "truck", "bus", "train", "plane", "boat"],
}


SPECIAL_ANCHORS = {
    "aquarium_fish": ["fish", "water", "sea", "ocean", "animal", "blue", "tank", "glass"],
    "bee": ["insect", "bug", "wing", "flower", "yellow", "black", "animal"],
    "beaver": ["animal", "fur", "tail", "water", "river", "wood", "tree"],
    "flatfish": ["fish", "water", "animal", "sea", "ocean"],
    "ray": ["fish", "water", "animal", "sea", "ocean"],
    "shark": ["fish", "water", "animal", "sea", "ocean"],
    "whale": ["water", "animal", "sea", "ocean"],
    "dolphin": ["water", "animal", "sea", "ocean"],
    "seal": ["water", "animal", "sea", "ocean"],
    "otter": ["water", "animal", "fur", "river"],
    "trout": ["fish", "water", "river", "sea", "animal"],
    "turtle": ["animal", "water", "sea", "shell"],
    "crab": ["animal", "sea", "water", "shell"],
    "lobster": ["animal", "sea", "water", "shell"],
}


def _heuristic_anchors(word: str) -> List[str]:
    anchors = []
    token = word.lower()
    if "tree" in token:
        anchors += ["tree", "plant", "leaf", "branch"]
    if "fish" in token:
        anchors += ["fish", "water", "sea", "ocean", "animal"]
    if any(k in token for k in ["shark", "whale", "dolphin", "seal", "otter", "ray", "trout"]):
        anchors += ["water", "sea", "ocean", "animal", "fish"]
    if any(k in token for k in ["flower", "tulip", "rose", "orchid", "poppy", "sunflower"]):
        anchors += ["flower", "plant", "petal", "yellow", "red"]
    if any(k in token for k in ["truck", "car", "bus", "train", "streetcar", "bicycle", "motorcycle", "tractor", "tank"]):
        anchors += ["vehicle", "car", "truck", "bus", "train"]
    if any(k in token for k in ["plane", "airplane", "rocket"]):
        anchors += ["plane", "airplane", "sky"]
    if any(k in token for k in ["snake", "lizard", "turtle", "crocodile", "dinosaur"]):
        anchors += ["animal", "tail", "leg"]
    if any(k in token for k in ["beetle", "butterfly", "caterpillar", "cockroach", "bee", "spider", "worm", "snail"]):
        anchors += ["animal", "insect", "bug", "wing", "flower", "yellow", "black"]
    if any(k in token for k in ["pear", "apple", "orange", "mushroom", "pepper"]):
        anchors += ["fruit", "food", "plant", "green"]
    if any(k in token for k in ["telephone", "keyboard", "lamp", "clock", "television"]):
        anchors += ["phone", "keyboard", "computer", "lamp", "clock"]
    return anchors


def build_semantic_anchor_map(
    target_words: List[str],
    vocab_set: Set[str],
    fine_to_coarse: Optional[Dict[str, str]] = None,
) -> Dict[str, List[str]]:
    anchors = {}
    for word in target_words:
        candidates = []
        if word in SPECIAL_ANCHORS:
            candidates.extend(SPECIAL_ANCHORS[word])

        if fine_to_coarse and word in fine_to_coarse:
            coarse = fine_to_coarse[word]
            candidates.extend(COARSE_ANCHOR_MAP.get(coarse, []))

        candidates.extend(_heuristic_anchors(word))

        parts = [p for p in re.split(r"[-_ ]+", word) if p]
        for p in parts:
            candidates.append(p)
            if p.endswith("s"):
                candidates.append(p[:-1])
            if p.endswith("es"):
                candidates.append(p[:-2])

        filtered = [c for c in candidates if c in vocab_set]
        if filtered:
            anchors[word] = list(dict.fromkeys(filtered))
    return anchors


SYNTHETIC_TEMPLATES = [
    "{target} {a} {b} {c}",
    "{a} {target} {b} {c}",
    "{a} {b} {target} {c}",
    "{a} {b} {c} {target}",
]


def build_synthetic_sentences(
    target_words: List[str],
    anchor_map: Dict[str, List[str]],
    vocab_set: Set[str],
    sentences_per_word: int = 6,
    anchors_per_sentence: int = 4,
    seed: int = 42,
) -> List[str]:
    """Generate lightweight synthetic sentences to enrich missing-word contexts."""
    rng = np.random.RandomState(seed)
    sentences: List[str] = []
    for word in target_words:
        anchors = [a for a in anchor_map.get(word, []) if a in vocab_set and a not in STOPWORDS]
        if len(anchors) < 2:
            continue
        target = word.replace("_", " ")
        for _ in range(sentences_per_word):
            k = min(anchors_per_sentence, len(anchors))
            sample = list(rng.choice(anchors, size=k, replace=len(anchors) < k))
            while len(sample) < 4:
                sample.append(sample[len(sample) % k])
            a, b, c, d = sample[:4]
            template = SYNTHETIC_TEMPLATES[rng.randint(0, len(SYNTHETIC_TEMPLATES))]
            sentences.append(template.format(target=target, a=a, b=b, c=c, d=d))
    return sentences


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5,
    return_counts: bool = False,
    extra_lines: Optional[List[str]] = None,
    extra_weight: float = 1.0
) -> Union[Dict[str, Counter], Tuple[Dict[str, Counter], Dict[str, int]]]:
    """
    Extract co-occurrence context statistics for target words from a text corpus.

    This function reads a corpus file line-by-line and tracks which words appear
    near specified target words. For each target word, it counts how many times
    each vocabulary word appears within a window around it.

    Args:
        text_file: Path to the corpus text file to analyze.
        target_words: List of words to extract contexts for.
        vocab_set: Set of valid vocabulary words (only count these as contexts).
        window: Number of words to look on each side of the target word.
        extra_lines: Optional list of synthetic text lines to inject.
        extra_weight: Weight multiplier for contexts from extra_lines.

    Returns:
        If return_counts is False (default):
            A dictionary mapping each target word to a Counter of context words and
            their frequencies.
        If return_counts is True:
            Tuple of (contexts, target_counts) where target_counts maps each target
            word to its occurrence count in the corpus.

    Example:
        >>> extract_word_contexts('corpus.txt', ['king', 'queen'], vocab, window=2)
        {'king': Counter({'royal': 5, 'crown': 3}),
         'queen': Counter({'royal': 4, 'throne': 2})}

    Implementation guidelines:
    --------------------------
    1. Initialize a dictionary `{word: Counter()}` for each target word.
    2. Convert `target_words` to a set for fast lookup.
    3. Stream through the file line-by-line (efficient for large corpora).
    4. For each line:
        - Tokenize using lowercase alphabetic words (regex: r"\\b[a-z]+\\b").
        - For each token that matches a target word:
            * Extract up to `window` tokens on both sides.
            * Exclude the target word itself.
            * Retain only context words that appear in `vocab_set`.
            * Skip very common stopwords to reduce noise.
            * Update the Counter for that target word.
    5. Handle edge cases: empty lines, start/end of token lists.
    6. Optionally print progress (e.g., every 50,000 lines) for user feedback.
    7. Return the dictionary of Counters.
    """

    contexts = {word: Counter() for word in target_words}
    target_counts = {word: 0.0 for word in target_words}
    parts_map = {word: [p for p in re.split(r"[-_ ]+", word) if p] for word in target_words}
    single_targets = {w for w, parts in parts_map.items() if len(parts) == 1}
    phrase_targets = {w: parts for w, parts in parts_map.items() if len(parts) > 1}
    phrase_first = {}
    for word, parts in phrase_targets.items():
        phrase_first.setdefault(parts[0], []).append((word, parts))

    def _process_tokens(tokens: List[str], weight: float) -> None:
        for i, tok in enumerate(tokens):
            if tok in single_targets:
                target_counts[tok] += weight
                left = max(0, i - window)
                right = min(len(tokens), i + window + 1)
                for j in range(left, right):
                    if j == i:
                        continue
                    ctx = tokens[j]
                    if ctx in STOPWORDS:
                        continue
                    if ctx in vocab_set:
                        contexts[tok][ctx] += weight

            if tok in phrase_first:
                for word, parts in phrase_first[tok]:
                    end = i + len(parts)
                    if end > len(tokens):
                        continue
                    if tokens[i:end] != parts:
                        continue
                    target_counts[word] += weight
                    left = max(0, i - window)
                    right = min(len(tokens), end + window)
                    for j in range(left, right):
                        if i <= j < end:
                            continue
                        ctx = tokens[j]
                        if ctx in STOPWORDS:
                            continue
                        if ctx in vocab_set:
                            contexts[word][ctx] += weight

    with open(text_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            tokens = re.findall(r"\b[a-z]+\b", line.lower())
            if not tokens:
                continue
            _process_tokens(tokens, 1.0)
            if line_idx % 50000 == 0:
                print(f"  processed {line_idx:,} lines...")

    if extra_lines:
        for line in extra_lines:
            tokens = re.findall(r"\b[a-z]+\b", line.lower())
            if not tokens:
                continue
            _process_tokens(tokens, float(extra_weight))

    if return_counts:
        return contexts, target_counts
    return contexts






# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    stats_dict: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute a three-term fitness score for a candidate word embedding vector.

    This function evaluates how well a candidate vector fits the learned
    embedding space by combining three complementary metrics:
    1. Corpus likelihood (how well it predicts observed contexts)
    2. Norm matching (how similar its magnitude is to typical embeddings)
    3. Anchor similarity (how similar it is to known reference words)

    Args:
        vec: Candidate embedding vector to evaluate.
        word: Target word (for reference, not used in computation).
        ctx_vecs: Context word vectors that co-occur with the target word.
                  Shape: (n_contexts, embedding_dim). May be None if no contexts.
        ctx_weights: Weights for each context (e.g., co-occurrence counts).
                     Shape: (n_contexts,). May be None if no contexts.
        neg_vecs: Negative sample vectors (words that don't co-occur).
                  Shape: (n_negatives, embedding_dim).
        anchor_vecs: Pre-normalized vectors of anchor words for comparison.
                     Shape: (n_anchors, embedding_dim). May be None.
        stats_dict: Dictionary containing embedding statistics:
                    - 'mean_norm': Average L2 norm of embeddings in the space
                    - 'std_norm': Standard deviation of embedding norms
                    - 'global_std': Global standard deviation (if needed)
        weights: Dictionary of weights for each fitness component:
                 - 'corpus': Weight for corpus likelihood term
                 - 'norm': Weight for norm matching term
                 - 'anchor': Weight for anchor similarity term

    Returns:
        Combined fitness score in the range [0, 1], where higher is better.

    Example:
        >>> vec = np.array([0.5, -0.3, 0.8, 0.1])
        >>> stats = {'mean_norm': 1.0, 'std_norm': 0.2, 'global_std': 0.5}
        >>> weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        >>> fitness = compute_fitness(vec, 'king', ctx_vecs, ctx_weights,
        ...                           neg_vecs, anchor_vecs, stats, weights)
        >>> print(f"Fitness: {fitness:.4f}")
        Fitness: 0.7234

    Implementation guidelines:
    --------------------------
    Term 1 - Corpus Likelihood (L_corpus_norm):
        - For positive contexts: sum over ctx_weights * log(sigmoid(ctx_vecs · vec))

        - For negative samples: sum over log(sigmoid(-neg_vecs · vec))
        - Add small epsilon (1e-10) inside log for numerical stability
        - Normalize by total samples, then apply sigmoid to map to [0, 1]
        - Default to 0.5 if no samples available

    Term 2 - Norm Match (S_norm):
        - Compute L2 norm of the candidate vector
        - Use Gaussian similarity: exp(-((norm - mean_norm)² / (2 * std_norm²)))
        - This rewards vectors with norms close to the typical embedding norm

    Term 3 - Anchor Similarity (S_anchor):
        - Normalize the candidate vector (divide by its norm + epsilon)
        - Compute dot products with all anchor vectors (they're pre-normalized)
        - Take the mean similarity across all anchors
        - Default to 0.5 if no anchors provided

    Final score:
        - Weighted sum: weights['corpus'] * L_corpus_norm +
                       weights['norm'] * S_norm +
                       weights['anchor'] * S_anchor

    Notes:
        - Handle None values for optional parameters (ctx_vecs, ctx_weights, anchor_vecs)
        - Use vectorized NumPy operations for efficiency
        - Add small epsilon values to prevent division by zero
    """
    eps = 1e-10
    vec = vec.astype(np.float32, copy=False)

    # Term 1: corpus likelihood
    corpus_term = None
    pos_term = 0.0
    neg_term = 0.0
    denom = 0

    if ctx_vecs is not None and ctx_weights is not None and len(ctx_vecs) > 0:
        pos_scores = np.dot(ctx_vecs, vec)
        pos_term = np.sum(ctx_weights * np.log(sigmoid(pos_scores) + eps))
        denom += len(ctx_vecs)

    if neg_vecs is not None and len(neg_vecs) > 0:
        neg_scores = np.dot(neg_vecs, vec)
        neg_term = np.sum(np.log(sigmoid(-neg_scores) + eps))
        denom += len(neg_vecs)

    if denom > 0:
        corpus_term = (pos_term + neg_term) / denom
        L_corpus_norm = sigmoid(corpus_term)
    else:
        L_corpus_norm = 0.5

    # Term 2: norm match
    vec_norm = np.linalg.norm(vec)
    std_norm = max(stats_dict.get("std_norm", 0.0), eps)
    mean_norm = stats_dict.get("mean_norm", 0.0)
    S_norm = np.exp(-((vec_norm - mean_norm) ** 2) / (2 * (std_norm ** 2)))

    # Term 3: anchor similarity
    if anchor_vecs is None or len(anchor_vecs) == 0:
        S_anchor = 0.5
    else:
        vec_unit = vec / (vec_norm + eps)
        anchor_sim = np.mean(np.dot(anchor_vecs, vec_unit))
        S_anchor = (anchor_sim + 1.0) / 2.0

    w_corpus = weights.get("corpus", 0.5)
    w_norm = weights.get("norm", 0.3)
    w_anchor = weights.get("anchor", 0.2)
    w_sum = w_corpus + w_norm + w_anchor
    if w_sum <= 0:
        return 0.0

    score = (w_corpus * L_corpus_norm + w_norm * S_norm + w_anchor * S_anchor) / w_sum
    return float(np.clip(score, 0.0, 1.0))


# ============================================================================
# GENETIC ALGORITHM (1+λ) EVOLUTION STRATEGY
# ============================================================================

def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    anchors: Optional[Dict[str, List[str]]] = None
) -> np.ndarray:
    """
    Initialize an embedding vector for a word using corpus bootstrap.

    This function creates an initial embedding by computing a weighted average
    of the embeddings of words that frequently co-occur with the target word.
    This provides a data-driven starting point that places the new word near
    semantically related words in the embedding space.

    Args:
        word: Target word to initialize an embedding for.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
        anchors: Optional anchor map to use when no contexts are available.

    Returns:
        Initial embedding vector for the word. Shape: (embedding_dim,).

    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30, 'castle': 20})}
        >>> embeddings = np.random.randn(1000, 300)  # 1000 words, 300 dims
        >>> word_to_idx = {'queen': 0, 'royal': 1, 'castle': 2, ...}
        >>> vec = initialize_embedding('king', contexts, embeddings, word_to_idx)
        >>> vec.shape
        (300,)

    Implementation guidelines:
    --------------------------
    1. Handle the no-context case:
       - If the word has no contexts (empty Counter), return the mean of all
         embeddings as a neutral starting point

    2. Get top context words:
       - Extract the top 20 most frequent context words using Counter.most_common()
       - This focuses on the strongest statistical relationships

    3. Compute weighted average:
       - Calculate the total weight (sum of all counts)
       - For each context word that exists in word_to_idx:
           * Get its embedding vector
           * Weight it by (count / weight_sum)
           * Add to running sum

    4. Validate the result:
       - Check if the resulting vector has non-zero norm
       - If zero (e.g., no valid context words found), fall back to mean embedding

    Notes:
        - Some context words may not be in word_to_idx; skip these
        - The weighted average naturally places the new word near its contexts
        - Using top 20 contexts balances informativeness with noise reduction
    """
    if word not in contexts or not contexts[word]:
        if anchors and word in anchors:
            anchor_vecs = [embeddings[word_to_idx[a]] for a in anchors[word] if a in word_to_idx]
            if anchor_vecs:
                return np.mean(anchor_vecs, axis=0).astype(np.float32)
        return embeddings.mean(axis=0)

    max_contexts = 40
    top_contexts = contexts[word].most_common(max_contexts)
    total = sum(np.log1p(count) for _, count in top_contexts)
    if total <= 0:
        return embeddings.mean(axis=0)

    vec = np.zeros(embeddings.shape[1], dtype=np.float32)
    for ctx_word, count in top_contexts:
        if ctx_word not in word_to_idx:
            continue
        weight = np.log1p(count) / total
        vec += weight * embeddings[word_to_idx[ctx_word]]

    if np.linalg.norm(vec) <= 1e-12:
        return embeddings.mean(axis=0)
    return vec


def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    anchors: Dict[str, List[str]],
    num_negatives: int = 15,
    context_weighting: str = "log",
    idf: Optional[Dict[str, float]] = None,
    neg_sampling_probs: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Precompute all vectors needed for fitness evaluation.

    This function extracts and prepares the three types of vectors used in
    fitness computation: positive context vectors, negative sample vectors,
    and anchor vectors. Precomputing these vectors once improves efficiency
    when evaluating fitness multiple times during optimization.

    Args:
        word: Target word being optimized.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
        vocab_list: List of all vocabulary words (for negative sampling).
        anchors: Dictionary mapping words to lists of semantically related anchor words.
        num_negatives: Number of negative samples to draw (default: 15).
        context_weighting: "log" (default), "count", or "tfidf" for context weights.
        idf: Optional IDF weights for context words (used with "tfidf").
        neg_sampling_probs: Optional unigram^power distribution aligned to vocab_list.

    Returns:
        Tuple of (ctx_vecs, ctx_weights, neg_vecs, anchor_vecs):
        - ctx_vecs: Context word embeddings. Shape: (n_contexts, dim) or None.
        - ctx_weights: Normalized context weights. Shape: (n_contexts,) or None.
        - neg_vecs: Negative sample embeddings. Shape: (num_negatives, dim).
        - anchor_vecs: Normalized anchor embeddings. Shape: (n_anchors, dim) or None.

    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30})}
        >>> anchors = {'king': ['queen', 'monarch', 'ruler']}
        >>> ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
        ...     'king', contexts, embeddings, word_to_idx, vocab_list, anchors
        ... )
        >>> ctx_v.shape  # Positive contexts
        (2, 300)
        >>> neg_v.shape  # Negative samples
        (15, 300)

    Implementation guidelines:
    --------------------------
    Part 1 - Positive Context Vectors:
        - Initialize ctx_vecs and ctx_weights to None (for no-context case)
        - If the word has contexts:
            * Iterate through contexts[word].items()
            * For each context word that exists in word_to_idx:
              - Collect its embedding vector
              - Collect its count
            * If any valid contexts found:
              - Convert lists to numpy arrays
              - Normalize weights to sum to 1.0

    Part 2 - Negative Sample Vectors:
        - Randomly sample num_negatives words from vocab_list (without replacement)
        - Look up their embeddings and stack into an array
        - Shape should be (num_negatives, embedding_dim)

    Part 3 - Anchor Vectors:
        - Initialize anchor_vecs to None (for no-anchor case)
        - If the word has anchors defined:
            * Filter to only anchors that exist in word_to_idx
            * If any valid anchors found:
              - Collect their embeddings into an array
              - Normalize each vector to unit length (L2 norm = 1)
              - Use np.linalg.norm with axis=1, keepdims=True
              - Add small epsilon (1e-10) to prevent division by zero

    Notes:
        - Handle missing words gracefully (skip if not in word_to_idx)
        - Return None for optional components if no valid data available
        - Negative samples should be random to avoid bias
        - Anchor normalization enables direct cosine similarity via dot product
    """
    ctx_vecs = None
    ctx_weights = None

    if word in contexts and contexts[word]:
        ctx_list = []
        weights = []
        max_contexts = 80
        for ctx_word, count in contexts[word].most_common(max_contexts):
            if ctx_word not in word_to_idx:
                continue
            ctx_list.append(embeddings[word_to_idx[ctx_word]])
            if context_weighting == "tfidf" and idf is not None:
                weight = float(np.log1p(count) * idf.get(ctx_word, 1.0))
            elif context_weighting == "count":
                weight = float(count)
            else:
                weight = float(np.log1p(count))
            weights.append(weight)
        if ctx_list:
            ctx_vecs = np.vstack(ctx_list)
            ctx_weights = np.array(weights, dtype=np.float32)
            ctx_weights = ctx_weights / (ctx_weights.sum() + 1e-10)

    if num_negatives <= 0:
        neg_vecs = np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    elif neg_sampling_probs is not None:
        probs = neg_sampling_probs
        if len(probs) != len(vocab_list):
            if len(probs) < len(vocab_list):
                pad = np.full(len(vocab_list) - len(probs), probs.min() * 0.01)
                probs = np.concatenate([probs, pad])
            else:
                probs = probs[:len(vocab_list)]
            probs = probs / probs.sum()
        neg_idx = np.random.choice(len(vocab_list), size=num_negatives, replace=True, p=probs)
        neg_vecs = embeddings[neg_idx]
    else:
        replace = len(vocab_list) < num_negatives
        neg_words = np.random.choice(vocab_list, size=num_negatives, replace=replace)
        neg_vecs = np.vstack([embeddings[word_to_idx[w]] for w in neg_words])

    anchor_vecs = None
    anchor_raw_vecs = None
    if word in anchors:
        anchor_list = [w for w in anchors[word] if w in word_to_idx]
        if anchor_list:
            anchor_raw_vecs = np.vstack([embeddings[word_to_idx[w]] for w in anchor_list])
            norms = np.linalg.norm(anchor_raw_vecs, axis=1, keepdims=True) + 1e-10
            anchor_vecs = anchor_raw_vecs / norms

    if (ctx_vecs is None or len(ctx_vecs) == 0) and anchor_raw_vecs is not None:
        ctx_vecs = anchor_raw_vecs
        ctx_weights = np.full(len(anchor_raw_vecs), 1.0 / len(anchor_raw_vecs), dtype=np.float32)

    return ctx_vecs, ctx_weights, neg_vecs, anchor_vecs


def evolve_embedding(word: str, contexts: Dict[str, Counter],
                    embeddings: np.ndarray, word_to_idx: Dict[str, int],
                    vocab_list: List[str], stats_dict: Dict[str, float],
                    anchors: Dict[str, List[str]], config: Dict,
                    idf: Optional[Dict[str, float]] = None,
                    neg_sampling_probs: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Evolve a single word embedding using (1+λ) Evolution Strategy.

    Args:
        word: Target word to insert
        contexts: Context word counts for all target words
        embeddings: Existing embedding matrix
        word_to_idx: Word to index mapping
        vocab_list: List of vocabulary words
        stats_dict: Embedding statistics
        anchors: Anchor words for semantic guidance
        config: Configuration dictionary

    Returns:
        Optimized embedding vector
    """
    print(f"\n  Evolving: '{word}'", end='')

    dim = embeddings.shape[1]
    base_sigma = config['ga_mutation_factor'] * stats_dict['global_std']
    min_sigma_ratio = config.get("ga_min_sigma_ratio", 0.25)
    sigma_decay = config.get("ga_sigma_decay", 0.5)

    # Initialize
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx, anchors)

    # Precompute vectors
    ctx_total = sum(contexts.get(word, {}).values()) if contexts and word in contexts else 0
    num_negatives = config.get("num_negatives", 15)
    if ctx_total == 0:
        num_negatives = int(config.get("num_negatives_zero_ctx", min(5, num_negatives)))
    low_threshold = config.get("low_context_threshold", 0)
    context_weighting = config.get("context_weighting", "log")
    if ctx_total == 0 or (low_threshold and ctx_total < low_threshold):
        context_weighting = config.get("context_weighting_low_ctx", "log")
    ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        word,
        contexts,
        embeddings,
        word_to_idx,
        vocab_list,
        anchors,
        num_negatives=num_negatives,
        context_weighting=context_weighting,
        idf=idf,
        neg_sampling_probs=neg_sampling_probs,
    )

    anchor_list = anchors.get(word) if anchors else None
    anchor_mean = None
    if anchor_list:
        anchor_vecs_raw = [embeddings[word_to_idx[a]] for a in anchor_list if a in word_to_idx]
        if anchor_vecs_raw:
            anchor_mean = np.mean(anchor_vecs_raw, axis=0).astype(np.float32)
    if (ctx_vecs is None or len(ctx_vecs) == 0) and (anchor_vecs is None or len(anchor_vecs) == 0):
        print(" (no contexts/anchors) -> mean")
        return best_vec

    if ctx_total == 0:
        if anchor_list:
            anchor_weight = float(config.get("anchor_weight_zero_ctx", 0.8))
            fitness_weights = {
                "corpus": max(0.0, 1.0 - anchor_weight - 0.1),
                "norm": 0.1,
                "anchor": anchor_weight,
            }
        else:
            fitness_weights = {"corpus": 0.1, "norm": 0.9, "anchor": 0.0}
    elif low_threshold and ctx_total < low_threshold and anchor_list:
        anchor_weight = float(config.get("anchor_weight_low_ctx", 0.6))
        fitness_weights = {
            "corpus": max(0.0, 1.0 - anchor_weight - 0.1),
            "norm": 0.1,
            "anchor": anchor_weight,
        }
    else:
        fitness_weights = dict(config["fitness_weights"])
        if not anchor_list:
            fitness_weights["anchor"] = 0.0

    if anchor_mean is not None and ctx_total > 0:
        blend_low = float(config.get("anchor_blend_low", 0.40))
        blend_high = float(config.get("anchor_blend_high", 0.20))
        blend = blend_low if (low_threshold and ctx_total < low_threshold) else blend_high
        best_vec = (1.0 - blend) * best_vec + blend * anchor_mean

    # Initial fitness
    best_fit = compute_fitness(best_vec, word, ctx_vecs, ctx_weights, neg_vecs,
                               anchor_vecs, stats_dict, fitness_weights)

    # Evolution loop
    total_gens = max(1, config['ga_generations'])
    sigma_scale = 1.0
    if ctx_total == 0:
        sigma_scale = float(config.get("sigma_scale_zero_ctx", 0.25))
    elif low_threshold and ctx_total < low_threshold:
        sigma_scale = float(config.get("sigma_scale_low_ctx", 0.6))
    effective_sigma = base_sigma * sigma_scale
    min_sigma = effective_sigma * min_sigma_ratio
    for gen in range(total_gens):
        frac = gen / max(1, total_gens - 1)
        mutation_sigma = max(min_sigma, effective_sigma * (1.0 - sigma_decay * frac))
        # Generate offspring and evaluate
        population = best_vec + np.random.normal(0, mutation_sigma, (config['ga_pop_size'], dim))
        all_candidates = np.vstack([best_vec, population])

        fitness_scores = [compute_fitness(vec, word, ctx_vecs, ctx_weights, neg_vecs,
                                         anchor_vecs, stats_dict, fitness_weights)
                         for vec in all_candidates]

        # Select best
        best_idx = np.argmax(fitness_scores)
        best_vec = all_candidates[best_idx].copy()
        best_fit = fitness_scores[best_idx]

        if gen % 50 == 0:
            print(f" G{gen}={best_fit:.4f}", end='')

    postprocess = config.get("postprocess_norm", False)
    if postprocess:
        mean_norm = stats_dict.get("mean_norm", None)
        norm_clip_ratio = config.get("norm_clip_ratio", 0.35)
        if mean_norm:
            vec_norm = np.linalg.norm(best_vec)
            if vec_norm > 1e-10:
                scale = mean_norm / vec_norm
                if norm_clip_ratio is not None:
                    scale = float(np.clip(scale, 1.0 - norm_clip_ratio, 1.0 + norm_clip_ratio))
                best_vec = best_vec * scale

    print(f" ✓ Final={best_fit:.4f}")
    return best_vec


def insert_cifar_embeddings(
    base_embeddings: np.ndarray,
    base_vocab: List[str],
    cifar_vocab: List[str],
    text_file: str,
    config: Dict,
    anchors: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True
) -> Tuple[List[str], np.ndarray]:
    """
    Insert CIFAR-100 words into an existing embedding space using GA optimization.
    Returns updated vocab list and embeddings matrix.
    """
    vocab_set = set(base_vocab)
    missing_words = analyze_vocabulary_overlap(cifar_vocab, base_vocab)
    if not missing_words:
        return base_vocab, base_embeddings

    if anchors is None:
        anchors = build_anchor_map(missing_words, vocab_set)

    fine_to_coarse = _load_cifar_fine_to_coarse()
    semantic_anchor_map = build_semantic_anchor_map(
        missing_words,
        vocab_set,
        fine_to_coarse=fine_to_coarse,
    )

    synthetic_lines = None
    if config.get("enable_synthetic_contexts", True):
        synth_anchor_map: Dict[str, List[str]] = {}
        for w in missing_words:
            combined: List[str] = []
            combined.extend(semantic_anchor_map.get(w, []))
            if anchors:
                combined.extend(anchors.get(w, []))
            if combined:
                synth_anchor_map[w] = list(dict.fromkeys(combined))
        synthetic_lines = build_synthetic_sentences(
            missing_words,
            synth_anchor_map,
            vocab_set,
            sentences_per_word=int(config.get("synthetic_sentences_per_word", 8)),
            anchors_per_sentence=int(config.get("synthetic_anchors_per_sentence", 4)),
            seed=int(config.get("synthetic_seed", 42)),
        )
        if verbose:
            print(f"✓ Synthetic lines: {len(synthetic_lines)}")

    contexts, target_counts = extract_word_contexts(
        text_file=text_file,
        target_words=missing_words,
        vocab_set=vocab_set,
        window=config["context_window"],
        return_counts=True,
        extra_lines=synthetic_lines,
        extra_weight=float(config.get("synthetic_weight", 8.0)),
    )
    token_counts = None
    idf = None
    neg_sampling_probs = None
    if config.get("use_token_counts", True) or config.get("context_weighting") == "tfidf":
        token_counts = build_token_counts(text_file, vocab_set=vocab_set)
    if token_counts:
        neg_sampling_probs = build_neg_sampling_probs(
            list(base_vocab),
            token_counts,
            power=float(config.get("neg_sampling_power", 0.75)),
        )
        if config.get("context_weighting") == "tfidf":
            idf = build_idf(token_counts)

    stats = compute_embedding_stats(base_embeddings)
    word_to_idx = {w: i for i, w in enumerate(base_vocab)}
    vocab_list = list(base_vocab)
    embeddings = base_embeddings.copy()
    anchor_context_boost = int(config.get("anchor_context_boost", 0))
    low_ctx_threshold = int(config.get("low_context_threshold", 0))
    if anchor_context_boost > 0:
        for word, anchors_for_word in semantic_anchor_map.items():
            if not anchors_for_word:
                continue
            cur_count = target_counts.get(word, 0)
            if cur_count == 0 or (low_ctx_threshold and cur_count < low_ctx_threshold):
                for anchor in anchors_for_word:
                    contexts[word][anchor] += anchor_context_boost
                target_counts[word] = cur_count + anchor_context_boost * len(anchors_for_word)
    insertion_order = sorted(
        missing_words,
        key=lambda w: target_counts.get(w, 0),
        reverse=True,
    )

    inserted = []
    for word in insertion_order:
        dynamic_vocab = set(vocab_list)
        dynamic_anchors = build_semantic_anchor_map(
            [word],
            dynamic_vocab,
            fine_to_coarse=fine_to_coarse,
        )
        merged = []
        if semantic_anchor_map.get(word):
            merged.extend(semantic_anchor_map[word])
        if anchors:
            merged.extend(anchors.get(word, []))
        merged.extend(dynamic_anchors.get(word, []))
        if config.get("link_coarse_words", True) and fine_to_coarse:
            coarse = fine_to_coarse.get(word)
            if coarse:
                same_coarse = [w for w in inserted if fine_to_coarse.get(w) == coarse]
                if same_coarse:
                    merged.extend(same_coarse)
        if merged:
            merged = list(dict.fromkeys(merged))
            dynamic_anchors[word] = merged
        elif anchors or semantic_anchor_map.get(word):
            dynamic_anchors[word] = list(dict.fromkeys((anchors.get(word, []) if anchors else []) + semantic_anchor_map.get(word, [])))
        if verbose:
            print(f"\n[INSERT] {word}")
        new_vec = evolve_embedding(
            word,
            contexts,
            embeddings,
            word_to_idx,
            vocab_list,
            stats,
            dynamic_anchors,
            config,
            idf=idf,
            neg_sampling_probs=neg_sampling_probs,
        )
        embeddings = np.vstack([embeddings, new_vec.astype(np.float32)])
        vocab_list.append(word)
        word_to_idx[word] = len(vocab_list) - 1
        inserted.append(word)

    if verbose:
        print(f"\n✓ Inserted {len(inserted)} new words.")
    return vocab_list, embeddings


def build_task5_embeddings(
    text_source: str = "vg_text.txt",
    base_checkpoint: str = "best_model.pth",
    output_checkpoint: str = "models/semantic_embeddings.pth",
    dropout: float = 0.1,
    config: Optional[Dict] = None,
    verbose: bool = True,
) -> str:
    """
    Full embedding expansion pipeline: load trained Skip-Gram, insert CIFAR-100 words, save checkpoint.
    """
    if config is None:
        config = {
            "ga_pop_size": 30,
            "ga_generations": 200,
            "ga_mutation_factor": 0.10,
            "context_window": 7,
            "fitness_weights": {"corpus": 0.60, "norm": 0.20, "anchor": 0.20},
            "low_context_threshold": 10,
            "num_negatives": 20,
            "ga_min_sigma_ratio": 0.25,
            "ga_sigma_decay": 0.6,
            "anchor_blend_low": 0.40,
            "anchor_blend_high": 0.20,
            "context_weighting": "tfidf",
            "context_weighting_low_ctx": "log",
            "neg_sampling_power": 0.75,
            "use_token_counts": True,
            "postprocess_norm": True,
            "norm_clip_ratio": 0.35,
            "anchor_context_boost": 30,
            "num_negatives_zero_ctx": 0,
            "anchor_weight_zero_ctx": 0.95,
            "anchor_weight_low_ctx": 0.7,
            "sigma_scale_zero_ctx": 0.25,
            "sigma_scale_low_ctx": 0.6,
            "enable_synthetic_contexts": True,
            "synthetic_sentences_per_word": 8,
            "synthetic_anchors_per_sentence": 4,
            "synthetic_weight": 8.0,
            "synthetic_seed": 42,
            "link_coarse_words": True,
        }

    if verbose:
        print("\n[STEP 1] Load trained model ")
    model, embeddings, nodes = load_trained_model_from_checkpoint(base_checkpoint, dropout)

    if verbose:
        print("\n[STEP 2] Insert CIFAR-100 words ")
    cifar_vocab = get_cifar100_vocabulary()
    updated_vocab, updated_embeddings = insert_cifar_embeddings(
        embeddings, nodes, cifar_vocab, text_source, config, verbose=verbose
    )

    word2idx = {w: i for i, w in enumerate(updated_vocab)}
    torch.save(
        {"word2idx": word2idx, "embeddings": torch.tensor(updated_embeddings, dtype=torch.float32)},
        output_checkpoint,
    )
    if verbose:
        print(f"\n✓ Saved embedding expansion checkpoint: {output_checkpoint}")
    return output_checkpoint


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_with_inserted_words(nodes: List[str], embeddings: np.ndarray,
                                  inserted_words: List[str],
                                  output_file: str = "embeddings_with_inserted.png",
                                  sample_size: int = 500):
    """Create t-SNE visualization highlighting inserted words."""
    print("\nGenerating t-SNE visualization with inserted words...")

    num_original = len(nodes) - len(inserted_words)
    inserted_indices = set(range(num_original, len(nodes)))

    # Sample: prioritize inserted words + random original
    if len(nodes) > sample_size:
        sample_indices = list(inserted_indices) + list(np.random.choice(
            num_original, min(sample_size - len(inserted_words), num_original), replace=False))
    else:
        sample_indices = list(range(len(nodes)))

    selected_embeddings = embeddings[sample_indices]
    selected_nodes = [nodes[i] for i in sample_indices]

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices)-1))
    projection = tsne.fit_transform(selected_embeddings)

    # Plot
    plt.figure(figsize=(14, 14))

    for i in range(len(projection)):
        is_inserted = sample_indices[i] in inserted_indices
        plt.scatter(projection[i, 0], projection[i, 1],
                   s=200 if is_inserted else 40,
                   alpha=1.0 if is_inserted else 0.6,
                   c='red' if is_inserted else 'steelblue')
        plt.annotate(selected_nodes[i], (projection[i, 0], projection[i, 1]),
                    fontsize=11 if is_inserted else 9,
                    alpha=1.0 if is_inserted else 0.8,
                    fontweight='bold' if is_inserted else 'normal')

    plt.title(f"t-SNE Visualization: {len(sample_indices)} Words "
              f"({sum(1 for i in sample_indices if i in inserted_indices)} Inserted)",
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE to {output_file}")
    plt.show()


def run_sanity_checks(model: torch.nn.Module, embeddings: np.ndarray,
                     nodes: List[str], word_to_idx: Dict[str, int]):
    """Run comprehensive sanity checks on loaded model and embeddings."""
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)

    print(f"\n1. Model Configuration:")
    print(f"   Training mode: {model.training}")
    print(f"   Device: {next(model.parameters()).device}")

    print(f"\n2. Embedding Quality:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")
    print(f"   Contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}")

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n3. Embedding Norms:")
    print(f"   Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")

    print(f"\n4. Vocabulary Test:")
    for test_word in ['man', 'woman', 'dog', 'car', 'blue']:
        if test_word in word_to_idx:
            word_idx = word_to_idx[test_word]
            print(f"   '{test_word:10s}' → idx={word_idx:4d}, norm={np.linalg.norm(embeddings[word_idx]):.4f}")
            similar = find_similar_words(test_word, nodes, embeddings, top_k=5)
            if similar:
                print(f"      Similar: {', '.join([f'{w}({s:.3f})' for w, s in similar])}")

    print("\n" + "="*70)
    print("✓ SANITY CHECKS COMPLETE")
    print("="*70)



"""
Don't forget to add tests!!!!
"""

def run_tests():
    """Run all unit tests."""
    class EmbeddingExpansionTests(unittest.TestCase):
        def test_extract_word_contexts(self):
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
                tf.write("cat dog cat mouse\n")
                tf_path = tf.name
            try:
                contexts = extract_word_contexts(
                    tf_path, ["cat"], {"cat", "dog", "mouse"}, window=1
                )
                self.assertEqual(contexts["cat"]["dog"], 2)
                self.assertEqual(contexts["cat"]["mouse"], 1)
            finally:
                os.remove(tf_path)

        def test_compute_fitness_range(self):
            vec = np.array([0.1, -0.2, 0.3], dtype=np.float32)
            ctx_vecs = np.array([[0.1, 0.0, 0.1]], dtype=np.float32)
            ctx_weights = np.array([1.0], dtype=np.float32)
            neg_vecs = np.array([[0.0, 0.2, -0.1]], dtype=np.float32)
            anchor_vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
            stats = {"mean_norm": 0.3, "std_norm": 0.2, "global_std": 0.1}
            weights = {"corpus": 0.5, "norm": 0.3, "anchor": 0.2}
            score = compute_fitness(vec, "x", ctx_vecs, ctx_weights, neg_vecs, anchor_vecs, stats, weights)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        def test_initialize_embedding(self):
            embeddings = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
            contexts = {"new": Counter({"a": 2, "b": 1})}
            word_to_idx = {"a": 0, "b": 1}
            vec = initialize_embedding("new", contexts, embeddings, word_to_idx)
            self.assertEqual(vec.shape, (2,))

        def test_precompute_fitness_vectors(self):
            embeddings = np.eye(3, dtype=np.float32)
            contexts = {"x": Counter({"a": 2})}
            word_to_idx = {"a": 0, "b": 1, "c": 2}
            vocab_list = ["a", "b", "c"]
            anchors = {"x": ["b"]}
            ctx_vecs, ctx_weights, neg_vecs, anchor_vecs = precompute_fitness_vectors(
                "x", contexts, embeddings, word_to_idx, vocab_list, anchors, num_negatives=2
            )
            self.assertEqual(ctx_vecs.shape[1], 3)
            self.assertEqual(neg_vecs.shape[1], 3)
            self.assertEqual(anchor_vecs.shape[1], 3)

    suite = unittest.TestLoader().loadTestsFromTestCase(EmbeddingExpansionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
