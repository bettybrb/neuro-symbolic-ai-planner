# Neuro-Symbolic AI Planner

An end-to-end neuro-symbolic AI system combining computer vision, learned semantic representations, evolutionary optimisation, and symbolic planning.

The project connects neural perception with symbolic reasoning: images are mapped into a shared semantic embedding space, recognised concepts are incorporated into a planning representation, and a PDDL-based planner searches for actions that transform an initial state into a requested goal state.

## System Overview

The system integrates three major AI components:

1. **Computer Vision** - a MobileNetV3-based encoder processes CIFAR-100 images and projects visual features into a semantic embedding space.
2. **Natural Language Processing** - Skip-Gram with Negative Sampling learns distributed word representations from textual relationships.
3. **Symbolic AI** - a PDDL-based planner represents predicates, actions and states and searches for valid action sequences.

These components form a neuro-symbolic pipeline in which learned representations from neural models can be used by an explicit symbolic reasoning system.

## Key Components

### Skip-Gram Semantic Embeddings

The project implements Skip-Gram with Negative Sampling in PyTorch to learn semantic representations. The embedding pipeline includes graph-based context construction, negative sampling, frequency-aware sampling and embedding visualisation.

### Evolutionary Embedding Expansion

A (1+lambda) evolutionary strategy is used to introduce previously unseen concepts into an existing embedding space while attempting to preserve its learned semantic structure. This allows CIFAR-100 concepts to be aligned with the language representation.

### CIFAR-100 Visual Encoder

The visual component uses a MobileNetV3-Small feature extractor with a trainable projection network. Image features are projected into the semantic representation space so that visual concepts can be associated with learned word embeddings.

### Symbolic Planning

The symbolic component implements PDDL-style predicates, actions and states together with action grounding and search-based planning. Preconditions and effects determine how actions transform the current state toward a specified goal.

### End-to-End Neuro-Symbolic Pipeline

The final pipeline accepts image or textual input, identifies the corresponding concept, constructs the symbolic planning problem, searches for a plan and validates the resulting action sequence.

## Technologies

- Python
- PyTorch
- Torchvision
- MobileNetV3
- NumPy
- NetworkX
- scikit-learn
- Matplotlib
- PDDL
- evolutionary optimisation

## Repository Structure

- `cw2.py` - end-to-end neuro-symbolic integration and planning pipeline
- `lab2.py` - graph and preprocessing utilities
- `lab6.py` - Skip-Gram with Negative Sampling
- `lab7.py` - evolutionary embedding expansion
- `lab8.py` - CIFAR-100 visual encoder and projection model
- `lab9.py` - PDDL representation and symbolic planning
- `best_skipgram_523words.pth` - trained semantic embedding checkpoint
- `best_cifar100_projection.pth` - trained visual projection checkpoint
- `requirements.txt` - Python dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Concepts Demonstrated

- Neuro-symbolic AI
- Representation learning
- Word embeddings
- Skip-Gram with Negative Sampling
- Transfer learning
- Computer vision
- Multimodal representation alignment
- Evolutionary optimisation
- State-space search
- PDDL parsing and planning
- A* search
- End-to-end AI system integration

## Motivation

Neural models are effective at learning representations from complex inputs, while symbolic systems provide explicit and interpretable reasoning. This project explores how these approaches can be connected: neural models provide semantic perception and symbolic planning uses those representations to reason about actions and goals.
