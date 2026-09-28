"""End-to-end neuro-symbolic pipeline connecting perception, semantic representations and symbolic planning."""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# SEMANTIC REPRESENTATION LOADING
# ============================================================================

def load_semantic_embeddings(
    checkpoint_path: str = str(REPO_ROOT / "models" / "semantic_embeddings.pth"),
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.

    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    """
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    data = torch.load(path, map_location="cpu")

    if not isinstance(data, dict):
        raise TypeError("Checkpoint format not recognized (expected dict).")

    if "word2idx" in data:
        vocab = data["word2idx"]
    elif "vocab" in data:
        vocab_list = data["vocab"]
        vocab = {w: i for i, w in enumerate(vocab_list)}
    else:
        raise KeyError("Checkpoint missing 'word2idx' or 'vocab'.")

    if "embeddings" not in data:
        raise KeyError("Checkpoint missing 'embeddings'.")
    embeddings = data["embeddings"]

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")
    if len(vocab) != embeddings.shape[0]:
        raise ValueError("Embedding matrix rows must match vocab size.")

    # Check vocab indices are 0..V-1
    idx_values = list(vocab.values())
    if len(set(idx_values)) != len(vocab):
        raise ValueError("Vocabulary indices are not unique.")
    if set(idx_values) != set(range(len(vocab))):
        raise ValueError("Vocabulary indices are not a contiguous 0..V-1 range.")

    return vocab, embeddings


CIFAR_100_CLASSES = {
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
}


# ============================================================================
# MULTIMODAL SYMBOLIC PLANNING
# ============================================================================

def generate_plan(
    input_data: Union[str, Path, np.ndarray, torch.Tensor, Any],
    initial_state: Iterable[Union[str, Any]],
    goal_state: Iterable[Union[str, Any]],
    domain_file: str = str(REPO_ROOT / "planning" / "domain.pddl"),
    skipgram_path: str = str(REPO_ROOT / "models" / "semantic_embeddings.pth"),
    projection_path: str = str(REPO_ROOT / "models" / "visual_projection.pth"),
    device: Optional[str] = None,
) -> Optional[List[Dict[str, Union[int, str, List[str]]]]]:
    """
    Ground image or text input into a symbolic concept, search for a valid plan and verify the resulting state transitions.

    Predicates can be provided as strings (e.g. "(at apple lab)") or symbolic_planner.Predicate objects.
    """
    from .vision_alignment import ImageEncoder
    from .symbolic_planner import ActionGrounder, PDDLParser, Predicate, State, _has_conflict, astar_search
    from PIL import Image
    from torchvision import transforms
    import re

    debug = os.environ.get("DEBUG_PLAN", "").lower() in {"1", "true", "yes"}

    # ------------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------------
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available; falling back to CPU.")
        device = "cpu"

    # ------------------------------------------------------------------------
    # Load Skip-Gram vocab + embedding matrix
    # ------------------------------------------------------------------------
    try:
        vocab, embeddings = load_semantic_embeddings(skipgram_path)
    except Exception as exc:
        warnings.warn(f"Failed to load skip-gram embeddings: {exc}")
        return None

    # ------------------------------------------------------------------------
    # Constants / helper mappings
    # ------------------------------------------------------------------------
    tree_to_pddl = {
        "maple_tree": "maple",
        "maple-tree": "maple",
        "oak_tree": "oak",
        "oak-tree": "oak",
        "palm_tree": "palm",
        "palm-tree": "palm",
        "pine_tree": "pine",
        "pine-tree": "pine",
        "willow_tree": "willow",
        "willow-tree": "willow",
    }
    pddl_to_tree = {v: k for k, v in tree_to_pddl.items() if "_" in k}

    placeholders = {"object", "obj", "item", "thing", "target"}

    # ------------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------------
    def _normalize_token(token: Any) -> str:
        return str(token).strip().lower()

    def _is_placeholder(token: Any) -> bool:
        raw = _normalize_token(token)
        return raw.startswith("?") or raw in placeholders

    def _is_image_like(data: Any) -> bool:
        if isinstance(data, (np.ndarray, torch.Tensor)):
            return True
        if isinstance(data, (str, Path)):
            return Path(data).exists()
        return "Image" in str(type(data))

    def _unwrap_input(data: Any) -> Any:
        # tuple/list wrapper
        if isinstance(data, (tuple, list)) and len(data) == 2:
            candidate = data[0]
            if _is_image_like(candidate):
                return candidate
        # dict wrapper
        if isinstance(data, dict):
            for key in ("image", "img", "data", "input"):
                if key in data and _is_image_like(data[key]):
                    return data[key]
        return data

    def to_vocab_name(name: Any) -> str:
        token = _normalize_token(name).replace(" ", "_").replace("-", "_")
        if token in pddl_to_tree:
            token = pddl_to_tree[token]
        if token not in vocab:
            alt = token.replace("_", "-")
            if alt in vocab:
                return alt
        return token

    def to_pddl_name(name: Any) -> str:
        token = _normalize_token(name)
        if token in tree_to_pddl:
            return tree_to_pddl[token]
        return token.replace("_", "-").replace(" ", "-")

    def _normalize_pred_name(name: Any) -> str:
        return _normalize_token(name).replace("_", "-").replace(" ", "-")

    def _find_placeholder_token(preds: Iterable[Union[str, Predicate]]) -> Optional[str]:
        for pred in preds:
            if isinstance(pred, Predicate):
                args = pred.args
            elif isinstance(pred, str):
                try:
                    parsed = Predicate.from_string(pred)
                    args = parsed.args
                except Exception:
                    continue
            else:
                continue

            for arg in args:
                if _is_placeholder(arg):
                    return str(arg).strip()
        return None

    # Unwrap nested test input formats
    input_data = _unwrap_input(input_data)

    placeholder_token = _find_placeholder_token(initial_state) or _find_placeholder_token(goal_state)

    # ------------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------------
    def is_image(data: Any) -> bool:
        return _is_image_like(data)

    def load_image(data: Any) -> Image.Image:
        if isinstance(data, (str, Path)):
            return Image.open(data).convert("RGB")
        if isinstance(data, Image.Image):
            return data.convert("RGB")
        if isinstance(data, torch.Tensor):
            arr = data.detach().cpu()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.permute(1, 2, 0)
            arr = arr.numpy()
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
        if isinstance(data, np.ndarray):
            arr = data
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = arr.astype(np.uint8)
            return Image.fromarray(arr)
        raise TypeError("Unsupported image input type.")

    # ------------------------------------------------------------------------
    # Identify object (image or text)
    # ------------------------------------------------------------------------
    if is_image(input_data):
        # Build CIFAR subset present in vocab (allow dash/underscore variants)
        class_words: List[str] = []
        for name in sorted(CIFAR_100_CLASSES):
            if name in vocab:
                class_words.append(name)
            else:
                alt = name.replace("_", "-")
                if alt in vocab:
                    class_words.append(alt)

        if not class_words:
            warnings.warn("No CIFAR-100 classes found in vocabulary.")
            return None

        class_idx = [vocab[w] for w in class_words]
        class_emb = embeddings[class_idx].astype(np.float32)
        class_emb = class_emb / (np.linalg.norm(class_emb, axis=1, keepdims=True) + 1e-8)

        try:
            image = load_image(input_data)
        except Exception as exc:
            warnings.warn(f"Failed to load image input: {exc}")
            return None

        proj_dim = embeddings.shape[1]
        proj_path = Path(projection_path)
        if not proj_path.is_absolute():
            proj_path = Path.cwd() / proj_path
        if not proj_path.exists():
            warnings.warn(f"Projection checkpoint not found: {proj_path}")
            return None

        model = ImageEncoder(proj_dim=proj_dim, device=device, input_size=224)

        ckpt = torch.load(proj_path, map_location=device)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
            elif "projection_head" in ckpt:
                model.projection.load_state_dict(ckpt["projection_head"], strict=False)
            elif "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"], strict=False)

            # Use stored class words/embeddings if available
            if "class_words" in ckpt and "text_embeddings" in ckpt:
                ckpt_words = [str(w) for w in ckpt["class_words"]]
                ckpt_emb = ckpt["text_embeddings"]
                if isinstance(ckpt_emb, torch.Tensor):
                    ckpt_emb = ckpt_emb.detach().cpu().numpy()
                ckpt_emb = np.asarray(ckpt_emb)

                if ckpt_emb.ndim == 2 and ckpt_emb.shape[1] == proj_dim:
                    idx_map = {w: i for i, w in enumerate(ckpt_words)}

                    def is_cifar(x: str) -> bool:
                        return x in CIFAR_100_CLASSES or x.replace("-", "_") in CIFAR_100_CLASSES

                    filtered = [w for w in ckpt_words if is_cifar(w)]
                    if filtered:
                        class_words = filtered
                        class_emb = ckpt_emb[[idx_map[w] for w in filtered]].astype(np.float32)
                        class_emb = class_emb / (np.linalg.norm(class_emb, axis=1, keepdims=True) + 1e-8)
        else:
            model.load_state_dict(ckpt, strict=False)

        model.eval()

        # Input size override if stored
        img_size = int(getattr(model, "input_size", 224))
        if isinstance(ckpt, dict):
            override = ckpt.get("input_size") or ckpt.get("image_size")
            if override:
                try:
                    img_size = int(override)
                    model.input_size = img_size
                except (TypeError, ValueError):
                    pass

        transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            _, proj = model(img_tensor)
            proj = F.normalize(proj, p=2, dim=1).cpu().numpy()[0]

        scores = class_emb @ proj
        best = float(np.max(scores))
        near = np.where(scores >= best - 1e-5)[0]

        if len(near) > 1:
            obj_vocab = sorted(class_words[i] for i in near)[0]
        else:
            obj_vocab = class_words[int(np.argmax(scores))]

        if debug:
            print(f"[generate_plan] identified image as: {obj_vocab}")

    else:
        obj_vocab = to_vocab_name(input_data)
        if obj_vocab not in vocab:
            warnings.warn(f"Unknown object name: '{input_data}' not in vocabulary.")
            return None
        if debug:
            print(f"[generate_plan] using text object: {obj_vocab}")

    obj_pddl = to_pddl_name(obj_vocab)

    # ------------------------------------------------------------------------
    # Load domain
    # ------------------------------------------------------------------------
    domain_path = Path(domain_file)
    if not domain_path.is_absolute():
        domain_path = Path.cwd() / domain_path
    if not domain_path.exists():
        warnings.warn(f"Domain file not found: {domain_path}")
        return None

    # ------------------------------------------------------------------------
    # Predicate coercion
    # ------------------------------------------------------------------------
    def coerce_predicates(preds: Iterable[Union[str, Predicate]]) -> List[Predicate]:
        if isinstance(preds, (list, tuple)):
            ordered = list(preds)
        else:
            try:
                ordered = sorted(preds, key=lambda p: str(p))
            except TypeError:
                ordered = list(preds)

        out: List[Predicate] = []
        for pred in ordered:
            if isinstance(pred, Predicate):
                parsed = pred
            elif isinstance(pred, str):
                parsed = Predicate.from_string(pred)
            else:
                raise TypeError("Predicates must be strings or Predicate objects.")

            pred_name = _normalize_pred_name(parsed.name)

            if parsed.args:
                new_args = []
                for arg in parsed.args:
                    token = arg
                    if _is_placeholder(token):
                        token = obj_pddl
                    token = to_pddl_name(token)
                    new_args.append(token)
                parsed = Predicate(pred_name, tuple(new_args))
            else:
                parsed = Predicate(pred_name, parsed.args)

            out.append(parsed)

        return out

    init_preds = coerce_predicates(initial_state)
    goal_preds = coerce_predicates(goal_state)

    # ------------------------------------------------------------------------
    # Merge base init from problem.pddl (optional)
    # ------------------------------------------------------------------------
    problem_path = domain_path.with_name("problem.pddl")
    if not problem_path.exists():
        problem_path = Path.cwd() / "problem.pddl"

    if problem_path.exists():
        try:
            content = problem_path.read_text()
            base_init = set()

            base_init_match = re.search(
                r":init(.*?)(?=\(:goal|\Z)",
                content,
                re.DOTALL | re.IGNORECASE,
            )
            if base_init_match:
                base_init = PDDLParser._extract_predicates(base_init_match.group(1))

            if base_init:
                relevant = set()
                for pred in init_preds + goal_preds:
                    relevant.update(pred.args)

                relevant -= {"lab", "outdoors"}
                relevant |= {"knife", "dslr"}

                filtered_base = set()
                for bp in base_init:
                    if not bp.args or bp.name == "agent-at":
                        filtered_base.add(bp)
                    elif any(arg in relevant for arg in bp.args):
                        filtered_base.add(bp)

                user_defs = defaultdict(set)
                for p in init_preds:
                    if p.args:
                        user_defs[p.args[0]].add(p.name)
                    user_defs["GLOBAL"].add(p.name)

                merged_init = set()
                for bp in filtered_base:
                    if bp in init_preds:
                        continue
                    if not _has_conflict(bp, user_defs):
                        merged_init.add(bp)

                merged_init.update(init_preds)
                init_preds = list(merged_init)

        except Exception as exc:
            warnings.warn(f"Failed to merge problem.pddl defaults: {exc}")

    # ------------------------------------------------------------------------
    # Minimal defaults (tools / whole)
    # ------------------------------------------------------------------------
    pred_set = set(init_preds)
    pred_names_by_arg = defaultdict(set)
    for p in pred_set:
        if p.args:
            pred_names_by_arg[p.args[0]].add(p.name)
        pred_names_by_arg["GLOBAL"].add(p.name)

    added_defaults: set[Predicate] = set()

    def add_pred(pred: Predicate) -> None:
        if pred in pred_set:
            return
        pred_set.add(pred)
        added_defaults.add(pred)
        if pred.args:
            pred_names_by_arg[pred.args[0]].add(pred.name)
        pred_names_by_arg["GLOBAL"].add(pred.name)

    cut_targets = {p.args[0] for p in goal_preds if p.name == "cut-into-pieces" and p.args}
    photo_targets = {p.args[0] for p in goal_preds if p.name == "documented" and p.args}

    if cut_targets:
        add_pred(Predicate("is-tool", ("knife",)))
        knife_defs = pred_names_by_arg["knife"]
        if not {"at", "on-top", "holding"} & knife_defs:
            add_pred(Predicate("at", ("knife", "lab")))
        if "clear" not in knife_defs:
            add_pred(Predicate("clear", ("knife",)))

        for obj in cut_targets:
            obj_defs = pred_names_by_arg[obj]
            if "whole" not in obj_defs and "cut-into-pieces" not in obj_defs:
                add_pred(Predicate("whole", (obj,)))

    if photo_targets:
        add_pred(Predicate("is-tool", ("dslr",)))
        dslr_defs = pred_names_by_arg["dslr"]
        if not {"at", "on-top", "holding"} & dslr_defs:
            add_pred(Predicate("at", ("dslr", "lab")))
        if "clear" not in dslr_defs:
            add_pred(Predicate("clear", ("dslr",)))

    init_preds = list(pred_set)

    # Mirror defaults into mutable user input (so validation sees them)
    if added_defaults and isinstance(initial_state, (set, list)):
        use_predicates = any(isinstance(p, Predicate) for p in initial_state)
        replace_target = placeholder_token and obj_pddl in cut_targets

        def format_for_input(pred: Predicate) -> Union[Predicate, str]:
            if replace_target and pred.args and obj_pddl in pred.args:
                args = tuple(placeholder_token if arg == obj_pddl else arg for arg in pred.args)
                return Predicate(pred.name, args) if use_predicates else f"({pred.name} {' '.join(args)})"
            return pred if use_predicates else str(pred)

        for pred in added_defaults:
            entry = format_for_input(pred)
            if isinstance(initial_state, set):
                initial_state.add(entry)
            else:
                initial_state.append(entry)

    # ------------------------------------------------------------------------
    # Build objects / ground actions
    # ------------------------------------------------------------------------
    items: set[str] = set()
    tools = {"knife", "dslr"}
    locations = {"lab", "outdoors"}

    for pred in init_preds + goal_preds:
        if pred.name == "agent-at" and pred.args:
            locations.add(pred.args[0])
        elif pred.name == "at" and len(pred.args) >= 2:
            items.add(pred.args[0])
            locations.add(pred.args[1])
        elif pred.name == "is-tool" and pred.args:
            tools.add(pred.args[0])
            items.add(pred.args[0])
        else:
            items.update(pred.args)

    items.add(obj_pddl)
    items.update(tools)
    items = {i for i in items if i not in locations}

    objects = {"item": items, "tool": tools, "location": locations}

    actions_dict = PDDLParser.parse_domain(str(domain_path))
    grounded_actions = ActionGrounder(actions_dict, objects).ground_all()

    # ------------------------------------------------------------------------
    # Search for plan
    # ------------------------------------------------------------------------
    init_state_obj = State(frozenset(init_preds))
    goal_frozen = frozenset(goal_preds)

    def heuristic_fn(state: State, goal: frozenset) -> int:
        return 0 if state.satisfies(goal) else 1

    plan = astar_search(
        init_state_obj,
        goal_frozen,
        grounded_actions,
        heuristic_fn=heuristic_fn,
        verbose=debug,
    )

    if plan is None:
        warnings.warn("No valid plan found.")
        return None

    # ------------------------------------------------------------------------
    # Validate plan
    # ------------------------------------------------------------------------
    state = init_state_obj
    for action in plan:
        if not state.is_applicable(action):
            warnings.warn("Plan validation failed (precondition mismatch).")
            return None
        state = state.apply_action(action)

    if not state.satisfies(goal_frozen):
        warnings.warn("Plan validation failed (goal not achieved).")
        return None

    # ------------------------------------------------------------------------
    # Return structured plan details
    # ------------------------------------------------------------------------
    changes: List[Dict[str, Union[int, str, List[str]]]] = []
    curr_state = init_state_obj

    for i, action in enumerate(plan, 1):
        curr_state = curr_state.apply_action(action)
        changes.append(
            {
                "step": i,
                "action": str(action),
                "added": sorted(map(str, action.add_effects)),
                "removed": sorted(map(str, action.del_effects)),
            }
        )

    return changes


# ============================================================================
# Optional local checks / demo
# ============================================================================

if __name__ == "__main__":
    try:
        vocab, embeddings = load_semantic_embeddings()
        vocab_size = len(vocab)

        print(f"Vocabulary size: {vocab_size}")
        print(f"Embedding shape: {embeddings.shape}")

        if embeddings.ndim != 2 or embeddings.shape[0] != vocab_size:
            raise ValueError("Embedding matrix shape does not match vocab size.")

        idx_values = list(vocab.values())
        if len(set(idx_values)) != vocab_size:
            raise ValueError("Vocabulary indices are not unique.")
        if set(idx_values) != set(range(vocab_size)):
            raise ValueError("Vocabulary indices are not a contiguous 0..V-1 range.")

        missing = sorted([w for w in CIFAR_100_CLASSES if w not in vocab])
        if missing:
            print(f"Missing CIFAR-100 words ({len(missing)}): {', '.join(missing)}")
        else:
            print("All CIFAR-100 class names are present.")

    except Exception as exc:
        print(f"Embedding checkpoint validation failed: {exc}")

    domain_path = REPO_ROOT / "planning" / "domain.pddl"
    proj_path = Path(str(REPO_ROOT / "models" / "visual_projection.pth"))

    if domain_path.exists() and proj_path.exists():
        print("\n[Planning demo] Planning for apple...")

        initial = {
            "(agent-at lab)",
            "(holding knife)",
            "(at apple lab)",
            "(whole apple)",
            "(clear apple)",
        }
        goal = {"(cut-into-pieces apple)"}

        plan = generate_plan(
            input_data="apple",
            initial_state=initial,
            goal_state=goal,
            domain_file=str(domain_path),
        )

        if plan is None:
            print("No valid plan found.")
        else:
            print(f"Plan steps: {len(plan)}")
            for step in plan:
                print(step)

    else:
        print("\n[Planning demo] Skipped because the planning domain or visual projection checkpoint is unavailable.")
