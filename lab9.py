"""
CIFAR-100 PDDL Planner - SUBOPTIMAL BFS VERSION
==========================================================
Breadth-First Search planner for CIFAR-100 domain.
"""

import re
import tempfile
import heapq
from typing import Set, List, Dict, Tuple, Optional, FrozenSet, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

# ==================== CONSTANTS ====================

CIFAR_100_CLASSES = {
    "apple", "aquarium-fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn-mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple", "motorcycle", "mountain", "mouse",
    "mushroom", "oak", "orange", "orchid", "otter", "palm", "pear",
    "pickup-truck", "pine", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet-pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow", "wolf", "woman", "worm"
}

TOOLS = {'knife', 'dslr'}
LOCATIONS = {'lab', 'outdoors'}
IGNORED_KEYWORDS = {'item', 'location', 'object', 'objects', '-', 'agent', 'define', 'problem', 'domain', 'init', 'goal'}
STATIC_PRED_NAMES = {'is-tool', 'is-surface'}

# ==================== DATA STRUCTURES ====================

@dataclass(frozen=True)
class Predicate:
    name: str
    args: Tuple[str, ...]
    
    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.args)})" if self.args else f"({self.name})"
    
    @staticmethod
    def from_string(s: str) -> 'Predicate':
        parts = s.strip().strip('()').split()
        return Predicate(parts[0], tuple(parts[1:]) if len(parts) > 1 else ())


@dataclass(frozen=True)
class Action:
    name: str
    parameters: Tuple[str, ...]
    preconditions: FrozenSet[Predicate]
    add_effects: FrozenSet[Predicate]
    del_effects: FrozenSet[Predicate]

    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.parameters)})"
    
    def instantiate(self, bindings: Dict[str, str]) -> 'Action':
        """
        Creates a concrete Action instance by replacing variables with objects.
        bindings example: {'?x': 'apple', '?loc': 'lab'}
        """
        # Helper lambda to substitute args in a single predicate
        
        def substitute_pred(pred: Predicate) -> Predicate:
            return Predicate(pred.name, tuple(bindings.get(arg, arg) for arg in pred.args))

        new_params = tuple(bindings.get(p, p) for p in self.parameters)
        new_preconds = frozenset(substitute_pred(p) for p in self.preconditions)
        new_adds = frozenset(substitute_pred(p) for p in self.add_effects)
        new_dels = frozenset(substitute_pred(p) for p in self.del_effects)
        return Action(self.name, new_params, new_preconds, new_adds, new_dels)


@dataclass(frozen=True)
class State:
    predicates: FrozenSet[Predicate]
    
    def apply_action(self, action: Action) -> 'State':
        """
        Returns a NEW State by applying the action's effects.
        Recall: Next State = (Current State - Del Effects) + Add Effects
        """
        # TODO: Implement this method
        new_preds = set(self.predicates)
        new_preds.difference_update(action.del_effects)
        new_preds.update(action.add_effects)
        return State(frozenset(new_preds))

    def is_applicable(self, action: Action) -> bool:
        """
        Checks if an action can be applied to this state.
        Recall: All preconditions must exist in the current state.
        """
        # TODO: Implement this method
        return action.preconditions.issubset(self.predicates)
        
    def satisfies(self, goal: FrozenSet[Predicate]) -> bool:
        return goal.issubset(self.predicates)


@dataclass(order=True)
class SearchNode:
    f_score: int
    state: State = field(compare=False)
    action: Optional[Action] = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False)
    g_score: int = field(compare=False)
    
    def get_plan(self) -> List[Action]:
        plan, node = [], self
        while node.parent:
            plan.append(node.action)
            node = node.parent
        return list(reversed(plan))

# ==================== PDDL PARSER ====================

class PDDLParser:
    last_discarded = set()
    
    @staticmethod
    def _extract_predicates(block: str) -> Set[Predicate]:
        """Extract predicates from a block, handling 'and' wrappers and nested parens."""
        block = block.strip()
        if block.startswith('(and'):
            depth, i = 0, 4
            while i < len(block):
                if block[i] == '(':
                    depth += 1
                elif block[i] == ')':
                    depth -= 1
                    if depth == -1:
                        block = block[4:i].strip()
                        break
                i += 1
        
        preds, depth, curr = set(), 0, []
        for c in block:
            if c == '(':
                depth += 1
                curr.append(c)
            elif c == ')':
                depth -= 1
                curr.append(c)
                if depth == 0 and curr:
                    s = ''.join(curr).strip()
                    if s and not s.startswith('(not') and not s.startswith('(forall'):
                        try:
                            preds.add(Predicate.from_string(s))
                        except:
                            pass
                    curr = []
            elif depth > 0:
                curr.append(c)
        return preds
    
    @staticmethod
    def _parse_effects(body: str) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Parse add and delete effects from action body."""
        m = re.search(r':effect\s*(\(.*)', body, re.DOTALL | re.IGNORECASE)
        if not m:
            return set(), set()
        
        effect_block = m.group(1).strip()
        depth, i = 0, 0
        while i < len(effect_block):
            if effect_block[i] == '(':
                depth += 1
            elif effect_block[i] == ')':
                depth -= 1
                if depth == 0:
                    effect_block = effect_block[:i+1]
                    break
            i += 1
        
        block = effect_block[4:].strip().rstrip(')') if effect_block.startswith('(and') else effect_block
        adds, dels, depth, curr = set(), set(), 0, []
        
        for c in block:
            if c == '(':
                depth += 1
                curr.append(c)
            elif c == ')':
                depth -= 1
                curr.append(c)
                if depth == 0 and curr:
                    s = ''.join(curr).strip()
                    if s.startswith('(not'):
                        inner = re.search(r'\(not\s+(\(.*?\))\s*\)', s, re.IGNORECASE)
                        if inner:
                            try:
                                dels.add(Predicate.from_string(inner.group(1)))
                            except:
                                pass
                    else:
                        try:
                            adds.add(Predicate.from_string(s))
                        except:
                            pass
                    curr = []
            elif depth > 0:
                curr.append(c)
        
        return adds, dels
    
    @staticmethod
    def parse_domain(path: str) -> Dict[str, Action]:
        """Parse domain file and return action schemas."""
        with open(path) as f:
            content = f.read()
        content = content.replace("\\n", "\n")
        content = re.sub(r';.*$', '', content, flags=re.MULTILINE)
        
        actions = {}
        for m in re.finditer(r'\(\s*:action\s+([^\s()]+)(.*?)(?=\(\s*:action|\Z)', content, re.DOTALL | re.IGNORECASE):
            name, body = m.groups()
            name = name.strip()
            
            params = []
            if params_match := re.search(r':parameters\s*\((.*?)\)', body, re.DOTALL | re.IGNORECASE):
                params = re.findall(r'\?[\w-]+', params_match.group(1))
            
            preconds = set()
            if precond_match := re.search(r':precondition\s*(\(.*?)(?=\s*:effect|\Z)', body, re.DOTALL | re.IGNORECASE):
                preconds = PDDLParser._extract_predicates(precond_match.group(1))
            
            adds, dels = PDDLParser._parse_effects(body)
            actions[name] = Action(name, tuple(params), frozenset(preconds), frozenset(adds), frozenset(dels))
        
        return actions
    
    @staticmethod
    def parse_problem(path: str) -> Tuple[Dict[str, Set[str]], State, FrozenSet[Predicate]]:
        """Parse problem file and return objects, initial state, and goal."""
        with open(path) as f:
            content = f.read()
        
        objs = defaultdict(set)
        PDDLParser.last_discarded = set()
        
        if om := re.search(r':objects(.*?)(?=\(:init)', content, re.DOTALL | re.IGNORECASE):
            ob = re.sub(r';.*$', '', om.group(1), flags=re.MULTILINE).strip()
            for token in re.findall(r'[^\s():;]+', ob):
                t_lower = token.lower()
                if t_lower in CIFAR_100_CLASSES:
                    objs['item'].add(t_lower)
                elif t_lower in TOOLS:
                    objs['item'].add(t_lower)
                    objs['tool'].add(t_lower)
                elif t_lower in LOCATIONS:
                    objs['location'].add(t_lower)
                elif t_lower not in IGNORED_KEYWORDS:
                    PDDLParser.last_discarded.add(t_lower)
        
        for t in TOOLS:
            objs['item'].add(t)
            objs['tool'].add(t)
        
        init = set()
        if im := re.search(r':init(.*?)(?=\(:goal|\Z)', content, re.DOTALL | re.IGNORECASE):
            init = PDDLParser._extract_predicates(im.group(1))
        
        goal = set()
        if gm := re.search(r':goal\s+(\(.*\))', content, re.DOTALL | re.IGNORECASE):
            goal = PDDLParser._extract_predicates(gm.group(1))
        
        return dict(objs), State(frozenset(init)), frozenset(goal)

# ==================== ACTION GROUNDING ====================

class ActionGrounder:
    def __init__(self, schemas: Dict[str, Action], objects: Dict[str, Set[str]]):
        self.schemas = schemas
        self.all_items = sorted(objects.get('item', set()))
        self.cifar_objects = sorted(set(self.all_items) - TOOLS)
        self.locations = sorted(LOCATIONS)
    
    def ground_all(self) -> List[Action]:
        """Ground all action schemas with concrete objects."""
        grounded = []
        
        for name, schema in self.schemas.items():
            if name == 'walk-between-rooms':
                grounded.extend(self._ground_walk(schema))
            elif name in ['pick-up', 'put-down']:
                grounded.extend(self._ground_item_location(schema))
            elif name in ['stack', 'unstack']:
                grounded.extend(self._ground_stack(schema))
            elif name in ['slice-object', 'clean-object', 'take-photo']:
                grounded.extend(self._ground_object_action(schema))
        
        return grounded
    
    def _ground_walk(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: l1, schema.parameters[1]: l2})
                for l1 in self.locations for l2 in self.locations if l1 != l2]
    
    def _ground_item_location(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: item, schema.parameters[1]: loc})
                for item in self.all_items for loc in self.locations]
    
    def _ground_stack(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: top, schema.parameters[1]: bottom, schema.parameters[2]: loc})
                for top in self.all_items for bottom in self.all_items 
                if top != bottom for loc in self.locations]
    
    def _ground_object_action(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: obj, schema.parameters[1]: loc})
                for obj in self.cifar_objects for loc in self.locations]

# ==================== ACTION PRUNING & HEURISTICS ====================

def prune_actions_static(actions: List[Action], init_state: State) -> List[Action]:
    """Remove grounded actions that violate static predicates (e.g., is-tool/is-surface)."""
    static_preds = {p for p in init_state.predicates if p.name in STATIC_PRED_NAMES}
    if not static_preds:
        return actions
    kept = []
    for action in actions:
        ok = True
        for p in action.preconditions:
            if p.name in STATIC_PRED_NAMES and p not in static_preds:
                ok = False
                break
        if ok:
            kept.append(action)
    return kept


def prune_actions_goal_relevant(actions: List[Action], goal: FrozenSet[Predicate]) -> List[Action]:
    """Remove actions that cannot contribute (directly or indirectly) to goal predicates."""
    if not goal:
        return actions
    relevant = set(goal)
    changed = True
    while changed:
        changed = False
        for action in actions:
            if any(p in relevant for p in action.add_effects):
                for pre in action.preconditions:
                    if pre not in relevant:
                        relevant.add(pre)
                        changed = True
    return [a for a in actions if any(p in relevant for p in a.add_effects)]


def build_action_index(actions: List[Action]) -> Dict[Predicate, List[Action]]:
    """Index actions by concrete preconditions for quick candidate filtering."""
    index = defaultdict(list)
    for action in actions:
        for p in action.preconditions:
            index[p].append(action)
    return index


def build_add_effect_index(actions: List[Action]) -> Dict[Predicate, List[Action]]:
    """Index actions by concrete add effects for heuristic lookup."""
    index = defaultdict(list)
    for action in actions:
        for p in action.add_effects:
            index[p].append(action)
    return index


def _candidate_actions(state: State, actions: List[Action],
                       action_index: Optional[Dict[Predicate, List[Action]]]) -> List[Action]:
    if not action_index:
        return actions
    candidates = set()
    for p in state.predicates:
        for a in action_index.get(p, ()):
            candidates.add(a)
    return list(candidates) if candidates else actions


def goal_count_heuristic(state: State, goal: FrozenSet[Predicate]) -> int:
    """Simple heuristic: count of unsatisfied goal predicates."""
    return len(goal - state.predicates)


def goal_precondition_heuristic(state: State, goal: FrozenSet[Predicate],
                                add_index: Optional[Dict[Predicate, List[Action]]]) -> int:
    """
    Heuristic: for each missing goal, estimate steps via best supporting action.
    Uses max over goals to keep the estimate conservative.
    """
    missing = [g for g in goal if g not in state.predicates]
    if not missing:
        return 0
    if not add_index:
        return len(missing)
    scores = []
    for g in missing:
        actions = add_index.get(g, [])
        if not actions:
            scores.append(len(goal) + 5)
            continue
        min_missing = min(sum(1 for p in a.preconditions if p not in state.predicates) for a in actions)
        scores.append(1 + min_missing)
    return max(scores) if scores else 0
# ==================== BFS PLANNER ====================

def bfs_search(initial: State, goal: FrozenSet[Predicate], actions: List[Action],
               max_iter: int = 50000, verbose: bool = True,
               action_index: Optional[Dict[Predicate, List[Action]]] = None) -> Optional[List[Action]]:
    
    # 1. Check if we are already there
    if initial.satisfies(goal):
        return []
    
    # 2. Setup Frontier and Visited set
    start_node = SearchNode(0, initial, None, None, 0)
    frontier = deque([start_node]) # FIFO Queue
    visited = {initial}
    
    if verbose:
        print(f"🔍 BFS Search: {len(actions)} actions, max {max_iter} iterations")

    iters = 0
    while frontier and iters < max_iter:
        iters += 1
        
        # TODO: Implement the expansion loop
        # 1. Pop the next node
        # 2. Check for goal
        # 3. Generate successors
        node = frontier.popleft()
        if node.state.satisfies(goal):
            plan = node.get_plan()
            if verbose:
                print(f"✓ Found plan! Iterations: {iters}, Plan length (Depth): {len(plan)}")
            return plan

        if verbose and iters % 1000 == 0:
            print(f"  Iter {iters}, frontier: {len(frontier)}, visited: {len(visited)}")

        for action in _candidate_actions(node.state, actions, action_index):
            if not node.state.is_applicable(action):
                continue
            next_state = node.state.apply_action(action)
            if next_state in visited:
                continue
            visited.add(next_state)
            frontier.append(SearchNode(node.g_score + 1, next_state, action, node, node.g_score + 1))

    if verbose:
        print(f"✗ No solution. Iterations: {iters}")
    return None


# ==================== A* PLANNER ====================

def astar_search(
    initial: State,
    goal: FrozenSet[Predicate],
    actions: List[Action],
    heuristic_fn: Optional[Callable[[State, FrozenSet[Predicate]], int]] = None,
    max_iter: int = 50000,
    verbose: bool = True,
    action_index: Optional[Dict[Predicate, List[Action]]] = None,
) -> Optional[List[Action]]:
    """A* search for optimal (shortest) plans. Uses an admissible heuristic by default."""
    if heuristic_fn is None:
        heuristic_fn = lambda state, goal: 0

    if initial.satisfies(goal):
        return []

    start_h = heuristic_fn(initial, goal)
    start_node = SearchNode(start_h, initial, None, None, 0)
    frontier = [start_node]
    g_scores = {initial: 0}
    visited = set()

    if verbose:
        print(f"🔍 A* Search: {len(actions)} actions, max {max_iter} iterations")

    iters = 0
    while frontier and iters < max_iter:
        iters += 1
        node = heapq.heappop(frontier)

        if node.state in visited:
            continue
        visited.add(node.state)

        if node.state.satisfies(goal):
            plan = node.get_plan()
            if verbose:
                print(f"✓ Found plan! Iterations: {iters}, Plan length (Depth): {len(plan)}")
            return plan

        if verbose and iters % 1000 == 0:
            print(f"  Iter {iters}, frontier: {len(frontier)}, visited: {len(visited)}")

        for action in _candidate_actions(node.state, actions, action_index):
            if not node.state.is_applicable(action):
                continue
            next_state = node.state.apply_action(action)
            tentative_g = node.g_score + 1
            if tentative_g >= g_scores.get(next_state, float("inf")):
                continue
            g_scores[next_state] = tentative_g
            f_score = tentative_g + heuristic_fn(next_state, goal)
            heapq.heappush(frontier, SearchNode(f_score, next_state, action, node, tentative_g))

    if verbose:
        print(f"✗ No solution. Iterations: {iters}")
    return None


# ==================== PROBLEM GENERATION ====================

def create_custom_problem(base_path: str, init_overrides: Set[str], 
                         goals: Set[str], name: str = "custom") -> str:
    """Generate custom problem file with conflict resolution."""
    with open(base_path) as f:
        content = f.read()
    
    objs_match = (re.search(r'(\(:objects.*?\)(?=\s*\(:init))', content, re.DOTALL | re.IGNORECASE) or
                  re.search(r'(\(:objects.*?\))', content, re.DOTALL | re.IGNORECASE))
    objs_sec = objs_match.group(1) if objs_match else "(:objects)"
    
    base_init = set()
    if base_init_match := re.search(r':init(.*?)(?=\(:goal|\Z)', content, re.DOTALL | re.IGNORECASE):
        base_init = PDDLParser._extract_predicates(base_init_match.group(1))
    
    user_init = {Predicate.from_string(p) for p in init_overrides}
    user_goal = {Predicate.from_string(p) for p in goals}
    
    # Build conflict map
    user_defs = defaultdict(set)
    for p in user_init:
        if p.args:
            user_defs[p.args[0]].add(p.name)
        user_defs['GLOBAL'].add(p.name)
    
    # Resolve conflicts
    final_init = set()
    for bp in base_init:
        if bp in user_init:
            continue
        
        conflict = _has_conflict(bp, user_defs)
        if not conflict:
            final_init.add(bp)
    
    final_init.update(user_init)
    
    problem_str = f"""(define (problem {name}-problem)
  (:domain cifar100-process)
  {objs_sec}
  (:init {chr(10).join(f"    {p}" for p in sorted(map(str, final_init)))} )
  (:goal (and {chr(10).join(f"      {p}" for p in sorted(map(str, user_goal)))} ))
)"""
    
    tf = tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False)
    tf.write(problem_str)
    tf.close()
    return tf.name


def _has_conflict(pred: Predicate, user_defs: Dict[str, Set[str]]) -> bool:
    """Check if base predicate conflicts with user definitions."""
    global_defs = user_defs['GLOBAL']
    
    if pred.name == 'hand-empty' and 'holding' in global_defs:
        return True
    if pred.name == 'holding' and 'hand-empty' in global_defs:
        return True
    if pred.name == 'agent-at' and 'agent-at' in global_defs:
        return True
    
    if pred.args:
        obj_defs = user_defs[pred.args[0]]
        if pred.name in ['at', 'on-top'] and {'at', 'on-top'} & obj_defs:
            return True
        if pred.name == 'clear' and {'at', 'on-top', 'clear'} & obj_defs:
            return True
        if pred.name == 'whole' and 'cut-into-pieces' in obj_defs:
            return True
        if pred.name == 'wet' and 'clean' in obj_defs:
            return True
        if pred.name == 'clean' and 'wet' in obj_defs:
            return True
    
    return False

# ==================== VALIDATION ====================

def validate_user_conditions(conditions: Set[str]):
    """Validate user input for syntax and vocabulary."""
    valid_vocab = CIFAR_100_CLASSES | TOOLS | LOCATIONS
    
    for s in conditions:
        s_clean = s.strip()
        if not (s_clean.startswith('(') and s_clean.endswith(')')):
            raise ValueError(f"SYNTAX ERROR: '{s}' is missing parentheses.")
        
        pred = Predicate.from_string(s_clean)
        for arg in pred.args:
            if arg not in valid_vocab:
                raise ValueError(
                    f"VOCABULARY ERROR: Unknown object '{arg}' in predicate '{s}'.\n"
                    f"   (Check spelling. Valid args are 100 CIFAR items, 2 tools, or 'lab'/'outdoors')"
                )

# ==================== VISUALIZATION ====================

def print_plan_execution(plan: List[Dict], object_name: str, initial_conditions: Set[str]):
    """Print formatted execution trace."""
    print("\n" + "="*80)
    print(f"📋 EXECUTION TRACE: {object_name.upper()}")
    print("="*80)
    print(f"{'STEP':<6} | {'ACTION':<35} | {'STATE CHANGES / INITIAL STATE':<40}")
    print("-" * 90)
    
    init_str = ", ".join(sorted(initial_conditions))
    print(f"{'0':<6} | {'(INITIAL STATE)':<35} | {init_str}")
    
    if not plan:
        print("-" * 90)
        print("✓ GOAL ALREADY ACHIEVED (No actions needed)")
        print("="*80 + "\n")
        return
    
    for step in plan:
        changes = []
        if step['added']:
            changes.append(f"++ {', '.join(step['added'])}")
        if step['removed']:
            changes.append(f"-- {', '.join(step['removed'])}")
        changes_str = " | ".join(changes)
        
        print(f"{step['step']:<6} | {step['action']:<35} | {changes_str}")
    
    print("-" * 90)
    print("✓ GOAL ACHIEVED")
    print("="*80 + "\n")
