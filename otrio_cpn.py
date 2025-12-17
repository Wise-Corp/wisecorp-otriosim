"""
Otrio Game - Colored Petri Net Implementation

Model structure:
- 2 initial places (Blue_Initial, Red_Initial) - each starts with 9 tokens (3 of each size)
- 18 board places (9 per player): Blue_A1..Blue_C3, Red_A1..Red_C3
- Token colors: SMALL (light-gray), MEDIUM (dark-gray), LARGE (black)
- Each board place can hold at most 1 token of each size

Win conditions:
1. Three nested pieces (all three sizes) in one space
2. Three same-sized pieces in a row/column/diagonal
3. Three ascending/descending sizes in a row/column/diagonal
"""

from cpnpy.cpn.cpn_imp import (
    CPN, Place, Transition, Arc, Marking, EvaluationContext,
    EnumeratedColorSet
)
from collections import defaultdict
import copy


# =============================================================================
# Color Set Definitions
# =============================================================================

# Token sizes (colors in CPN terminology)
SIZE_COLORSET = EnumeratedColorSet("Size", ["SMALL", "MEDIUM", "LARGE"])


# =============================================================================
# Board Positions
# =============================================================================

ROWS = ["A", "B", "C"]
COLS = ["1", "2", "3"]
POSITIONS = [f"{r}{c}" for r in ROWS for c in COLS]  # A1, A2, ..., C3

# Win lines: rows, columns, diagonals
WIN_LINES = [
    # Rows
    ["A1", "A2", "A3"],
    ["B1", "B2", "B3"],
    ["C1", "C2", "C3"],
    # Columns
    ["A1", "B1", "C1"],
    ["A2", "B2", "C2"],
    ["A3", "B3", "C3"],
    # Diagonals
    ["A1", "B2", "C3"],
    ["A3", "B2", "C1"],
]


# =============================================================================
# Game State Representation (for state space exploration)
# =============================================================================

class OtrioGameState:
    """
    Represents the state of an Otrio game.

    Board representation:
    - board[player][position] = set of sizes placed
    - initial[player] = multiset of remaining pieces
    """

    def __init__(self):
        # Each player's board positions
        self.board = {
            "BLUE": {pos: set() for pos in POSITIONS},
            "RED": {pos: set() for pos in POSITIONS}
        }
        # Initial pieces for each player (3 of each size)
        self.initial = {
            "BLUE": {"SMALL": 3, "MEDIUM": 3, "LARGE": 3},
            "RED": {"SMALL": 3, "MEDIUM": 3, "LARGE": 3}
        }
        self.current_player = "BLUE"
        self.winner = None
        self.move_history = []

    def copy(self):
        """Create a deep copy of the game state."""
        new_state = OtrioGameState()
        new_state.board = {
            player: {pos: sizes.copy() for pos, sizes in positions.items()}
            for player, positions in self.board.items()
        }
        new_state.initial = {
            player: counts.copy() for player, counts in self.initial.items()
        }
        new_state.current_player = self.current_player
        new_state.winner = self.winner
        new_state.move_history = self.move_history.copy()
        return new_state

    def get_valid_moves(self):
        """Return list of valid moves for current player: (position, size)."""
        if self.winner:
            return []

        moves = []
        player = self.current_player
        opponent = "RED" if player == "BLUE" else "BLUE"

        for size in ["SMALL", "MEDIUM", "LARGE"]:
            if self.initial[player][size] > 0:
                for pos in POSITIONS:
                    # Can only place if that size isn't already there for EITHER player
                    # (opponent's piece blocks that size at that position)
                    if size not in self.board[player][pos] and size not in self.board[opponent][pos]:
                        moves.append((pos, size))

        return moves

    def make_move(self, position, size):
        """
        Make a move: place a piece of given size at position.
        Returns new state (immutable style).
        """
        if self.winner:
            raise ValueError("Game already won")

        player = self.current_player
        opponent = "RED" if player == "BLUE" else "BLUE"

        if self.initial[player][size] <= 0:
            raise ValueError(f"No {size} pieces left for {player}")

        if size in self.board[player][position]:
            raise ValueError(f"{size} already at {position} for {player}")

        # Check if opponent already has that size there (blocking rule)
        if size in self.board[opponent][position]:
            raise ValueError(f"{size} at {position} is blocked by {opponent}")

        # Create new state
        new_state = self.copy()
        new_state.initial[player][size] -= 1
        new_state.board[player][position].add(size)
        new_state.move_history.append((player, position, size))

        # Check for winner
        new_state.winner = new_state.check_winner(player)

        # Switch player
        new_state.current_player = "RED" if player == "BLUE" else "BLUE"

        return new_state

    def check_winner(self, player):
        """Check if the given player has won."""
        board = self.board[player]

        # Win condition 1: Three nested pieces in one space
        for pos in POSITIONS:
            if len(board[pos]) == 3:  # Has SMALL, MEDIUM, LARGE
                return player

        # Win condition 2 & 3: Three in a row (same size or ordered sizes)
        for line in WIN_LINES:
            # Check same size in a row
            for size in ["SMALL", "MEDIUM", "LARGE"]:
                if all(size in board[pos] for pos in line):
                    return player

            # Check ascending order (SMALL -> MEDIUM -> LARGE)
            if ("SMALL" in board[line[0]] and
                "MEDIUM" in board[line[1]] and
                "LARGE" in board[line[2]]):
                return player

            # Check descending order (LARGE -> MEDIUM -> SMALL)
            if ("LARGE" in board[line[0]] and
                "MEDIUM" in board[line[1]] and
                "SMALL" in board[line[2]]):
                return player

        return None

    def to_tuple(self):
        """Convert state to hashable tuple for state space exploration."""
        board_tuple = tuple(
            (player, tuple((pos, tuple(sorted(sizes)))
                          for pos, sizes in sorted(positions.items())))
            for player, positions in sorted(self.board.items())
        )
        initial_tuple = tuple(
            (player, tuple(sorted(counts.items())))
            for player, counts in sorted(self.initial.items())
        )
        return (board_tuple, initial_tuple, self.current_player)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __str__(self):
        """Pretty print the game state."""
        lines = []
        lines.append(f"Current player: {self.current_player}")
        if self.winner:
            lines.append(f"WINNER: {self.winner}")

        lines.append("\nBLUE board:")
        for row in ROWS:
            row_str = f"  {row}: "
            for col in COLS:
                pos = f"{row}{col}"
                sizes = self.board["BLUE"][pos]
                size_str = "".join(s[0] for s in sorted(sizes))  # S, M, L
                row_str += f"[{size_str:3s}] "
            lines.append(row_str)

        lines.append(f"  Remaining: {self.initial['BLUE']}")

        lines.append("\nRED board:")
        for row in ROWS:
            row_str = f"  {row}: "
            for col in COLS:
                pos = f"{row}{col}"
                sizes = self.board["RED"][pos]
                size_str = "".join(s[0] for s in sorted(sizes))
                row_str += f"[{size_str:3s}] "
            lines.append(row_str)

        lines.append(f"  Remaining: {self.initial['RED']}")

        return "\n".join(lines)


# =============================================================================
# CPN Model Builder (Original - Separate Places per Player)
# =============================================================================

def build_otrio_cpn():
    """
    Build the Colored Petri Net model for Otrio (original model).

    Structure:
    - Places: Blue_Initial, Red_Initial, Blue_A1..Blue_C3, Red_A1..Red_C3
    - Transitions: Blue_to_A1..Blue_to_C3, Red_to_A1..Red_to_C3
    - Arcs connect initial places to board places via transitions

    Note: This model doesn't enforce turn-taking or blocking in the CPN structure.
    Those constraints are handled in the game logic (OtrioGameState).
    """
    cpn = CPN()

    # Create places dictionary for reference
    places = {}

    # Create initial places (hold the player's unplaced pieces)
    places["Blue_Initial"] = Place("Blue_Initial", SIZE_COLORSET)
    places["Red_Initial"] = Place("Red_Initial", SIZE_COLORSET)
    cpn.add_place(places["Blue_Initial"])
    cpn.add_place(places["Red_Initial"])

    # Create board places for each player
    for player in ["Blue", "Red"]:
        for pos in POSITIONS:
            place_name = f"{player}_{pos}"
            places[place_name] = Place(place_name, SIZE_COLORSET)
            cpn.add_place(places[place_name])

    # Create transitions dictionary
    transitions = {}

    # Create transitions and arcs
    for player in ["Blue", "Red"]:
        initial_place = places[f"{player}_Initial"]
        for pos in POSITIONS:
            trans_name = f"{player}_to_{pos}"
            transition = Transition(trans_name, variables=["x"])
            transitions[trans_name] = transition
            cpn.add_transition(transition)

            # Input arc: from initial place (consume token x)
            input_arc = Arc(initial_place, transition, "x")
            cpn.add_arc(input_arc)

            # Output arc: to board place (produce token x)
            output_arc = Arc(transition, places[f"{player}_{pos}"], "x")
            cpn.add_arc(output_arc)

    return cpn, places, transitions


# =============================================================================
# Improved CPN Model (Shared Board Places + Turn Control + Blocking)
# =============================================================================

# Define color sets for the improved model
TOKEN_COLORSET = EnumeratedColorSet("Token", ["BLUE_SMALL", "BLUE_MEDIUM", "BLUE_LARGE",
                                               "RED_SMALL", "RED_MEDIUM", "RED_LARGE"])
TURN_COLORSET = EnumeratedColorSet("Turn", ["BLUE", "RED"])


def build_improved_otrio_cpn():
    """
    Build an improved CPN model for Otrio with proper constraints.

    Key improvements:
    1. SHARED BOARD PLACES: One place per cell (Board_A1, etc.) holding tokens
       that encode both player and size: (BLUE_SMALL, RED_MEDIUM, etc.)

    2. TURN CONTROL: A "Turn" place holds a single token (BLUE or RED)
       indicating whose turn it is. Transitions consume and produce turn tokens.

    3. BLOCKING RULE: Implemented via guards on transitions that check
       if the same size (regardless of player) already exists at target position.

    Structure:
    - Places:
        * Blue_Pool, Red_Pool: Unplaced pieces (9 each)
        * Board_A1..Board_C3: Shared board cells (9 places)
        * Turn: Current player token (1 place)

    - Transitions:
        * Move_Blue_A1_S, Move_Blue_A1_M, Move_Blue_A1_L, ... (27 per player = 54 total)
        * Each transition is specific to (player, position, size)

    - Guards:
        * Check that the target cell doesn't already have a token of the same size
    """
    cpn = CPN()
    places = {}
    transitions = {}

    # ----- PLACES -----

    # Player pools (unplaced pieces)
    places["Blue_Pool"] = Place("Blue_Pool", SIZE_COLORSET)
    places["Red_Pool"] = Place("Red_Pool", SIZE_COLORSET)
    cpn.add_place(places["Blue_Pool"])
    cpn.add_place(places["Red_Pool"])

    # Shared board places (one per cell)
    for pos in POSITIONS:
        place_name = f"Board_{pos}"
        places[place_name] = Place(place_name, TOKEN_COLORSET)
        cpn.add_place(places[place_name])

    # Turn control place
    places["Turn"] = Place("Turn", TURN_COLORSET)
    cpn.add_place(places["Turn"])

    # ----- TRANSITIONS -----
    # Create one transition per (player, position, size) combination
    # This explicit enumeration allows us to enforce blocking via guards

    for player in ["Blue", "Red"]:
        pool_place = places[f"{player}_Pool"]
        player_upper = player.upper()

        for pos in POSITIONS:
            board_place = places[f"Board_{pos}"]

            for size in ["SMALL", "MEDIUM", "LARGE"]:
                trans_name = f"Move_{player}_{pos}_{size[0]}"
                token_value = f"{player_upper}_{size}"

                # Create transition with guard
                # Guard: no token with same size exists at this position
                guard = f"no_{size}_at_{pos}"

                transition = Transition(trans_name, variables=["turn"], guard=guard)
                transitions[trans_name] = transition
                cpn.add_transition(transition)

                # Input arc from pool: consume the specific size token
                input_pool_arc = Arc(pool_place, transition, size)
                cpn.add_arc(input_pool_arc)

                # Input arc from Turn: consume turn token (must be this player's turn)
                input_turn_arc = Arc(places["Turn"], transition, player_upper)
                cpn.add_arc(input_turn_arc)

                # Output arc to board: produce player+size token
                output_board_arc = Arc(transition, board_place, token_value)
                cpn.add_arc(output_board_arc)

                # Output arc to Turn: produce opponent's turn token
                opponent = "RED" if player == "Blue" else "BLUE"
                output_turn_arc = Arc(transition, places["Turn"], opponent)
                cpn.add_arc(output_turn_arc)

    return cpn, places, transitions


def create_improved_initial_marking(cpn):
    """Create initial marking for the improved CPN model."""
    marking = Marking()

    # Each player starts with 3 SMALL, 3 MEDIUM, 3 LARGE in their pool
    initial_tokens = ["SMALL"] * 3 + ["MEDIUM"] * 3 + ["LARGE"] * 3
    marking.set_tokens("Blue_Pool", initial_tokens)
    marking.set_tokens("Red_Pool", initial_tokens)

    # All board places start empty
    for pos in POSITIONS:
        marking.set_tokens(f"Board_{pos}", [])

    # BLUE goes first
    marking.set_tokens("Turn", ["BLUE"])

    return marking


def check_blocking_guard(marking, position, size):
    """
    Guard function for blocking rule.
    Returns True if the move is allowed (no token of same size at position).
    """
    board_place = f"Board_{position}"
    tokens = [t.value for t in marking.get_multiset(board_place).tokens]

    # Check if any token has the same size (regardless of player)
    for token in tokens:
        # Token format: "BLUE_SMALL", "RED_MEDIUM", etc.
        token_size = token.split("_")[1]
        if token_size == size:
            return False  # Blocked!

    return True  # Move allowed


def get_enabled_transitions_improved(cpn, marking, current_player):
    """
    Get all enabled transitions for the current player in the improved model.
    Checks turn control and blocking guards.
    """
    enabled = []

    # Check if it's this player's turn
    turn_tokens = [t.value for t in marking.get_multiset("Turn").tokens]
    if current_player not in turn_tokens:
        return []  # Not this player's turn

    player = "Blue" if current_player == "BLUE" else "Red"
    pool_place = f"{player}_Pool"
    pool_tokens = [t.value for t in marking.get_multiset(pool_place).tokens]

    for pos in POSITIONS:
        for size in ["SMALL", "MEDIUM", "LARGE"]:
            # Check if player has this size in pool
            if size not in pool_tokens:
                continue

            # Check blocking guard
            if not check_blocking_guard(marking, pos, size):
                continue

            trans_name = f"Move_{player}_{pos}_{size[0]}"
            enabled.append((trans_name, pos, size))

    return enabled


def print_improved_cpn_info():
    """Print information about the improved CPN model."""
    print("""
IMPROVED CPN MODEL FOR OTRIO
============================

This model properly captures the game constraints in the CPN structure:

1. SHARED BOARD PLACES
   - 9 places (Board_A1, Board_A2, ... Board_C3)
   - Tokens encode player AND size: BLUE_SMALL, RED_MEDIUM, etc.
   - Both players' pieces coexist in the same place

2. TURN CONTROL PLACE
   - Single "Turn" place holds one token (BLUE or RED)
   - Transitions consume current player's turn token
   - Transitions produce opponent's turn token
   - Enforces strict alternation in the CPN itself

3. BLOCKING VIA GUARDS
   - Each transition has a guard checking for blocking
   - Guard: "no token with same SIZE at target position"
   - If BLUE_SMALL is at A1, RED cannot place RED_SMALL at A1

4. TRANSITION STRUCTURE
   - 54 transitions total (27 per player)
   - Format: Move_Blue_A1_S, Move_Red_B2_M, etc.
   - Each transition is specific to (player, position, size)

PLACE COUNT: 12 (2 pools + 9 board + 1 turn)
TRANSITION COUNT: 54 (9 positions × 3 sizes × 2 players)
ARC COUNT: 216 (54 transitions × 4 arcs each)
""")


def create_initial_marking(cpn):
    """Create the initial marking with 9 pieces per player."""
    marking = Marking()

    # Each player starts with 3 SMALL, 3 MEDIUM, 3 LARGE
    initial_tokens = ["SMALL"] * 3 + ["MEDIUM"] * 3 + ["LARGE"] * 3

    marking.set_tokens("Blue_Initial", initial_tokens)
    marking.set_tokens("Red_Initial", initial_tokens)

    # All board places start empty
    for player in ["Blue", "Red"]:
        for pos in POSITIONS:
            marking.set_tokens(f"{player}_{pos}", [])

    return marking


# =============================================================================
# CPN-based Simulation
# =============================================================================

def get_enabled_transitions(cpn, marking, context, current_player):
    """Get all enabled transitions for the current player."""
    enabled = []
    player = "Blue" if current_player == "BLUE" else "Red"

    for pos in POSITIONS:
        trans_name = f"{player}_to_{pos}"
        transition = cpn.get_transition_by_name(trans_name)

        # Find all valid bindings for this transition
        all_bindings = cpn._find_all_bindings(transition, marking, context)

        for binding in all_bindings:
            # Check if the size isn't already in the target place
            target_place = f"{player}_{pos}"
            target_tokens = [t.value for t in marking.get_multiset(target_place).tokens]
            if binding["x"] not in target_tokens:
                enabled.append((transition, binding, pos))

    return enabled


def check_cpn_winner(marking, player):
    """Check if a player has won based on the CPN marking."""
    player_prefix = "Blue" if player == "BLUE" else "Red"

    # Get all tokens for each position
    board = {}
    for pos in POSITIONS:
        place_name = f"{player_prefix}_{pos}"
        tokens = [t.value for t in marking.get_multiset(place_name).tokens]
        board[pos] = set(tokens)

    # Win condition 1: Three nested pieces in one space
    for pos in POSITIONS:
        if len(board[pos]) == 3:
            return player

    # Win condition 2 & 3: Three in a row
    for line in WIN_LINES:
        # Same size in a row
        for size in ["SMALL", "MEDIUM", "LARGE"]:
            if all(size in board[pos] for pos in line):
                return player

        # Ascending order
        if ("SMALL" in board[line[0]] and
            "MEDIUM" in board[line[1]] and
            "LARGE" in board[line[2]]):
            return player

        # Descending order
        if ("LARGE" in board[line[0]] and
            "MEDIUM" in board[line[1]] and
            "SMALL" in board[line[2]]):
            return player

    return None


def simulate_cpn_game(cpn, seed=None):
    """Simulate a game using the CPN model."""
    import random
    if seed is not None:
        random.seed(seed)

    marking = create_initial_marking(cpn)
    context = EvaluationContext()
    current_player = "BLUE"
    move_num = 0

    print("Starting CPN-based Otrio simulation...")
    print_cpn_state(marking, current_player)

    while True:
        enabled = get_enabled_transitions(cpn, marking, context, current_player)

        if not enabled:
            print("\nNo more moves available - Draw!")
            break

        # Pick a random enabled transition
        transition, binding, pos = random.choice(enabled)

        print(f"\nMove {move_num + 1}: {current_player} places {binding['x']} at {pos}")

        # Fire the transition
        cpn.fire_transition(transition, marking, context, binding)
        move_num += 1

        # Check for winner
        winner = check_cpn_winner(marking, current_player)
        if winner:
            print(f"\n{'='*40}")
            print(f"WINNER: {winner}")
            print_cpn_state(marking, current_player)
            return winner

        # Switch player
        current_player = "RED" if current_player == "BLUE" else "BLUE"

    print_cpn_state(marking, current_player)
    return None


def print_cpn_state(marking, current_player):
    """Print the current state of the CPN."""
    print(f"\nCurrent player: {current_player}")

    for player, prefix in [("BLUE", "Blue"), ("RED", "Red")]:
        print(f"\n{player} board:")
        for row in ROWS:
            row_str = f"  {row}: "
            for col in COLS:
                pos = f"{row}{col}"
                place_name = f"{prefix}_{pos}"
                tokens = [t.value for t in marking.get_multiset(place_name).tokens]
                size_str = "".join(s[0] for s in sorted(tokens))
                row_str += f"[{size_str:3s}] "
            print(row_str)

        initial_tokens = [t.value for t in marking.get_multiset(f"{prefix}_Initial").tokens]
        remaining = {"SMALL": 0, "MEDIUM": 0, "LARGE": 0}
        for t in initial_tokens:
            remaining[t] += 1
        print(f"  Remaining: {remaining}")


# =============================================================================
# State Space Exploration
# =============================================================================

def explore_state_space(max_states=100000, max_depth=18, verbose=True):
    """
    Explore all reachable states of the Otrio game.

    Returns statistics about the state space.
    """
    initial_state = OtrioGameState()

    visited = set()
    to_visit = [(initial_state, 0)]  # (state, depth)

    stats = {
        "total_states": 0,
        "winning_states": {"BLUE": 0, "RED": 0},
        "terminal_states": 0,
        "max_depth_reached": 0,
        "states_by_depth": defaultdict(int),
        "wins_by_depth": defaultdict(lambda: {"BLUE": 0, "RED": 0}),
    }

    while to_visit and stats["total_states"] < max_states:
        state, depth = to_visit.pop()

        state_hash = state.to_tuple()
        if state_hash in visited:
            continue

        visited.add(state_hash)
        stats["total_states"] += 1
        stats["states_by_depth"][depth] += 1
        stats["max_depth_reached"] = max(stats["max_depth_reached"], depth)

        if verbose and stats["total_states"] % 10000 == 0:
            print(f"Explored {stats['total_states']} states...")

        # Check for terminal state
        if state.winner:
            stats["winning_states"][state.winner] += 1
            stats["wins_by_depth"][depth][state.winner] += 1
            stats["terminal_states"] += 1
            continue

        # Get valid moves and explore
        moves = state.get_valid_moves()
        if not moves:
            stats["terminal_states"] += 1
            continue

        if depth < max_depth:
            for pos, size in moves:
                new_state = state.make_move(pos, size)
                new_hash = new_state.to_tuple()
                if new_hash not in visited:
                    to_visit.append((new_state, depth + 1))

    return stats, visited


def print_stats(stats):
    """Print exploration statistics."""
    print("\n" + "=" * 60)
    print("OTRIO STATE SPACE EXPLORATION RESULTS")
    print("=" * 60)
    print(f"\nTotal states explored: {stats['total_states']:,}")
    print(f"Terminal states: {stats['terminal_states']:,}")
    print(f"Max depth reached: {stats['max_depth_reached']}")
    print(f"\nWinning states:")
    print(f"  BLUE wins: {stats['winning_states']['BLUE']:,}")
    print(f"  RED wins: {stats['winning_states']['RED']:,}")

    print(f"\nStates by depth:")
    for depth in sorted(stats['states_by_depth'].keys()):
        count = stats['states_by_depth'][depth]
        blue_wins = stats['wins_by_depth'][depth]['BLUE']
        red_wins = stats['wins_by_depth'][depth]['RED']
        print(f"  Depth {depth:2d}: {count:8,} states "
              f"(BLUE wins: {blue_wins:,}, RED wins: {red_wins:,})")


# =============================================================================
# Winning Strategy Search (Minimax)
# =============================================================================

def find_winning_strategy(player="BLUE", max_depth=18, verbose=True, max_nodes=500000):
    """
    Find a guaranteed winning strategy for the specified player.

    Uses minimax with alpha-beta pruning to find moves that guarantee
    a win regardless of opponent's responses.

    Args:
        player: "BLUE" or "RED"
        max_depth: Maximum search depth
        verbose: Print progress
        max_nodes: Maximum nodes to explore (memory limit)

    Returns:
        - result: 1 = player wins, -1 = opponent wins, 0 = unknown/draw
        - best_move: The best first move (position, size) or None
        - win_depth: Depth to win
        - stats: Search statistics
        - cache: Memoization cache for strategy lookup
    """
    initial_state = OtrioGameState()
    opponent = "RED" if player == "BLUE" else "BLUE"

    # Memoization cache: state -> (result, best_move, depth_to_win)
    # result: 1 = player wins, -1 = opponent wins, 0 = draw/unknown
    cache = {}
    stats = {"nodes_explored": 0, "cache_hits": 0, "max_depth_seen": 0, "cutoff": False}

    def minimax(state, depth, alpha, beta, maximizing):
        """
        Minimax with alpha-beta pruning.
        Returns: (result, best_move, depth_to_win)
        """
        # Check node limit
        if stats["nodes_explored"] >= max_nodes:
            stats["cutoff"] = True
            return (0, None, depth)

        stats["nodes_explored"] += 1
        stats["max_depth_seen"] = max(stats["max_depth_seen"], depth)

        if verbose and stats["nodes_explored"] % 100000 == 0:
            print(f"  Explored {stats['nodes_explored']:,} nodes (cache size: {len(cache):,})...")

        # Check cache
        state_key = (state.to_tuple(), maximizing)
        if state_key in cache:
            stats["cache_hits"] += 1
            return cache[state_key]

        # Terminal states
        if state.winner == player:
            result = (1, None, depth)  # Player wins
            cache[state_key] = result
            return result
        if state.winner == opponent:
            result = (-1, None, depth)  # Opponent wins
            cache[state_key] = result
            return result

        moves = state.get_valid_moves()
        if not moves or depth >= max_depth:
            result = (0, None, depth)  # Draw or max depth
            cache[state_key] = result
            return result

        best_move = None
        best_depth = max_depth + 1

        if maximizing:  # Player's turn - maximize
            max_eval = -2
            for pos, size in moves:
                new_state = state.make_move(pos, size)
                eval_result, _, win_depth = minimax(new_state, depth + 1, alpha, beta, False)

                # Prefer faster wins
                if eval_result > max_eval or (eval_result == max_eval and win_depth < best_depth):
                    max_eval = eval_result
                    best_move = (pos, size)
                    best_depth = win_depth

                alpha = max(alpha, eval_result)
                if beta <= alpha:
                    break  # Pruning

            result = (max_eval, best_move, best_depth)
            cache[state_key] = result
            return result

        else:  # Opponent's turn - minimize
            min_eval = 2
            for pos, size in moves:
                new_state = state.make_move(pos, size)
                eval_result, _, win_depth = minimax(new_state, depth + 1, alpha, beta, True)

                if eval_result < min_eval or (eval_result == min_eval and win_depth > best_depth):
                    min_eval = eval_result
                    best_move = (pos, size)
                    best_depth = win_depth

                beta = min(beta, eval_result)
                if beta <= alpha:
                    break  # Pruning

            result = (min_eval, best_move, best_depth)
            cache[state_key] = result
            return result

    if verbose:
        print(f"Searching for winning strategy for {player}...")
        print(f"Max depth: {max_depth}")

    # Start search (BLUE moves first, so maximizing if player is BLUE)
    maximizing = (player == "BLUE")
    result, best_move, win_depth = minimax(initial_state, 0, -2, 2, maximizing)

    if verbose:
        print(f"\nSearch complete!")
        print(f"  Nodes explored: {stats['nodes_explored']:,}")
        print(f"  Cache hits: {stats['cache_hits']:,}")
        print(f"  Max depth seen: {stats['max_depth_seen']}")

    return result, best_move, win_depth, stats, cache


def build_strategy_tree(player, cache, max_moves=10, verbose=True):
    """
    Build a strategy tree from the minimax cache.
    Shows the optimal moves for the player at each decision point.
    """
    initial_state = OtrioGameState()
    opponent = "RED" if player == "BLUE" else "BLUE"

    def get_strategy(state, depth, path):
        if depth > max_moves:
            return None

        if state.winner:
            return {"winner": state.winner, "path": path}

        moves = state.get_valid_moves()
        if not moves:
            return {"winner": "DRAW", "path": path}

        is_player_turn = (state.current_player == player)
        state_key = (state.to_tuple(), is_player_turn)

        if state_key in cache:
            result, best_move, win_depth = cache[state_key]

            if best_move and is_player_turn:
                pos, size = best_move
                new_state = state.make_move(pos, size)
                new_path = path + [(state.current_player, pos, size)]

                # Get opponent's responses
                if new_state.winner:
                    return {
                        "move": best_move,
                        "result": "WIN" if new_state.winner == player else "LOSE",
                        "path": new_path
                    }

                # Show what happens for each opponent response
                responses = {}
                for opp_pos, opp_size in new_state.get_valid_moves()[:5]:  # Limit responses shown
                    opp_state = new_state.make_move(opp_pos, opp_size)
                    sub_strategy = get_strategy(opp_state, depth + 2,
                                               new_path + [(opponent, opp_pos, opp_size)])
                    responses[(opp_pos, opp_size)] = sub_strategy

                return {
                    "move": best_move,
                    "win_in": win_depth - depth,
                    "responses": responses
                }

        return None

    return get_strategy(initial_state, 0, [])


def print_winning_strategy(result, best_move, win_depth, player="BLUE", cache=None, stats=None):
    """Print the winning strategy result."""
    print("\n" + "=" * 60)
    print(f"WINNING STRATEGY ANALYSIS FOR {player}")
    print("=" * 60)

    if stats and stats.get("cutoff"):
        print(f"\n[WARNING: Search was cut off at {stats['nodes_explored']:,} nodes]")
        print(f"[Results may be incomplete - increase --max-nodes for full search]")

    if result == 1:
        print(f"\n*** {player} HAS A GUARANTEED WIN! ***")
        print(f"Best first move: {best_move[1]} at {best_move[0]}")
        print(f"Wins in at most {win_depth} moves")

        # Show the winning path
        if cache:
            print("\n" + "-" * 60)
            print("WINNING STRATEGY (one example path):")
            print("-" * 60)
            show_winning_path(player, cache)

    elif result == -1:
        opponent = "RED" if player == "BLUE" else "BLUE"
        print(f"\n{opponent} has a guaranteed win with perfect play.")
        print(f"{player} cannot force a win.")
    else:
        print(f"\nNo guaranteed win found within search limits.")
        if best_move:
            print(f"Best move found so far: {best_move[1]} at {best_move[0]}")
        print(f"Game may be a draw with perfect play, or deeper search needed.")


def show_winning_path(player, cache):
    """
    Display the winning path from the cache.
    Shows player's optimal moves and one example of opponent's response.
    """
    state = OtrioGameState()
    opponent = "RED" if player == "BLUE" else "BLUE"
    move_num = 1

    print(f"\nNote: >>> marks {player}'s optimal moves")
    print(f"      {opponent}'s moves shown are just ONE example response.\n")

    while not state.winner:
        current = state.current_player
        is_maximizing = (current == player)
        state_key = (state.to_tuple(), is_maximizing)

        if state_key not in cache:
            print(f"  (strategy not cached for this state)")
            break

        result, best_move, _ = cache[state_key]

        if best_move is None:
            break

        pos, size = best_move

        # Display move
        marker = ">>>" if current == player else "   "
        print(f"{marker} {move_num}. {current}: {size[0]} at {pos}")

        state = state.make_move(pos, size)
        move_num += 1

        if state.winner:
            print(f"\n    === {state.winner} WINS! ===")
            break

        # Safety limit
        if move_num > 20:
            print("  (path truncated)")
            break

    # Show the full strategy tree for key decision points
    print("\n" + "-" * 60)
    print(f"FULL STRATEGY TREE (what {player} plays vs any {opponent} response):")
    print("-" * 60)
    show_strategy_tree(player, cache, max_depth=6)


def show_strategy_tree(player, cache, max_depth=6):
    """
    Display strategy tree showing player's responses to all opponent moves.
    """
    opponent = "RED" if player == "BLUE" else "BLUE"

    def get_optimal_move(state):
        """Get the optimal move for current player from cache."""
        is_maximizing = (state.current_player == player)
        state_key = (state.to_tuple(), is_maximizing)
        if state_key in cache:
            _, best_move, _ = cache[state_key]
            return best_move
        return None

    def format_move(pos, size):
        return f"{size[0]}@{pos}"

    def print_tree(state, depth, indent, last_move_str=""):
        if depth > max_depth or state.winner:
            if state.winner:
                print(f"{indent}  -> {state.winner} WINS")
            return

        current = state.current_player
        optimal = get_optimal_move(state)

        if optimal is None:
            return

        if current == player:
            # Player's turn: show the optimal move
            pos, size = optimal
            print(f"{indent}{player}: {format_move(pos, size)}")
            new_state = state.make_move(pos, size)
            print_tree(new_state, depth + 1, indent, format_move(pos, size))
        else:
            # Opponent's turn: show all responses (limited) and player's counter
            moves = state.get_valid_moves()
            # Limit number of branches shown
            shown_moves = moves[:5] if depth < 4 else moves[:2]

            for i, (pos, size) in enumerate(shown_moves):
                prefix = "|-" if i < len(shown_moves) - 1 else "+-"
                new_indent = indent + ("| " if i < len(shown_moves) - 1 else "  ")

                new_state = state.make_move(pos, size)
                print(f"{indent}{prefix} if {opponent} plays {format_move(pos, size)}:")
                print_tree(new_state, depth + 1, new_indent, format_move(pos, size))

            if len(moves) > len(shown_moves):
                print(f"{indent}   ... and {len(moves) - len(shown_moves)} more {opponent} options (all lead to {player} win)")

    # Start from initial state
    initial = OtrioGameState()
    print_tree(initial, 0, "")


# =============================================================================
# Compact State Encoding for Efficient BFS
# =============================================================================

# Position indices: A1=0, A2=1, A3=2, B1=3, B2=4, B3=5, C1=6, C2=7, C3=8
POS_TO_IDX = {f"{r}{c}": i for i, (r, c) in enumerate(
    [(r, c) for r in "ABC" for c in "123"]
)}
IDX_TO_POS = {i: pos for pos, i in POS_TO_IDX.items()}

# Size indices: SMALL=0, MEDIUM=1, LARGE=2
SIZE_TO_IDX = {"SMALL": 0, "MEDIUM": 1, "LARGE": 2}
IDX_TO_SIZE = {0: "SMALL", 1: "MEDIUM", 2: "LARGE"}


def encode_state(state):
    """
    Encode game state as a 55-bit integer.

    Layout (55 bits total):
    - Bits 0-26: BLUE board (9 positions × 3 sizes = 27 bits)
    - Bits 27-53: RED board (9 positions × 3 sizes = 27 bits)
    - Bit 54: Turn (0=BLUE, 1=RED)

    Each position has 3 bits: bit 0=SMALL present, bit 1=MEDIUM, bit 2=LARGE
    """
    encoded = 0

    # Encode BLUE board (bits 0-26)
    for pos_idx in range(9):
        pos = IDX_TO_POS[pos_idx]
        sizes = state.board["BLUE"][pos]
        for size_idx, size in enumerate(["SMALL", "MEDIUM", "LARGE"]):
            if size in sizes:
                bit_pos = pos_idx * 3 + size_idx
                encoded |= (1 << bit_pos)

    # Encode RED board (bits 27-53)
    for pos_idx in range(9):
        pos = IDX_TO_POS[pos_idx]
        sizes = state.board["RED"][pos]
        for size_idx, size in enumerate(["SMALL", "MEDIUM", "LARGE"]):
            if size in sizes:
                bit_pos = 27 + pos_idx * 3 + size_idx
                encoded |= (1 << bit_pos)

    # Encode turn (bit 54)
    if state.current_player == "RED":
        encoded |= (1 << 54)

    return encoded


def decode_move(move_int):
    """Decode a move from compact integer (0-26) to (position, size)."""
    pos_idx = move_int // 3
    size_idx = move_int % 3
    return (IDX_TO_POS[pos_idx], IDX_TO_SIZE[size_idx])


def encode_move(pos, size):
    """Encode a move (position, size) to compact integer (0-26)."""
    return POS_TO_IDX[pos] * 3 + SIZE_TO_IDX[size]


def apply_moves(moves_list):
    """Apply a sequence of encoded moves and return the final state."""
    state = OtrioGameState()
    for move_int in moves_list:
        pos, size = decode_move(move_int)
        state = state.make_move(pos, size)
    return state


def get_valid_moves_fast(state):
    """Get valid moves as list of encoded integers (0-26)."""
    moves = []
    player = state.current_player
    opponent = "RED" if player == "BLUE" else "BLUE"

    for size_idx, size in enumerate(["SMALL", "MEDIUM", "LARGE"]):
        if state.initial[player][size] > 0:
            for pos_idx in range(9):
                pos = IDX_TO_POS[pos_idx]
                # Check blocking rule
                if size not in state.board[player][pos] and size not in state.board[opponent][pos]:
                    moves.append(pos_idx * 3 + size_idx)
    return moves


def compact_bfs(max_depth=5, verbose=True):
    """
    Memory-efficient BFS using compact state encoding.

    Uses 55-bit integers for state keys instead of nested tuples.
    Implements iterative deepening to find shortest wins first.
    """
    from collections import deque

    if verbose:
        print(f"Compact BFS (max_depth={max_depth})...")

    initial = OtrioGameState()
    initial_key = encode_state(initial)

    # visited: state_key (int) -> depth
    visited = {initial_key: 0}

    # Queue: (state, depth, moves_sequence)
    # moves_sequence is a tuple of encoded move integers
    queue = deque([(initial, 0, ())])

    stats = {
        "total": 0,
        "by_depth": defaultdict(int),
        "blue_wins": 0,
        "red_wins": 0,
        "first_blue_win_depth": None,
        "first_red_win_depth": None,
        "first_blue_win_moves": None,
        "first_red_win_moves": None,
    }

    while queue:
        state, depth, moves_seq = queue.popleft()
        stats["total"] += 1
        stats["by_depth"][depth] += 1

        if verbose and stats["total"] % 50000 == 0:
            print(f"  States: {stats['total']:,}, depth={depth}, queue={len(queue):,}, visited={len(visited):,}")

        # Check for win
        if state.winner:
            if state.winner == "BLUE":
                stats["blue_wins"] += 1
                if stats["first_blue_win_depth"] is None:
                    stats["first_blue_win_depth"] = depth
                    stats["first_blue_win_moves"] = moves_seq
                    if verbose:
                        print(f"  *** BLUE wins at depth {depth}! ***")
            else:
                stats["red_wins"] += 1
                if stats["first_red_win_depth"] is None:
                    stats["first_red_win_depth"] = depth
                    stats["first_red_win_moves"] = moves_seq
                    if verbose:
                        print(f"  *** RED wins at depth {depth}! ***")
            continue

        # Don't expand beyond max_depth
        if depth >= max_depth:
            continue

        # Get valid moves and expand
        for move_int in get_valid_moves_fast(state):
            pos, size = decode_move(move_int)
            new_state = state.make_move(pos, size)
            new_key = encode_state(new_state)

            if new_key not in visited:
                visited[new_key] = depth + 1
                queue.append((new_state, depth + 1, moves_seq + (move_int,)))

    if verbose:
        print(f"\nCompact BFS Complete:")
        print(f"  Total states: {stats['total']:,}")
        print(f"  Unique states visited: {len(visited):,}")
        print(f"  BLUE wins: {stats['blue_wins']}")
        print(f"  RED wins: {stats['red_wins']}")
        print(f"  States by depth: {dict(stats['by_depth'])}")

        if stats["first_blue_win_moves"]:
            print(f"\n  First BLUE win at depth {stats['first_blue_win_depth']}:")
            print(f"    Moves: {[decode_move(m) for m in stats['first_blue_win_moves']]}")

        if stats["first_red_win_moves"]:
            print(f"\n  First RED win at depth {stats['first_red_win_depth']}:")
            print(f"    Moves: {[decode_move(m) for m in stats['first_red_win_moves']]}")

    return visited, stats


def iterative_deepening_bfs(max_depth=10, target_player="BLUE", verbose=True):
    """
    Iterative deepening BFS to find the shortest winning path.

    Searches depth 1, 2, 3, ... until finding a guaranteed win for target_player.
    Stops as soon as a win is found (shortest path).
    """
    from collections import deque

    if verbose:
        print(f"Iterative Deepening BFS for {target_player} win...")
        print(f"  Max depth: {max_depth}")

    for current_max_depth in range(1, max_depth + 1):
        if verbose:
            print(f"\n--- Searching depth {current_max_depth} ---")

        initial = OtrioGameState()
        initial_key = encode_state(initial)

        visited = {initial_key: 0}
        queue = deque([(initial, 0, ())])

        states_explored = 0
        wins_found = []

        while queue:
            state, depth, moves_seq = queue.popleft()
            states_explored += 1

            if verbose and states_explored % 100000 == 0:
                print(f"    States: {states_explored:,}, queue={len(queue):,}")

            # Check for target player win
            if state.winner == target_player:
                wins_found.append(moves_seq)
                # Don't expand further from winning states
                continue

            # Check for opponent win (dead end for us)
            if state.winner:
                continue

            # Don't expand beyond current depth limit
            if depth >= current_max_depth:
                continue

            # Expand
            for move_int in get_valid_moves_fast(state):
                pos, size = decode_move(move_int)
                new_state = state.make_move(pos, size)
                new_key = encode_state(new_state)

                if new_key not in visited:
                    visited[new_key] = depth + 1
                    queue.append((new_state, depth + 1, moves_seq + (move_int,)))

        if verbose:
            print(f"    Explored {states_explored:,} states")
            print(f"    {target_player} wins found: {len(wins_found)}")

        if wins_found:
            if verbose:
                print(f"\n*** Found {len(wins_found)} winning path(s) at depth {current_max_depth}! ***")
                # Show first winning path
                first_win = wins_found[0]
                print(f"\nShortest winning sequence ({len(first_win)} moves):")
                state = OtrioGameState()
                for i, move_int in enumerate(first_win):
                    pos, size = decode_move(move_int)
                    player = state.current_player
                    print(f"  {i+1}. {player}: {size} at {pos}")
                    state = state.make_move(pos, size)
                print(f"\n  Result: {state.winner} WINS!")

            return current_max_depth, wins_found

    if verbose:
        print(f"\nNo {target_player} win found within depth {max_depth}")

    return None, []


# =============================================================================
# BFS-Based Shortest Path Strategy (Finding Optimal Winning Moves)
# =============================================================================

def simple_bfs_explore(max_depth=5, verbose=True):
    """
    Simple, lightweight BFS exploration of the game tree.

    Only stores minimal data: state_key -> depth
    Prints statistics as it goes.

    Args:
        max_depth: Maximum depth to explore
        verbose: Print progress

    Returns:
        - visited: dict of state_key -> depth
        - stats: exploration statistics
    """
    from collections import deque

    if verbose:
        print(f"Simple BFS exploration (max_depth={max_depth})...")

    initial = OtrioGameState()
    initial_key = initial.to_tuple()

    # Only store: state_key -> depth (minimal memory)
    visited = {initial_key: 0}

    # Queue contains: (state, depth) - we recreate states as needed
    queue = deque([(initial, 0)])

    stats = {
        "total": 0,
        "by_depth": defaultdict(int),
        "blue_wins": 0,
        "red_wins": 0,
        "terminals": 0,
    }

    while queue:
        state, depth = queue.popleft()
        stats["total"] += 1
        stats["by_depth"][depth] += 1

        if verbose and stats["total"] % 10000 == 0:
            print(f"  States: {stats['total']:,}, depth={depth}, queue={len(queue):,}")

        # Check for win
        if state.winner:
            stats["terminals"] += 1
            if state.winner == "BLUE":
                stats["blue_wins"] += 1
            else:
                stats["red_wins"] += 1
            continue

        # Don't expand beyond max_depth
        if depth >= max_depth:
            stats["terminals"] += 1
            continue

        # Get valid moves and expand
        for pos, size in state.get_valid_moves():
            new_state = state.make_move(pos, size)
            new_key = new_state.to_tuple()

            # Only add if not seen before
            if new_key not in visited:
                visited[new_key] = depth + 1
                queue.append((new_state, depth + 1))

    if verbose:
        print(f"\nBFS Complete:")
        print(f"  Total states: {stats['total']:,}")
        print(f"  BLUE wins: {stats['blue_wins']}")
        print(f"  RED wins: {stats['red_wins']}")
        print(f"  States by depth: {dict(stats['by_depth'])}")

    return visited, stats


def find_shortest_path_wins(verbose=True, max_states=0, max_depth=10):
    """
    Use BFS to find the shortest path from any state to a winning terminal state.

    This implements backward induction:
    1. First, find all terminal states (wins) using BFS
    2. Then, propagate distances backward to find optimal moves

    Args:
        verbose: Print progress
        max_states: Maximum states to explore (0 = no limit, explore all)
        max_depth: Maximum depth (moves) to explore (default: 10)

    Returns:
        - win_distances: dict mapping state -> (winner, min_moves_to_win)
        - optimal_moves: dict mapping state -> best_move for current player
        - stats: exploration statistics
    """
    from collections import deque

    initial_state = OtrioGameState()

    # Phase 1: BFS to discover all reachable states and their depths
    if verbose:
        print("Phase 1: BFS exploration to find all states...")
        if max_depth > 0:
            print(f"  (limited to depth {max_depth})")
        if max_states > 0:
            print(f"  (limited to {max_states:,} states)")

    visited = {}  # state_tuple -> (state, depth, parent_state, move_to_reach)
    terminal_states = []  # [(state, depth, winner)]

    queue = deque([(initial_state, 0, None, None)])  # (state, depth, parent, move)

    stats = {
        "total_states": 0,
        "terminal_states": 0,
        "blue_wins": 0,
        "red_wins": 0,
        "draws": 0,
        "min_win_depth": {"BLUE": float('inf'), "RED": float('inf')},
        "states_by_depth": defaultdict(int),
    }

    while queue:
        # Check state limit
        if max_states > 0 and stats["total_states"] >= max_states:
            if verbose:
                print(f"  Reached state limit ({max_states:,}), stopping exploration...")
            break

        state, depth, parent, move = queue.popleft()
        state_key = state.to_tuple()

        if state_key in visited:
            continue

        visited[state_key] = (state, depth, parent, move)
        stats["total_states"] += 1
        stats["states_by_depth"][depth] += 1

        if verbose and stats["total_states"] % 50000 == 0:
            print(f"  Explored {stats['total_states']:,} states (depth {depth})...")

        # Check terminal state
        if state.winner:
            terminal_states.append((state, depth, state.winner))
            stats["terminal_states"] += 1
            stats[f"{state.winner.lower()}_wins"] += 1
            stats["min_win_depth"][state.winner] = min(stats["min_win_depth"][state.winner], depth)
            continue

        moves = state.get_valid_moves()
        if not moves:
            terminal_states.append((state, depth, None))  # Draw
            stats["terminal_states"] += 1
            stats["draws"] += 1
            continue

        # Check depth limit - treat as terminal if at max depth
        if max_depth > 0 and depth >= max_depth:
            # At max depth, don't expand further - mark as unknown terminal
            stats["terminal_states"] += 1
            continue

        # Explore all successors (BFS guarantees shortest path)
        for pos, size in moves:
            new_state = state.make_move(pos, size)
            new_key = new_state.to_tuple()
            if new_key not in visited:
                queue.append((new_state, depth + 1, state_key, (pos, size)))

    if verbose:
        print(f"  Found {stats['total_states']:,} states, {stats['terminal_states']} terminal")
        print(f"  Earliest BLUE win: move {stats['min_win_depth']['BLUE']}")
        print(f"  Earliest RED win: move {stats['min_win_depth']['RED']}")

    # Phase 2: Backward induction to compute optimal values
    if verbose:
        print("\nPhase 2: Backward induction for optimal strategy...")

    # For each state, compute: (game_value, moves_to_end, best_move)
    # game_value: 1 = BLUE wins, -1 = RED wins, 0 = draw
    # moves_to_end: minimum moves to reach terminal state with optimal play
    state_values = {}  # state_key -> (value, moves_to_end, best_move)

    # Initialize terminal states
    for state, depth, winner in terminal_states:
        state_key = state.to_tuple()
        if winner == "BLUE":
            state_values[state_key] = (1, 0, None)
        elif winner == "RED":
            state_values[state_key] = (-1, 0, None)
        else:
            state_values[state_key] = (0, 0, None)

    # Process states in reverse BFS order (highest depth first)
    max_depth_found = max(stats["states_by_depth"].keys())

    for depth in range(max_depth_found, -1, -1):
        if verbose and depth % 5 == 0:
            print(f"  Processing depth {depth}...")

        # Get all states at this depth
        states_at_depth = [(k, v) for k, v in visited.items()
                          if v[1] == depth and k not in state_values]

        for state_key, (state, d, parent, move) in states_at_depth:
            moves = state.get_valid_moves()

            if not moves:
                # Terminal state (draw) - should already be in state_values
                continue

            player = state.current_player

            # Compute value based on successors
            best_value = None
            best_moves_to_end = None
            best_move = None

            for pos, size in moves:
                new_state = state.make_move(pos, size)
                new_key = new_state.to_tuple()

                if new_key not in state_values:
                    # Successor not yet computed - this shouldn't happen in reverse BFS
                    continue

                succ_value, succ_moves, _ = state_values[new_key]
                moves_to_end = succ_moves + 1

                if player == "BLUE":
                    # BLUE maximizes value, prefers faster wins
                    if best_value is None:
                        best_value = succ_value
                        best_moves_to_end = moves_to_end
                        best_move = (pos, size)
                    elif succ_value > best_value:
                        best_value = succ_value
                        best_moves_to_end = moves_to_end
                        best_move = (pos, size)
                    elif succ_value == best_value:
                        # Same value: prefer shorter path for wins, longer for losses
                        if succ_value > 0 and moves_to_end < best_moves_to_end:
                            best_moves_to_end = moves_to_end
                            best_move = (pos, size)
                        elif succ_value < 0 and moves_to_end > best_moves_to_end:
                            best_moves_to_end = moves_to_end
                            best_move = (pos, size)
                else:
                    # RED minimizes value, prefers faster wins
                    if best_value is None:
                        best_value = succ_value
                        best_moves_to_end = moves_to_end
                        best_move = (pos, size)
                    elif succ_value < best_value:
                        best_value = succ_value
                        best_moves_to_end = moves_to_end
                        best_move = (pos, size)
                    elif succ_value == best_value:
                        # Same value: prefer shorter path for wins, longer for losses
                        if succ_value < 0 and moves_to_end < best_moves_to_end:
                            best_moves_to_end = moves_to_end
                            best_move = (pos, size)
                        elif succ_value > 0 and moves_to_end > best_moves_to_end:
                            best_moves_to_end = moves_to_end
                            best_move = (pos, size)

            if best_value is not None:
                state_values[state_key] = (best_value, best_moves_to_end, best_move)

    if verbose:
        initial_key = initial_state.to_tuple()
        if initial_key in state_values:
            value, moves, best_move = state_values[initial_key]
            winner = "BLUE" if value > 0 else ("RED" if value < 0 else "DRAW")
            print(f"\n  Initial state value: {value} ({winner})")
            print(f"  Optimal play ends in {moves} moves")
            if best_move:
                print(f"  Best opening move: {best_move[1]} at {best_move[0]}")

    return state_values, visited, stats


def find_shortest_win_strategy(player="BLUE", verbose=True, max_states=0, max_depth=10):
    """
    Find the shortest-path winning strategy for a player using complete BFS.

    This explores the game tree up to max_depth and uses backward induction to find
    the optimal strategy that minimizes moves to win.

    Args:
        player: "BLUE" or "RED"
        verbose: Print progress
        max_states: Maximum states to explore (0 = no limit)
        max_depth: Maximum depth (moves) to explore (default: 10)

    Returns:
        - result: 1 = player can force win, -1 = opponent wins, 0 = draw
        - optimal_moves: dict mapping state_key -> (best_pos, best_size)
        - win_in_moves: number of moves to win with optimal play
        - stats: exploration statistics
        - state_values: computed state values for reuse
        - visited: visited states dict for reuse
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"SHORTEST-PATH WINNING STRATEGY FOR {player}")
        print(f"{'='*60}")

    state_values, visited, stats = find_shortest_path_wins(verbose, max_states, max_depth)

    initial_state = OtrioGameState()
    initial_key = initial_state.to_tuple()

    if initial_key not in state_values:
        if verbose:
            print("\nError: Initial state not in computed values!")
        return 0, {}, 0, stats

    value, moves_to_end, best_move = state_values[initial_key]

    # Extract optimal moves for the target player
    optimal_moves = {}
    for state_key, (val, moves, best) in state_values.items():
        if best is not None:
            state, depth, _, _ = visited[state_key]
            if state.current_player == player:
                optimal_moves[state_key] = best

    # Determine result from player's perspective
    if player == "BLUE":
        result = value
    else:
        result = -value

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULT")
        print(f"{'='*60}")
        if result > 0:
            print(f"*** {player} HAS A GUARANTEED WIN ***")
            print(f"With optimal play, {player} wins in {moves_to_end} moves")
        elif result < 0:
            opponent = "RED" if player == "BLUE" else "BLUE"
            print(f"*** {opponent} HAS A GUARANTEED WIN ***")
            print(f"{player} cannot force a win")
        else:
            print(f"Game is a DRAW with optimal play")

        if best_move:
            print(f"\nBest opening move: {best_move[1]} at {best_move[0]}")

        print(f"\nStatistics:")
        print(f"  Total states explored: {stats['total_states']:,}")
        print(f"  Terminal states: {stats['terminal_states']}")
        print(f"  BLUE wins: {stats['blue_wins']}")
        print(f"  RED wins: {stats['red_wins']}")
        print(f"  Draws: {stats['draws']}")

    return result, optimal_moves, moves_to_end, stats, state_values, visited


def show_shortest_win_path(player="BLUE", state_values=None, visited=None):
    """
    Show the shortest winning path for a player.
    """
    if state_values is None:
        state_values, visited, _ = find_shortest_path_wins(verbose=False)

    state = OtrioGameState()
    opponent = "RED" if player == "BLUE" else "BLUE"
    move_num = 0

    print(f"\n{'='*60}")
    print(f"SHORTEST WINNING PATH FOR {player}")
    print(f"{'='*60}")
    print(f"\nNote: >>> marks {player}'s optimal moves")
    print(f"      {opponent}'s moves are also optimal (best defense)\n")

    while not state.winner:
        state_key = state.to_tuple()

        if state_key not in state_values:
            print(f"  (state not in computed values)")
            break

        value, moves_left, best_move = state_values[state_key]

        if best_move is None:
            print(f"  (no move found)")
            break

        pos, size = best_move
        current = state.current_player

        marker = ">>>" if current == player else "   "
        value_str = f"(value={value:+d}, {moves_left} to end)"
        print(f"{marker} {move_num + 1}. {current}: {size[0]} at {pos} {value_str}")

        state = state.make_move(pos, size)
        move_num += 1

        if state.winner:
            print(f"\n    === {state.winner} WINS in {move_num} moves! ===")
            break

        if move_num > 20:
            print("  (path truncated)")
            break


# =============================================================================
# Strategy Verification - Exhaustive Check Against All Opponent Responses
# =============================================================================

def verify_strategy(verbose=True):
    """
    Verify the user's winning strategy by exploring ALL possible RED responses.

    The strategy is:
    1B: BLUE plays M at B2 (center)
    1R: RED plays anything
    2B: BLUE responds based on RED's move:
        - If RED plays S or L at B2: BLUE plays L elsewhere (e.g., A1)
        - If RED plays elsewhere: BLUE plays same size contiguous to RED's move
    2R: RED must defend
    3B: BLUE plays S at same position as previous L (creating nested or double threat)

    This function exhaustively explores ALL RED responses to verify BLUE always wins.
    """

    print("=" * 70)
    print("STRATEGY VERIFICATION: Checking BLUE wins against ALL RED responses")
    print("=" * 70)

    # Step 1: BLUE plays M at B2
    initial = OtrioGameState()
    state_after_1b = initial.make_move("B2", "MEDIUM")

    print(f"\n1B: BLUE plays MEDIUM at B2 (center)")

    # Get ALL possible RED responses
    red_moves = state_after_1b.get_valid_moves()
    print(f"\n1R: RED has {len(red_moves)} possible responses")

    total_branches = 0
    blue_wins = 0
    red_wins = 0
    red_blocks = []  # Track any RED move that blocks the strategy

    # Explore ALL RED responses
    for red_pos, red_size in red_moves:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Analyzing: RED plays {red_size} at {red_pos}")
            print(f"{'='*60}")

        state_after_1r = state_after_1b.make_move(red_pos, red_size)

        # Determine BLUE's response based on the strategy
        if red_pos == "B2":
            # RED plays at center - BLUE plays L elsewhere
            # Try A1 first, or find another position
            blue_2b_moves = []
            for pos in POSITIONS:
                if pos != "B2":
                    blue_2b_moves.append((pos, "LARGE"))

            # For each possible BLUE response, check if we can win
            for blue_2b_pos, blue_2b_size in blue_2b_moves:
                result = analyze_branch(
                    state_after_1r,
                    blue_2b_pos, blue_2b_size,
                    f"RED {red_size} at {red_pos}",
                    verbose=verbose
                )
                total_branches += 1
                if result == "BLUE":
                    blue_wins += 1
                    break  # Found a winning response for this RED move
                elif result == "RED":
                    red_wins += 1
        else:
            # RED plays elsewhere - BLUE plays same size contiguous to RED's move
            # Find contiguous positions to RED's move
            contiguous = get_contiguous_positions(red_pos)

            blue_response_found = False
            for cont_pos in contiguous:
                # Try to play RED's size at contiguous position
                try:
                    state_test = state_after_1r.copy()
                    # Check if this move is valid
                    moves = state_test.get_valid_moves()
                    if (cont_pos, red_size) in moves:
                        result = analyze_branch(
                            state_after_1r,
                            cont_pos, red_size,
                            f"RED {red_size} at {red_pos}",
                            verbose=verbose
                        )
                        total_branches += 1
                        if result == "BLUE":
                            blue_wins += 1
                            blue_response_found = True
                            break
                        elif result == "RED":
                            red_wins += 1
                except:
                    pass

            if not blue_response_found:
                # Try alternative responses - play L at any position to create threat
                for pos in POSITIONS:
                    if pos != red_pos and pos != "B2":
                        moves = state_after_1r.get_valid_moves()
                        if (pos, "LARGE") in moves:
                            result = analyze_branch(
                                state_after_1r,
                                pos, "LARGE",
                                f"RED {red_size} at {red_pos}",
                                verbose=verbose
                            )
                            total_branches += 1
                            if result == "BLUE":
                                blue_wins += 1
                                blue_response_found = True
                                break

            if not blue_response_found:
                red_blocks.append((red_pos, red_size))
                if verbose:
                    print(f"  WARNING: No winning response found for RED {red_size} at {red_pos}")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"Total RED responses analyzed: {len(red_moves)}")
    print(f"Total branches explored: {total_branches}")
    print(f"BLUE wins: {blue_wins}")
    print(f"RED wins: {red_wins}")

    if red_blocks:
        print(f"\nRED blocking moves found ({len(red_blocks)}):")
        for pos, size in red_blocks:
            print(f"  - {size} at {pos}")
        return False
    else:
        print(f"\n*** STRATEGY VERIFIED: BLUE wins against ALL RED responses! ***")
        return True


def get_contiguous_positions(pos):
    """Get positions contiguous to the given position (same row, column, or diagonal)."""
    row = pos[0]
    col = pos[1]

    contiguous = []

    # Same row
    for c in "123":
        if c != col:
            contiguous.append(f"{row}{c}")

    # Same column
    for r in "ABC":
        if r != row:
            contiguous.append(f"{r}{col}")

    # Diagonals
    row_idx = "ABC".index(row)
    col_idx = "123".index(col)

    # Main diagonal (A1-B2-C3)
    if row_idx == col_idx or abs(row_idx - col_idx) <= 1:
        for i in range(3):
            r = "ABC"[i]
            c = "123"[i]
            if f"{r}{c}" != pos and f"{r}{c}" not in contiguous:
                # Only add if on same diagonal
                if i - row_idx == int(c) - 1 - col_idx:
                    contiguous.append(f"{r}{c}")

    # Anti-diagonal (A3-B2-C1)
    if row_idx + col_idx == 2 or abs((row_idx + col_idx) - 2) <= 1:
        for i in range(3):
            r = "ABC"[i]
            c = "123"[2-i]
            if f"{r}{c}" != pos and f"{r}{c}" not in contiguous:
                if i + (2-i) == row_idx + col_idx:
                    contiguous.append(f"{r}{c}")

    return contiguous


def analyze_branch(state, blue_pos, blue_size, context, verbose=True, depth=0, max_depth=6):
    """
    Analyze a branch of the game tree from the given state.
    BLUE plays (blue_pos, blue_size), then we explore ALL RED responses.

    Uses minimax to determine if BLUE can force a win from this position.
    Returns: "BLUE", "RED", or "DRAW"
    """
    if depth > max_depth:
        return "UNKNOWN"

    indent = "  " * depth

    # BLUE makes the move
    try:
        state_after_blue = state.make_move(blue_pos, blue_size)
    except ValueError as e:
        if verbose:
            print(f"{indent}Invalid move: {blue_size} at {blue_pos}: {e}")
        return "INVALID"

    if verbose and depth <= 2:
        move_num = depth // 2 + 2
        print(f"{indent}{move_num}B: BLUE plays {blue_size} at {blue_pos}")

    # Check if BLUE won
    if state_after_blue.winner == "BLUE":
        if verbose and depth <= 2:
            print(f"{indent}    -> BLUE WINS!")
        return "BLUE"

    # Get RED's possible responses
    red_moves = state_after_blue.get_valid_moves()

    if not red_moves:
        return "DRAW"

    # For each RED response, find BLUE's best counter
    # BLUE wins if there exists a winning response for ALL RED moves
    all_blue_wins = True

    for red_pos, red_size in red_moves:
        try:
            state_after_red = state_after_blue.make_move(red_pos, red_size)
        except:
            continue

        if verbose and depth <= 1:
            move_num = depth // 2 + 2
            print(f"{indent}  {move_num}R: if RED plays {red_size} at {red_pos}...")

        # Check if RED won
        if state_after_red.winner == "RED":
            if verbose and depth <= 1:
                print(f"{indent}      -> RED WINS (strategy fails here)")
            return "RED"

        # Find BLUE's best response to this RED move
        blue_responses = state_after_red.get_valid_moves()

        if not blue_responses:
            all_blue_wins = False
            continue

        # Try each BLUE response - we need at least ONE winning response
        found_winning_response = False
        for b_pos, b_size in blue_responses:
            result = analyze_branch(
                state_after_red,
                b_pos, b_size,
                f"{context} -> RED {red_size}@{red_pos}",
                verbose=False,  # Reduce verbosity in deep recursion
                depth=depth + 2,
                max_depth=max_depth
            )
            if result == "BLUE":
                found_winning_response = True
                if verbose and depth <= 1:
                    print(f"{indent}      -> BLUE wins with {b_size} at {b_pos}")
                break

        if not found_winning_response:
            all_blue_wins = False
            if verbose and depth <= 1:
                print(f"{indent}      -> No winning response found")

    return "BLUE" if all_blue_wins else "UNKNOWN"


def explore_game_tree(output_file, verbose=True, max_depth=None):
    """
    Exhaustively explore ALL game sequences until terminal states.
    Streams directly to JSON file to minimize memory usage.

    JSON structure:
    - Internal: {"m": "SA1", "c": [...]}
    - Leaf: {"m": "LC3", "r": "B"|"R"|"D"|"L"}
      r = B (BLUE wins), R (RED wins), D (DRAW), L (depth Limit)
    """
    print("=" * 60)
    print("EXHAUSTIVE GAME TREE (streaming)")
    print("=" * 60)
    print(f"Output: {output_file}")
    if max_depth:
        print(f"Max depth: {max_depth}")
    print()

    stats = {"nodes": 0, "B": 0, "R": 0, "D": 0, "L": 0}
    last_report = [0]

    def get_mem_mb():
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0

    def report():
        mem = get_mem_mb()
        mem_str = f"  Mem: {mem:.0f}MB" if mem > 0 else ""
        limit_str = f" L:{stats['L']:,}" if max_depth else ""
        print(f"  Nodes: {stats['nodes']:,}  B:{stats['B']:,} R:{stats['R']:,} D:{stats['D']:,}{limit_str}{mem_str}")

    def write_tree(f, state, is_first_child, depth):
        """Stream tree node directly to file."""
        stats["nodes"] += 1

        if stats["nodes"] - last_report[0] >= 100000:
            last_report[0] = stats["nodes"]
            if verbose:
                report()

        # Get move that led here (for non-root)
        if state.move_history:
            _, pos, size = state.move_history[-1]
            move_str = f"{size[0]}{pos}"
        else:
            move_str = None

        # Comma separator for non-first children
        if not is_first_child:
            f.write(",")

        # Terminal: BLUE wins
        if state.winner == "BLUE":
            stats["B"] += 1
            f.write(f'{{"m":"{move_str}","r":"B"}}')
            return

        # Terminal: RED wins
        if state.winner == "RED":
            stats["R"] += 1
            f.write(f'{{"m":"{move_str}","r":"R"}}')
            return

        moves = state.get_valid_moves()

        # Terminal: DRAW
        if not moves:
            stats["D"] += 1
            f.write(f'{{"m":"{move_str}","r":"D"}}')
            return

        # Terminal: Depth limit (depth = number of moves played)
        if max_depth and depth >= max_depth:
            stats["L"] += 1
            f.write(f'{{"m":"{move_str}","r":"L"}}')
            return

        # Internal node
        if move_str:
            f.write(f'{{"m":"{move_str}","c":[')
        else:
            f.write('{"c":[')

        # Write children
        for i, (pos, size) in enumerate(moves):
            new_state = state.make_move(pos, size)
            write_tree(f, new_state, i == 0, depth + 1)

        f.write("]}\n")

    print("Streaming to file...")
    if verbose:
        report()

    with open(output_file, 'w') as f:
        f.write('{"tree":')
        write_tree(f, OtrioGameState(), True, 0)
        f.write(f'\n,"stats":{{"nodes":{stats["nodes"]},"B":{stats["B"]},"R":{stats["R"]},"D":{stats["D"]},"L":{stats["L"]}}}}}\n')

    print()
    print("=" * 60)
    print(f"Nodes: {stats['nodes']:,}")
    print(f"BLUE wins: {stats['B']:,}")
    print(f"RED wins: {stats['R']:,}")
    print(f"Draws: {stats['D']:,}")
    if max_depth:
        print(f"Depth limit: {stats['L']:,}")
    print(f"\nWrote to {output_file}")


def explore_game_tree_parallel(output_file, verbose=True, max_depth=None, num_workers=None):
    """
    Multi-threaded exhaustive game tree exploration.
    Parallelizes at the opening move level (27 moves across N workers).
    Each worker writes to a temp file, then files are merged.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import tempfile
    import os
    import time

    if num_workers is None:
        num_workers = mp.cpu_count()

    print("=" * 60)
    print("EXHAUSTIVE GAME TREE (parallel)")
    print("=" * 60)
    print(f"Output: {output_file}")
    print(f"Workers: {num_workers}")
    if max_depth:
        print(f"Max depth: {max_depth}")
    print()

    # Get all opening moves
    initial_state = OtrioGameState()
    opening_moves = initial_state.get_valid_moves()
    print(f"Parallelizing {len(opening_moves)} opening moves...")
    print()

    start_time = time.time()

    # Process each opening move in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for pos, size in opening_moves:
            move_str = f"{size[0]}{pos}"
            state = initial_state.make_move(pos, size)
            future = executor.submit(_explore_subtree, state, move_str, max_depth, verbose)
            futures[future] = move_str

        results = {}
        total_stats = {"nodes": 0, "B": 0, "R": 0, "D": 0, "L": 0}

        for future in as_completed(futures):
            move_str = futures[future]
            try:
                json_str, stats = future.result()
                results[move_str] = json_str
                for k in total_stats:
                    total_stats[k] += stats[k]
                elapsed = time.time() - start_time
                print(f"  {move_str} done: nodes={stats['nodes']:,} B={stats['B']:,} R={stats['R']:,} D={stats['D']:,} L={stats['L']:,} [{elapsed:.1f}s]")
            except Exception as e:
                print(f"  {move_str} FAILED: {e}")

    # Merge results in move order
    print()
    print("Merging results...")

    with open(output_file, 'w') as f:
        f.write('{"tree":{"c":[')
        first = True
        for pos, size in opening_moves:
            move_str = f"{size[0]}{pos}"
            if move_str in results:
                if not first:
                    f.write(",")
                f.write(results[move_str])
                first = False
        f.write(']}\n')
        f.write(f',"stats":{{"nodes":{total_stats["nodes"]},"B":{total_stats["B"]},"R":{total_stats["R"]},"D":{total_stats["D"]},"L":{total_stats["L"]}}}}}\n')

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"Nodes: {total_stats['nodes']:,}")
    print(f"BLUE wins: {total_stats['B']:,}")
    print(f"RED wins: {total_stats['R']:,}")
    print(f"Draws: {total_stats['D']:,}")
    if max_depth:
        print(f"Depth limit: {total_stats['L']:,}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nWrote to {output_file}")


def _explore_subtree(state, move_str, max_depth, verbose):
    """
    Worker function: explore a subtree and return JSON string + stats.
    Called by ProcessPoolExecutor.
    """
    import io

    stats = {"nodes": 0, "B": 0, "R": 0, "D": 0, "L": 0}

    def build_tree(state, depth):
        """Build tree as nested dict, then serialize."""
        stats["nodes"] += 1

        # Get move string
        if state.move_history:
            _, pos, size = state.move_history[-1]
            m = f"{size[0]}{pos}"
        else:
            m = None

        # Terminal: BLUE wins
        if state.winner == "BLUE":
            stats["B"] += 1
            return {"m": m, "r": "B"}

        # Terminal: RED wins
        if state.winner == "RED":
            stats["R"] += 1
            return {"m": m, "r": "R"}

        moves = state.get_valid_moves()

        # Terminal: DRAW
        if not moves:
            stats["D"] += 1
            return {"m": m, "r": "D"}

        # Terminal: Depth limit
        if max_depth and depth >= max_depth:
            stats["L"] += 1
            return {"m": m, "r": "L"}

        # Internal node
        children = []
        for pos, size in moves:
            new_state = state.make_move(pos, size)
            children.append(build_tree(new_state, depth + 1))

        return {"m": m, "c": children}

    # Build tree starting at depth 1 (opening move already made)
    tree = build_tree(state, 1)

    # Serialize to compact JSON
    def to_json(node):
        if "r" in node:
            return f'{{"m":"{node["m"]}","r":"{node["r"]}"}}'
        children_json = ",".join(to_json(c) for c in node["c"])
        return f'{{"m":"{node["m"]}","c":[{children_json}]}}\n'

    return to_json(tree), stats


def show_winning_lines(state, player):
    """Show all winning lines (threats) for a player."""
    board = state.board[player]
    threats = []

    for line in WIN_LINES:
        # Check same-size threats
        for size in ["SMALL", "MEDIUM", "LARGE"]:
            count = sum(1 for pos in line if size in board[pos])
            if count == 2:
                empty_pos = [pos for pos in line if size not in board[pos]][0]
                # Check if this position is actually playable
                opponent = "RED" if player == "BLUE" else "BLUE"
                if size not in state.board[opponent][empty_pos]:
                    threats.append((line, f"Same {size[0]}", empty_pos, size))

        # Check ascending sequence threats
        sizes_in_line = []
        for i, pos in enumerate(line):
            expected = ["SMALL", "MEDIUM", "LARGE"][i]
            if expected in board[pos]:
                sizes_in_line.append(i)

        if len(sizes_in_line) == 2:
            missing_idx = 3 - sum(sizes_in_line)  # Find missing position
            if 0 <= missing_idx < 3:
                missing_pos = line[missing_idx]
                missing_size = ["SMALL", "MEDIUM", "LARGE"][missing_idx]
                opponent = "RED" if player == "BLUE" else "BLUE"
                if missing_size not in state.board[opponent][missing_pos]:
                    threats.append((line, "Ascending", missing_pos, missing_size))

        # Check descending sequence threats
        sizes_in_line = []
        for i, pos in enumerate(line):
            expected = ["LARGE", "MEDIUM", "SMALL"][i]
            if expected in board[pos]:
                sizes_in_line.append(i)

        if len(sizes_in_line) == 2:
            missing_idx = 3 - sum(sizes_in_line)
            if 0 <= missing_idx < 3:
                missing_pos = line[missing_idx]
                missing_size = ["LARGE", "MEDIUM", "SMALL"][missing_idx]
                opponent = "RED" if player == "BLUE" else "BLUE"
                if missing_size not in state.board[opponent][missing_pos]:
                    threats.append((line, "Descending", missing_pos, missing_size))

    # Check nested threats (2 sizes in same position)
    for pos in POSITIONS:
        if len(board[pos]) == 2:
            missing = [s for s in ["SMALL", "MEDIUM", "LARGE"] if s not in board[pos]][0]
            opponent = "RED" if player == "BLUE" else "BLUE"
            if missing not in state.board[opponent][pos]:
                threats.append(([pos], "Nested", pos, missing))

    return threats


def analyze_move_rankings_bfs(state=None, verbose=True):
    """
    Rank all possible moves from the current state using BFS shortest-path analysis.

    Returns moves ranked by:
    1. Win probability (guaranteed win = 100%, guaranteed loss = 0%)
    2. Moves to end (fewer is better for wins, more is better to delay losses)
    """
    if state is None:
        state = OtrioGameState()

    state_values, visited, stats = find_shortest_path_wins(verbose=False)

    player = state.current_player
    moves = state.get_valid_moves()

    rankings = []

    for pos, size in moves:
        new_state = state.make_move(pos, size)
        new_key = new_state.to_tuple()

        if new_state.winner == player:
            # Immediate win
            rankings.append({
                "move": (pos, size),
                "result": "WIN",
                "value": 1 if player == "BLUE" else -1,
                "moves_to_end": 1,
                "description": "Immediate WIN!"
            })
        elif new_key in state_values:
            value, moves_to_end, _ = state_values[new_key]

            # Adjust value to be from current player's perspective
            if player == "RED":
                value = -value

            if value > 0:
                result = "WIN"
                desc = f"Forces win in {moves_to_end + 1} moves"
            elif value < 0:
                result = "LOSS"
                desc = f"Leads to loss in {moves_to_end + 1} moves"
            else:
                result = "DRAW"
                desc = f"Leads to draw in {moves_to_end + 1} moves"

            rankings.append({
                "move": (pos, size),
                "result": result,
                "value": value,
                "moves_to_end": moves_to_end + 1,
                "description": desc
            })
        else:
            rankings.append({
                "move": (pos, size),
                "result": "UNKNOWN",
                "value": 0,
                "moves_to_end": 999,
                "description": "Not in computed states"
            })

    # Sort: wins first (by fastest), then draws, then losses (by slowest)
    def sort_key(r):
        if r["result"] == "WIN":
            return (0, r["moves_to_end"])  # Wins first, faster better
        elif r["result"] == "DRAW":
            return (1, 0)  # Draws second
        elif r["result"] == "LOSS":
            return (2, -r["moves_to_end"])  # Losses last, slower better
        else:
            return (3, 0)  # Unknown last

    rankings.sort(key=sort_key)

    if verbose:
        print(f"\n{'='*60}")
        print(f"MOVE RANKINGS FOR {player} (BFS Shortest-Path Analysis)")
        print(f"{'='*60}")

        for i, r in enumerate(rankings):
            pos, size = r["move"]
            print(f"{i+1:3}. {size[0]}@{pos:2} -> {r['result']:4} | {r['description']}")

    return rankings


# =============================================================================
# Sample Game Simulation (using OtrioGameState)
# =============================================================================

def simulate_random_game(seed=None):
    """Simulate a random game and show the progression."""
    import random
    if seed is not None:
        random.seed(seed)

    state = OtrioGameState()
    print("Starting Otrio game simulation...")
    print(state)

    move_num = 0
    while not state.winner:
        moves = state.get_valid_moves()
        if not moves:
            print("\nNo more moves available - Draw!")
            break

        pos, size = random.choice(moves)
        print(f"\nMove {move_num + 1}: {state.current_player} places {size} at {pos}")
        state = state.make_move(pos, size)
        move_num += 1

    print("\n" + "=" * 40)
    print("FINAL STATE:")
    print(state)

    return state


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Otrio Game - Colored Petri Net Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python otrio_cpn.py                    # Run with defaults (50,000 states)
  python otrio_cpn.py -n 200000          # Explore 200,000 states
  python otrio_cpn.py --max-states 1000000  # Explore 1 million states
  python otrio_cpn.py --no-simulate      # Skip random game simulation
  python otrio_cpn.py --solve            # Find winning strategy for BLUE
  python otrio_cpn.py --solve --player red  # Find winning strategy for RED
        """
    )
    parser.add_argument(
        "-n", "--max-states",
        type=int,
        default=50000,
        help="Maximum number of states to explore (default: 50000)"
    )
    parser.add_argument(
        "--no-simulate",
        action="store_true",
        help="Skip the random game simulation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for game simulation (default: 42)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Find a guaranteed winning strategy (minimax search)"
    )
    parser.add_argument(
        "--player",
        choices=["blue", "red"],
        default="blue",
        help="Player to find winning strategy for (default: blue)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=18,
        help="Maximum search depth for --solve (default: 18)"
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=500000,
        help="Maximum nodes to explore for --solve (default: 500000)"
    )
    parser.add_argument(
        "--cpn-info",
        action="store_true",
        help="Show information about the improved CPN model with blocking"
    )
    parser.add_argument(
        "--bfs-solve",
        action="store_true",
        help="Find winning strategy using BFS shortest-path analysis"
    )
    parser.add_argument(
        "--simple-bfs",
        action="store_true",
        help="Run simple BFS exploration (just counts states)"
    )
    parser.add_argument(
        "--compact-bfs",
        action="store_true",
        help="Run compact BFS with 55-bit state encoding"
    )
    parser.add_argument(
        "--find-win",
        action="store_true",
        help="Use iterative deepening to find shortest win"
    )
    parser.add_argument(
        "--bfs-depth",
        type=int,
        default=5,
        help="Maximum depth (moves ahead) for BFS exploration (default: 5)"
    )
    parser.add_argument(
        "--show-path",
        action="store_true",
        help="Show the shortest winning path (use with --bfs-solve)"
    )
    parser.add_argument(
        "--tree",
        type=str,
        metavar="FILE",
        help="Export full game tree to JSON file (explores ALL sequences)"
    )
    parser.add_argument(
        "--depth",
        type=int,
        metavar="N",
        help="Limit tree depth to N moves (use with --tree for debugging)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        nargs="?",
        const=0,  # 0 means auto-detect
        metavar="N",
        help="Use N parallel workers (use with --tree). Default: auto-detect CPU count"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OTRIO - Colored Petri Net Simulation")
    print("=" * 60)

    # Export full game tree
    if args.tree:
        if args.parallel is not None:
            num_workers = args.parallel if args.parallel > 0 else None
            explore_game_tree_parallel(args.tree, verbose=not args.quiet, max_depth=args.depth, num_workers=num_workers)
        else:
            explore_game_tree(args.tree, verbose=not args.quiet, max_depth=args.depth)
        return

    # Find shortest win using iterative deepening
    if args.find_win:
        player = args.player.upper()
        print(f"\nSearching for shortest {player} win...")
        print("-" * 40)

        win_depth, wins = iterative_deepening_bfs(
            max_depth=args.bfs_depth,
            target_player=player,
            verbose=True
        )

        print("\n" + "=" * 60)
        if win_depth:
            print(f"Found {player} win at depth {win_depth}!")
        else:
            print(f"No {player} win found within depth {args.bfs_depth}")
        return

    # Compact BFS mode: memory-efficient exploration
    if args.compact_bfs:
        print(f"\nCompact BFS exploration...")
        print(f"  Depth limit: {args.bfs_depth} moves")
        print("-" * 40)

        visited, stats = compact_bfs(max_depth=args.bfs_depth, verbose=True)

        print("\n" + "=" * 60)
        print("Compact BFS complete!")
        return

    # Simple BFS mode: just explore and count states
    if args.simple_bfs:
        print(f"\nSimple BFS exploration...")
        print(f"  Depth limit: {args.bfs_depth} moves")
        print("-" * 40)

        visited, stats = simple_bfs_explore(max_depth=args.bfs_depth, verbose=True)

        print("\n" + "=" * 60)
        print("Simple BFS complete!")
        return

    # BFS solve mode: complete game tree analysis with shortest paths
    if args.bfs_solve:
        player = args.player.upper()
        print(f"\nBFS Shortest-Path Strategy Analysis for {player}...")
        print(f"  Depth limit: {args.bfs_depth} moves")
        print("-" * 40)

        result, optimal_moves, win_in_moves, stats, state_values, visited = find_shortest_win_strategy(
            player=player,
            verbose=not args.quiet,
            max_states=args.max_states,
            max_depth=args.bfs_depth
        )

        if args.show_path:
            show_shortest_win_path(player, state_values, visited)

        print("\n" + "=" * 60)
        print("BFS analysis complete!")
        return

    # CPN info mode: show improved model details
    if args.cpn_info:
        print_improved_cpn_info()
        print("\nBuilding improved CPN model...")
        cpn, places, transitions = build_improved_otrio_cpn()
        print(f"  Places: {len(places)} ({', '.join(sorted(places.keys())[:6])}...)")
        print(f"  Transitions: {len(transitions)}")
        print(f"  Arcs: {len(cpn.arcs)}")

        print("\nCreating initial marking...")
        marking = create_improved_initial_marking(cpn)
        turn = [t.value for t in marking.get_multiset("Turn").tokens]
        print(f"  Turn: {turn[0]}")
        print(f"  Each player has 9 pieces (3 SMALL, 3 MEDIUM, 3 LARGE)")

        print("\nEnabled transitions for BLUE at start:")
        enabled = get_enabled_transitions_improved(cpn, marking, "BLUE")
        print(f"  {len(enabled)} moves available")
        return

    # Solve mode: find winning strategy
    if args.solve:
        player = args.player.upper()
        print(f"\nSearching for winning strategy for {player}...")
        print("-" * 40)

        result, best_move, win_depth, search_stats, cache = find_winning_strategy(
            player=player,
            max_depth=args.max_depth,
            verbose=not args.quiet,
            max_nodes=args.max_nodes
        )
        print_winning_strategy(result, best_move, win_depth, player, cache, search_stats)

        print("\n" + "=" * 60)
        print("Search complete!")
        return

    # Build the CPN model
    print("\n1. Building CPN model...")
    cpn, places, transitions = build_otrio_cpn()
    print(f"   Places: {len(cpn.places)}")
    print(f"   Transitions: {len(cpn.transitions)}")
    print(f"   Arcs: {len(cpn.arcs)}")

    # Create initial marking
    print("\n2. Creating initial marking...")
    marking = create_initial_marking(cpn)
    blue_tokens = [t.value for t in marking.get_multiset("Blue_Initial").tokens]
    red_tokens = [t.value for t in marking.get_multiset("Red_Initial").tokens]
    print(f"   Blue initial tokens: {blue_tokens}")
    print(f"   Red initial tokens: {red_tokens}")

    # Simulate a game using CPN
    if not args.no_simulate:
        print("\n3. Simulating a game using CPN model...")
        print("-" * 40)
        simulate_cpn_game(cpn, seed=args.seed)

    # Explore state space
    print(f"\n\n4. Exploring state space (max {args.max_states:,} states)...")
    print("-" * 40)
    stats, visited = explore_state_space(
        max_states=args.max_states,
        verbose=not args.quiet
    )
    print_stats(stats)

    print("\n" + "=" * 60)
    print("Simulation complete!")


if __name__ == "__main__":
    main()
