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

        for size in ["SMALL", "MEDIUM", "LARGE"]:
            if self.initial[player][size] > 0:
                for pos in POSITIONS:
                    # Can only place if that size isn't already there
                    if size not in self.board[player][pos]:
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

        if self.initial[player][size] <= 0:
            raise ValueError(f"No {size} pieces left for {player}")

        if size in self.board[player][position]:
            raise ValueError(f"{size} already at {position} for {player}")

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
# CPN Model Builder
# =============================================================================

def build_otrio_cpn():
    """
    Build the Colored Petri Net model for Otrio.

    Structure:
    - Places: Blue_Initial, Red_Initial, Blue_A1..Blue_C3, Red_A1..Red_C3
    - Transitions: Blue_to_A1..Blue_to_C3, Red_to_A1..Red_to_C3
    - Arcs connect initial places to board places via transitions
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

def find_winning_strategy(player="BLUE", max_depth=18, verbose=True):
    """
    Find a guaranteed winning strategy for the specified player.

    Uses minimax with alpha-beta pruning to find moves that guarantee
    a win regardless of opponent's responses.

    Returns:
        - winning_move: The best first move (position, size) or None
        - strategy_tree: Dict with the full winning strategy
        - stats: Search statistics
    """
    initial_state = OtrioGameState()
    opponent = "RED" if player == "BLUE" else "BLUE"

    # Memoization cache: state -> (result, best_move, depth_to_win)
    # result: 1 = player wins, -1 = opponent wins, 0 = draw/unknown
    cache = {}
    stats = {"nodes_explored": 0, "cache_hits": 0, "max_depth_seen": 0}

    def minimax(state, depth, alpha, beta, maximizing):
        """
        Minimax with alpha-beta pruning.
        Returns: (result, best_move, depth_to_win)
        """
        stats["nodes_explored"] += 1
        stats["max_depth_seen"] = max(stats["max_depth_seen"], depth)

        if verbose and stats["nodes_explored"] % 100000 == 0:
            print(f"  Explored {stats['nodes_explored']:,} nodes...")

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


def print_winning_strategy(result, best_move, win_depth, player="BLUE", cache=None):
    """Print the winning strategy result."""
    print("\n" + "=" * 60)
    print(f"WINNING STRATEGY ANALYSIS FOR {player}")
    print("=" * 60)

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
        print(f"\nNo guaranteed win found within search depth.")
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

    args = parser.parse_args()

    print("=" * 60)
    print("OTRIO - Colored Petri Net Simulation")
    print("=" * 60)

    # Solve mode: find winning strategy
    if args.solve:
        player = args.player.upper()
        print(f"\nSearching for winning strategy for {player}...")
        print("-" * 40)

        result, best_move, win_depth, stats, cache = find_winning_strategy(
            player=player,
            max_depth=args.max_depth,
            verbose=not args.quiet
        )
        print_winning_strategy(result, best_move, win_depth, player, cache)

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
