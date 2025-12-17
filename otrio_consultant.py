"""
Otrio Game Consultant - Interactive Move Advisor

This script acts as an in-game consultant for the player.
It explores the state space after hypothetical moves and provides
win probability estimates to help the player make informed decisions.

Usage:
    python otrio_consultant.py [--max-states N] [--player blue|red]
"""

import argparse
from collections import defaultdict
from otrio_cpn import (
    OtrioGameState,
    POSITIONS,
    ROWS,
    COLS,
    WIN_LINES,
    find_winning_strategy,
)


# =============================================================================
# State Space Analysis for Win Probability
# =============================================================================

def analyze_move(state, position, size, max_states=10000):
    """
    Analyze a potential move by exploring the resulting state space.
    Returns win/loss statistics for the player making the move.
    """
    player = state.current_player

    # Make the hypothetical move
    try:
        new_state = state.make_move(position, size)
    except ValueError as e:
        return {"error": str(e)}

    # If the move wins immediately
    if new_state.winner == player:
        return {
            "immediate_win": True,
            "player": player,
            "win_type": get_win_type(new_state, player, position)
        }

    # Explore state space from this position
    stats = explore_from_state(new_state, max_states)

    return {
        "immediate_win": False,
        "player": player,
        "opponent": "RED" if player == "BLUE" else "BLUE",
        "states_explored": stats["total_states"],
        "player_wins": stats["winning_states"][player],
        "opponent_wins": stats["winning_states"]["RED" if player == "BLUE" else "BLUE"],
        "draws": stats["draws"],
        "win_rate": calculate_win_rate(stats, player),
    }


def get_win_type(state, player, last_position):
    """Determine how the player won."""
    board = state.board[player]

    # Check nested win at last position
    if len(board[last_position]) == 3:
        return f"Nested win at {last_position} (all three sizes)"

    # Check line wins
    for line in WIN_LINES:
        if last_position not in line:
            continue

        # Same size
        for size in ["SMALL", "MEDIUM", "LARGE"]:
            if all(size in board[pos] for pos in line):
                return f"Three {size} in a row: {' -> '.join(line)}"

        # Ascending
        if ("SMALL" in board[line[0]] and
            "MEDIUM" in board[line[1]] and
            "LARGE" in board[line[2]]):
            return f"Ascending order: {line[0]}(S) -> {line[1]}(M) -> {line[2]}(L)"

        # Descending
        if ("LARGE" in board[line[0]] and
            "MEDIUM" in board[line[1]] and
            "SMALL" in board[line[2]]):
            return f"Descending order: {line[0]}(L) -> {line[1]}(M) -> {line[2]}(S)"

    return "Win condition met"


def explore_from_state(initial_state, max_states=10000):
    """
    Explore state space starting from a given state.
    Returns statistics about win outcomes.
    """
    visited = set()
    to_visit = [(initial_state, 0)]

    stats = {
        "total_states": 0,
        "winning_states": {"BLUE": 0, "RED": 0},
        "terminal_states": 0,
        "draws": 0,
        "max_depth": 0,
    }

    while to_visit and stats["total_states"] < max_states:
        state, depth = to_visit.pop()

        state_hash = state.to_tuple()
        if state_hash in visited:
            continue

        visited.add(state_hash)
        stats["total_states"] += 1
        stats["max_depth"] = max(stats["max_depth"], depth)

        if state.winner:
            stats["winning_states"][state.winner] += 1
            stats["terminal_states"] += 1
            continue

        moves = state.get_valid_moves()
        if not moves:
            stats["terminal_states"] += 1
            stats["draws"] += 1
            continue

        for pos, size in moves:
            new_state = state.make_move(pos, size)
            new_hash = new_state.to_tuple()
            if new_hash not in visited:
                to_visit.append((new_state, depth + 1))

    return stats


def calculate_win_rate(stats, player):
    """Calculate win rate for a player."""
    total_terminal = stats["terminal_states"]
    if total_terminal == 0:
        return 0.0
    player_wins = stats["winning_states"][player]
    return (player_wins / total_terminal) * 100


# =============================================================================
# Perfect Play Strategy (Verified Winning Strategy for First Player)
# =============================================================================

# Diagonal opposites for strategy calculations
DIAGONAL_OPPOSITE = {
    "A1": "C3", "C3": "A1",
    "A3": "C1", "C1": "A3",
}

# Corner positions
CORNERS = ["A1", "A3", "C1", "C3"]


def get_perfect_move(state):
    """
    Get the optimal move using the verified winning strategy for BLUE.

    The strategy guarantees a win for the first player (BLUE) by:
    1. Opening with MEDIUM at B2 (center)
    2. Responding to opponent with LARGE to create diagonal threats
    3. Creating double threats that opponent cannot defend

    Returns: (position, size, explanation) or None if strategy doesn't apply
    """
    player = state.current_player
    move_count = len(state.move_history)

    # Strategy only works for BLUE (first player)
    if player != "BLUE":
        return None

    # Move 1: Open with MEDIUM at B2
    if move_count == 0:
        if is_move_valid(state, "B2", "MEDIUM"):
            return ("B2", "MEDIUM", "Opening: Center control with MEDIUM creates 4 potential win lines")
        return None

    # Move 2 (BLUE's second move, after RED's first)
    if move_count == 2:
        # Find RED's first move
        red_pos = state.move_history[1][1]  # (player, pos, size)

        # Strategy: place LARGE to create diagonal threat through B2
        if red_pos in DIAGONAL_OPPOSITE:
            # RED is on a corner - threaten the opposite diagonal
            target = DIAGONAL_OPPOSITE[red_pos]
            if is_move_valid(state, target, "LARGE"):
                return (target, "LARGE",
                        f"Counter-corner: LARGE at {target} creates ascending/descending threat through B2")

        # Default: LARGE at A1 to create diagonal threat
        if is_move_valid(state, "A1", "LARGE"):
            return ("A1", "LARGE",
                    "Diagonal threat: L@A1 + M@B2 threatens S@C3 for ascending win")

        # Fallback to other corners
        for corner in ["C3", "A3", "C1"]:
            if is_move_valid(state, corner, "LARGE"):
                return (corner, "LARGE",
                        f"Diagonal threat: LARGE at {corner} creates win threat through B2")

    # Move 3+ (BLUE's third move and beyond): Look for immediate wins or double threats
    if move_count >= 4 and player == "BLUE":
        # Check for immediate winning moves
        for pos, size in state.get_valid_moves():
            test_state = state.make_move(pos, size)
            if test_state.winner == "BLUE":
                return (pos, size, f"WINNING MOVE: {size} at {pos}!")

        # Look for moves that create double threats
        best_threat_move = find_double_threat_move(state)
        if best_threat_move:
            pos, size, threat_count = best_threat_move
            return (pos, size, f"Creates {threat_count} simultaneous threats - opponent cannot block all!")

        # Continue with threats towards ascending/descending wins
        move = find_forcing_move(state)
        if move:
            return move

    return None


def is_move_valid(state, position, size):
    """Check if a move is valid."""
    player = state.current_player
    opponent = "RED" if player == "BLUE" else "BLUE"

    if state.initial[player][size] <= 0:
        return False
    if size in state.board[player][position]:
        return False
    if size in state.board[opponent][position]:
        return False
    return True


def count_threats(state, player):
    """Count how many winning threats a player has."""
    threats = []
    board = state.board[player]

    for line in WIN_LINES:
        # Nested win threat (2 of 3 sizes in same position)
        for pos in line:
            if len(board[pos]) == 2:
                missing = {"SMALL", "MEDIUM", "LARGE"} - board[pos]
                for size in missing:
                    if is_move_valid_for(state, pos, size, player):
                        threats.append(("nested", pos, size))

        # Same-size win threat
        for size in ["SMALL", "MEDIUM", "LARGE"]:
            positions_with_size = [pos for pos in line if size in board[pos]]
            if len(positions_with_size) == 2:
                missing_pos = [p for p in line if p not in positions_with_size][0]
                if is_move_valid_for(state, missing_pos, size, player):
                    threats.append(("same_size", missing_pos, size))

        # Ascending threat (S-M-L)
        asc_present = []
        for i, pos in enumerate(line):
            expected = ["SMALL", "MEDIUM", "LARGE"][i]
            if expected in board[pos]:
                asc_present.append(i)

        if len(asc_present) == 2:
            for i in range(3):
                if i not in asc_present:
                    missing_pos = line[i]
                    missing_size = ["SMALL", "MEDIUM", "LARGE"][i]
                    if is_move_valid_for(state, missing_pos, missing_size, player):
                        threats.append(("ascending", missing_pos, missing_size))

        # Descending threat (L-M-S)
        desc_present = []
        for i, pos in enumerate(line):
            expected = ["LARGE", "MEDIUM", "SMALL"][i]
            if expected in board[pos]:
                desc_present.append(i)

        if len(desc_present) == 2:
            for i in range(3):
                if i not in desc_present:
                    missing_pos = line[i]
                    missing_size = ["LARGE", "MEDIUM", "SMALL"][i]
                    if is_move_valid_for(state, missing_pos, missing_size, player):
                        threats.append(("descending", missing_pos, missing_size))

    return threats


def is_move_valid_for(state, position, size, player):
    """Check if a move would be valid for a specific player."""
    opponent = "RED" if player == "BLUE" else "BLUE"

    if state.initial[player][size] <= 0:
        return False
    if size in state.board[player][position]:
        return False
    if size in state.board[opponent][position]:
        return False
    return True


def find_double_threat_move(state):
    """Find a move that creates 2+ threats (unblockable)."""
    player = state.current_player
    moves = state.get_valid_moves()

    best_move = None
    best_threat_count = 1

    for pos, size in moves:
        new_state = state.make_move(pos, size)
        if new_state.winner:
            continue  # This is an immediate win, not a threat

        threats = count_threats(new_state, player)
        if len(threats) > best_threat_count:
            best_threat_count = len(threats)
            best_move = (pos, size, len(threats))

    return best_move


def find_forcing_move(state):
    """Find a move that creates a threat forcing opponent to respond."""
    player = state.current_player
    moves = state.get_valid_moves()

    # Prioritize moves that create ascending/descending threats through B2
    for pos, size in moves:
        new_state = state.make_move(pos, size)
        if new_state.winner:
            continue

        threats = count_threats(new_state, player)
        if threats:
            # Prefer threats involving the diagonal
            for threat_type, threat_pos, threat_size in threats:
                if threat_type in ("ascending", "descending"):
                    return (pos, size, f"Creates {threat_type} threat at {threat_pos}")

    # Any threat is better than none
    for pos, size in moves:
        new_state = state.make_move(pos, size)
        if not new_state.winner:
            threats = count_threats(new_state, player)
            if threats:
                return (pos, size, f"Creates {len(threats)} threat(s)")

    return None


# =============================================================================
# Minimax Strategy Functions (for --strategy mode)
# =============================================================================

def get_strategy_move(state, strategy_cache, strategy_player):
    """
    Get the optimal move from the precomputed strategy cache.
    Returns (pos, size, result, win_depth) or None if not in cache.
    """
    is_maximizing = (state.current_player == strategy_player)
    state_key = (state.to_tuple(), is_maximizing)

    if state_key in strategy_cache:
        result, best_move, win_depth = strategy_cache[state_key]
        if best_move:
            return best_move[0], best_move[1], result, win_depth
    return None


def print_strategy_hint(state, strategy_cache, strategy_player, your_player):
    """Print strategy recommendation based on precomputed minimax."""
    current = state.current_player
    is_maximizing = (current == strategy_player)
    state_key = (state.to_tuple(), is_maximizing)

    if state_key not in strategy_cache:
        print("  [Strategy: position not in cache]")
        return

    result, best_move, win_depth = strategy_cache[state_key]

    if best_move is None:
        return

    pos, size = best_move

    print("\n" + "=" * 50)
    print("STRATEGY RECOMMENDATION")
    print("=" * 50)

    if current == your_player:
        # It's your turn - show recommendation
        if result == 1:
            moves_to_win = win_depth - len(state.move_history)
            print(f"  Optimal move: {size[0]} at {pos}")
            print(f"  Result: GUARANTEED WIN in {moves_to_win} moves!")

            # Show the winning path
            print(f"\n  Winning path from here:")
            show_strategy_path(state, strategy_cache, strategy_player, max_moves=6)
        elif result == -1:
            print(f"  Best defensive move: {size[0]} at {pos}")
            print(f"  Result: Opponent has forced win (try to extend game)")
        else:
            print(f"  Suggested move: {size[0]} at {pos}")
            print(f"  Result: Draw with perfect play")
    else:
        # Opponent's turn - show what we expect
        if result == 1:
            print(f"  Opponent's best: {size[0]} at {pos}")
            print(f"  But YOU still have a winning strategy!")
        elif result == -1:
            print(f"  Warning: Opponent can force a win")
            print(f"  Their optimal: {size[0]} at {pos}")
        else:
            print(f"  Opponent's likely move: {size[0]} at {pos}")


def show_strategy_path(state, cache, strategy_player, max_moves=6):
    """Show the expected winning path from current state."""
    current_state = state
    moves_shown = 0
    move_num = len(state.move_history) + 1

    while not current_state.winner and moves_shown < max_moves:
        current = current_state.current_player
        is_maximizing = (current == strategy_player)
        state_key = (current_state.to_tuple(), is_maximizing)

        if state_key not in cache:
            break

        result, best_move, _ = cache[state_key]
        if best_move is None:
            break

        pos, size = best_move
        marker = ">>>" if current == strategy_player else "   "
        print(f"    {marker} {move_num}. {current}: {size[0]} at {pos}")

        current_state = current_state.make_move(pos, size)
        move_num += 1
        moves_shown += 1

        if current_state.winner:
            print(f"    === {current_state.winner} WINS! ===")
            break


# =============================================================================
# Display Functions
# =============================================================================

def print_board(state):
    """Print the current game state."""
    print("\n" + "=" * 50)
    print(f"Current player: {state.current_player}")
    if state.winner:
        print(f"WINNER: {state.winner}")
    print("=" * 50)

    # Print boards side by side
    print("\n       BLUE                    RED")
    print("   1     2     3          1     2     3")

    for row in ROWS:
        blue_row = ""
        red_row = ""
        for col in COLS:
            pos = f"{row}{col}"
            blue_sizes = state.board["BLUE"][pos]
            red_sizes = state.board["RED"][pos]
            blue_str = "".join(s[0] for s in sorted(blue_sizes))
            red_str = "".join(s[0] for s in sorted(red_sizes))
            blue_row += f"[{blue_str:3s}] "
            red_row += f"[{red_str:3s}] "
        print(f"{row}  {blue_row}     {row}  {red_row}")

    print(f"\nBLUE remaining: S={state.initial['BLUE']['SMALL']}, "
          f"M={state.initial['BLUE']['MEDIUM']}, L={state.initial['BLUE']['LARGE']}")
    print(f"RED remaining:  S={state.initial['RED']['SMALL']}, "
          f"M={state.initial['RED']['MEDIUM']}, L={state.initial['RED']['LARGE']}")


def print_analysis(analysis, position, size):
    """Print move analysis results."""
    print("\n" + "-" * 50)
    print(f"Analysis for: {size} at {position}")
    print("-" * 50)

    if "error" in analysis:
        print(f"Invalid move: {analysis['error']}")
        return

    if analysis["immediate_win"]:
        print(f"*** IMMEDIATE WIN! ***")
        print(f"Win type: {analysis['win_type']}")
        return

    print(f"States explored: {analysis['states_explored']:,}")
    print(f"\nOutcome distribution:")
    print(f"  Your wins ({analysis['player']}): {analysis['player_wins']:,}")
    print(f"  Opponent wins ({analysis['opponent']}): {analysis['opponent_wins']:,}")
    print(f"  Draws: {analysis['draws']:,}")
    print(f"\n  >>> Win probability: {analysis['win_rate']:.1f}% <<<")


def print_valid_moves(state):
    """Print all valid moves for the current player."""
    moves = state.get_valid_moves()
    if not moves:
        print("No valid moves available!")
        return

    player = state.current_player
    opponent = "RED" if player == "BLUE" else "BLUE"

    print(f"\nValid moves for {player}:")

    # Group by position
    by_position = defaultdict(list)
    for pos, size in moves:
        by_position[pos].append(size)

    # Track blocked positions
    fully_blocked = []

    for pos in POSITIONS:
        if pos in by_position:
            sizes = ", ".join(s[0] for s in by_position[pos])
            # Check what's blocked (by self or opponent)
            own_blocked = [s for s in ["SMALL", "MEDIUM", "LARGE"]
                         if s in state.board[player][pos]]
            opp_blocked = [s for s in ["SMALL", "MEDIUM", "LARGE"]
                         if s in state.board[opponent][pos]]

            notes = []
            if own_blocked:
                notes.append(f"own:{','.join(s[0] for s in own_blocked)}")
            if opp_blocked:
                notes.append(f"opp:{','.join(s[0] for s in opp_blocked)}")

            if notes:
                print(f"  {pos}: [{sizes}] ({'; '.join(notes)})")
            else:
                print(f"  {pos}: [{sizes}]")
        else:
            # Position has no valid moves - track why
            own_blocked = [s for s in ["SMALL", "MEDIUM", "LARGE"]
                         if s in state.board[player][pos]]
            opp_blocked = [s for s in ["SMALL", "MEDIUM", "LARGE"]
                         if s in state.board[opponent][pos]]
            if own_blocked or opp_blocked:
                fully_blocked.append((pos, own_blocked, opp_blocked))

    if fully_blocked:
        print(f"\n  Fully blocked positions:")
        for pos, own, opp in fully_blocked[:3]:  # Show max 3
            parts = []
            if own:
                parts.append(f"your {','.join(s[0] for s in own)}")
            if opp:
                parts.append(f"opponent's {','.join(s[0] for s in opp)}")
            print(f"    {pos}: {' + '.join(parts)}")


def print_perfect_move(state):
    """Print the perfect play recommendation for BLUE."""
    if state.current_player != "BLUE":
        print("\n  [Perfect play only available for BLUE's turn]")
        return None

    result = get_perfect_move(state)
    if result:
        pos, size, explanation = result
        print("\n" + "=" * 50)
        print("PERFECT PLAY RECOMMENDATION")
        print("=" * 50)
        print(f"  Move: {size[0]} at {pos}")
        print(f"  Why:  {explanation}")
        print("=" * 50)
        return (pos, size)
    else:
        print("\n  [No perfect play move found - use 'analyze all' for suggestions]")
        return None


def print_help(strategy_mode=False, perfect_mode=False):
    """Print help information."""
    print("""
Commands:
  <position> <size>  - Analyze a move (e.g., "A1 S" or "B2 LARGE")
  confirm            - Confirm your last analyzed move
  opponent <pos> <size> - Record opponent's move
  board              - Show current board
  moves              - Show valid moves
  analyze all        - Analyze all valid moves
  undo               - Undo last move
  help               - Show this help
  quit               - Exit the consultant""")

    if perfect_mode:
        print("""
Perfect play commands (BLUE only):
  perfect            - Show winning strategy move
  autowin            - Auto-play perfect moves until win""")

    if strategy_mode:
        print("""
Strategy mode commands:
  strategy           - Show optimal move from precomputed strategy
  path               - Show winning path from current position
  auto               - Auto-play optimal move (your turn only)""")

    print("""
Position format: A1, A2, A3, B1, B2, B3, C1, C2, C3
Size format: S/SMALL, M/MEDIUM, L/LARGE
""")


# =============================================================================
# Input Parsing
# =============================================================================

def parse_move(move_str):
    """Parse a move string like 'A1 S' or 'B2 MEDIUM'."""
    parts = move_str.upper().strip().split()
    if len(parts) != 2:
        return None, None

    position = parts[0]
    if position not in POSITIONS:
        return None, None

    size_input = parts[1]
    size_map = {
        "S": "SMALL", "SMALL": "SMALL",
        "M": "MEDIUM", "MEDIUM": "MEDIUM",
        "L": "LARGE", "LARGE": "LARGE",
    }
    size = size_map.get(size_input)

    return position, size


# =============================================================================
# Main Interactive Loop
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Otrio Game Consultant - Interactive Move Advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python otrio_consultant.py                # Standard mode with move analysis
  python otrio_consultant.py --perfect      # Perfect play mode (guaranteed win for BLUE)
  python otrio_consultant.py --strategy     # Precompute winning strategy (slow)
  python otrio_consultant.py --player red   # Play as RED
        """
    )
    parser.add_argument(
        "-n", "--max-states",
        type=int,
        default=10000,
        help="Maximum states to explore per analysis (default: 10000)"
    )
    parser.add_argument(
        "--player",
        choices=["blue", "red"],
        default="blue",
        help="Which player you are (default: blue)"
    )
    parser.add_argument(
        "--perfect",
        action="store_true",
        help="Enable perfect play mode (verified winning strategy for BLUE)"
    )
    parser.add_argument(
        "--strategy",
        action="store_true",
        help="Precompute winning strategy using minimax (shows optimal moves)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("OTRIO GAME CONSULTANT")
    print("=" * 60)
    print(f"\nYou are playing as: {args.player.upper()}")

    # Perfect play mode (fast, uses verified strategy)
    perfect_mode = args.perfect

    if perfect_mode:
        print("\n*** PERFECT PLAY MODE ENABLED ***")
        print("Using verified winning strategy for BLUE (first player)")
        print("Strategy: M@B2 -> L@corner -> create double threats")
        print("\nType 'perfect' to see optimal move, 'autowin' for auto-play")

    # Strategy mode: precompute the winning strategy (slower, full minimax)
    strategy_cache = None
    strategy_player = "BLUE"  # Strategy is computed for first player

    if args.strategy:
        print("\nPrecomputing winning strategy (this may take a moment)...")
        result, best_move, win_depth, stats, strategy_cache = find_winning_strategy(
            player=strategy_player,
            max_depth=18,
            verbose=True
        )
        if result == 1:
            print(f"\n*** BLUE has a guaranteed win in {win_depth} moves! ***")
            print(f"Opening move: {best_move[1]} at {best_move[0]}")
        elif result == -1:
            print(f"\n*** RED has a guaranteed win with perfect play ***")
        else:
            print(f"\nGame is likely a draw with perfect play.")

        print("\nStrategy mode ENABLED - type 'strategy' for recommendations")
    elif not perfect_mode:
        print(f"Analysis depth: {args.max_states:,} states per move")
        print("\nTip: Run with --perfect for verified winning strategy")

    print("\nType 'help' for commands, 'quit' to exit")

    state = OtrioGameState()
    your_player = args.player.upper()
    last_analysis = None
    last_move = None
    move_history = []

    print_board(state)

    # Show initial strategy hint if in strategy mode
    if strategy_cache and your_player == "BLUE":
        print_strategy_hint(state, strategy_cache, strategy_player, your_player)

    # Show initial perfect play hint
    if perfect_mode and your_player == "BLUE":
        print_perfect_move(state)

    while True:
        if state.winner:
            print(f"\n*** GAME OVER - {state.winner} WINS! ***")
            break

        prompt = f"\n[{state.current_player}'s turn] > "
        try:
            user_input = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Parse commands
        if user_input == "quit" or user_input == "exit":
            print("Goodbye!")
            break

        elif user_input == "help":
            print_help(strategy_mode=strategy_cache is not None, perfect_mode=perfect_mode)

        elif user_input == "board":
            print_board(state)

        elif user_input == "moves":
            print_valid_moves(state)

        elif user_input == "undo":
            if move_history:
                state = move_history.pop()
                print("Move undone.")
                print_board(state)
                if strategy_cache:
                    print_strategy_hint(state, strategy_cache, strategy_player, your_player)
                if perfect_mode and state.current_player == "BLUE":
                    print_perfect_move(state)
            else:
                print("No moves to undo.")

        # Strategy mode commands
        elif user_input == "strategy":
            if not strategy_cache:
                print("Strategy mode not enabled. Restart with --strategy flag.")
            else:
                print_strategy_hint(state, strategy_cache, strategy_player, your_player)

        elif user_input == "path":
            if not strategy_cache:
                print("Strategy mode not enabled. Restart with --strategy flag.")
            else:
                print("\nWinning path from current position:")
                show_strategy_path(state, strategy_cache, strategy_player, max_moves=10)

        elif user_input == "auto":
            if not strategy_cache:
                print("Strategy mode not enabled. Restart with --strategy flag.")
            elif state.current_player != your_player:
                print("It's not your turn. Record opponent's move first.")
            else:
                move_info = get_strategy_move(state, strategy_cache, strategy_player)
                if move_info:
                    pos, size, result, win_depth = move_info
                    move_history.append(state)
                    state = state.make_move(pos, size)
                    print(f"\nAuto-played optimal move: {size} at {pos}")
                    print_board(state)

                    if state.winner:
                        continue

                    print(f"\nNow waiting for opponent's move...")
                    print("Use: opponent <position> <size>")

                    # Show what we expect from opponent
                    if strategy_cache:
                        print_strategy_hint(state, strategy_cache, strategy_player, your_player)
                else:
                    print("No optimal move found in strategy cache.")

        # Perfect play mode commands
        elif user_input == "perfect":
            if not perfect_mode:
                print("Perfect play mode not enabled. Restart with --perfect flag.")
            else:
                result = print_perfect_move(state)
                if result:
                    last_move = result

        elif user_input == "autowin":
            if not perfect_mode:
                print("Perfect play mode not enabled. Restart with --perfect flag.")
            elif your_player != "BLUE":
                print("Autowin only works when playing as BLUE (first player).")
            else:
                print("\n*** AUTO-WIN MODE ***")
                print("Playing perfect strategy until win...\n")

                while not state.winner:
                    if state.current_player == "BLUE":
                        # BLUE's turn - play perfect move
                        result = get_perfect_move(state)
                        if result:
                            pos, size, explanation = result
                            move_history.append(state)
                            state = state.make_move(pos, size)
                            print(f"BLUE plays: {size[0]} at {pos}")
                            print(f"  ({explanation})")

                            if state.winner:
                                break

                            print_board(state)
                            print()
                        else:
                            # Fallback to analysis if strategy doesn't cover this state
                            print("  [Using analysis for remaining moves...]")
                            moves = state.get_valid_moves()
                            if moves:
                                # Try to find winning or best move
                                for p, s in moves:
                                    test = state.make_move(p, s)
                                    if test.winner == "BLUE":
                                        move_history.append(state)
                                        state = test
                                        print(f"BLUE plays: {s[0]} at {p} -> WIN!")
                                        break
                                else:
                                    # Just pick first move
                                    p, s = moves[0]
                                    move_history.append(state)
                                    state = state.make_move(p, s)
                                    print(f"BLUE plays: {s[0]} at {p}")
                                    print_board(state)
                            else:
                                print("No moves available!")
                                break
                    else:
                        # RED's turn - wait for input
                        print("\nRED's turn - enter opponent's move:")
                        try:
                            opp_input = input("  opponent > ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\nAuto-win interrupted.")
                            break

                        pos, size = parse_move(opp_input)
                        if pos and size:
                            try:
                                move_history.append(state)
                                state = state.make_move(pos, size)
                                print(f"RED plays: {size[0]} at {pos}")
                                print_board(state)
                            except ValueError as e:
                                move_history.pop()
                                print(f"Invalid move: {e}")
                        else:
                            print("Invalid format. Use: A1 S")

                if state.winner:
                    print("\n" + "=" * 50)
                    print(f"*** {state.winner} WINS! ***")
                    print("=" * 50)

        elif user_input == "analyze all":
            moves = state.get_valid_moves()
            print(f"\nAnalyzing {len(moves)} possible moves...")

            results = []
            for pos, size in moves:
                analysis = analyze_move(state, pos, size, args.max_states)
                if "error" not in analysis:
                    if analysis["immediate_win"]:
                        results.append((pos, size, 100.0, True))
                    else:
                        results.append((pos, size, analysis["win_rate"], False))

            # Sort by win rate
            results.sort(key=lambda x: x[2], reverse=True)

            print("\n" + "=" * 50)
            print("MOVE RANKINGS (best to worst)")
            print("=" * 50)
            for i, (pos, size, win_rate, immediate) in enumerate(results[:10], 1):
                if immediate:
                    print(f"  {i}. {pos} {size[0]} -> *** IMMEDIATE WIN ***")
                else:
                    print(f"  {i}. {pos} {size[0]} -> {win_rate:.1f}% win rate")

            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more moves")

        elif user_input == "confirm":
            if last_move is None:
                print("No move to confirm. Analyze a move first.")
            elif state.current_player != your_player:
                print("It's not your turn. Record opponent's move first.")
            else:
                pos, size = last_move
                move_history.append(state)
                state = state.make_move(pos, size)
                print(f"\nMove confirmed: {size} at {pos}")
                last_move = None
                last_analysis = None
                print_board(state)

                if state.winner:
                    continue

                print(f"\nNow waiting for opponent's move...")
                print("Use: opponent <position> <size>")

                # Show strategy hint for opponent's turn
                if strategy_cache:
                    print_strategy_hint(state, strategy_cache, strategy_player, your_player)

        elif user_input.startswith("opponent "):
            if state.current_player == your_player:
                print("It's your turn, not the opponent's.")
                continue

            move_str = user_input[9:]
            pos, size = parse_move(move_str)

            if pos is None or size is None:
                print("Invalid move format. Use: opponent <position> <size>")
                print("Example: opponent A1 S")
                continue

            try:
                move_history.append(state)
                state = state.make_move(pos, size)
                print(f"\nOpponent played: {size} at {pos}")
                print_board(state)

                if state.winner:
                    continue

                print("\nYour turn! Analyze a move or type 'moves' to see options.")

                # Show strategy recommendation for your turn
                if strategy_cache:
                    print_strategy_hint(state, strategy_cache, strategy_player, your_player)
                if perfect_mode and state.current_player == "BLUE":
                    print_perfect_move(state)
            except ValueError as e:
                move_history.pop()
                print(f"Invalid move: {e}")

        else:
            # Try to parse as a move
            pos, size = parse_move(user_input)

            if pos is None or size is None:
                print("Unknown command. Type 'help' for available commands.")
                continue

            if state.current_player != your_player:
                print("It's not your turn. Record opponent's move first.")
                continue

            # Analyze the move
            print(f"\nAnalyzing {size} at {pos}...")
            analysis = analyze_move(state, pos, size, args.max_states)
            print_analysis(analysis, pos, size)

            if "error" not in analysis:
                last_move = (pos, size)
                last_analysis = analysis
                print("\nType 'confirm' to make this move, or analyze another.")


if __name__ == "__main__":
    main()
