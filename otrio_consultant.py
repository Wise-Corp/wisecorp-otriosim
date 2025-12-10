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

    print(f"\nValid moves for {state.current_player}:")

    # Group by position
    by_position = defaultdict(list)
    for pos, size in moves:
        by_position[pos].append(size)

    for pos in POSITIONS:
        if pos in by_position:
            sizes = ", ".join(s[0] for s in by_position[pos])
            print(f"  {pos}: [{sizes}]")


def print_help():
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
  quit               - Exit the consultant

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
        description="Otrio Game Consultant - Interactive Move Advisor"
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

    args = parser.parse_args()

    print("=" * 60)
    print("OTRIO GAME CONSULTANT")
    print("=" * 60)
    print(f"\nYou are playing as: {args.player.upper()}")
    print(f"Analysis depth: {args.max_states:,} states per move")
    print("\nType 'help' for commands, 'quit' to exit")

    state = OtrioGameState()
    your_player = args.player.upper()
    last_analysis = None
    last_move = None
    move_history = []

    print_board(state)

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
            print_help()

        elif user_input == "board":
            print_board(state)

        elif user_input == "moves":
            print_valid_moves(state)

        elif user_input == "undo":
            if move_history:
                state = move_history.pop()
                print("Move undone.")
                print_board(state)
            else:
                print("No moves to undo.")

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
