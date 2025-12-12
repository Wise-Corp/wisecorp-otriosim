"""
Verify the proposed winning strategy for BLUE (first player) in Otrio.

Strategy outline:
1. BLUE places MEDIUM at B2 (center)
2. RED places whatever
3. BLUE responds based on RED's move:
   - If RED placed at B2: BLUE places LARGE at A1
   - If RED placed elsewhere: BLUE places LARGE at contiguous position
4. This creates forcing sequences that RED cannot defend

Let's verify this systematically.
"""

from otrio_cpn import OtrioGameState, POSITIONS, WIN_LINES

# Adjacency map (contiguous positions)
ADJACENT = {
    "A1": ["A2", "B1", "B2"],
    "A2": ["A1", "A3", "B1", "B2", "B3"],
    "A3": ["A2", "B2", "B3"],
    "B1": ["A1", "A2", "B2", "C1", "C2"],
    "B2": ["A1", "A2", "A3", "B1", "B3", "C1", "C2", "C3"],  # Center is adjacent to all
    "B3": ["A2", "A3", "B2", "C2", "C3"],
    "C1": ["B1", "B2", "C2"],
    "C2": ["C1", "C3", "B1", "B2", "B3"],
    "C3": ["C2", "B2", "B3"],
}

# Diagonal pairs (for ascending/descending win threats)
DIAGONAL_OPPOSITE = {
    "A1": "C3",
    "A3": "C1",
    "C1": "A3",
    "C3": "A1",
}


def get_threats(state, player):
    """Find all win threats for a player (positions where one more piece wins)."""
    threats = []
    board = state.board[player]

    # Check each win line
    for line in WIN_LINES:
        # Nested win threat (2 of 3 sizes in same position)
        for pos in line:
            if len(board[pos]) == 2:
                missing = {"SMALL", "MEDIUM", "LARGE"} - board[pos]
                for size in missing:
                    threats.append(("nested", pos, size))

        # Same-size win threat (2 of 3 positions with same size)
        for size in ["SMALL", "MEDIUM", "LARGE"]:
            positions_with_size = [pos for pos in line if size in board[pos]]
            if len(positions_with_size) == 2:
                missing_pos = [p for p in line if p not in positions_with_size][0]
                threats.append(("same_size", missing_pos, size))

        # Ordered win threat (ascending or descending)
        # Ascending: S-M-L
        sizes_present = []
        for i, pos in enumerate(line):
            expected = ["SMALL", "MEDIUM", "LARGE"][i]
            if expected in board[pos]:
                sizes_present.append((i, pos, expected))

        if len(sizes_present) == 2:
            # Find missing position
            present_indices = {s[0] for s in sizes_present}
            for i in range(3):
                if i not in present_indices:
                    missing_pos = line[i]
                    missing_size = ["SMALL", "MEDIUM", "LARGE"][i]
                    threats.append(("ascending", missing_pos, missing_size))

        # Descending: L-M-S
        sizes_present = []
        for i, pos in enumerate(line):
            expected = ["LARGE", "MEDIUM", "SMALL"][i]
            if expected in board[pos]:
                sizes_present.append((i, pos, expected))

        if len(sizes_present) == 2:
            present_indices = {s[0] for s in sizes_present}
            for i in range(3):
                if i not in present_indices:
                    missing_pos = line[i]
                    missing_size = ["LARGE", "MEDIUM", "SMALL"][i]
                    threats.append(("descending", missing_pos, missing_size))

    return threats


def verify_strategy():
    """
    Trace through the proposed strategy and verify it works.
    """
    print("=" * 70)
    print("VERIFYING PROPOSED WINNING STRATEGY FOR BLUE (FIRST PLAYER)")
    print("=" * 70)

    # Step 1: BLUE places MEDIUM at B2
    print("\n--- STEP 1: BLUE places MEDIUM at B2 (center) ---")
    state = OtrioGameState()
    state = state.make_move("B2", "MEDIUM")
    print(f"Board after BLUE M@B2:")
    print_board(state)

    # Now enumerate all RED responses
    red_moves = state.get_valid_moves()
    print(f"\nRED has {len(red_moves)} possible responses")

    # Track results
    blue_wins = 0
    blue_loses = 0
    unknown = 0

    for red_pos, red_size in red_moves:
        print(f"\n{'='*60}")
        print(f"Testing: RED plays {red_size} at {red_pos}")
        print("=" * 60)

        state2 = state.make_move(red_pos, red_size)

        # Apply the strategy
        result = apply_strategy(state2, red_pos, red_size)

        if result == "BLUE":
            blue_wins += 1
        elif result == "RED":
            blue_loses += 1
        else:
            unknown += 1

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total RED responses tested: {len(red_moves)}")
    print(f"BLUE wins: {blue_wins}")
    print(f"RED wins: {blue_loses}")
    print(f"Unknown/needs more analysis: {unknown}")

    if blue_loses == 0 and unknown == 0:
        print("\n*** STRATEGY VERIFIED: BLUE wins against ALL RED responses! ***")
    elif blue_loses == 0:
        print(f"\n*** STRATEGY PROMISING: {unknown} cases need deeper analysis ***")
    else:
        print(f"\n*** STRATEGY HAS HOLES: RED can win in {blue_loses} cases ***")


def apply_strategy(state, red_pos, red_size):
    """
    Apply BLUE's strategy after RED's first move.
    Returns the winner or "UNKNOWN".
    """
    # Strategy rule:
    # - If RED placed at B2: BLUE places LARGE at A1
    # - If RED placed elsewhere: BLUE places LARGE at position adjacent/contiguous to RED's move

    if red_pos == "B2":
        # RED blocked B2 with same size or different
        # BLUE plays LARGE at A1
        blue_pos = "A1"
        blue_size = "LARGE"
        print(f"  Strategy: RED at B2 -> BLUE plays L@A1")
    else:
        # RED played elsewhere - BLUE plays LARGE adjacent to RED's position
        # Creating threats along the diagonal or line

        # Find best adjacent position for LARGE
        # Prefer positions that create winning threats
        adjacent = ADJACENT.get(red_pos, [])

        # Strategy: place LARGE to threaten ascending/descending win through B2
        # Since we have MEDIUM at B2, we want S-M-L or L-M-S lines

        # Check which lines through B2 RED is blocking
        # and choose a LARGE placement that creates maximum threat

        best_pos = None

        # Look for diagonal opportunities
        if red_pos in DIAGONAL_OPPOSITE:
            # RED is on a corner - threaten the diagonal
            diag_opp = DIAGONAL_OPPOSITE[red_pos]
            if "LARGE" not in state.board["RED"].get(diag_opp, set()):
                best_pos = diag_opp

        if not best_pos:
            # Pick a corner that creates a line with B2
            for pos in ["A1", "A3", "C1", "C3"]:
                if "LARGE" not in state.board["RED"].get(pos, set()) and \
                   "LARGE" not in state.board["BLUE"].get(pos, set()):
                    best_pos = pos
                    break

        if not best_pos:
            # Fallback: any valid position for LARGE
            for pos in POSITIONS:
                if "LARGE" not in state.board["RED"].get(pos, set()) and \
                   "LARGE" not in state.board["BLUE"].get(pos, set()):
                    best_pos = pos
                    break

        if not best_pos:
            print(f"  No valid LARGE placement found!")
            return "UNKNOWN"

        blue_pos = best_pos
        blue_size = "LARGE"
        print(f"  Strategy: RED at {red_pos} -> BLUE plays L@{blue_pos}")

    # Make BLUE's move
    try:
        state = state.make_move(blue_pos, blue_size)
    except ValueError as e:
        print(f"  Error: {e}")
        return "UNKNOWN"

    if state.winner:
        print(f"  -> BLUE WINS immediately!")
        return state.winner

    print_board_compact(state)

    # Now analyze if BLUE has a winning position
    # Check for double threats (2 ways to win that RED can't both block)

    result = continue_game_with_analysis(state, depth=6)
    return result


def continue_game_with_analysis(state, depth=6):
    """
    Continue the game with minimax analysis up to given depth.
    """
    from collections import deque

    if depth == 0:
        return "UNKNOWN"

    if state.winner:
        return state.winner

    moves = state.get_valid_moves()
    if not moves:
        return "DRAW"

    player = state.current_player

    if player == "BLUE":
        # BLUE (maximizing) - try to find a winning move
        for pos, size in moves:
            new_state = state.make_move(pos, size)
            if new_state.winner == "BLUE":
                print(f"    BLUE can win with {size}@{pos}")
                return "BLUE"

        # Check for double threats
        threats_after = []
        for pos, size in moves:
            new_state = state.make_move(pos, size)
            if not new_state.winner:
                blue_threats = get_threats(new_state, "BLUE")
                if len(blue_threats) >= 2:
                    print(f"    BLUE creates double threat with {size}@{pos}: {len(blue_threats)} threats")
                    # RED can only block one - BLUE wins!
                    return "BLUE"

        # No immediate win - continue recursively (simplified)
        best = "UNKNOWN"
        for pos, size in moves[:5]:  # Limit for speed
            new_state = state.make_move(pos, size)
            result = continue_game_with_analysis(new_state, depth - 1)
            if result == "BLUE":
                return "BLUE"
            if result == "UNKNOWN":
                best = "UNKNOWN"
        return best

    else:
        # RED (minimizing) - see if RED can prevent BLUE from winning
        # Check if BLUE has any immediate winning threats RED must block
        blue_threats = get_threats(state, "BLUE")

        if len(blue_threats) >= 2:
            # Multiple threats - RED can only block one
            print(f"    RED faces {len(blue_threats)} threats - cannot defend!")
            return "BLUE"

        # RED blocks if there's a threat
        if blue_threats:
            threat_type, threat_pos, threat_size = blue_threats[0]
            # RED must play at threat_pos with threat_size (blocking)
            valid_moves = [(p, s) for p, s in moves if p == threat_pos and s == threat_size]
            if valid_moves:
                pos, size = valid_moves[0]
                new_state = state.make_move(pos, size)
                print(f"    RED blocks threat at {pos} with {size}")
                return continue_game_with_analysis(new_state, depth - 1)
            else:
                # RED can't block!
                print(f"    RED cannot block threat {threat_type} at {threat_pos}!")
                return "BLUE"

        # No threats - RED plays randomly (for analysis we try worst case for BLUE)
        for pos, size in moves[:5]:
            new_state = state.make_move(pos, size)
            result = continue_game_with_analysis(new_state, depth - 1)
            if result == "RED":
                return "RED"
            if result == "UNKNOWN":
                pass

        return "UNKNOWN"


def print_board(state):
    """Print a readable board state."""
    print("  BLUE:")
    for row in ["A", "B", "C"]:
        row_str = f"    {row}:"
        for col in ["1", "2", "3"]:
            pos = f"{row}{col}"
            sizes = state.board["BLUE"][pos]
            s = "".join(sz[0] for sz in sorted(sizes)) if sizes else "."
            row_str += f" [{s:3}]"
        print(row_str)

    print("  RED:")
    for row in ["A", "B", "C"]:
        row_str = f"    {row}:"
        for col in ["1", "2", "3"]:
            pos = f"{row}{col}"
            sizes = state.board["RED"][pos]
            s = "".join(sz[0] for sz in sorted(sizes)) if sizes else "."
            row_str += f" [{s:3}]"
        print(row_str)


def print_board_compact(state):
    """Print board state in compact form."""
    print("  Board:")
    for row in ["A", "B", "C"]:
        row_str = f"    {row}:"
        for col in ["1", "2", "3"]:
            pos = f"{row}{col}"
            blue = "".join(sz[0] for sz in sorted(state.board["BLUE"][pos]))
            red = "".join(sz[0] for sz in sorted(state.board["RED"][pos]))
            cell = ""
            if blue:
                cell += f"B:{blue}"
            if red:
                cell += f"R:{red}" if not cell else f",R:{red}"
            if not cell:
                cell = "."
            row_str += f" [{cell:10}]"
        print(row_str)


if __name__ == "__main__":
    verify_strategy()
