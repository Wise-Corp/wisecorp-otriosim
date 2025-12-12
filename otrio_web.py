"""
Otrio Game - Streamlit Web Interface

A visual web interface for playing Otrio with AI-powered move analysis.

Usage:
    streamlit run otrio_web.py
"""

import streamlit as st
from otrio_cpn import OtrioGameState, POSITIONS, ROWS, COLS, WIN_LINES
from otrio_consultant import analyze_move, explore_from_state


# =============================================================================
# SVG Board Rendering
# =============================================================================

def generate_board_svg(state, selected_pos=None, hover_pos=None):
    """Generate SVG representation of the Otrio board."""

    # SVG dimensions
    cell_size = 120
    board_size = cell_size * 3
    margin = 40
    total_size = board_size + 2 * margin

    # Colors
    colors = {
        "BLUE": {"SMALL": "#3498db", "MEDIUM": "#2980b9", "LARGE": "#1a5276"},
        "RED": {"SMALL": "#e74c3c", "MEDIUM": "#c0392b", "LARGE": "#922b21"},
    }
    ring_radii = {"SMALL": 15, "MEDIUM": 30, "LARGE": 45}

    svg_parts = [
        f'<svg width="{total_size}" height="{total_size}" xmlns="http://www.w3.org/2000/svg">',
        # Background
        f'<rect width="{total_size}" height="{total_size}" fill="#f5f5dc"/>',
    ]

    # Draw grid
    for i in range(4):
        x = margin + i * cell_size
        svg_parts.append(f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{margin + board_size}" stroke="#8b7355" stroke-width="2"/>')
        svg_parts.append(f'<line x1="{margin}" y1="{x}" x2="{margin + board_size}" y2="{x}" stroke="#8b7355" stroke-width="2"/>')

    # Draw cells and pieces
    for row_idx, row in enumerate(ROWS):
        for col_idx, col in enumerate(COLS):
            pos = f"{row}{col}"
            cx = margin + col_idx * cell_size + cell_size // 2
            cy = margin + row_idx * cell_size + cell_size // 2

            # Highlight selected or hovered cell
            if pos == selected_pos:
                svg_parts.append(f'<rect x="{margin + col_idx * cell_size + 2}" y="{margin + row_idx * cell_size + 2}" '
                               f'width="{cell_size - 4}" height="{cell_size - 4}" fill="#fffacd" rx="5"/>')
            elif pos == hover_pos:
                svg_parts.append(f'<rect x="{margin + col_idx * cell_size + 2}" y="{margin + row_idx * cell_size + 2}" '
                               f'width="{cell_size - 4}" height="{cell_size - 4}" fill="#f0f0f0" rx="5"/>')

            # Draw concentric circle guides (light gray)
            for size, radius in ring_radii.items():
                svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#ddd" stroke-width="1" stroke-dasharray="3,3"/>')

            # Draw pieces for both players
            for player in ["BLUE", "RED"]:
                for size in state.board[player][pos]:
                    radius = ring_radii[size]
                    color = colors[player][size]
                    stroke_width = 8 if size == "SMALL" else 6 if size == "MEDIUM" else 5
                    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="{color}" stroke-width="{stroke_width}"/>')

    # Draw row/column labels
    for idx, row in enumerate(ROWS):
        y = margin + idx * cell_size + cell_size // 2 + 5
        svg_parts.append(f'<text x="15" y="{y}" font-family="Arial" font-size="16" font-weight="bold" fill="#333">{row}</text>')

    for idx, col in enumerate(COLS):
        x = margin + idx * cell_size + cell_size // 2
        svg_parts.append(f'<text x="{x}" y="25" font-family="Arial" font-size="16" font-weight="bold" fill="#333" text-anchor="middle">{col}</text>')

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def generate_piece_legend():
    """Generate SVG legend showing piece sizes."""
    svg = '''
    <svg width="300" height="60" xmlns="http://www.w3.org/2000/svg">
        <text x="10" y="35" font-family="Arial" font-size="14" fill="#333">Sizes:</text>

        <circle cx="80" cy="30" r="15" fill="none" stroke="#666" stroke-width="6"/>
        <text x="80" y="55" font-family="Arial" font-size="10" fill="#333" text-anchor="middle">Small</text>

        <circle cx="150" cy="30" r="25" fill="none" stroke="#666" stroke-width="5"/>
        <text x="150" y="55" font-family="Arial" font-size="10" fill="#333" text-anchor="middle">Medium</text>

        <circle cx="230" cy="30" r="35" fill="none" stroke="#666" stroke-width="4"/>
        <text x="230" y="55" font-family="Arial" font-size="10" fill="#333" text-anchor="middle">Large</text>
    </svg>
    '''
    return svg


# =============================================================================
# Streamlit App
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'game_state' not in st.session_state:
        st.session_state.game_state = OtrioGameState()
    if 'move_history' not in st.session_state:
        st.session_state.move_history = []
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'max_states' not in st.session_state:
        st.session_state.max_states = 10000


def reset_game():
    """Reset the game to initial state."""
    st.session_state.game_state = OtrioGameState()
    st.session_state.move_history = []
    st.session_state.analysis_results = None


def undo_move():
    """Undo the last move."""
    if st.session_state.move_history:
        st.session_state.game_state = st.session_state.move_history.pop()
        st.session_state.analysis_results = None


def make_move(position, size):
    """Make a move and update game state."""
    try:
        st.session_state.move_history.append(st.session_state.game_state)
        st.session_state.game_state = st.session_state.game_state.make_move(position, size)
        st.session_state.analysis_results = None
        return True
    except ValueError as e:
        st.error(f"Invalid move: {e}")
        return False


def analyze_all_moves():
    """Analyze all possible moves and return ranked results."""
    state = st.session_state.game_state
    moves = state.get_valid_moves()

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (pos, size) in enumerate(moves):
        status_text.text(f"Analyzing {pos} {size[0]}... ({i+1}/{len(moves)})")
        analysis = analyze_move(state, pos, size, st.session_state.max_states)

        if "error" not in analysis:
            if analysis.get("immediate_win"):
                results.append((pos, size, 100.0, True, analysis))
            else:
                results.append((pos, size, analysis["win_rate"], False, analysis))

        progress_bar.progress((i + 1) / len(moves))

    status_text.empty()
    progress_bar.empty()

    # Sort by win rate
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def main():
    st.set_page_config(
        page_title="Otrio Game Consultant",
        page_icon="🎯",
        layout="wide"
    )

    init_session_state()
    state = st.session_state.game_state

    # Header
    st.title("🎯 Otrio Game Consultant")
    st.markdown("*AI-powered move analysis for the Otrio board game*")

    # Sidebar
    with st.sidebar:
        st.header("Game Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 New Game", use_container_width=True):
                reset_game()
                st.rerun()
        with col2:
            if st.button("↩️ Undo", use_container_width=True, disabled=len(st.session_state.move_history) == 0):
                undo_move()
                st.rerun()

        st.divider()

        st.header("Analysis Settings")
        st.session_state.max_states = st.slider(
            "Analysis depth (states)",
            min_value=1000,
            max_value=100000,
            value=st.session_state.max_states,
            step=1000,
            help="More states = better analysis but slower"
        )

        st.divider()

        st.header("Game Info")
        if state.winner:
            st.success(f"🏆 {state.winner} WINS!")
        else:
            player_color = "🔵" if state.current_player == "BLUE" else "🔴"
            st.info(f"{player_color} {state.current_player}'s turn")

        st.markdown("**Pieces remaining:**")
        for player in ["BLUE", "RED"]:
            icon = "🔵" if player == "BLUE" else "🔴"
            pieces = state.initial[player]
            st.markdown(f"{icon} S:{pieces['SMALL']} M:{pieces['MEDIUM']} L:{pieces['LARGE']}")

        st.divider()

        st.header("Legend")
        st.markdown(generate_piece_legend(), unsafe_allow_html=True)

    # Main content
    col_board, col_controls = st.columns([2, 1])

    with col_board:
        st.subheader("Game Board")

        # Render SVG board
        svg = generate_board_svg(state)
        st.markdown(svg, unsafe_allow_html=True)

        # Move history
        if st.session_state.move_history:
            with st.expander("Move History", expanded=False):
                for i, (player, pos, size) in enumerate(state.move_history):
                    icon = "🔵" if player == "BLUE" else "🔴"
                    st.markdown(f"{i+1}. {icon} {player}: {size} at {pos}")

    with col_controls:
        if state.winner:
            st.subheader("🏆 Game Over!")
            st.markdown(f"**{state.winner}** wins the game!")
            if st.button("Play Again", use_container_width=True):
                reset_game()
                st.rerun()
        else:
            st.subheader(f"Make a Move ({state.current_player})")

            # Position selection
            position = st.selectbox(
                "Position",
                options=POSITIONS,
                format_func=lambda x: f"{x}"
            )

            # Size selection - only show available sizes
            available_sizes = [s for s in ["SMALL", "MEDIUM", "LARGE"]
                             if state.initial[state.current_player][s] > 0
                             and s not in state.board[state.current_player][position]]

            if available_sizes:
                size = st.selectbox(
                    "Size",
                    options=available_sizes,
                    format_func=lambda x: {"SMALL": "Small (S)", "MEDIUM": "Medium (M)", "LARGE": "Large (L)"}[x]
                )

                col_analyze, col_play = st.columns(2)

                with col_analyze:
                    if st.button("🔍 Analyze", use_container_width=True):
                        with st.spinner("Analyzing..."):
                            analysis = analyze_move(state, position, size, st.session_state.max_states)
                            st.session_state.analysis_results = (position, size, analysis)

                with col_play:
                    if st.button("✅ Play Move", use_container_width=True, type="primary"):
                        if make_move(position, size):
                            st.rerun()

                # Show analysis results
                if st.session_state.analysis_results:
                    a_pos, a_size, analysis = st.session_state.analysis_results

                    st.divider()
                    st.markdown(f"**Analysis: {a_size} at {a_pos}**")

                    if "error" in analysis:
                        st.error(analysis["error"])
                    elif analysis.get("immediate_win"):
                        st.success("⭐ IMMEDIATE WIN!")
                        st.markdown(f"*{analysis.get('win_type', 'Win condition met')}*")
                    else:
                        win_rate = analysis["win_rate"]

                        # Color-coded win rate
                        if win_rate >= 60:
                            st.success(f"Win probability: **{win_rate:.1f}%**")
                        elif win_rate >= 40:
                            st.warning(f"Win probability: **{win_rate:.1f}%**")
                        else:
                            st.error(f"Win probability: **{win_rate:.1f}%**")

                        st.caption(f"States analyzed: {analysis['states_explored']:,}")
            else:
                st.warning("No valid moves for this position!")

            st.divider()

            # Analyze all moves
            if st.button("📊 Analyze All Moves", use_container_width=True):
                results = analyze_all_moves()

                st.subheader("Move Rankings")
                for i, (pos, size, win_rate, immediate, _) in enumerate(results[:10], 1):
                    if immediate:
                        st.markdown(f"**{i}. {pos} {size[0]}** → ⭐ IMMEDIATE WIN")
                    else:
                        bar_color = "green" if win_rate >= 60 else "orange" if win_rate >= 40 else "red"
                        st.markdown(f"**{i}. {pos} {size[0]}** → {win_rate:.1f}%")
                        st.progress(win_rate / 100)

                if len(results) > 10:
                    st.caption(f"...and {len(results) - 10} more moves")

    # Footer
    st.divider()
    st.caption("Otrio Game Consultant | Powered by Colored Petri Net analysis")


if __name__ == "__main__":
    main()
