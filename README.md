# OtrioSim

A Colored Petri Net (CPN) simulation of **Otrio**, a strategic tic-tac-toe variant board game.

## Game Overview

Otrio is played on a 3x3 grid where players place pieces of three different sizes (small, medium, large). A player wins by achieving one of the following:

1. **Nested Win**: Place all three sizes (small, medium, large) in the same space
2. **Same Size Win**: Place three pieces of the same size in a row, column, or diagonal
3. **Ordered Win**: Place three pieces in ascending or descending size order in a row, column, or diagonal

## CPN Model Design

The game is modeled as a Colored Petri Net with the following structure:

![Otrio CPN Model](docs/otrio_cpn_diagram.png)

### Model Architecture

```
                    ┌─────────────────┐
                    │  Blue_Initial   │ ← 9 tokens: 3×SMALL, 3×MEDIUM, 3×LARGE
                    │   (△△△ △△△ ▲▲▲) │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │Blue_to_ │        │Blue_to_ │        │Blue_to_ │  ... (9 transitions)
    │   A1    │        │   A2    │        │   A3    │
    └────┬────┘        └────┬────┘        └────┬────┘
         │                   │                   │
         ▼                   ▼                   ▼
    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ Blue_A1 │        │ Blue_A2 │        │ Blue_A3 │
    └─────────┘        └─────────┘        └─────────┘

    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ Blue_B1 │        │ Blue_B2 │        │ Blue_B3 │  (9 board places per player)
    └─────────┘        └─────────┘        └─────────┘

    ┌─────────┐        ┌─────────┐        ┌─────────┐
    │ Blue_C1 │        │ Blue_C2 │        │ Blue_C3 │
    └─────────┘        └─────────┘        └─────────┘

                    ┌─────────────────┐
                    │  Red_Initial    │ ← 9 tokens: 3×SMALL, 3×MEDIUM, 3×LARGE
                    │   (△△△ △△△ ▲▲▲) │
                    └────────┬────────┘
                             │
                            ...  (same structure for Red player)
```

### Components

| Component | Count | Description |
|-----------|-------|-------------|
| **Places** | 20 | 2 initial + 18 board positions (9 per player) |
| **Transitions** | 18 | One per position per player |
| **Arcs** | 36 | Input/output arcs connecting places to transitions |
| **Color Set** | 1 | Size = {SMALL, MEDIUM, LARGE} |

### Token Colors

| Token | Symbol | Description |
|-------|--------|-------------|
| SMALL | △ (light gray) | Smallest piece |
| MEDIUM | △ (dark gray) | Medium piece |
| LARGE | ▲ (black) | Largest piece |

## Installation

```bash
# Clone the repository
git clone https://gitlab.com/wise-corp/techteam/devops/otriosim.git
cd otriosim

# Install dependencies
pip install cpnpy streamlit
```

## Usage

### 1. Run the Simulation

```bash
python otrio_cpn.py                      # Default: simulate + explore 50k states
python otrio_cpn.py -n 200000            # Explore 200,000 states
python otrio_cpn.py -n 1000000 -q        # 1M states, quiet mode
python otrio_cpn.py --no-simulate        # Skip random game, only explore states
python otrio_cpn.py -n 0                 # Only simulate, no state exploration
```

### 2. Web Interface (Recommended)

Launch the visual web interface with interactive board:

```bash
streamlit run otrio_web.py
```

Then open http://localhost:8501 in your browser.

**Features:**
- Visual SVG board with colored pieces
- Click to select position and size
- Real-time move analysis with win probabilities
- "Analyze All Moves" to rank every possible move
- Move history and undo support

![Otrio Web Interface](docs/otrio_web_screenshot.png)

### 3. Command-Line Consultant

Use the CLI consultant for move advice during a real game:

```bash
python otrio_consultant.py               # Play as BLUE (default)
python otrio_consultant.py --player red  # Play as RED
python otrio_consultant.py -n 50000      # More thorough analysis (50k states)
```

**Consultant Commands:**

| Command | Description |
|---------|-------------|
| `A1 S` | Analyze placing SMALL at A1 |
| `B2 LARGE` | Analyze placing LARGE at B2 |
| `analyze all` | Rank all possible moves by win probability |
| `confirm` | Confirm your analyzed move |
| `opponent B1 M` | Record opponent's move |
| `moves` | Show all valid moves |
| `board` | Show current board state |
| `undo` | Undo last move |
| `help` | Show help |
| `quit` | Exit |

**Example Session:**

```
[BLUE's turn] > analyze all

MOVE RANKINGS (best to worst)
==================================================
  1. B2 M -> 64.7% win rate
  2. C2 M -> 56.3% win rate
  3. C3 M -> 56.3% win rate
  ...

[BLUE's turn] > B2 M
Analysis for: MEDIUM at B2
  >>> Win probability: 64.7% <<<

[BLUE's turn] > confirm
Move confirmed: MEDIUM at B2

[RED's turn] > opponent A1 S
Opponent played: SMALL at A1
```

### 4. Using as a Library

```python
from otrio_cpn import (
    build_otrio_cpn,
    create_initial_marking,
    simulate_cpn_game,
    explore_state_space,
    OtrioGameState
)

# Build the CPN model
cpn, places, transitions = build_otrio_cpn()

# Create initial marking
marking = create_initial_marking(cpn)

# Simulate a game
simulate_cpn_game(cpn, seed=42)

# Explore state space
stats, visited = explore_state_space(max_states=100000)
```

## State Space Analysis

The state space exploration provides insights into the game:

| Metric | Value (sample) |
|--------|----------------|
| Total states explored | 50,000 |
| Terminal states | ~39,000 |
| BLUE wins | ~16,400 |
| RED wins | ~20,000 |
| Minimum depth to win | 5 moves |
| Maximum depth | 18 moves |

## Win Conditions

### 1. Nested Win (all sizes in one space)

```
Position A1: [S M L]  ← SMALL + MEDIUM + LARGE = WIN
```

### 2. Same Size Win (three in a row)

```
  1   2   3
A [S] [S] [S]  ← Three SMALL in row A = WIN
B [ ] [ ] [ ]
C [ ] [ ] [ ]
```

### 3. Ordered Win (ascending/descending)

```
  1   2   3
A [S] [ ] [ ]
B [ ] [M] [ ]  ← SMALL → MEDIUM → LARGE diagonal = WIN
C [ ] [ ] [L]
```

## Project Structure

```
otriosim/
├── README.md              # This file
├── CLAUDE.md              # Project context for Claude Code
├── otrio_cpn.py           # Main CPN implementation & state space explorer
├── otrio_consultant.py    # CLI game consultant
├── otrio_web.py           # Streamlit web interface
├── .gitlab-ci.yml         # CI/CD configuration
└── docs/
    └── otrio_cpn_diagram.png  # CPN diagram
```

## License

Internal use - WiseCorp TechTeam

## References

- [Otrio Board Game Rules](https://www.spinmaster.com/en-US/brands/marbles-brain-workshop/otrio)
- [cpn-py: Python Colored Petri Net Library](https://github.com/fit-alessandro-berti/cpn-py)
- [Colored Petri Nets (Jensen, 1997)](https://link.springer.com/book/10.1007/978-3-662-03241-1)
