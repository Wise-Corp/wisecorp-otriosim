# OtrioSim

## Project Overview
A simulation/exploration project for **Otrio**, a strategic tic-tac-toe variant board game. Otrio is played on a 3x3 grid where players place pieces of three different sizes (small, medium, large) and win by getting three of their pieces in a row by:
- Same size in a row/column/diagonal
- Ascending/descending sizes in a row/column/diagonal
- Three nested sizes (small, medium, large) in the same space

## Goals
- Explore game mechanics and strategies
- Simulate gameplay scenarios
- Analyze winning conditions and optimal moves

## Repository Structure
- `.gitlab-ci.yml` - GitLab CI/CD pipeline configuration
- `otrio_cpn.py` - Colored Petri Net implementation of Otrio

## Tech Stack
- **Language**: Python 3.10+
- **CPN Framework**: cpn-py (Colored Petri Net library)
- **CI/CD**: GitLab CI
- **Version Control**: Git (hosted on GitLab)

## CPN Model Architecture
The game is modeled as a Colored Petri Net with:
- **20 Places**: 2 initial places (Blue_Initial, Red_Initial) + 18 board places (9 per player)
- **18 Transitions**: One transition per position per player (Blue_to_A1, Red_to_A1, etc.)
- **36 Arcs**: Input/output arcs connecting initial places to board positions
- **Color Set**: Size = {SMALL, MEDIUM, LARGE}

## Running the Simulation
```bash
# Install dependencies
pip install cpnpy

# Run simulation and state space exploration
python otrio_cpn.py
```

## Development Workflow
- Main branch: `master`
- Remote: `https://gitlab.com/wise-corp/techteam/devops/otriosim.git`

## Game Rules Reference
- 2-4 players
- Each player has 9 pieces: 3 small, 3 medium, 3 large
- Players take turns placing one piece on the board
- Win conditions:
  1. Three same-sized pieces in a row/column/diagonal
  2. Three ascending or descending sizes in a row/column/diagonal
  3. Three nested pieces (all three sizes) in one space
