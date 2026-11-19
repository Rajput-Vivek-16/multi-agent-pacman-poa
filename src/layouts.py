"""
Maze layouts for Multi-Agent Pac-Man.
Representation:
  '%' or '#' : Wall
  '.' : Pellet
  ' ' : Empty open space
  'P' : Pac-Man spawn position placeholder
  'G' : Ghost spawn position placeholder
"""

LAYOUTS = {
    # 6x6 Toy Maze: compact, great for exact analysis and equilibrium verification
    "toy": [
        "######",
        "#P..P#",
        "#.##.#",
        "#.GG.#",
        "#P..P#",
        "######"
    ],

    # 9x7 Small Grid: classic loops and multiple corridors
    "small_grid": [
        "#########",
        "#P.....P#",
        "#.###.#.#",
        "#...G...#",
        "#.#.###.#",
        "#P.....P#",
        "#########"
    ],

    # 13x9 Medium Classic: standard CS188-style multi-room environment
    "medium_classic": [
        "#############",
        "#P....#....P#",
        "#.###.#.###.#",
        "#...........#",
        "#.###.G.###.#",
        "#.....#.....#",
        "#.###.#.###.#",
        "#P....#....P#",
        "#############"
    ],

    # 11x11 Open Contention Maze: high contention area in the center
    "open_contention": [
        "###########",
        "#P.......P#",
        "#.##...##.#",
        "#.........#",
        "#..##G##..#",
        "#....G....#",
        "#..## ##..#",
        "#.........#",
        "#.##...##.#",
        "#P.......P#",
        "###########"
    ]
}


def get_layout(name: str):
    """Retrieve layout by name."""
    if name not in LAYOUTS:
        raise ValueError(f"Unknown layout: {name}. Available: {list(LAYOUTS.keys())}")
    return LAYOUTS[name]
