import numpy as np

instances_data = {
    "NL4": {
        "team_names": ["ATL", "NYM", "PHI", "MON"],
        "dist_matrix": np.array([
            [0, 745, 665, 929],
            [745, 0, 80, 337],
            [665, 80, 0, 380],
            [929, 337, 380, 0]
        ], dtype=float)
    },
    "NL6": {
        "team_names": ["ATL", "NYM", "PHI", "MON", "FLA", "PIT"],
        "dist_matrix": np.array([
            [0, 745, 665, 929, 605, 521],
            [745, 0, 80, 337, 1090, 315],
            [665, 80, 0, 380, 1020, 257],
            [929, 337, 380, 0, 1380, 408],
            [605, 1090, 1020, 1380, 0, 1010],
            [521, 315, 257, 408, 1010, 0]
        ], dtype=float)
    },
    "CIRC4": {
        "team_names": ["T1", "T2", "T3", "T4"],
        "dist_matrix": np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ], dtype=float)
    },
    "CON4": {
        "team_names": ["T1", "T2", "T3", "T4"],
        "dist_matrix": np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ], dtype=float)
    },
    "CIRC6": {
        "team_names": ["T1", "T2", "T3", "T4", "T5", "T6"],
        "dist_matrix": np.array([
            [0, 1, 2, 3, 2, 1],
            [1, 0, 1, 2, 3, 2],
            [2, 1, 0, 1, 2, 3],
            [3, 2, 1, 0, 1, 2],
            [2, 3, 2, 1, 0, 1],
            [1, 2, 3, 2, 1, 0]
        ], dtype=float)
    },
    "INCR4": {
        "team_names": ["T1", "T2", "T3", "T4"],
        "dist_matrix": np.array([
            [0, 1, 3, 6],
            [1, 0, 2, 5],
            [3, 2, 0, 3],
            [6, 5, 3, 0]
        ], dtype=float)
    },
    "LINE4": {
        "team_names": ["T1", "T2", "T3", "T4"],
        "dist_matrix": np.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0]
        ], dtype=float)
    }
}

