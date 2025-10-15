# %% [md]
"""
# Column Generation for the TTP
"""

# %%
import numpy as np
from copy import deepcopy
import time, math, heapq
from itertools import product, combinations
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# %% [md]
"""
## Instance data
"""

# %%
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


# %% [md]
"""
## Column Generation
"""

# %%
"""
TTP column generation. 

The Master LP is solved with Gurobi.
The pricing problem for each team is solved by dynamic progrmaming that finds the schedule minimizing travel_cost_exact - pair_round_dual_terms where the reduced_cost = DP_obj - team_dual.
Main constraints:
    * Valid Double Round Robin
    * no immediate rematch (non-repeater),
    * at most K consecutive home / away (tracked as streaks),
    * optional return-to-home cost after final round (default True).
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

class TeamScheduleColumn:
    def __init__(self, team, opp_round, home, cost):
        self.team = int(team)
        self.opp_round = list(opp_round)
        self.home = list(home)
        self.cost = float(cost)

# Computes the cost of a schedule
def schedule_cost_exact(team, opp_round, home_flags, dist_matrix, include_return_home=True):
    prev = team
    total = 0.0
    for r, opp in enumerate(opp_round):
        curr = team if home_flags[r] else opp
        total += dist_matrix[prev, curr]
        prev = curr
    if include_return_home:
        total += dist_matrix[prev, team]
    return total

# Solves the MP with gurobi
def solve_master_gurobi(columns, n_teams, n_rounds, gurobi_params=None):
    m = gp.Model("master_lp")
    m.setParam("OutputFlag", 0)
    if gurobi_params:
        for k, v in gurobi_params.items():
            try:
                m.setParam(k, v)
            except Exception:
                pass

    x_vars = [m.addVar(lb=0.0, name=f"x_{idx}", obj=c.cost) for idx, c in enumerate(columns)]
    m.modelSense = GRB.MINIMIZE
    m.update()

    team_constrs = {}
    for i in range(n_teams):
        idxs = [k for k, c in enumerate(columns) if c.team == i]
        if not idxs:
            raise ValueError(f"No initial column for team {i}")
        team_constrs[i] = m.addConstr(gp.quicksum(x_vars[k] for k in idxs) == 1.0, name=f"team_{i}")

    pair_constrs = {}
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            for r in range(n_rounds):
                idxs_i = [k for k, c in enumerate(columns) if c.team == i and c.opp_round[r] == j]
                idxs_j = [k for k, c in enumerate(columns) if c.team == j and c.opp_round[r] == i]
                pair_constrs[(i, j, r)] = m.addConstr(
                    gp.quicksum(x_vars[k] for k in idxs_i) - gp.quicksum(x_vars[k] for k in idxs_j) == 0.0,
                    name=f"pair_{i}_{j}_{r}"
                )
    m.update()
    m.optimize()

    if m.status != GRB.OPTIMAL:
        raise RuntimeError(f"Master LP not optimal (status {m.status}).")

    x_vals = np.array([v.x for v in m.getVars()], dtype=float)
    team_duals = [team_constrs[i].pi for i in range(len(team_constrs))]
    pair_duals = {k: float(constr.pi or 0.0) for k, constr in pair_constrs.items()}
    return x_vals, {'team': team_duals, 'pair_round': pair_duals}, float(m.objVal)

# Solves the pricing problem
def price_team_exact_dp(team, dist_matrix, duals, forbid_repeat=True,
                        max_consec_home=None, max_consec_away=None,
                        include_return_home=True):
    n_teams = dist_matrix.shape[0]
    R = 2 * (n_teams - 1)
    opponents = [j for j in range(n_teams) if j != team]
    m = len(opponents)
    full_mask = (1 << m) - 1

    init_state = (0, 0, team, -1, 0, -1)
    dp_curr = {init_state: 0.0}
    parent = {}

    for r in range(R):
        dp_next = {}
        for state, cost_so_far in dp_curr.items():
            home_mask, away_mask, prev_loc, streak_type, streak_len, last_opp = state
            for bit_idx, opp in enumerate(opponents):
                if forbid_repeat and last_opp == opp:
                    continue

                # HOME option
                if not (home_mask >> bit_idx) & 1:
                    ok_home = not (max_consec_home is not None and streak_type == 1 and streak_len >= max_consec_home)
                    if ok_home:
                        new_prev = team
                        travel = dist_matrix[prev_loc, new_prev]
                        pi = duals['pair_round'].get((team, opp, r), 0.0) if team < opp else -duals['pair_round'].get((opp, team, r), 0.0)
                        new_cost = cost_so_far + travel - pi
                        new_streak_len = streak_len + 1 if streak_type == 1 else 1
                        new_state = (home_mask | (1 << bit_idx), away_mask, new_prev, 1, new_streak_len, opp)
                        if new_cost < dp_next.get(new_state, float('inf')):
                            dp_next[new_state] = new_cost
                            parent[(r + 1, new_state)] = (state, opp, True)

                # AWAY option
                if not (away_mask >> bit_idx) & 1:
                    ok_away = not (max_consec_away is not None and streak_type == 0 and streak_len >= max_consec_away)
                    if ok_away:
                        new_prev = opp
                        travel = dist_matrix[prev_loc, new_prev]
                        pi = duals['pair_round'].get((team, opp, r), 0.0) if team < opp else -duals['pair_round'].get((opp, team, r), 0.0)
                        new_cost = cost_so_far + travel - pi
                        new_streak_len = streak_len + 1 if streak_type == 0 else 1
                        new_state = (home_mask, away_mask | (1 << bit_idx), new_prev, 0, new_streak_len, opp)
                        if new_cost < dp_next.get(new_state, float('inf')):
                            dp_next[new_state] = new_cost
                            parent[(r + 1, new_state)] = (state, opp, False)
        dp_curr = dp_next
        if not dp_curr: return None, None

    best_val, best_final_state = min(
        ((cost + (dist_matrix[s[2], team] if include_return_home else 0.0)), s)
        for s, cost in dp_curr.items() if s[0] == full_mask and s[1] == full_mask
    ) if any(s[0] == full_mask and s[1] == full_mask for s in dp_curr) else (None, None)

    if best_final_state is None: return None, None

    opp_round, home_flags = [-1] * R, [False] * R
    cur, r_idx = best_final_state, R
    while r_idx > 0:
        prev_state, opp, is_home = parent[(r_idx, cur)]
        opp_round[r_idx - 1], home_flags[r_idx - 1] = opp, is_home
        cur, r_idx = prev_state, r_idx - 1

    exact_cost = schedule_cost_exact(team, opp_round, home_flags, dist_matrix, include_return_home)
    reduced_cost = best_val - (duals['team'][team] if 'team' in duals and len(duals['team']) > team else 0.0)
    return TeamScheduleColumn(team, opp_round, home_flags, exact_cost), reduced_cost

# Generates initial columns
def circle_method_double_round_robin(n_teams):
    teams = list(range(n_teams))
    single_rr = []
    for r in range(n_teams - 1):
        pairs = [(teams[i], teams[n_teams - 1 - i]) if r % 2 == 0 else (teams[n_teams - 1 - i], teams[i]) for i in range(n_teams // 2)]
        single_rr.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return single_rr + [[(a, h) for h, a in p] for p in single_rr]

# Main function
def column_generation_exact_pricing(dist_matrix, max_iters=1000, tol=1e-8,
                                    forbid_repeat=True, max_consec_home=None, max_consec_away=None,
                                    include_return_home=True, gurobi_master_params=None):
    n_teams = dist_matrix.shape[0]
    R = 2 * (n_teams - 1)
    
    rr = circle_method_double_round_robin(n_teams)
    columns = []
    for t in range(n_teams):
        opp_round, home_flags = [-1] * R, [False] * R
        for r, round_pairs in enumerate(rr):
            for h, a in round_pairs:
                if h == t: opp_round[r], home_flags[r] = a, True; break
                if a == t: opp_round[r], home_flags[r] = h, False; break
        cost = schedule_cost_exact(t, opp_round, home_flags, dist_matrix, include_return_home)
        columns.append(TeamScheduleColumn(t, opp_round, home_flags, cost))

    for it in range(max_iters):
        x_vals, duals, master_obj = solve_master_gurobi(columns, n_teams, R, gurobi_params=gurobi_master_params)
        print(f"[iter {it}] master obj = {master_obj:.6f}, #columns = {len(columns)}")

        new_cols = []
        for t in range(n_teams):
            col, rc = price_team_exact_dp(t, dist_matrix, duals,
                                          forbid_repeat, max_consec_home, max_consec_away,
                                          include_return_home)
            if col and rc < -tol:
                # print(f"  team {t}: found negative reduced-cost column rc={rc:.9f}")
                new_cols.append(col)
        
        if not new_cols:
            print("\nNo negative reduced-cost columns found. LP optimality reached.")
            return columns, x_vals, master_obj
        columns.extend(new_cols)

    raise RuntimeError("Reached max iterations without convergence.")

# Prints the final schedule, not used for just the column generation but was used for branch-and-bound
def print_final_schedule(columns, x_vals, team_names, tol=1e-5):
    n_teams = len(team_names)
    R = 2 * (n_teams - 1)

    chosen_schedules = {}
    is_fractional = False
    for i, col in enumerate(columns):
        if x_vals[i] > tol:
            if abs(x_vals[i] - 1.0) > tol:
                is_fractional = True
                break 
            chosen_schedules[col.team] = col

    if is_fractional or len(chosen_schedules) != n_teams:
        print("\n--- SOLUTION IS FRACTIONAL ---")
        print("Cannot construct a single valid schedule from the LP solution.")
        print("This is common for larger problems and requires a Branch-and-Price algorithm for a guaranteed integer solution.")
        return

    print("\n--- FINAL TOURNAMENT SCHEDULE (INTEGER SOLUTION) ---")
    full_schedule = [[] for _ in range(R)]
    for r in range(R):
        seen_teams_in_round = set()
        for team_idx in range(n_teams):
            if team_idx in seen_teams_in_round:
                continue
            
            schedule = chosen_schedules[team_idx]
            opponent_idx = schedule.opp_round[r]
            
            home_team, away_team = (team_names[team_idx], team_names[opponent_idx]) if schedule.home[r] else (team_names[opponent_idx], team_names[team_idx])
            
            full_schedule[r].append(f"  {away_team} @ {home_team}")
            seen_teams_in_round.add(team_idx)
            seen_teams_in_round.add(opponent_idx)

    for r in range(R):
        print(f"\n--- Round {r+1} ---")
        for match in full_schedule[r]:
            print(match)

# Run CG on some instance
def run_ttp_instance(instance_name, team_names, dist_matrix, max_consec=3, include_return_home=True):
    print(f"\n{'='*25} Running TTP Instance: {instance_name} {'='*25}\n")
    
    cols, x_vals, obj = column_generation_exact_pricing(
        dist_matrix,
        max_iters=500,  
        tol=1e-7,
        forbid_repeat=True,
        max_consec_home=max_consec,
        max_consec_away=max_consec,
        include_return_home=include_return_home,
        gurobi_master_params={'OutputFlag': 0, 'Method': 1}
    )

    print("\nFractional master solution (non-zero columns):")
    for k, c in enumerate(cols):
        w = x_vals[k] if k < len(x_vals) else 0.0
        if w > 1e-9:
            rounds_repr = [(team_names[c.opp_round[r]], 'H' if c.home[r] else 'A') for r in range(len(c.opp_round))]
            print(f" col {k:3d}: team {team_names[c.team]:4s} weight={w:.6f} cost={c.cost:.1f}")
    
    print(f"\nOptimal LP objective for {instance_name} = {obj:.6f}")
    
    print_final_schedule(cols, x_vals, team_names)

    print(f"\n{'='*25} Finished {instance_name} {'='*25}")

# Run all CG on all instances in instances_data
for name, data in instances_data.items():
    run_ttp_instance(name, data["team_names"], data["dist_matrix"])
