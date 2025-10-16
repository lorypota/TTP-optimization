import numpy as np
import xml.etree.ElementTree as ET
import math
import time
from gurobipy import Model, GRB, quicksum

def load_xml(path: str):
    tree = ET.parse(path)
    root = tree.getroot()

    id_to_name = {}
    team_elems = root.findall(".//Teams/team")
    for t in team_elems:
        tid = int(t.attrib["id"])
        id_to_name[tid] = t.attrib.get("name", f"T{tid}")
    n = len(id_to_name)

    D = np.zeros((n, n), dtype= float)
    for d in root.findall(".//Distances/distance"):
        i = int(d.attrib["team1"])
        j = int(d.attrib["team2"])
        D[i, j] = float(d.attrib["dist"])
    for i in range(n):
        D[i, i] = 0.0

    max_run_A = 1
    max_run_H = 1
    min_run_A = 1
    min_run_H = 1

    for ca in root.findall(".//CapacityConstraints/CA3"):
        if ca.attrib.get("mode1") == "H":
            max_run_H = ca.attrib.get("max")
            min_run_H = ca.attrib.get("min")
        if ca.attrib.get("mode1") == "A":
            max_run_A = ca.attrib.get("max")
            min_run_A = ca.attrib.get("min")

    min_sep = 0

    se = root.find(".//SeparationConstraints/SE1")
    min_sep = se.attrib.get("min")

    return n, id_to_name, D, [int(max_run_A), int(min_run_A), int(max_run_H), int(min_run_H)], int(min_sep)

def create_rounds(n, min_sep):
    teams =  list(range(n))
    fixed = n-1
    teams.remove(fixed)
    circle = teams
    rounds = []

    for i in range(n-1):
        team = circle[0]
        circle.remove(team)
        circle.append(team)
        temp = circle + [fixed]
        matches = []

        for j in range(int(n/2)):
            matches.append([temp[j], temp[n-j-1]])
        rounds.append(matches)

    mirror_rounds = []
    for i in range(len(rounds)):
        mirror_rounds.append(rounds[len(rounds) - 1 -i])

    if min_sep > n/2:
        min_sep = math.floor(n/2)

    while min_sep > 0:
        min_sep -= 1
        temp_round = mirror_rounds[0]
        mirror_rounds.remove(temp_round)
        mirror_rounds.append(temp_round)

    rounds = rounds + mirror_rounds
    return rounds

def assign_home_away(
        rounds,
        n_teams,
        max_run_A,
        min_run_A,
        max_run_H,
        min_run_H
):
    R = len(rounds)
    m = Model("assign_home_away")

    opponents = {(t,r): None for t in range(n_teams) for r in range(R)}
    for r, games in enumerate(rounds):
        for i, j in games:
            opponents[(i,r)] = j
            opponents[(j,r)] = i

    home = {(t,r): m.addVar(vtype=GRB.BINARY) for t in range(n_teams) for r in range(R)}
    
    for r, games in enumerate(rounds):
        for i, j in games:
            m.addConstr(home[(i,r)] + home[(j,r)] == 1)

    pair_rounds = {}
    for r, games in enumerate(rounds):
        for i,j in games:
            a, b = (i, j) if i < j else (j, i)
            pair_rounds.setdefault((a, b), []).append(r)

    for (a, b), r in pair_rounds.items():
        m.addConstr(home[(a, r[0])] + home[(a, r[1])] == 1)

    window = max_run_H + 1
    for t in range(n_teams):
        for i in range(R - window + 1):
            m.addConstr(quicksum(home[(t,r)] for r in range(i, i + window)) <= max_run_H)

    window = max_run_A + 1
    for t in range(n_teams):
        for i in range(R - window + 1):
            m.addConstr(quicksum(1 - home[(t,r)] for r in range(i, i + window)) <= max_run_A)

    consecutive_home = {(t, r): m.addVar(vtype=GRB.BINARY) for t in range(n_teams) for r in range(R - 1)}
    for t in range(n_teams):
        for r in range(R-1):
            m.addConstr(consecutive_home[(t, r)] <= home[(t, r)])
            m.addConstr(consecutive_home[(t, r)] <= home[(t, r + 1)])
            m.addConstr(consecutive_home[(t, r)] >= home[(t, r)] + home[(t, r + 1)] - 1)
    m.setObjective(quicksum(consecutive_home.values()), GRB.MAXIMIZE)

    m.optimize()

    plays_home_by_team = {t: [] for t in range(n_teams)}
    plays_home = {(t,r): home[(t,r)].X for t in range(n_teams) for r in range(R)}
    for r, games in enumerate(rounds):
        for i, j in games:
            plays_home_by_team[i].append((j, plays_home[(i,r)]))
            plays_home_by_team[j].append((i, plays_home[(j,r)]))

    return plays_home_by_team

def assign_team(
        D,
        matches,
        id_to_name
):
    n = len(matches)
    placeholders = list(range(n))
    real_teams = list(range(n))

    m = Model("assign_team")
    
    assignment = {(r, p): m.addVar(vtype= GRB.BINARY) for r in real_teams for p in placeholders}

    for p in placeholders:
        m.addConstr(quicksum(assignment[r, p] for r in real_teams) == 1)

    for r in real_teams:
        m.addConstr(quicksum(assignment[r, p] for p in placeholders) == 1)

    travel_times = [0 for i in range(n)]
    for p in placeholders:
        team_schedule  = matches[p]

        if team_schedule[0][1] == 0:
            travel_times[p] += quicksum(D[x][y] * assignment[(x, p)] * assignment[y, team_schedule[0][0]] for x in real_teams for y in real_teams)

        for round in range(1,len(matches[0])):
            if team_schedule[round-1][1] == 0:
                from_placeholder = team_schedule[round -1][0]
            else:
                from_placeholder = p
            
            if team_schedule[round][1] == 0:
                to_placeholder = team_schedule[round][0]
            else:
                to_placeholder = p

            travel_times[p] += quicksum(D[x][y] * assignment[(x, from_placeholder)] * assignment[(y, to_placeholder)] for x in real_teams for y in real_teams)

        if team_schedule[-1][1] == 0:
            travel_times[p] += quicksum(D[x][y] * assignment[(x, team_schedule[-1][0])] * assignment[(y, p)] for x in real_teams for y in real_teams)

        
    m.setObjective(quicksum(travel_times), GRB.MINIMIZE)
    m.optimize()
    
    ps = sorted({p for (_, p) in assignment.keys()})
    rs = sorted({r for (r, _) in assignment.keys()})
    assignment_map = {}
    for p in ps:
        for r in rs:
            if assignment[(r, p)].X > 0.5:
                assignment_map[p] = r
                break

    travel_times_value = [tt.getValue() for tt in travel_times]
    travel_times_value = { id_to_name[assignment_map[p]]: travel_times_value[p] for p in ps }

    return assignment_map, travel_times_value

def rename_matches(matches, assignment_map, id_to_name):
    named_matches = {}
    assigned_matches = {}

    for placeholder, schedule in matches.items():
        real = assignment_map[placeholder]
        assigned_matches[real] = []
        real_name = id_to_name[real]
        named_matches[real_name] = []

        for opponent, flag in schedule:
            real_opponent = assignment_map[opponent]
            named_opponent = id_to_name[real_opponent]

            if flag == 1:
                home_flag = "Home"
            else:
                home_flag = "Away"

            assigned_matches[real].append((real_opponent, home_flag))
            named_matches[real_name].append((named_opponent, home_flag))
    
    return named_matches, assigned_matches

def test(dest):
    n, id_to_name, D, [max_run_A, min_run_A, max_run_H, min_run_H], min_sep = load_xml(dest)

    rounds = create_rounds(n, min_sep)

    matches = assign_home_away(rounds, n, max_run_A, min_run_A, max_run_H, min_run_H)
    assignment_map, travel_times = assign_team(D, matches, id_to_name)
    named_matches, assigned_matches = rename_matches(matches, assignment_map, id_to_name)

    total_travel_time = 0
    for i in named_matches:
        total_travel_time += travel_times[i]
        print(i, named_matches[i], travel_times[i])
    print(total_travel_time)
    return(total_travel_time)

dest = "Path the XML File"

test(dest)