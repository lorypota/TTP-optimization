import sys
import os
import time
import xml.etree.ElementTree as ET
from gurobipy import Model, GRB, quicksum
import csv

#Instance parsing from .xml files
def parse_instance(xml_file):
    """Parse TTP XML data file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    teams = root.findall(".//Teams/team")
    n = len(teams)
    team_names = {int(t.attrib["id"]): t.attrib["name"] for t in teams}

    distances = [[0] * n for _ in range(n)]
    for d in root.findall(".//Distances/distance"):
        i = int(d.attrib["team1"])
        j = int(d.attrib["team2"])
        distances[i][j] = float(d.attrib["dist"])

    slots = root.findall(".//Slots/slot")
    rounds = len(slots)

    cap = root.find(".//CapacityConstraints/CA3")
    L = int(cap.attrib["min"]) if cap is not None else 0
    U = int(cap.attrib["max"]) if cap is not None else n - 1

    print(f"Parsed instance: n={n}, rounds={rounds}, L={L}, U={U}")
    return {"n": n, "distances": distances, "rounds": rounds, "L": L, "U": U, "team_names": team_names}


#model
def build_and_solve(data, time_limit=300, mipgap=0.01):
    n = data["n"]
    d = data["distances"]
    rounds = data["rounds"]
    L, U = data["L"], data["U"]

    m = Model("TTP")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mipgap

    x = m.addVars(n, n, rounds, vtype=GRB.BINARY, name="x")  # i plays away @ j in round k
    y = m.addVars(n, n, n, rounds - 1, vtype=GRB.BINARY, name="y")  # t travels i->j between k and k+1

    # Constraints
    # 1. No team can play against itself
    for i in range(n):
        for k in range(rounds):
            m.addConstr(x[i, i, k] == 0)

    # 2. Each team plays exactly one match per round
    for i in range(n):
        for k in range(rounds):
            m.addConstr(quicksum(x[i, j, k] + x[j, i, k] for j in range(n)) == 1)

    # 3. Each pair of teams plays exactly twice (once home, once away)
    for i in range(n):
        for j in range(n):
            if i != j:
                m.addConstr(quicksum(x[i, j, k] for k in range(rounds)) == 1)

    # 4. L/U consecutive away games constraint
    #    Ensures no team has more than U consecutive away games and at least L consecutive away games
    for i in range(n):
        for k in range(rounds - U):
            m.addConstr(
                quicksum(x[i, j, k + l] for l in range(U + 1) for j in range(n) if i != j) <= U,
                name=f"maxAway_{i}_{k}"
            )

        if L > 0:
            for k in range(rounds - L + 1):
                m.addConstr(
                    quicksum(x[i, j, k + l] for l in range(L) for j in range(n) if i != j) >= L,
                    name=f"minAway_{i}_{k}"
                )

    # 5. Non-repeater constraint
    #    Prevents two teams from playing against each other in consecutive rounds
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(rounds - 1):
                m.addConstr(
                    x[i, j, k] + x[j, i, k] + x[i, j, k + 1] + x[j, i, k + 1] <= 1,
                    name=f"nonRepeater_{i}_{j}_{k}"
                )

    # Objective fct is to minimize total travel distance
    obj = (
        quicksum(d[i][j] * x[i, j, 0] for i in range(n) for j in range(n)) +
        quicksum(d[i][j] * y[t, i, j, k] for t in range(n)
                 for i in range(n) for j in range(n)
                 for k in range(rounds - 1) if i != j and t != i and t != j) +
        quicksum(d[j][i] * x[i, j, rounds - 1] for i in range(n) for j in range(n))
    )
    m.setObjective(obj, GRB.MINIMIZE)

    print("Model built, optimizing...")
    start_time = time.time()
    m.optimize()
    solve_time = time.time() - start_time

    if m.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        print(f"Objective value: {m.objVal:.2f}")
        print(f"Solve time: {solve_time:.2f} seconds")
    else:
        print("No optimal solution found.")

    return m, x, solve_time


#Output generation
def print_schedule_matrix(x, n, rounds, team_names, filename="schedule_matrix.csv",
                          objective_value=None, solve_time=None):
    """Print and save a matrix: rows = teams, columns = rounds, with metrics."""

    matrix = [[None for _ in range(rounds)] for _ in range(n)]

    home_counts = [0] * n
    away_counts = [0] * n

    for k in range(rounds):
        for i in range(n):
            for j in range(n):
                if i != j and x[i, j, k].X > 0.5:
                    matrix[i][k] = f"{team_names[j]}(A)"
                    matrix[j][k] = f"{team_names[i]}(H)"
                    away_counts[i] += 1
                    home_counts[j] += 1

    print("\nSchedule Matrix (Team × Round):")
    header = ["Round " + str(k + 1) for k in range(rounds)]
    print(f"{'Team':<10}", end="")
    for h in header:
        print(f"{h:<12}", end="")
    print()

    for i in range(n):
        print(f"{team_names[i]:<10}", end="")
        for k in range(rounds):
            print(f"{matrix[i][k]:<12}", end="")
        print()

    if objective_value is not None:
        print(f"\nTotal travel distance (objective value): {objective_value:.2f}")
    if solve_time is not None:
        print(f"Solve time: {solve_time:.2f} seconds\n")

    # Save CSV
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        if objective_value is not None:
            writer.writerow(["Total travel distance (objective value):", f"{objective_value:.2f}"])
        if solve_time is not None:
            writer.writerow(["Solve time (seconds):", f"{solve_time:.2f}"])
        writer.writerow([])
        writer.writerow(["Team"] + header)
        for i in range(n):
            writer.writerow([team_names[i]] + matrix[i])

    return home_counts, away_counts


#datasets folder processing
def process_folder(input_folder="datasets", output_folder="outputs"):
    """Process all XML datasets in the input folder and create a summary CSV."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    xml_files = [f for f in os.listdir(input_folder) if f.endswith(".xml")]
    if not xml_files:
        print(f"No XML files found in folder '{input_folder}'")
        return

    summary_file = os.path.join(output_folder, "summary_results.csv")
    with open(summary_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Instance", "Teams", "Rounds", "Objective", "Solve Time (s)", "Status", "Home Counts", "Away Counts"])

    for xml_file in xml_files:
        xml_path = os.path.join(input_folder, xml_file)
        base_name = os.path.splitext(xml_file)[0]
        csv_filename = os.path.join(output_folder, f"{base_name}_schedule.csv")

        print(f"\n==============================")
        print(f" Processing: {xml_file}")
        print(f"==============================")

        data = parse_instance(xml_path)
        model, x, solve_time = build_and_solve(data)

        obj_val = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None
        home_counts, away_counts = print_schedule_matrix(
            x,
            data["n"],
            data["rounds"],
            data["team_names"],
            filename=csv_filename,
            objective_value=obj_val,
            solve_time=solve_time,
        )

        with open(summary_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                base_name,
                data["n"],
                data["rounds"],
                f"{obj_val:.2f}" if obj_val else "N/A",
                f"{solve_time:.2f}" if solve_time else "N/A",
                model.status,
                ",".join(map(str, home_counts)),
                ",".join(map(str, away_counts))
            ])

    print(f"\nSummary of all runs saved to '{summary_file}'")


def main():
    if len(sys.argv) == 1:
        process_folder("datasets", "outputs")

    elif len(sys.argv) == 2:
        xml_file = sys.argv[1]
        data = parse_instance(xml_file)
        model, x, solve_time = build_and_solve(data)

        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        csv_filename = f"{base_name}_schedule.csv"
        obj_val = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None
        home_counts, away_counts = print_schedule_matrix(
            x,
            data["n"],
            data["rounds"],
            data["team_names"],
            filename=csv_filename,
            objective_value=obj_val,
            solve_time=solve_time,
        )

        summary_file = "summary_results.csv"
        if not os.path.exists(summary_file):
            with open(summary_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Instance", "Teams", "Rounds", "Objective", "Solve Time (s)", "Status", "Home Counts", "Away Counts"])
        with open(summary_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                base_name,
                data["n"],
                data["rounds"],
                f"{obj_val:.2f}" if obj_val else "N/A",
                f"{solve_time:.2f}" if solve_time else "N/A",
                model.status,
                ",".join(map(str, home_counts)),
                ",".join(map(str, away_counts))
            ])
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
