import sys
import os
import time
import xml.etree.ElementTree as ET
from gurobipy import Model, GRB, quicksum
import csv


# 1. DATA PARSING
def parse_instance(xml_file):
    """
    Parses a TTP instance from an XML file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    teams_data = root.findall(".//Teams/team")
    n = len(teams_data)
    team_names = {int(t.attrib["id"]): t.attrib["name"] for t in teams_data}

    distances = [[0] * n for _ in range(n)]
    for d in root.findall(".//Distances/distance"):
        i, j = int(d.attrib["team1"]), int(d.attrib["team2"])
        distances[i][j] = float(d.attrib["dist"])

    rounds = len(root.findall(".//Slots/slot"))

    # Only parse the 'max' consecutive games constraint (U).
    # The 'min' (L) constraint is not used in standard benchmarks.
    U = n - 1
    cap_constraint = root.find(".//CapacityConstraints/CA3")
    if cap_constraint is not None:
        U = int(cap_constraint.attrib["max"])

    print(f"Parsed instance: n={n}, rounds={rounds}, U={U}")
    return {
        "n": n,
        "distances": distances,
        "rounds": rounds,
        "U": U,
        "team_names": team_names
    }


# 2. MIP MODEL DEFINITION (Movement-Based Formulation)
def create_variables(model, n, R):
    """Creates and returns the decision variables for the movement model."""
    # o[i, j, k] = 1 if team i plays away at team j's venue in round k.
    o = model.addVars(n, n, R, vtype=GRB.BINARY, name="o")
    # h[i, k] = 1 if team i plays at home in round k.
    h = model.addVars(n, R, vtype=GRB.BINARY, name="h")
    # zAA[i, j, l, k] = 1 if team i is away at j in k and away at l in k+1.
    zAA = model.addVars(n, n, n, R - 1, vtype=GRB.BINARY, name="zAA")
    # zHA[i, j, k] = 1 if team i is home in k and away at j in k+1.
    zHA = model.addVars(n, n, R - 1, vtype=GRB.BINARY, name="zHA")
    # zAH[i, j, k] = 1 if team i is away at j in k and home in k+1.
    zAH = model.addVars(n, n, R - 1, vtype=GRB.BINARY, name="zAH")
    return o, h, zAA, zHA, zAH


def add_scheduling_constraints(model, o, h, n, R):
    """Adds fundamental tournament scheduling constraints."""
    # Each team plays exactly one match per round.
    model.addConstrs((
        quicksum(o[i, j, k] + o[j, i, k] for j in range(n) if j != i) == 1
        for i in range(n) for k in range(R)), name="one_match_per_round")

    # Each team plays every other team exactly once away (defines double round-robin).
    model.addConstrs((
        quicksum(o[i, j, k] for k in range(R)) == 1
        for i in range(n) for j in range(n) if i != j), name="double_round_robin")

    # Link home and away status: a team is either home or away in a round.
    model.addConstrs((
        h[i, k] + quicksum(o[i, j, k] for j in range(n) if j != i) == 1
        for i in range(n) for k in range(R)), name="home_away_link")


def add_pattern_constraints(model, o, h, n, R, U):
    """Adds constraints on home/away patterns."""
    # No team plays against the same opponent in two consecutive rounds.
    model.addConstrs((
        o[i, j, k] + o[j, i, k] + o[i, j, k + 1] + o[j, i, k + 1] <= 1
        for i in range(n) for j in range(i + 1, n) for k in range(R - 1)), name="no_repeater")

    # Maximum run-length constraints.
    for i in range(n):
        for k in range(R - U):
            # To prevent >U consecutive away games, at least 1 home game must be in the window.
            model.addConstr(quicksum(h[i, r] for r in range(k, k + U + 1)) >= 1, name=f"max_away_{i}_{k}")
            # To prevent >U consecutive home games, at least 1 away game must be in the window.
            model.addConstr(quicksum(1 - h[i, r] for r in range(k, k + U + 1)) >= 1, name=f"max_home_{i}_{k}")


def add_travel_constraints(model, o, h, zAA, zHA, zAH, n, R):
    """Adds constraints to link movement variables to the schedule."""
    # Away -> Away movement
    model.addConstrs((
        zAA[i, j, l, k] >= o[i, j, k] + o[i, l, k + 1] - 1
        for i in range(n) for k in range(R - 1)
        for j in range(n) if j != i for l in range(n) if l != i), name="link_zAA")

    # Home -> Away movement
    model.addConstrs((
        zHA[i, j, k] >= h[i, k] + o[i, j, k + 1] - 1
        for i in range(n) for k in range(R - 1)
        for j in range(n) if j != i), name="link_zHA")

    # Away -> Home movement
    model.addConstrs((
        zAH[i, j, k] >= o[i, j, k] + h[i, k + 1] - 1
        for i in range(n) for k in range(R - 1)
        for j in range(n) if j != i), name="link_zAH")


def set_objective_function(model, o, zAA, zHA, zAH, distances, n, R):
    """Defines the objective function to minimize total travel distance."""
    d = distances

    # Sum of travel for each team i
    total_travel = quicksum(
        # 1. Travel from home to the first away game
        quicksum(d[i][j] * o[i, j, 0] for j in range(n) if j != i) +

        # 2. Travel from the last away game back to home
        quicksum(d[j][i] * o[i, j, R - 1] for j in range(n) if j != i) +

        # 3. Intermediate travel legs, calculated by the movement variables
        quicksum(
            d[j][l] * zAA[i, j, l, k] for k in range(R - 1) for j in range(n) if j != i for l in range(n) if l != i) +
        quicksum(d[i][j] * zHA[i, j, k] for k in range(R - 1) for j in range(n) if j != i) +
        quicksum(d[j][i] * zAH[i, j, k] for k in range(R - 1) for j in range(n) if j != i)

        for i in range(n)
    )

    model.setObjective(total_travel, GRB.MINIMIZE)


def build_and_solve_movement(data, time_limit, mipgap):
    """Builds and solves the TTP optimization model using the movement formulation."""
    n, R, U = data["n"], data["rounds"], data["U"]

    model = Model("TTP_Movement")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mipgap

    # 1. Create variables
    o, h, zAA, zHA, zAH = create_variables(model, n, R)

    # 2. Add constraints
    add_scheduling_constraints(model, o, h, n, R)
    add_pattern_constraints(model, o, h, n, R, U)
    add_travel_constraints(model, o, h, zAA, zHA, zAH, n, R)

    # 3. Set objective function
    set_objective_function(model, o, zAA, zHA, zAH, data["distances"], n, R)

    # 4. Solve the model
    print("Model built. Starting optimization...")
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time

    # 5. Print summary
    if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT):
        print(f"Objective value: {model.objVal:.2f}")
        print(f"Solve time: {solve_time:.2f} seconds")
    else:
        print("No feasible solution found.")

    return model, o, solve_time


# 3. RESULTS AND OUTPUT
def format_and_save_schedule(o, data, output_filename, obj_val, solve_time):
    """Formats the schedule into a matrix and saves it to a CSV file."""
    n, R, team_names = data["n"], data["rounds"], data["team_names"]

    matrix = [["" for _ in range(R)] for _ in range(n)]
    # Use (H)ome and (A)way markers for clarity
    for k in range(R):
        for i in range(n):
            for j in range(n):
                if i != j and o[i, j, k].X > 0.5:
                    matrix[i][k] = f"{team_names[j]} (A)"
                    matrix[j][k] = f"{team_names[i]} (H)"

    # Console output
    header = [f"R{k + 1}" for k in range(R)]
    print(f"\n--- Schedule Matrix ---\n{'Team':<12}" + "".join(f"{h:<15}" for h in header))
    for i in range(n):
        row_str = "".join(f"{matrix[i][k]:<15}" for k in range(R))
        print(f"{team_names[i]:<12}{row_str}")
    print("-" * (12 + 15 * R))

    # CSV output
    with open(output_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Objective Value:", f"{obj_val:.2f}" if obj_val is not None else "N/A"])
        writer.writerow(["Solve Time (s):", f"{solve_time:.2f}" if solve_time is not None else "N/A"])
        writer.writerow([])
        writer.writerow(["Team"] + header)
        for i in range(n):
            writer.writerow([team_names[i]] + matrix[i])
    print(f"Schedule saved to {output_filename}")


def append_to_summary(summary_file, result_data):
    """Appends a result row to the summary CSV file."""
    file_exists = os.path.isfile(summary_file)
    with open(summary_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Instance", "Teams", "Rounds", "Objective", "Solve Time (s)", "Status"])
        writer.writerow([
            result_data["name"],
            result_data["n"],
            result_data["rounds"],
            f"{result_data['obj_val']:.2f}" if result_data['obj_val'] is not None else "N/A",
            f"{result_data['solve_time']:.2f}",
            result_data["status"]
        ])


# 4. MAIN EXECUTION LOGIC
def process_single_instance(xml_path, output_folder, time_limit, mipgap):
    """Parses, solves, and reports results for a single TTP instance."""
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    print(f"\n{'=' * 30}\n Processing: {base_name}\n{'=' * 30}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Solve model
    data = parse_instance(xml_path)
    model, o, solve_time = build_and_solve_movement(data, time_limit, mipgap)
    obj_val = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None

    # Format and save results
    schedule_filename = os.path.join(output_folder, f"{base_name}_schedule.csv")
    if obj_val is not None:
        format_and_save_schedule(o, data, schedule_filename, obj_val, solve_time)

    # Append to summary log
    summary_filename = os.path.join(output_folder, "summary_results.csv")
    result_data = {
        "name": base_name, "n": data["n"], "rounds": data["rounds"],
        "obj_val": obj_val, "solve_time": solve_time, "status": model.status
    }
    append_to_summary(summary_filename, result_data)


def main():
    """Main entry point for the TTP solver script."""
    # Default parameters
    input_arg = "datasets"  # Default folder
    output_folder = "outputs_movement_model"
    time_limit = 3600
    mipgap = 0.001

    if len(sys.argv) > 1:
        input_arg = sys.argv[1]

    if os.path.isdir(input_arg):
        xml_files = [os.path.join(input_arg, f) for f in os.listdir(input_arg) if f.endswith(".xml")]
        if not xml_files:
            print(f"No XML files found in directory: {input_arg}")
            return
        for xml_file in sorted(xml_files):
            process_single_instance(xml_file, output_folder, time_limit, mipgap)
    elif os.path.isfile(input_arg):
        process_single_instance(input_arg, output_folder, time_limit, mipgap)
    else:
        print(f"Error: Path not found '{input_arg}'")


if __name__ == "__main__":
    main()
