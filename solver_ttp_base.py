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

    Args:
        xml_file (str): Path to the XML instance file.

    Returns:
        dict: A dictionary containing instance data (n, distances, rounds, etc.).
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

    # Default to no constraints if not specified
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


# 2. MIP MODEL DEFINITION
def create_variables(model, n, rounds):
    """Creates and returns the decision variables for the TTP model."""
    # x[i, j, k] = 1 if team i plays away at team j's venue in round k.
    x = model.addVars(n, n, rounds, vtype=GRB.BINARY, name="x")
    # y[t, i, j, k] = 1 if team t travels from venue i to venue j between rounds k and k+1.
    y = model.addVars(n, n, n, rounds - 1, vtype=GRB.BINARY, name="y")
    # z[t, i, k] = 1 if team t is at venue i for its match in round k.
    z = model.addVars(n, n, rounds, vtype=GRB.BINARY, name="z")
    return x, y, z


def add_scheduling_constraints(model, x, n, rounds):
    """Adds fundamental tournament scheduling constraints to the model."""
    # Each team plays exactly one match per round.
    model.addConstrs((
        quicksum(x[i, j, k] + x[j, i, k] for j in range(n)) == 1
        for i in range(n) for k in range(rounds)), name="one_match_per_round")

    # Over the season, each team plays every other team exactly once away.
    model.addConstrs((
        quicksum(x[i, j, k] for k in range(rounds)) == 1
        for i in range(n) for j in range(n) if i != j), name="double_round_robin")


def add_pattern_constraints(model, x, n, rounds, U):
    """Adds constraints on home/away patterns (consecutive games, no-repeater)."""
    # No more than U consecutive away games.
    model.addConstrs((
        quicksum(x[i, j, k + l] for l in range(U + 1) for j in range(n)) <= U
        for i in range(n) for k in range(rounds - U)), name="max_away")

    # No more than U consecutive home games.
    model.addConstrs((
        quicksum(x[j, i, k + l] for j in range(n) for l in range(U + 1)) <= U
        for i in range(n) for k in range(rounds - U)), name="max_home")

    # No team plays against the same opponent in two consecutive rounds.
    model.addConstrs((
        x[i, j, k] + x[j, i, k] + x[i, j, k + 1] + x[j, i, k + 1] <= 1
        for i in range(n) for j in range(i + 1, n) for k in range(rounds - 1)), name="no_repeater")


def add_travel_constraints(model, x, y, z, n, rounds):
    """Adds constraints to define team locations (z) and travel (y)."""
    # Define location z based on schedule x.
    # If team t plays away at venue i, its location is i.
    model.addConstrs((z[t, i, k] == x[t, i, k]
                      for t in range(n) for i in range(n) if t != i for k in range(rounds)), name="z_def_away")

    # If team t plays at home, its location is t.
    model.addConstrs((z[t, t, k] == quicksum(x[j, t, k] for j in range(n))
                      for t in range(n) for k in range(rounds)), name="z_def_home")

    # Link travel variable y to locations z.
    # y[t,i,j,k] is 1 if team t is at venue i in round k AND at venue j in round k+1.
    model.addConstrs((
        y[t, i, j, k] >= z[t, i, k] + z[t, j, k + 1] - 1
        for t in range(n) for i in range(n) for j in range(n) if i != j
        for k in range(rounds - 1)), name="travel_link")


def set_objective_function(model, x, y, distances, n, rounds):
    """Defines and sets the objective function to minimize total travel distance."""
    d = distances
    # Travel from home to the first away game.
    first_round_travel = quicksum(d[i][j] * x[i, j, 0] for i in range(n) for j in range(n))

    # Travel between venues in intermediate rounds.
    intermediate_travel = quicksum(d[i][j] * y[t, i, j, k]
                                   for t in range(n) for i in range(n) for j in range(n)
                                   for k in range(rounds - 1))

    # Travel from the last away game back to home.
    last_round_travel = quicksum(d[j][i] * x[i, j, rounds - 1] for i in range(n) for j in range(n))

    model.setObjective(first_round_travel + intermediate_travel + last_round_travel, GRB.MINIMIZE)


def build_and_solve(data, time_limit, mipgap):
    """Builds and solves the TTP optimization model."""
    n, rounds, U = data["n"], data["rounds"], data["U"]

    model = Model("TTP")
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mipgap

    # 1. Create variables
    x, y, z = create_variables(model, n, rounds)

    # 2. Add constraints
    add_scheduling_constraints(model, x, n, rounds)
    add_pattern_constraints(model, x, n, rounds, U)
    add_travel_constraints(model, x, y, z, n, rounds)

    # 3. Set objective function
    set_objective_function(model, x, y, data["distances"], n, rounds)

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

    return model, x, solve_time


# 3. RESULTS AND OUTPUT
def format_and_save_schedule(x, data, output_filename, obj_val, solve_time):
    """Formats the schedule into a matrix and saves it to a CSV file."""
    n, rounds, team_names = data["n"], data["rounds"], data["team_names"]

    matrix = [["" for _ in range(rounds)] for _ in range(n)]
    for k in range(rounds):
        for i in range(n):
            for j in range(n):
                if i != j and x[i, j, k].X > 0.5:
                    matrix[i][k] = f"@{team_names[j]}"
                    matrix[j][k] = f"vs {team_names[i]}"

    # Console output
    header = [f"R{k + 1}" for k in range(rounds)]
    print(f"\n--- Schedule Matrix ---\n{'Team':<12}" + "".join(f"{h:<12}" for h in header))
    for i in range(n):
        row_str = "".join(f"{matrix[i][k]:<12}" for k in range(rounds))
        print(f"{team_names[i]:<12}{row_str}")
    print("-" * (12 * (rounds + 1)))

    # CSV output
    with open(output_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Objective Value:", f"{obj_val:.2f}" if obj_val else "N/A"])
        writer.writerow(["Solve Time (s):", f"{solve_time:.2f}" if solve_time else "N/A"])
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
    model, x, solve_time = build_and_solve(data, time_limit, mipgap)
    obj_val = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None

    # Format and save results
    schedule_filename = os.path.join(output_folder, f"{base_name}_schedule.csv")
    format_and_save_schedule(x, data, schedule_filename, obj_val, solve_time)

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
    output_folder = "outputs"
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
