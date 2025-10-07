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
def build_movement_solver(data, time_limit=300, mipgap=0.01):
    n = data["n"]
    d = data["distances"]
    R = data["rounds"]
    L, U = data["L"], data["U"]

    m = Model("TTP_Movement")
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mipgap

    #Decision variables
    o = m.addVars(n, n, R, vtype=GRB.BINARY, name="o")  # away
    h = m.addVars(n, R, vtype=GRB.BINARY, name="h")     # home
    zAA = m.addVars(n, n, n, R-1, vtype=GRB.BINARY, name="zAA")
    zHA = m.addVars(n, n, R-1, vtype=GRB.BINARY, name="zHA")
    zAH = m.addVars(n, n, R-1, vtype=GRB.BINARY, name="zAH")

    if L > 1:
        sA = m.addVars(n, R, vtype=GRB.BINARY, name="sA")
        sH = m.addVars(n, R, vtype=GRB.BINARY, name="sH")

    #Constraints
    # 1. Exactly one game per team per round
    for i in range(n):
        for k in range(R):
            m.addConstr(quicksum(o[i,j,k] + o[j,i,k] for j in range(n) if j != i) == 1)

    # 2. Each ordered pair occurs once (double round-robin)
    for i in range(n):
        for j in range(n):
            if i != j:
                m.addConstr(quicksum(o[i,j,k] for k in range(R)) == 1)

    # 3. No repeaters
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for k in range(R-1):
                m.addConstr(o[i,j,k] + o[j,i,k] + o[i,j,k+1] + o[j,i,k+1] <= 1)

    # 4. Home indicator definition
    for i in range(n):
        for k in range(R):
            m.addConstr(h[i,k] + quicksum(o[i,j,k] for j in range(n) if j != i) == 1)

    # 5. Maximum run-length constraints (sliding windows)
    if U > 0:
        for i in range(n):
            for k in range(R - U):
                # max away run
                m.addConstr(quicksum(1 - h[i,r] for r in range(k, k+U+1)) <= U)
                # max home run
                m.addConstr(quicksum(h[i,r] for r in range(k, k+U+1)) <= U)

    # 6. Minimum run-length constraints using sA, sH
    if L > 1:
        for i in range(n):
            for k in range(R):
                # start-of-run link
                prev_h = 0 if k==0 else h[i,k-1]
                m.addConstr(sA[i,k] >= 1 - h[i,k] - (1 - prev_h))
                m.addConstr(sA[i,k] <= 1 - h[i,k])
                m.addConstr(sA[i,k] <= prev_h)
                m.addConstr(sH[i,k] >= h[i,k] - prev_h)
                m.addConstr(sH[i,k] <= h[i,k])
                m.addConstr(sH[i,k] <= 1 - prev_h)

                # enforce minima
                end_idx = min(k + L - 1, R - 1)
                m.addConstr(quicksum(1 - h[i,r] for r in range(k, end_idx+1)) >= L * sA[i,k])
                m.addConstr(quicksum(h[i,r] for r in range(k, end_idx+1)) >= L * sH[i,k])

    # 7. Movement linearization
    for i in range(n):
        for k in range(R-1):
            for j in range(n):
                if j == i: continue
                for l in range(n):
                    if l == i: continue
                    m.addConstr(zAA[i,j,l,k] >= o[i,j,k] + o[i,l,k+1] - 1)
                m.addConstr(zHA[i,j,k] >= h[i,k] + o[i,j,k+1] - 1)
                m.addConstr(zAH[i,j,k] >= o[i,j,k] + h[i,k+1] - 1)

    # Objective fct is to minimize total travel distance
    obj = quicksum(
        quicksum(d[i][j]*o[i,j,0] for j in range(n) if j!=i) +
        quicksum(d[j][i]*o[i,j,R-1] for j in range(n) if j!=i) +
        quicksum(d[j][l]*zAA[i,j,l,k] for k in range(R-1) for j in range(n) if j!=i for l in range(n) if l!=i) +
        quicksum(d[i][j]*zHA[i,j,k] for k in range(R-1) for j in range(n) if j!=i) +
        quicksum(d[j][i]*zAH[i,j,k] for k in range(R-1) for j in range(n) if j!=i)
        for i in range(n)
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

    return m, o, h, solve_time

#Output generation
def print_movement_schedule(o, h, n, R, team_names, filename="schedule_matrix.csv",
                            objective_value=None, solve_time=None):
    """Print and save schedule matrix."""
    matrix = [[None for _ in range(R)] for _ in range(n)]
    home_counts = [0]*n
    away_counts = [0]*n

    for k in range(R):
        for i in range(n):
            for j in range(n):
                if i!=j and o[i,j,k].X > 0.5:
                    matrix[i][k] = f"{team_names[j]}(A)"
                    matrix[j][k] = f"{team_names[i]}(H)"
                    away_counts[i] += 1
                    home_counts[j] += 1

    print("\nSchedule Matrix (Team × Round):")
    header = ["Round " + str(k+1) for k in range(R)]
    print(f"{'Team':<10}", end="")
    for h_str in header:
        print(f"{h_str:<12}", end="")
    print()

    for i in range(n):
        print(f"{team_names[i]:<10}", end="")
        for k in range(R):
            print(f"{matrix[i][k]:<12}", end="")
        print()

    if objective_value is not None:
        print(f"\nTotal travel distance (objective value): {objective_value:.2f}")
    if solve_time is not None:
        print(f"Solve time: {solve_time:.2f} seconds\n")

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
def process_movement_folder(input_folder="datasets", output_folder="outputs_new"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    xml_files = [f for f in os.listdir(input_folder) if f.endswith(".xml")]
    if not xml_files:
        print(f"No XML files found in folder '{input_folder}'")
        return

    summary_file = os.path.join(output_folder, "summary_results.csv")
    with open(summary_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Instance","Teams","Rounds","Objective","Solve Time (s)","Status","Home Counts","Away Counts"])

    for xml_file in xml_files:
        xml_path = os.path.join(input_folder, xml_file)
        base_name = os.path.splitext(xml_file)[0]
        csv_filename = os.path.join(output_folder, f"{base_name}_schedule.csv")

        print(f"\n==============================")
        print(f" Processing: {xml_file}")
        print(f"==============================")

        data = parse_instance(xml_path)
        model, o, h, solve_time = build_movement_solver(data)

        obj_val = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None
        home_counts, away_counts = print_movement_schedule(
            o, h, data["n"], data["rounds"], data["team_names"],
            filename=csv_filename,
            objective_value=obj_val,
            solve_time=solve_time
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
        process_movement_folder("datasets","outputs_new")
    elif len(sys.argv) == 2:
        xml_file = sys.argv[1]
        data = parse_instance(xml_file)
        model, o, h, solve_time = build_movement_solver(data)
        base_name = os.path.splitext(os.path.basename(xml_file))[0]
        csv_filename = f"{base_name}_schedule.csv"
        obj_val = model.objVal if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) else None
        home_counts, away_counts = print_movement_schedule(
            o, h, data["n"], data["rounds"], data["team_names"],
            filename=csv_filename,
            objective_value=obj_val,
            solve_time=solve_time
        )
    else:
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        process_movement_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
