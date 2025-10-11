import os, sys, glob, xml.etree.ElementTree as ET
import gurobipy as gp
from gurobipy import GRB, quicksum
import requests
from datetime import datetime

# ---------- Base Integer Program ----------
def build_ttp_base_ip(d, L_min=1, U_max=3, name="TTP_Base"):
    """
    Builds the TTP model based on the standard literature formulation (e.g., Ribeiro et al., 2012).
    Uses x_ijk for game assignments and y_tijk for travel between rounds.
    """
    n = len(d); assert n % 2 == 0, "n must be even"
    R = 2*(n-1)
    T = range(n); K = range(R); K_last = range(R-1)
    m = gp.Model(name)

    # --- Decision Variables ---
    # x[i,j,k] = 1 if team i plays away against team j in round k
    x = m.addVars(((i,j,k) for i in T for j in T if i!=j for k in K), vtype=GRB.BINARY, name="x")

    # y[t,i,j,k] = 1 if team t travels from venue i to venue j between rounds k and k+1
    y = m.addVars(((t,i,j,k) for t in T for i in T for j in T for k in K_last), vtype=GRB.BINARY, name="y")

    # Helper variable: z[t,i,k] = 1 if team t is at venue i in round k
    z = m.addVars(((t,i,k) for t in T for i in T for k in K), vtype=GRB.BINARY, name="z")

    # Helper variable: h[i,k] = 1 if team i plays home in round k
    h = m.addVars(((i,k) for i in T for k in K), vtype=GRB.BINARY, name="h")

    # --- Objective Function ---
    # 1. Travel from home to first away game
    obj = quicksum(d[i][j] * x[i,j,0] for i in T for j in T if i!=j)
    # 2. Travel between rounds
    obj += quicksum(d[i][j] * y[t,i,j,k] for t in T for i in T for j in T for k in K_last)
    # 3. Travel from last away game back to home
    obj += quicksum(d[j][i] * x[i,j,R-1] for i in T for j in T if i!=j)
    m.setObjective(obj, GRB.MINIMIZE)

    # --- Constraints ---
    # One game per team per round
    for i in T:
        for k in K:
            # Team i is either away (sum(x[i,j,k])) or home (sum(x[j,i,k]))
            m.addConstr(quicksum(x[i,j,k] for j in T if j!=i) + quicksum(x[j,i,k] for j in T if j!=i) == 1)

    # Each ordered pair occurs once (Double Round Robin)
    for i in T:
        for j in T:
            if i!=j:
                m.addConstr(quicksum(x[i,j,k] for k in K) == 1)

    # No repeaters
    for i in T:
        for j in T:
            if i<j:
                for k in K_last:
                    m.addConstr(x[i,j,k]+x[j,i,k]+x[i,j,k+1]+x[j,i,k+1] <= 1)

    # Link home indicator variable 'h'
    for i in T:
        for k in K:
            # Team i is home if some other team j plays away against it
            m.addConstr(h[i,k] == quicksum(x[j,i,k] for j in T if j!=i))

    # Run caps via sliding windows (using the 'h' variable)
    if U_max is not None and U_max >= 1:
        for i in T:
            for k in range(R - U_max):
                # Max away runs: at least 1 home game in any window of size U_max+1
                m.addConstr(quicksum(h[i,r] for r in range(k, k+U_max+1)) >= 1)
                # Max home runs: at most U_max home games in any window of size U_max+1
                m.addConstr(quicksum(1 - h[i,r] for r in range(k, k+U_max+1)) >= 1)


    # --- Travel Logic Constraints ---
    # Define location z[t,i,k]
    for t in T:
        for k in K:
            # Team t is at its own venue 't' if it's playing home
            m.addConstr(z[t,t,k] == h[t,k])
            # Team t is at venue 'i' if it's playing away against team i
            for i in T:
                if i != t:
                    m.addConstr(z[t,i,k] == x[t,i,k])

    # Link location 'z' to travel 'y'
    for t in T:
        for i in T:
            for j in T:
                for k in K_last:
                    # If team t is at venue i in round k AND at venue j in round k+1, then y[t,i,j,k] must be 1
                    m.addConstr(y[t,i,j,k] >= z[t,i,k] + z[t,j,k+1] - 1)

    # Small odd-set cuts (|S|=3) per round (cheap & useful)
    for k in K:
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # at most floor(3/2)=1 internal match among {a,b,c} in round k
                    m.addConstr((x[a,b,k]+x[b,a,k]) + (x[a,c,k]+x[c,a,k]) + (x[b,c,k]+x[c,b,k]) <= 1)
    
    # Return model and the main game variable for solution printing
    return m, dict(x=x)

# ---------- XML loader ----------
def load_itc_ttp_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    teams = []
    for t in root.findall(".//Resources/Teams/team"):
        teams.append((int(t.attrib["id"]), t.attrib.get("name", f"Team{t.attrib['id']}")))
    teams.sort(key=lambda x: x[0])
    id2idx = {tid:i for i,(tid,_) in enumerate(teams)}
    team_names = [nm for _,nm in teams]
    n = len(team_names)
    d = [[0.0]*n for _ in range(n)]
    for e in root.findall(".//Data/Distances/distance"):
        i = id2idx[int(e.attrib["team1"])]
        j = id2idx[int(e.attrib["team2"])]
        d[i][j] = float(e.attrib["dist"])
    slots = root.findall(".//Resources/Slots/slot")
    R_xml = len(slots)
    nrr = root.findtext(".//Structure/Format/numberRoundRobin")
    if nrr is not None and int(nrr) != 2:
        print(f"[WARN] numberRoundRobin={nrr} (model expects DRR); continuing.")
    U_home = U_away = None
    for ca3 in root.findall(".//Constraints/CapacityConstraints/CA3"):
        mode = ca3.attrib.get("mode1","").upper()
        M = int(ca3.attrib.get("max","0"))
        if mode == "H": U_home = M
        elif mode == "A": U_away = M
    U_max = None
    if U_home is not None and U_away is not None:
        U_max = min(U_home, U_away)
    return d, team_names, R_xml, U_max

# ---------- Fetch instance directly from RobinX repository ----------
def fetch_instance_xml(name, cache_dir="instances"):
    base_url = "https://robinxval.ugent.be/RobinX/Repository/TravelOptimization/Instances/"
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, name)
    if os.path.exists(local_path):
        print(f"[CACHE] Using local copy: {local_path}")
        return local_path
    url = base_url + name
    print(f"[FETCH] Downloading {url} ...")
    r = requests.get(url)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch {url} (HTTP {r.status_code})")
    with open(local_path, "wb") as f:
        f.write(r.content)
    print(f"[FETCH] Saved to {local_path}")
    return local_path

# ---------- Per-instance result logger ----------
def save_txt_result(xml_name, n, obj, runtime, gap, status, rounds_output, output_dir="results_exact_sol_base"):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(xml_name)[0]
    txt_path = os.path.join(output_dir, f"{base}_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== Traveling Tournament Problem Results (Base Model) ===\n")
        f.write(f"Instance: {xml_name}\n")
        f.write(f"Teams: {n}\n")
        f.write(f"Date: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        if obj is not None: f.write(f"Objective (total distance): {obj:.2f}\n")
        if runtime is not None: f.write(f"Runtime (sec): {runtime:.2f}\n")
        if gap is not None: f.write(f"MIP Gap: {gap:.4f}\n")
        f.write(f"Status: {status}\n\n")
        if rounds_output:
            f.write("=== Schedule ===\n")
            for line in rounds_output:
                f.write(line + "\n")
    print(f"[LOG] Results saved to {txt_path}")

# ---------- Solve helpers ----------
def solve_instance(xml_path, time_limit=GRB.INFINITY, mip_focus=1, quiet=False):
    d, team_names, R_xml, U_cap = load_itc_ttp_xml(xml_path)
    n = len(team_names); R = 2*(n-1)
    if R_xml and R_xml != R:
        print(f"[WARN] XML has {R_xml} slots but DRR implies {R}. Using DRR={R} in the model.")

    m, V = build_ttp_base_ip(d, L_min=1, U_max=U_cap if U_cap is not None else 3,
                             name=os.path.basename(xml_path))

    if time_limit: m.Params.TimeLimit = time_limit
    if mip_focus is not None: m.Params.MIPFocus = mip_focus
    m.Params.OutputFlag = 0 if quiet else 1

    m.optimize()

    if m.SolCount == 0:
        print(f"[{os.path.basename(xml_path)}] No feasible solution found.")
        return

    print(f"\n[{os.path.basename(xml_path)}] Objective (total distance): {m.ObjVal:.2f}")

    x = V["x"]
    
    rounds_output = []
    for k in range(R):
        pairs, used = [], set()
        for i in range(n):
            if i in used: continue
            for j in range(n):
                if i==j or j in used: continue
                # Check away i @ j
                if x[i,j,k].X > 0.5:
                    pairs.append(f"{team_names[i]} @ {team_names[j]}")
                    used.add(i); used.add(j); break
                # Check away j @ i
                if x[j,i,k].X > 0.5:
                    pairs.append(f"{team_names[j]} @ {team_names[i]}")
                    used.add(i); used.add(j); break
        line = f"Round {k:2d}: " + ", ".join(pairs)
        print(line)
        rounds_output.append(line)
    
    runtime = m.Runtime
    gap = m.MIPGap if m.MIPGap is not None else 0
    status = m.Status
    
    save_txt_result(os.path.basename(xml_path), n, m.ObjVal, runtime, gap, status, rounds_output)

# ---------- Main entry point ----------
def main():
    if len(sys.argv) < 2:
        print("Usage: python solve_ttp_base_ip.py <instance.xml | folder>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.xml")))
        if not files:
            print("No .xml files found in folder.")
            sys.exit(1)
        for f in files:
            solve_instance(f)
    else:
        if not os.path.exists(target) and not target.startswith("/"):
            target = fetch_instance_xml(target)
        solve_instance(target)

if __name__ == "__main__":
    main()