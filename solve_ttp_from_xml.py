# solve_ttp_from_xml.py
import os, sys, glob, xml.etree.ElementTree as ET
import gurobipy as gp
from gurobipy import GRB, quicksum

# ---------- Movement-based MILP (same as earlier, trimmed a bit) ----------
def build_ttp_movement_mip(d, L_min=1, U_max=3, add_symmetry=True, oddset_sizes=(3,), name="TTP"):
    n = len(d); assert n % 2 == 0, "n must be even"
    R = 2*(n-1)
    T = range(n); K = range(R); K_last = range(R-1)
    m = gp.Model(name)

    o = m.addVars(((i,j,k) for i in T for j in T if i!=j for k in K), vtype=GRB.BINARY, name="o")
    h = m.addVars(((i,k) for i in T for k in K), vtype=GRB.BINARY, name="h")
    zAA = m.addVars(((i,j,ell,k) for i in T for j in T if j!=i for ell in T if ell!=i and ell!=j for k in K_last),
                    vtype=GRB.BINARY, name="zAA")
    zHA = m.addVars(((i,j,k) for i in T for j in T if j!=i for k in K_last), vtype=GRB.BINARY, name="zHA")
    zAH = m.addVars(((i,j,k) for i in T for j in T if j!=i for k in K_last), vtype=GRB.BINARY, name="zAH")

    # Objective
    expr = quicksum(d[i][j]*o[i,j,0] for i in T for j in T if i!=j)
    expr += quicksum(d[j][ell]*zAA[i,j,ell,k] for i in T for j in T if j!=i for ell in T if ell!=i and ell!=j for k in K_last)
    expr += quicksum(d[i][j]*zHA[i,j,k] for i in T for j in T if j!=i for k in K_last)
    expr += quicksum(d[j][i]*zAH[i,j,k] for i in T for j in T if j!=i for k in K_last)
    expr += quicksum(d[j][i]*o[i,j,R-1] for i in T for j in T if i!=j)
    m.setObjective(expr, GRB.MINIMIZE)

    # One game per team per round
    for i in T:
        for k in K:
            m.addConstr(quicksum(o[i,j,k] for j in T if j!=i) + quicksum(o[j,i,k] for j in T if j!=i) == 1)

    # Each ordered pair once
    for i in T:
        for j in T:
            if i!=j:
                m.addConstr(quicksum(o[i,j,k] for k in K) == 1)

    # No repeaters
    for i in T:
        for j in T:
            if i<j:
                for k in K_last:
                    m.addConstr(o[i,j,k]+o[j,i,k]+o[i,j,k+1]+o[j,i,k+1] <= 1)

    # Home indicator
    for i in T:
        for k in K:
            m.addConstr(h[i,k] + quicksum(o[i,j,k] for j in T if j!=i) == 1)

    # Run caps via sliding windows of size U_max+1 (if provided)
    if U_max is not None and U_max >= 1:
        for i in T:
            for k in range(R - U_max):
                m.addConstr(quicksum(h[i,r] for r in range(k, k+U_max+1)) >= 1)       # cap away runs
                m.addConstr(quicksum(h[i,r] for r in range(k, k+U_max+1)) <= U_max)   # cap home runs

    # Movement linearizations
    for i in T:
        for k in K_last:
            for j in T:
                if j==i: continue
                for ell in T:
                    if ell==i or ell==j: continue
                    m.addConstr(zAA[i,j,ell,k] >= o[i,j,k] + o[i,ell,k+1] - 1)
    for i in T:
        for j in T:
            if j==i: continue
            for k in K_last:
                m.addConstr(zHA[i,j,k] >= h[i,k] + o[i,j,k+1] - 1)
                m.addConstr(zAH[i,j,k] >= o[i,j,k] + h[i,k+1] - 1)

    # Symmetry breaking (optional)
    if add_symmetry:
        m.addConstr(h[0,0] == 1)         # team 0 home in round 0
        m.addConstr(o[1,0,0] == 1)       # team 1 @ team 0 in round 0

    # Small odd-set cuts (|S|=3) per round (cheap & useful)
    for k in K:
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    # at most floor(3/2)=1 internal match among {a,b,c} in round k
                    m.addConstr((o[a,b,k]+o[b,a,k]) + (o[a,c,k]+o[c,a,k]) + (o[b,c,k]+o[c,b,k]) <= 1)

    return m, dict(o=o, h=h, zAA=zAA, zHA=zHA, zAH=zAH)

# ---------- XML loader ----------
def load_itc_ttp_xml(xml_path):
    """
    Parses an ITC/TTPlib-like XML and returns:
      d: n x n distance matrix (floats)
      team_names: list of names by index
      R_xml: number of slots in <Slots> (sanity check)
      U_max: derived from <CA3 ... mode1="H/A"> with intp=W and max=M  => U_max := min(M_H, M_A)
    Assumptions:
      - Double round robin (numberRoundRobin=2)
      - SE1 handles no-repeaters (we enforce no-repeaters anyway in the model)
      - CA3 with mode1="H" and "A" exist and share the same window length
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Teams (id -> index, by ascending id)
    teams = []
    for t in root.findall(".//Resources/Teams/team"):
        teams.append((int(t.attrib["id"]), t.attrib.get("name", f"Team{t.attrib['id']}")))
    teams.sort(key=lambda x: x[0])
    id2idx = {tid:i for i,(tid,_) in enumerate(teams)}
    team_names = [nm for _,nm in teams]
    n = len(team_names)

    # Distances
    d = [[0.0]*n for _ in range(n)]
    for e in root.findall(".//Data/Distances/distance"):
        i = id2idx[int(e.attrib["team1"])]
        j = id2idx[int(e.attrib["team2"])]
        d[i][j] = float(e.attrib["dist"])

    # Slots / rounds
    slots = root.findall(".//Resources/Slots/slot")
    R_xml = len(slots)

    # numberRoundRobin (should be 2)
    nrr = root.findtext(".//Structure/Format/numberRoundRobin")
    if nrr is not None and int(nrr) != 2:
        print(f"[WARN] numberRoundRobin={nrr} (model expects DRR); continuing.")

    # Run caps from CA3 (window=intp, max=M)
    U_home = U_away = None
    for ca3 in root.findall(".//Constraints/CapacityConstraints/CA3"):
        mode = ca3.attrib.get("mode1","").upper()
        W = int(ca3.attrib.get("intp","0"))  # window length
        M = int(ca3.attrib.get("max","0"))   # max home/away in any window W
        # Interpreting classical CA3: "in any window of W slots, at most M home (or away) games"
        # That implies max run length <= M.
        if mode == "H":
            U_home = M
        elif mode == "A":
            U_away = M
    U_max = None
    if U_home is not None and U_away is not None:
        U_max = min(U_home, U_away)

    return d, team_names, R_xml, U_max

# ---------- Solve helpers ----------
def solve_instance(xml_path, time_limit=60, mip_focus=1, quiet=False):
    d, team_names, R_xml, U_cap = load_itc_ttp_xml(xml_path)
    n = len(team_names); R = 2*(n-1)
    if R_xml and R_xml != R:
        print(f"[WARN] XML has {R_xml} slots but DRR implies {R}. Using DRR={R} in the model.")

    m, V = build_ttp_movement_mip(d, L_min=1, U_max=U_cap if U_cap is not None else 3,
                                  add_symmetry=True, oddset_sizes=(3,),
                                  name=os.path.basename(xml_path))

    if time_limit: m.Params.TimeLimit = time_limit
    if mip_focus is not None: m.Params.MIPFocus = mip_focus
    m.Params.OutputFlag = 0 if quiet else 1

    m.optimize()

    if m.SolCount == 0:
        print(f"[{os.path.basename(xml_path)}] No feasible solution found.")
        return

    print(f"\n[{os.path.basename(xml_path)}] Objective (total distance): {m.ObjVal:.2f}")
    o = V["o"]; R = 2*(n-1)
    for k in range(R):
        pairs, used = [], set()
        for i in range(n):
            if i in used: continue
            found = False
            for j in range(n):
                if i==j or j in used: continue
                if o[i,j,k].X > 0.5:
                    pairs.append(f"{team_names[i]} @ {team_names[j]}")
                    used.add(i); used.add(j); found = True; break
                if o[j,i,k].X > 0.5:
                    pairs.append(f"{team_names[j]} @ {team_names[i]}")
                    used.add(i); used.add(j); found = True; break
            if not found:
                pass
        print(f"Round {k:2d}: " + ", ".join(pairs))

def main():
    if len(sys.argv) < 2:
        print("Usage: python solve_ttp_from_xml.py <instance.xml | folder>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        files = sorted(glob.glob(os.path.join(target, "*.xml")))
        if not files:
            print("No .xml files found in folder.")
            sys.exit(1)
        for f in files:
            solve_instance(f, time_limit=60)
    else:
        solve_instance(target, time_limit=60)

if __name__ == "__main__":
    main()
