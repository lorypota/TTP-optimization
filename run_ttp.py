# TTP True Branch-and-Price with C++ Pricer and Final Optimizations

import numpy as np
from copy import deepcopy
import time, math, heapq
from itertools import product, combinations
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import ttp_pricer
import ttp_instances as instances

# ----------------------- USER INPUT -----------------------
predefined_instance = "CIRC6"

# Load instance data based on the selection
if predefined_instance in instances.instances_data:
    team_names = instances.instances_data[predefined_instance]["team_names"]
    dist_matrix = instances.instances_data[predefined_instance]["dist_matrix"]
else:
    raise ValueError(f"Unknown instance name: {predefined_instance}")
# ----------------------------------------------------------------

n = len(team_names)
m_slots = 2 * (n - 1)
pair_list = list(combinations(range(n), 2))
pair_to_idx = {tuple(sorted(p)): i for i, p in enumerate(pair_list)}
P = len(pair_list)
print(f"Setup: n={n}, slots={m_slots}, pairs={P}")
print(f"Solving for {predefined_instance} instance.")


def build_master_model_gurobi(pool, forbid_triplets, forbid_signatures, require_list, phase=2):
    included_cols = [c for c in pool if not ((c['team'], tuple(c['tour']['opp_seq']), tuple(c['tour']['home_flags'])) in forbid_signatures or any((c['team'], o, s) in forbid_triplets for s, o in enumerate(c['tour']['opp_seq'])))]
    
    model = gp.Model("TTP_Master"); model.Params.OutputFlag = 0
    model.setParam('Method', 2)
    tour_costs = [float(c['tour']['cost']) if phase == 2 else 0.0 for c in included_cols]
    artificial_cost = 1.0 if phase == 1 else 1e6

    xvars = model.addVars(len(included_cols), lb=0.0, obj=tour_costs, name="x")
    constrs = []
    
    for t in range(n):
        constrs.append(model.addConstr(gp.quicksum(xvars[k] for k, c in enumerate(included_cols) if c['team'] == t) == 1.0))

    num_pair_slot_constrs = P * m_slots
    num_home_cons_constrs = P * m_slots

    for constr_idx in range(num_pair_slot_constrs + num_home_cons_constrs):
        if phase == 1:
            s_plus = model.addVar(lb=0.0, obj=artificial_cost)
            s_minus = model.addVar(lb=0.0, obj=artificial_cost)
            model.update() # To get the variables into the model before getting their index
            constrs.append(model.addConstr(gp.LinExpr() - s_plus + s_minus == 0.0))
        else:
            constrs.append(model.addConstr(gp.LinExpr() == 0.0))
    
    for k, col in enumerate(included_cols):
        t, tour = col['team'], col['tour']
        for s, opp in enumerate(tour['opp_seq']):
            i, j = tuple(sorted((t, opp)))
            pair_idx = pair_to_idx[(i,j)]
            
            pair_constr = constrs[n + pair_idx*m_slots + s]
            model.chgCoeff(pair_constr, xvars[k], 1.0 if t==i else -1.0)
            
            home_constr = constrs[n + num_pair_slot_constrs + pair_idx*m_slots + s]
            if t == i:
                model.chgCoeff(home_constr, xvars[k], (1.0 if tour['home_flags'][s] else 0.0) - 1.0)
            else: # t == j
                model.chgCoeff(home_constr, xvars[k], (1.0 if tour['home_flags'][s] else 0.0))

    for i,j,s in require_list:
        constrs.append(model.addConstr(gp.quicksum(xvars[k] for k, c in enumerate(included_cols) if c['team'] == i and c['tour']['delta'][j, s] == 1) == 1.0))
        
    model.ModelSense = GRB.MINIMIZE; model.update()
    return model, list(xvars.values()), included_cols, constrs


def column_generation_gurobi(node_pool, forbid_triplets=None, forbid_signatures=None, require_list=None, cg_max_iters=1500):
    pool = deepcopy(node_pool)
    
    for phase in [1, 2]:
        for _ in range(cg_max_iters):
            model, xvars, included_cols, constrs = build_master_model_gurobi(pool, forbid_triplets, forbid_signatures, require_list, phase=phase)
            model.optimize()
            if model.Status != GRB.OPTIMAL: return None, None, None, None, None
            if phase == 1 and model.ObjVal < 1e-6: break

            all_duals = np.array([c.Pi for c in constrs])
            duals = {
                'team': all_duals[:n],
                'pair_slot': all_duals[n : n + P*m_slots],
                'home_cons': all_duals[n + P*m_slots : n + 2*P*m_slots]
            }
            
            new_cols_this_iter = []
            for t in range(n):
                new_tour = ttp_pricer.solve(t, duals, dist_matrix, n, m_slots, pair_to_idx, phase=phase)
                if new_tour:
                    new_tour['delta'] = np.array(new_tour['delta'], dtype=int).reshape(n, m_slots)
                    sig = (t, tuple(new_tour['opp_seq']), tuple(new_tour['home_flags']))
                    if not any((c['team'], tuple(c['tour']['opp_seq']), tuple(c['tour']['home_flags'])) == sig for c in pool):
                        new_cols_this_iter.append({'team': t, 'tour': new_tour})
            
            if not new_cols_this_iter:
                if phase == 1: return None, None, None, None, None # Infeasible
                else: return model, xvars, included_cols, np.array([v.X for v in xvars]), model.ObjVal
            pool.extend(new_cols_this_iter)
        else: # If loop finishes without break
            if phase == 1: return None, None, None, None, None # Infeasible
            else: print("CG reached max iterations in Phase 2.")
    
    model, xvars, included_cols, constrs = build_master_model_gurobi(pool, forbid_triplets, forbid_signatures, require_list, phase=2)
    model.optimize()
    if model.Status != GRB.OPTIMAL: return None, None, None, None, None
    return model, xvars, included_cols, np.array([v.X for v in xvars]), model.ObjVal


def is_solution_integral(x_vals, tol=1e-6): return all(abs(v - round(v)) < tol for v in x_vals)
def compute_pairslot_z_from_solution(included_cols, x_vals):
    zvals = {}; [zvals.update({(i,j,s): sum(x_vals[k] for k,c in enumerate(included_cols) if c['team']==i and c['tour']['delta'][j,s]==1)}) for i,j in pair_list for s in range(m_slots)]; return zvals

def branch_and_price_gurobi(initial_pool, time_limit_nodes=3600.0, max_nodes=2000):
    start_time = time.time()
    node_id_gen = 0
    heap = [(0.0, 0, initial_pool, frozenset(), frozenset(), tuple())]
    best_int_obj, best_int_solution, nodes_processed = float('inf'), None, 0
    
    while heap and (time.time() - start_time) < time_limit_nodes and nodes_processed < max_nodes:
        lb, node_id, node_pool, forbid_triplets, forbid_signatures, require_list = heapq.heappop(heap)
        if lb >= best_int_obj: continue
        nodes_processed += 1
        print(f"\nNode {node_id} (processed {nodes_processed}) lb={lb:.2f} |B_T|={len(forbid_triplets)} |B_S|={len(forbid_signatures)} |R|={len(require_list)}")
        
        res = column_generation_gurobi(list(node_pool), set(forbid_triplets), set(forbid_signatures), list(require_list))
        if res[0] is None: print(" -> Node LP infeasible, prune."); continue
        model, xvars, included_cols, x_vals, true_lb = res
        
        print(f" -> Node LP obj (True LB) = {true_lb:.2f}, columns = {len(included_cols)}")
        
        if true_lb >= best_int_obj: print(" -> Bound >= incumbent, prune."); continue
            
        if is_solution_integral(x_vals):
            solution_cols = [c for k, c in enumerate(included_cols) if x_vals[k] > 0.5]
            true_obj = sum(c['tour']['cost'] for c in solution_cols)
            print(f" -> VALID Integer solution found with TRUE cost = {true_obj:.2f}!")
            if true_obj < best_int_obj:
                print(f"    (New best solution, updating incumbent from {best_int_obj:.2f} to {true_obj:.2f})")
                best_int_obj, best_int_solution = true_obj, solution_cols
                heap = [(plb,pid,pp,pft,pfs,preq) for plb,pid,pp,pft,pfs,preq in heap if plb < best_int_obj]
                heapq.heapify(heap)
            continue

        pruned_pool = [c for k,c in enumerate(included_cols) if x_vals[k] > 1e-6 or (xvars and hasattr(xvars[k], 'RC') and abs(xvars[k].RC) < 1e-6)]

        zvals = compute_pairslot_z_from_solution(included_cols, x_vals)
        frac_items = sorted([(abs(val-0.5),(p,val)) for p,val in zvals.items() if 1e-6<val<1-1e-6])
        if frac_items:
            _, ((i,j,s), val) = frac_items[0]
            print(f" -> Branch on pair-slot z({i},{j},{s}) = {val:.4f}")
            child1_req = tuple(list(require_list) + [(i,j,s)])
            heapq.heappush(heap, (true_lb, node_id_gen+1, pruned_pool, forbid_triplets, forbid_signatures, child1_req)); 
            child2_trip = set(forbid_triplets); child2_trip.add((i,j,s)); child2_trip.add((j,i,s))
            heapq.heappush(heap, (true_lb, node_id_gen+2, pruned_pool, frozenset(child2_trip), forbid_signatures, require_list)); node_id_gen += 2
        elif not is_solution_integral(x_vals):
            frac_cols = [(k, v) for k, v in enumerate(x_vals) if 1e-6 < v < 1-1e-6]
            if not frac_cols: continue
            by_team = {t:[] for t in range(n)}; [by_team[included_cols[k]['team']].append((v, k)) for k,v in frac_cols]
            target_team = max(by_team, key=lambda t: sum(v for v,k in by_team[t]))
            _, k_forbid = max(by_team[target_team])
            sig_forbid = (included_cols[k_forbid]['team'], tuple(included_cols[k_forbid]['tour']['opp_seq']), tuple(included_cols[k_forbid]['tour']['home_flags']))
            print(f" -> Fallback branch on team {team_names[target_team]} (x_{k_forbid}={x_vals[k_forbid]:.4f})")
            child1_sigs = set(forbid_signatures); child1_sigs.add(sig_forbid)
            heapq.heappush(heap, (true_lb, node_id_gen+1, pruned_pool, forbid_triplets, frozenset(child1_sigs), require_list)); 
            other_frac = [c for c in by_team[target_team] if c[1] != k_forbid]
            if other_frac:
                _, k2_forbid = max(other_frac)
                sig2_forbid = (included_cols[k2_forbid]['team'], tuple(included_cols[k2_forbid]['tour']['opp_seq']), tuple(included_cols[k2_forbid]['tour']['home_flags']))
                child2_sigs = set(forbid_signatures); child2_sigs.add(sig2_forbid)
                heapq.heappush(heap, (true_lb, node_id_gen+2, pruned_pool, forbid_triplets, frozenset(child2_sigs), require_list)); 
            node_id_gen += 2

    print(f"\nBranch-and-price finished. Nodes processed: {nodes_processed}")
    if best_int_solution: print(f"Best integer obj = {best_int_obj:.2f}")
    else: print("No integer solution found within limits.")
    return best_int_obj, best_int_solution
    
if __name__ == "__main__":
    print("Generating a valid initial pool using the C++ pricer...")
    initial_pool = []
    dummy_duals = {'team': np.zeros(n), 'pair_slot': np.zeros(P*m_slots), 'home_cons': np.zeros(P*m_slots)}
    for t in range(n):
        # Using phase=0 for initial generation
        initial_tour = ttp_pricer.solve(t, dummy_duals, dist_matrix, n, m_slots, pair_to_idx, phase=0)
        if initial_tour:
            initial_tour['delta'] = np.array(initial_tour['delta'], dtype=int).reshape(n, m_slots)
            initial_pool.append({'team': t, 'tour': initial_tour})
        else:
            raise RuntimeError(f"Could not generate a valid initial tour for team {team_names[t]}.")

    print(f"Initial pool size: {len(initial_pool)}")
    start_total = time.time()
    best_obj, best_solution = branch_and_price_gurobi(initial_pool, time_limit_nodes=3600.0, max_nodes=5000)
    print(f"\nTotal elapsed time: {time.time() - start_total:.2f} seconds")

    if best_solution:
        sel = [{'team': team_names[c['team']], 'cost': c['tour']['cost'],
                'opp_seq': tuple(team_names[o] for o in c['tour']['opp_seq']),
                'home_flags': tuple(['H' if h else 'A' for h in c['tour']['home_flags']])}
               for c in sorted(best_solution, key=lambda c: c['team'])]
        df = pd.DataFrame(sel)
        print("\nBest integer solution found:\n"); print(df.to_string(index=False))
    else:
        print("\nNo integer solution was found.")

