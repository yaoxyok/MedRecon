import gurobipy as gp
from gurobipy import GRB
import itertools

# BIG M constant for linearizing constraints
M = 10 

# Step 1: Optimal List Model
def optimal_list_model(D,C,Dk,E,I,N,t):
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)  # or env.setParam('LogToConsole', 0)
        env.start()
        with gp.Model("OptimalListModel", env=env) as model:

            # Decision variables
            o = model.addVars(D, vtype=GRB.BINARY, name="o")  # Binary variables for the optimal list

            # Objective: Minimize the total number of drugs in the optimal list
            model.setObjective(gp.quicksum(o[d] for d in range(D)), GRB.MINIMIZE)

            # Constraints:
            # 1. Treatment goals must be achieved for each condition
            for c in range(C):
                model.addConstr(gp.quicksum(E[d, c] * o[d] for d in range(D)) >= t[c], f"UnderPrescribing_c{c}")
            for k in Dk:
                model.addConstr(gp.quicksum(o[d] for d in Dk[k]) <= 1, f"MaxOneDrug_k{k}")

            # 2. Avoid drug-drug interactions
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    model.addConstr(o[d1] + o[d2] <= 2 - I[d1, d2], f"DrugInteraction_{d1}_{d2}")

            # 3. Avoid contraindicated drugs
            for c in range(C):
                for d in range(D):
                    model.addConstr(o[d] <= 1 - N[d, c], f"Contraindication_d{d}_c{c}")

            # Solve the model
            model.optimize()

            if model.status == GRB.OPTIMAL:
                optimal_drug_list = [d for d in range(D) if o[d].x > 0.5]
                print(f"Optimal drug list: {optimal_drug_list}")
                return [o[d].x for d in range(D)], model.ObjVal
            else:
                print("No optimal solution found.")
                return [], 0

# Step 2: Reconciled List Model
def reconciled_list_model(o_sum,D,C,Dk,E,I,N,t,D_c,D_p,m):

    with gp.Env(empty=True) as env:
        
        env.setParam('OutputFlag', 0)  # or env.setParam('LogToConsole', 0)
        env.start()
        
        with gp.Model("ReconciledListModel", env=env) as model:

            # Decision variables
            x = model.addVars(D, vtype=GRB.BINARY, name="x")  # Binary variables for reconciled list
            D_upper = list(itertools.combinations(range(D), 2))
            zxx = model.addVars(D_upper, vtype=GRB.BINARY, name="zxx")  # Binary variables for linearize obj function
            zxo = model.addVars(D,D, vtype=GRB.BINARY, name="zxo")  # Binary variables for linearize obj function
            zoo = model.addVars(D_upper, vtype=GRB.BINARY, name="zoo")  # Binary variables for linearize obj function
            y = model.addVars(D, C, vtype=GRB.BINARY, name="y")  # Auxiliary binary variables
            o = model.addVars(D, vtype=GRB.BINARY, name="o")  # Binary variables for the optimal list

            S = m @ m.T

            # Objective: Minimize the number of changes
            f1 = gp.quicksum(x[d] for d in D_p) - gp.quicksum(x[d] for d in D_c)
            # f1_alt = gp.quicksum(m[d].dot(m[d]) * x[d] for d in D) - gp.quicksum(m[d].dot(m[d]) for d in Dc)
            f2 = gp.quicksum(
                    S[d, d] * (x[d] - 2*zxo[d, d] + o[d])
                for d in range(D)) + \
                2*gp.quicksum(
                    S[d, d_prime] * (zxx[d, d_prime] - zxo[d, d_prime] - zxo[d_prime,d] + zoo[d,d_prime])
                for d,d_prime in D_upper)  # Outer summation over d


            for d, d_prime in D_upper:
                model.addConstr(zxx[d, d_prime] >= x[d] + x[d_prime] -1 , f"AuxVar1_z_d{d}_d'{d_prime}")
                model.addConstr(zxx[d, d_prime] <= x[d]  , f"AuxVar2_z_d{d}_d'{d_prime}")
                model.addConstr(zxx[d, d_prime] <= x[d_prime] , f"AuxVar3_z_d{d}_d'{d_prime}")

            for d in range(D):
                for d_prime in range(D):
                    model.addConstr(zxo[d, d_prime] >= x[d] + o[d_prime] -1 , f"AuxVar1_zo_d{d}_d'{d_prime}")
                    model.addConstr(zxo[d, d_prime] <= x[d], f"AuxVar2_zo_d{d}_d'{d_prime}")
                    model.addConstr(zxo[d, d_prime] <= o[d_prime], f"AuxVar3_zo_d{d}_d'{d_prime}")

            for d, d_prime in D_upper:
                model.addConstr(zoo[d, d_prime] >= o[d] + o[d_prime] -1 , f"AuxVar1_oo_d{d}_d'{d_prime}")
                model.addConstr(zoo[d, d_prime] <= o[d] , f"AuxVar2_oo_d{d}_d'{d_prime}")
                model.addConstr(zoo[d, d_prime] <= o[d_prime] , f"AuxVar3_oo_d{d}_d'{d_prime}")

            print(o_sum)
            model.addConstr(gp.quicksum(o[d] for d in range(D)) == o_sum, f"MinimizeOptimalList")
            # Constraints:
            # 1. Treatment goals must be achieved for each condition
            for c in range(C):
                model.addConstr(gp.quicksum(E[d, c] * x[d] for d in range(D)) >= t[c], f"UnderPrescribing_c{c}")

            # 2. Over-prescribing constraints (linearized)
            for d in range(D):
                model.addConstr(x[d] <= gp.quicksum(y[d, c] for c in range(C)), f"OverPrescribing_d{d}")
                for c in range(C):
                    model.addConstr(y[d, c] <= x[d], f"AuxVar_y_d{d}_c{c}")
                    model.addConstr(gp.quicksum(E[d_prime, c] * x[d_prime] for d_prime in range(D) if d_prime != d) <= t[c] -1 + M * (1 - y[d, c]), f"TreatmentWithout_d{d}_c{c}")

            for k in Dk:
                model.addConstr(gp.quicksum(o[d] for d in Dk[k]) <= 1, f"MaxOneDrug_o{k}")
                model.addConstr(gp.quicksum(x[d] for d in Dk[k]) <= 1, f"MaxOneDrug_x{k}")

            # 3. Avoid drug-drug interactions
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    model.addConstr(x[d1] + x[d2] <= 2 - I[d1, d2], f"DrugInteraction_{d1}_{d2}")

            # 4. Avoid contraindicated drugs
            for c in range(C):
                for d in range(D):
                    model.addConstr(x[d] <= 1 - N[d, c], f"Contraindication_d{d}_c{c}")

            ### For the optimal list
            # 1. Treatment goals must be achieved for each condition
            for c in range(C):
                model.addConstr(gp.quicksum(E[d, c] * o[d] for d in range(D)) >= t[c], f"UnderPrescribing_c{c}")

            # 2. Avoid drug-drug interactions
            for d1 in range(D):
                for d2 in range(d1 + 1, D):
                    model.addConstr(o[d1] + o[d2] <= 2 - I[d1, d2], f"DrugInteraction_{d1}_{d2}")

            # 3. Avoid contraindicated drugs
            for c in range(C):
                for d in range(D):
                    model.addConstr(o[d] <= 1 - N[d, c], f"Contraindication_d{d}_c{c}")
            
            # Set up a list to store solutions on the Pareto frontier
            pareto_solutions = []
            def weighted_optimize(w1):
                model.setObjective(w1 * f1 + (1-w1) * f2, GRB.MINIMIZE)
                # Optimize the model
                model.optimize()
                if model.status == GRB.OPTIMAL:
                    # Store the solution
                    reconciled_ls = [d for d in range(D) if x[d].x > 0.5]
                    optimal_ls = [d for d in range(D) if o[d].x > 0.5]
                    return (f1.getValue(), f2.getValue(), w1, reconciled_ls, optimal_ls)
                else:
                    return (None, None, None, None, None)
            
            pareto_solutions.append(weighted_optimize(1))
            pareto_solutions.append(weighted_optimize(0))

            if pareto_solutions[0][0] == pareto_solutions[1][0]:
                pareto_solutions.pop(0)
            elif pareto_solutions[0][1] == pareto_solutions[1][1]:
                pareto_solutions.pop(1)

            i = 0
            # Iterate over adjacent pairs
            while i < len(pareto_solutions) - 1:
                f11,f21,_,_,_ = pareto_solutions[i]
                f12,f22,_,_,_ = pareto_solutions[i + 1]
                # print('f11-f21-f12+f22',f11-f21-f12+f22)
                if -1e-5 < f11-f21-f12+f22 < 1e-5:
                    i += 1
                    continue
                alpha = (f22-f21) / (f11-f12+f22-f21)

                # Solve weighted problem
                f1_new,f2_new,_,x_new,o_new = weighted_optimize(alpha)
                print('f1_new-f2_new',f1_new,f2_new, (f1_new, f2_new) != (f11, f21), (f1_new, f2_new) != (f12, f22))
                if (f1_new, f2_new) != (f11, f21) and (f1_new, f2_new) != (f12, f22):
                    # Insert new point between left and right
                    pareto_solutions.insert(i+1, (f1_new,f2_new,alpha,x_new,o_new))
                else:
                    # Move to next pair
                    i += 1
            
            return pareto_solutions
    

