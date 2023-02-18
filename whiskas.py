"""
Whiskas Model Python Formulation for the PuLP Modeller [1]

Optimisation problem whose aim is to provide a minimisation
of costs with constraints given by nutritional values.

References
----------
[1] https://coin-or.github.io/pulp/CaseStudies/a_blending_problem.html
"""
import pandas as pd
from pulp import *

# ingredients data
ingredient_df = pd.DataFrame.from_dict({
    "chicken": [0.100, 0.080, 0.001, 0.002],
    "beef": [0.200, 0.100, 0.005, 0.005],
    "mutton": [0.150, 0.110, 0.003, 0.007],
    "rice": [0.000, 0.010, 0.100, 0.002],
    "wheat_bran": [0.040, 0.010, 0.150, 0.008],
    "gel": [0.000, 0.000, 0.000, 0.00]
},
    orient='index',
    columns=["protein", "fat", "fibre", "salt"])

# cost data
costs_df = pd.DataFrame.from_dict({
    "chicken": [0.013],
    "beef": [0.008],
    "mutton": [0.010],
    "rice": [0.002],
    "wheat_bran": [0.005],
    "gel": [0.001]
    },
    orient='index',
    columns=["cost"])

# Decision variables
decision_var_lst = costs_df.index.to_list()

print("Ingredient data:")
print(ingredient_df.head())

print("Cost data:")
print(costs_df.head())

# Initialisation of the problem
prob = LpProblem("The Whiskas Problem", LpMinimize)

# Define decision variables
ingredient_vars = LpVariable.dicts(
    "ingr",
    indices=decision_var_lst,
    lowBound=0,
    upBound=None,
    cat='Continuous'
)

# Objective function
prob += lpSum([costs_df.loc[idx] * ingredient_vars[idx] for idx in decision_var_lst])

# Constraints
prob += lpSum([ingredient_vars[i] for i in decision_var_lst]) == 100, "PercentagesSum"
prob += (
    lpSum([ingredient_df.loc[i, "protein"] * ingredient_vars[i] for i in decision_var_lst]) >= 8.0,
    "ProteinRequirement",
)
prob += (
    lpSum([ingredient_df.loc[i, "fat"] * ingredient_vars[i] for i in decision_var_lst]) >= 6.0,
    "FatRequirement",
)
prob += (
    lpSum([ingredient_df.loc[i, "fibre"] * ingredient_vars[i] for i in decision_var_lst]) <= 2.0,
    "FibreRequirement",
)
prob += (
    lpSum([ingredient_df.loc[i, "salt"] * ingredient_vars[i] for i in decision_var_lst]) <= 0.4,
    "SaltRequirement",
)

# # The problem data is written to an .lp file
# prob.writeLP("WhiskasModel.lp")

# The problem is solved using PuLP's choice of Solver
print(f"Available solvers: {listSolvers(onlyAvailable=True)}")

# Select the solver and run the solution
# solver = PULP_CBC_CMD(msg=True, warmStart=True)
solver = getSolver('PULP_CBC_CMD')
sol = prob.solve(solver)

# Solution status
print(f"Solution status: {LpStatus[sol]}")

# Get optimal solution (percentage of each ingredient)
for var in decision_var_lst:
    print(f"{var}: {ingredient_vars[var].varValue}")
