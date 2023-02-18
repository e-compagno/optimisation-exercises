"""
The Beer Distribution Problem [1].

References
----------
[1] https://coin-or.github.io/pulp/CaseStudies/a_transportation_problem.html
"""
#%%
import pandas as pd
from pulp import *

# define variables warehouse and house
warehouses = ["A", "B"]
houses = [str(idx + 1) for idx in range(5)]

# Decision variables are all possible routes
routes = [(w, h) for w in warehouses for h in houses]  #  all possible routes

#%% Production: how many units can be produced
production_df = pd.DataFrame.from_dict({"A": 1000,
                                        "B": 4000
                                        },
                                       orient='index',
                                       columns=["req"])
print("Production data")
print(production_df.head())

#%% demand data
demand_df = pd.DataFrame.from_dict({
    "1": 500,
    "2": 900,
    "3": 1800,
    "4": 200,
    "5": 700,
},
    orient='index',
    columns=['demand']
)

print("\nDemand data")
print(demand_df.head())

#%% Cost data
# cost per unit from warehouse to house
costs_df = pd.DataFrame([  # Bars
    # 1 2 3 4 5
    [2, 4, 5, 2, 1],  # A   Warehouses
    [3, 1, 3, 2, 3],  # B
], columns=houses, index=warehouses)

print("\nCost data")
print(costs_df.head())

#%% Modelling phase
# Initialise the problem
prob = LpProblem("Beer Distribution Problem", LpMinimize)

# Decision variables (number of units transported in that route)
# nested dictionary with {'A': {1: route_A_1, ...}, 'B': {1: route_B_1, ...}}
vars = LpVariable.dicts("route", (warehouses, houses), lowBound=0, upBound=None, cat="Integer")

# Objective function: total transport cost
# sum of all costs in each route (cost_per_unit * n_units: cost_A_1 * A_1 + ...
prob += (
    lpSum([vars[w][h] * costs_df.loc[w, h] for (w, h) in routes]),
    "Total_transport_cost"
)

# Constraints
# each warehouse cannot supply more than the max production
for w in warehouses:
    prob += (
        lpSum([vars[w][h] for h in houses]) <= production_df.loc[w, 'req'],
        f"Sum_production_of_warehouse_{w}"
    )

# each house needs to have at least the demand
for h in houses:
    prob += (
        lpSum([vars[w][h] for w in warehouses]) >= demand_df.loc[h, 'demand'],
        f"Sum_delivered_to_house_{h}"
    )

#%% Solve the optimisation problem
# Select the solver and run the solution
# solver = PULP_CBC_CMD(msg=True, warmStart=True)
solver = getSolver('PULP_CBC_CMD')
sol = prob.solve(solver)

# Solution status
print(f"Solution status: {LpStatus[sol]}")

# Get optimal solution (units transported in each route)
for (w, h) in routes:
    print(f"{vars[w][h]}: {vars[w][h].varValue}")



