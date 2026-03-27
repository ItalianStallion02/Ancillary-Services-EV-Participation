import gurobipy as gp
from gurobipy import GRB
import numpy as np
from config import *

n_vehicles = 100
T = 15

# charger efficiency
mu = 0.1

# final charge error
epsilon = 0.02

# charging station upper and lower power bounds
p_min = 0
p_max = 50

# get charging matrix
plain_v_DA, q_arr, q_dep = get_DA_vehicles()

# get soc data
soc0, soc_ref = adjusted_soc_data(get_initial_soc(), get_target_soc())

# fill charging matrix with reference soc values
v_DA = fill_charging_matrix(plain_v_DA, soc_ref)

# create gurobi model
model = gp.Model("da_scheduling")

# create EV decsion variables
POP = model.addMVar(shape=(n_veichles,96), vtype=GRB.CONTINUOUS, name="POP")
SOC = model.addMVar(shape=(n_veichles,96), vtype=GRB.CONTINUOUS, name="SOC")
error = model.addMVar(shape=(n_veichles,96), vtype=GRB.CONTINUOUS, name="error")
b = model.addMVar(shape=(n_vehicles,96), vtype=gp.GRB.BINARY, name="b")

# define objective function
obj = 0

for i in range(n_vehicles):
    for j in range(96):
        obj += error[i,j]

model.setObjective(obj, GRB.MINIMIZE)

# error dynamics contraints
model.addConstrs((error[:,i+1] == (error[:,i] - POP[:,i]*(1-mu)*T) for i in range(95)), name="error_dynamics")

# energy withdrawal constraint (not more than residual battery)
model.addConstrs((POP[:,i]*(1-mu)*T <= error[:,i] for i in range(96)), name="energy withdrawal bound for residual energy")

# charger operating point bounds
model.addConstrs((p_min*b[:,i] <= POP[:,i] for i in range(96)), name="Charger operating point - lower bound")
model.addConstrs((POP[:,i] <= p_max*b[:,i] for i in range(96)), name="Charger operating point - upper bound")

# setting integer variable and error inital condition (only once the associated EV has arrived)
for i in range(n_vehicles):
    for j in range(96):
        if (q_arr[i]<=j):
            if (q_arr[i] == j): model.addConstr(error[i,j] == (soc_ref-soc0)[i])
            model.addConstr(error[i,j] - epsilon <= b[i]*((soc_ref-soc0)[i]))
            model.addConstr(b[i,j]*((soc_ref-soc0)[i]) <= ((soc_ref-soc0)[i]) + error[i,j] - epsilon)
        else:
            model.addConstr(error[i,j] == 0)
            model.addConstr(b[i,j] == 0)

# final charging error constraint
for i in range(n_vehicles):
    if (q_dep[i] != 96):
        model.addConstr(error[i,q_dep[i]] <= epsilon)

model.optimize()