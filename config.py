import numpy as np
from random import random, randint, triangular, weibullvariate
import sys

def get_quarters_in_a_day(T):
    return 24*60/T

# sampling time (in minutes)
T = 15

# number of quarters in a day (according to sampling time)
q = 96

# number of distinct vehicles at charging point during the day
n_veichles = 100

# number of vehicles split among time of charge (according to mobility survey)
n_morning = round(n_veichles * 0.25)
n_evening = round(n_veichles * 0.5)
n_generic = round(n_veichles * 0.25)


# number of predicted vehicles in day-ahead
def get_DA_vehicles():

    # create matrix with:
    # ---> as many rows as distinct number of vehicles during day
    # ---> as many columns as quarters in a day
    n_vehicles = n_morning+n_evening+n_generic
    v_DA = np.zeros((n_vehicles, q))

    # create array of arrival times to fill
    q_arr = np.zeros(n_veichles)

    # fill arrays that indicate the arrival time of each class of driver
    q_arr[0:n_morning] = morning_arrivals = np.array([randint(28,42) for _ in range(n_morning)])
    q_arr[n_morning:n_morning+n_evening] = evening_arrivals = np.array([randint(64, 84) for _ in range(n_evening)])
    q_arr[n_morning+n_evening:] = generic_arrivals = np.array([round(triangular(0, 96, 48)) for _ in range(n_generic)])

    # fill array with dwelling time for each vehicle in each class
    morning_dwelling = np.array([round(weibullvariate(9.02,2)) for _ in range(n_morning)])
    
    evening_dwelling = np.zeros(n_evening)
    for i in range(n_evening):
        same_day = 1 if random() <= 0.4 else 0
        evening_dwelling[i] = randint(2, 88-evening_arrivals[i]) if same_day else 96-evening_arrivals[i]

    generic_dwelling = np.array([randint(2, 10) for _ in range(n_generic)])

    # create vector of departure quarters to fill
    q_dep = np.zeros(n_vehicles)

    # fill matrix with dwelling time:
    # ---> v_DA[i, j] = 1: vehicle i present at time j
    # ---> v_DA[i, j] = 0: vehicle i not present at quarter j
    for i in range(n_morning):
        arrival = morning_arrivals[i]
        dwelling = morning_dwelling[i]
        #v_DA[i, arrival:arrival+dwelling] = 1
        v_DA[i, arrival:] = 1
        q_dep[i] = arrival+dwelling

    for i in range(n_evening):
        arrival = round(evening_arrivals[i])
        dwelling = round(evening_dwelling[i])
        #v_DA[i+n_morning, arrival:arrival+dwelling] = 1
        v_DA[i+n_morning, arrival:] = 1
        q_dep[i+n_morning] = arrival+dwelling

    for i in range(n_generic):
        arrival = round(generic_arrivals[i])
        dwelling = round(generic_dwelling[i])
        #v_DA[i+n_morning+n_evening, arrival:arrival+dwelling] = 1
        v_DA[i+n_morning+n_evening, arrival:] = 1
        q_dep[i+n_morning+n_evening] = arrival+dwelling

    return v_DA, q_arr, q_dep

# get initial soc
def get_initial_soc():
    soc0 = np.array([np.random.normal(0.25, 0.08) for _ in range(n_veichles)])
    soc0 = np.array([0.02 if soc0[i]<0 else soc0[i] for i in range(n_veichles)])
    return soc0

# get target soc
def get_target_soc():
    soc_ref = np.array([np.random.normal(0.9, 0.17) for _ in range(n_veichles)])
    soc_ref = np.array([1 if soc_ref[i]>1 else soc_ref[i] for i in range(n_veichles)])
    return soc_ref

# make sure target soc is higher that initial soc
def adjusted_soc_data(x0, x_ref):
    for i in range(n_veichles):
        if x_ref[i] <= x0[i]:
            x0[i] = 0.95 * x_ref[i]
    return x0, x_ref

# fill charging matrix with reference state of charge
def fill_charging_matrix(v_DA, soc_ref):
    for i in range(n_veichles):
        v_DA[i, :] = v_DA[i, :] * soc_ref[i]
    return v_DA