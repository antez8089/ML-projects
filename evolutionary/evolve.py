#==============================================================================
#==============================EVOLUTIONARY====================================
#===============================ALGORITHMS=====================================
#==============================================================================
import numpy as np
from cec2017.functions import f2, f13
from random import randint

UPPER_BOUND = 100
DIMENSIONALITY = 10
POPULATION_SIZE = 2

x = [np.random.uniform(-UPPER_BOUND, UPPER_BOUND, size=DIMENSIONALITY) for i in range(POPULATION_SIZE)]



def evaluate(f, x):
    """
    Function to evaluate the value of the function f in the point x
    """
    true_val = f(x)
    for i in range(len(x)):
        if x[i] > UPPER_BOUND:
            true_val += 10e6
        elif x[i] < -UPPER_BOUND:
            true_val += 10e6
    return true_val

def mutations(x, p_mutation, sigma=3):
    """
    Function to apply the genetic operators, such
    as crossover and mutations to the population
    """
    for i in range(len(x)):
        if randint(0,1) <= p_mutation:
            add = np.random.normal(0, 1, size=DIMENSIONALITY)
            np.multiply(add, sigma)
            x[i] =x[i] + add
    return x

def tourney_selection(o, o_new):
    """
    Function that implements the tourney selection
    """
    knights = o + o_new
    winners = []
    while len(knights) > 1:
        contestan1 = knights.pop(randint(0, len(knights) -1))
        contestan2 =knights.pop(randint(0, len(knights)-1))
        winners.append(min(contestan1, contestan2, key=lambda x: x[1]))
    return winners

def evolve(x, f, p_mutation, budget=10000, sigma=1.8):
    """
    Implementation of the classical evolutionary algorithm without crossover
    """
    tmax = budget/len(x)
    t = 0
    o = [(individual, evaluate(f, individual)) for individual in x]
    best_individual, best_value = min(o, key=lambda x: x[1])
    while t < tmax:
        x_new = mutations(x, p_mutation, sigma)
        o_new =  [(individual, evaluate(f, individual)) for individual in x_new]
        new_best_individual, new_best_value = min(o_new, key=lambda x: x[1])
        if new_best_value < best_value:
            best_individual = new_best_individual
            best_value = new_best_value
        o = tourney_selection(o, o_new)
        x = [individual for individual, value in o]
        t += 1
    return best_individual, best_value


print(evolve(x, f2, 0.55))