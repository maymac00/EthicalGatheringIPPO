import numpy as np

from CommonsGame.new_utils import *
from CommonsGame.ValueIteration import value_iteration
from CommonsGame.constants import DONATION_BOX_CAPACITY

v_folder = "v_functions/"

def SolveSOMDP(discount_factor, weight_vector, algorithm, learner=0, mode="scalarisation"):

    if algorithm == 'VI':
        q_function, value_function = value_iteration(learner, discount_factor=discount_factor, mode=mode, weights=weight_vector)
    else:
        return "Fatal Error. Unknown algorithm."
    return value_function[learner]

def ethical_embedding_state(hull):
    """
    Ethical embedding operation for a single state. Considers the points in the hull of a given state and returns
    the ethical weight that guarantees optimality for the ethical point of the hull

    :param hull: set of two 2-D points, coded as a numpy array
    :return: the etical weight w, a positive real number
    """

    w = 0.0
    second_best_ethical = hull[0]
    best_ethical = hull[1]

    ethical_delta = best_ethical[1] - second_best_ethical[1]

    threshold = 0.01 # to avoid possible numerical errors we set a threshold for the ethical value

    if ethical_delta > threshold:
        eth = best_ethical[1]
        uneth = second_best_ethical[1]
        ind = best_ethical[0]
        unind = second_best_ethical[0]
    else:
        eth = uneth = ind = unind = 0

    return w, eth, uneth, ind, unind


def ethical_embedding(hull, epsilon):
    """
    Repeats the ethical embedding process for each state in order to select the ethical weight that guarantees
    that all optimal policies are ethical.

    :param hull: the convex-hull-value function storing a partial convex hull for each state. The states are adapted
    to the public civility game.
    :param epsilon: the epsilon positive number considered in order to guarantee ethical optimality (it does not matter
    its value as long as it is greater than 0).
    :return: the desired ethical weight
    """
    ""


    positions_with_apples = [[2, 1], [2, 2], [3, 1]]

    accumulated_ethical = 0
    accumulated_unethical = 0
    accumulated_individual = 0
    accumulated_unindividual = 0

    ## We only consider the initial states
    for pos in agent_positions:
        state = new_state(1, [0], True, forced_agent_apples=0, forced_grass=[True, True, True], forced_ag_pos=pos)

        if check_redundant_states(positions_with_apples, pos, [True, True, True]):
            continue

        w_temp, eth, uneth, ind, unind = ethical_embedding_state([hull[0][state], hull[1][state]])
        accumulated_ethical += eth
        accumulated_unethical += uneth

        accumulated_individual += ind
        accumulated_unindividual += unind

    #if accumulated_individual > 0.0000:
    #    print("Individual Similarity: ", accumulated_individual, accumulated_unindividual, (accumulated_unindividual)/accumulated_individual*100, "%")
    #if accumulated_ethical > 0.0000:
    #    print("Ethical Similarity: ", accumulated_unethical, accumulated_ethical, (accumulated_unethical)/accumulated_ethical*100, "%")

    individual_delta = accumulated_unindividual - accumulated_individual
    ethical_delta = accumulated_ethical - accumulated_unethical

    if ethical_delta < 0.0001: # to avoid dividing by zero
        w_ethical = 0.0
    else:
        w_ethical = max(0.0, individual_delta/ethical_delta)

    w_ethical = round(w_ethical  + epsilon, 1)
    return w_ethical




def Optimistic_Linear_Support(agent, epsilon):
    """
    Computes the ethical weight using Optimistic Linear Support.
    :param env: the MOMDP
    :return: the ethical weight
    """

    try:
        v_ethical = np.load(v_folder +"V_" + str(agent) +"_C" + str(DONATION_BOX_CAPACITY) + "_" + str(3000.0) + ".npy")
    except:
        v_ethical = SolveSOMDP(discount_factor=0.8, weight_vector=[1.0, 3000.0], algorithm="VI", learner=agent, mode="scalarisation")
        np.save(v_folder +"V_" + str(agent) + "_C" + str(DONATION_BOX_CAPACITY) + "_" + str(3000.0) + ".npy", v_ethical)
    print("Ethical policy computed.")

    try:
        v_unethical = np.load(v_folder +"V_" + str(agent) +"_C" + str(DONATION_BOX_CAPACITY) + "_" + str(0.0) + ".npy")
    except:
        v_unethical = SolveSOMDP(discount_factor=0.8, weight_vector=[1.0, 0.0], algorithm="VI", learner=agent, mode="scalarisation")
        np.save(v_folder +"V_" + str(agent) + "_C" + str(DONATION_BOX_CAPACITY) + "_" + str(0.0) + ".npy", v_unethical)
    print("Unethical policy computed.")

    hull = [v_unethical, v_ethical]

    new_ethical_weight = ethical_embedding(hull, epsilon=epsilon)
    ethical_weight = -99999

    while new_ethical_weight > epsilon:

        ethical_weight = new_ethical_weight
        print()
        print("The two policies are different. So far the ethical weight is: ", ethical_weight)
        print()

        try:
            v_unethical = np.load(v_folder +"V_" + str(agent) + "_C" + str(DONATION_BOX_CAPACITY) + "_" + str(ethical_weight) + ".npy")
        except:
            v_unethical = SolveSOMDP(discount_factor=0.8, weight_vector=[1.0, ethical_weight], algorithm="VI", learner=agent, mode="scalarisation")
            np.save(v_folder +"V_" + str(agent) + "_C" + str(DONATION_BOX_CAPACITY) + "_" + str(ethical_weight) + ".npy", v_unethical)

        print("Possible ethical policy computed. Testing.")
        hull[0] = v_unethical
        new_ethical_weight = ethical_embedding(hull, epsilon=epsilon)


    print("Found ethical weight for agent " + str(agent) + " : " + str(ethical_weight))
    print("--------")
    print()
    return ethical_weight
