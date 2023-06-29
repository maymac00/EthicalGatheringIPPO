import numpy as np
import gym
from CommonsGame.new_utils import number_of_agents, new_action_space, policy_creator, policy_NULL
from CommonsGame.constants import DONATION_BOX_CAPACITY, tinyMap
from CommonsGame.OLS import Optimistic_Linear_Support as SAEEP
from CommonsGame.ValueIteration import value_iteration

policy_folder = "policies/"
v_folder = "v_functions/"

def ethical_equilibrium_computation():
    """
    Computes the first step of the MAEE process.

    Computes a best-ethical equilibrium within the ethical MOMG of the Ethical Gathering Game

    :return: the joint ethical policy
    """

    ethical_equilibrium = list()

    random_joint_policy = [policy_NULL, policy_NULL]

    print(v_folder)

    for agent in range(number_of_agents):
        try:
            ethical_equilibrium.append(np.load(policy_folder +"ref_policy" + str(agent) +"_C" + str(DONATION_BOX_CAPACITY) + ".npy"))
        except:
            environment = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=tinyMap,
                                   visualRadius=3, fullState=False, tabularState=True)


            total_action_space = [i for i in range(environment.action_space.n)]
            action_space = new_action_space(total_action_space, environment)

            Q_functions, _ = value_iteration(agent, weights=[1.0, 7000.0], equilibrium=random_joint_policy)
            policy = policy_creator(Q_functions[agent], action_space, weights=[1.0, 7000.0])
            ethical_equilibrium.append(policy)
            np.save("ref_policy" + str(agent) + "_C" + str(DONATION_BOX_CAPACITY) + ".npy", policy)

    return ethical_equilibrium


def solution_weight_vector_computation():
    """

    Computes the second step of our MAEE process, the solution weight vector for the ethical MOMG.

    :return: the solution ethical weight, a number
    """

    solution_ethical_weight = 0.0
    epsilon = 0.1
    for agent in range(number_of_agents):
        agent_weight = SAEEP(agent, epsilon)

        solution_ethical_weight = max(solution_ethical_weight, agent_weight)

    return solution_ethical_weight


def multi_agent_ethical_embedding_process():

    ethical_eq = ethical_equilibrium_computation()

    for agent in range(number_of_agents):
        np.save("ref_policy" + str(agent) + "_C" + str(DONATION_BOX_CAPACITY) + ".npy", ethical_eq[agent])

    ethical_weight = solution_weight_vector_computation()

    print("The solution weight vector for a common pool of capacity C = ", DONATION_BOX_CAPACITY, " is : ", ethical_weight)


if __name__ == '__main__':

    multi_agent_ethical_embedding_process()
