import numpy as np
import gym
import random
import argparse

from CommonsGame.constants import tinyMap, smallMap, bigMap, SURVIVAL_THRESHOLD, DONATION_BOX_CAPACITY, MAP_NAME
from CommonsGame.new_utils import *

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

if MAP_NAME == "tiny":
    current_map = tinyMap
else:
    current_map = smallMap


def learning_policy(state, action_space, Q_function, epsilon, mode, weights):
    """
    Epsilon-greedy policy.

    :param state:  typically, a list of integers
    :param action_space: a list of integers representing the possible actions that an agent can do
    :param Q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
    :return: a recommended action
    """

    choose_randomly = np.random.random() < epsilon
    if choose_randomly:
        index = np.random.randint(low=0, high=len(action_space))
    else:
        if mode == "lex":
            _, _, index = lexicographic_Qs(action_space, Q_function[state][action_space])

            return index
        elif mode == "scalarisation":
            index = np.argmax(scalarised_Qs(len(action_space), Q_function[state][action_space], weights))
        else:
            print("Uh oh!!!!!")
            index = 0

    return action_space[index]


def update_Q(Q, s, a, r, s_prima, alpha, gamma, action_space, mode, weights):
    """

    As a template, it updates Q(s,a) with the current reward

    :param Q_function: a numpy 2-d array (in tabular RL) that contains information about each state-action pair
    :param current_state: a list of integers
    :param current_action: an integer
    :param current_reward: an integer
    :param next_state: a list of integers
    :return: an updated Q_function
    """

    if mode == "lex":
        V_ind, V_eth, _ = lexicographic_Qs(action_space, Q[s_prima][action_space])
        V_prima = np.array([V_ind, V_eth])
    elif mode == "scalarisation":
        index = np.argmax(scalarised_Qs(len(action_space), Q[s_prima][action_space], weights))
        V_prima = Q[s_prima][action_space[index]]
    else:
        print("Fatal error. Unrecognised mode.")
        V_prima = [0., 0.]

    Q[s][a] += alpha * (r + gamma * V_prima - Q[s][a])

    return Q


def learning_loop(tabularRL, learning_rate=0.9, discount_factor=0.8, mode="lex", weights=[1.0, 0.0]):
    """
    Adapted for Q-learning

    :param environment: the environment already configured
    :param tabularRL: boolean to know if you will be using tabular RL or deep RL
    :return:
    """
    environment = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3,
                           fullState=False, tabularState=tabularRL)
    info_states = np.zeros((number_of_agents, len_state_space))
    print(len_state_space)
    info_state_actions = np.zeros((number_of_agents, len_state_space, environment.action_space.n))

    Q_functions = np.zeros((number_of_agents, len_state_space, environment.action_space.n, number_of_objectives))

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)
    min_epsilon = 0.97  # the bigger it is, the more random the agents' behaviour

    episodes = 10000
    timesteps = 500
    total_episodes = 0
    epsilon = [min_epsilon, min_epsilon]

    possible_ag_apples = [0, SURVIVAL_THRESHOLD - 1, SURVIVAL_THRESHOLD, SURVIVAL_THRESHOLD + 1]
    possible_pool_apples = [0]

    for episode in range(episodes):

        chosen_pos = False

        while not chosen_pos:
            random_pos_index = random.randint(0, len(agent_positions) - 1)

            ag_pos = agent_positions[random_pos_index]

            if ag_pos[0][0] == ag_pos[1][0] and ag_pos[0][1] == ag_pos[1][1]:
                pass
            elif ag_pos[0] in positions_with_apples:
                pass
            elif ag_pos[1] in positions_with_apples:
                pass
            else:
                chosen_pos = True

        total_episodes += 1

        if total_episodes % 20 == 0:
            print("--Episode", episode, ag_pos, total_episodes, "/", episodes)

            policy0 = policy_creator(Q_functions[0], action_space, mode=mode, weights=weights)
            policy1 = policy_creator(Q_functions[1], action_space, mode=mode, weights=weights)

            np.save("Q0_temp.npy", Q_functions[0])
            np.save("Q1_temp.npy", Q_functions[1])
            np.save("policy0_temp.npy", policy0)
            np.save("policy1_temp.npy", policy1)

        environment = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3,
                               fullState=False, tabularState=tabularRL, agent_pos=ag_pos)

        for agg in possible_ag_apples:
            for poo in possible_pool_apples:
                initial_state = environment.reset(num_apples=[agg, agg], common_pool=poo)
                state = []

                for agent in range(number_of_agents):
                    nu_state = new_state(agent, initial_state[agent], tabularRL)
                    state.append(nu_state)

                for timestep in range(timesteps):
                    nActions = list()

                    for agent in range(number_of_agents):
                        action_i = learning_policy(state[agent], action_space, Q_functions[agent], epsilon[agent], mode,
                                                   weights)
                        nActions.append(action_i)
                        info_state_actions[agent][state[agent]][action_i] += 1

                    nObservations, nRewards, _, _ = environment.step(nActions)

                    next_state = []

                    for agent in range(number_of_agents):
                        next_state.append(new_state(agent, nObservations[agent], tabularRL))

                        info_states[agent][next_state[agent]] += 1

                        buffer_count = 1
                        if info_state_actions[agent][state[agent]][nActions[agent]] < buffer_count:
                            new_alpha = learning_rate
                        else:
                            new_alpha = max(0.05, learning_rate / (
                                    learning_rate + info_state_actions[agent][state[agent]][
                                nActions[agent]] - buffer_count))

                        Q_functions[agent] = update_Q(Q_functions[agent], state[agent], nActions[agent],
                                                      nRewards[agent], next_state[agent], new_alpha, discount_factor,
                                                      action_space, mode, weights)

                    state = next_state
                    for agent in range(number_of_agents):
                        epsilon[agent] = max(0.4, min_epsilon ** info_states[agent][next_state[agent]])

    return Q_functions


if __name__ == '__main__':
    tabularRL = True
    evaluating = False

    policy_folder = "/../policies/"

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-ethical_weight', metavar='--W', type=float, help='an integer for the accumulator',
                        default=1.4)

    args = parser.parse_args()

    ethical_weight = 2.6

    weights = [1.0, ethical_weight]

    if not evaluating:

        for mode in ["scalarisation"]:
            env = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3,
                           fullState=False, tabularState=tabularRL)
            total_action_space = [i for i in range(env.action_space.n)]
            action_space = new_action_space(total_action_space, env)

            Qs = learning_loop(tabularRL, mode=mode)

            policy0 = policy_creator(Qs[0], action_space, mode=mode, weights=weights)
            policy1 = policy_creator(Qs[1], action_space, mode=mode, weights=weights)

            what = "_" + str(weights[1])

            np.save("policy0_C" + str(DONATION_BOX_CAPACITY) + "_" + what + ".npy", policy0)
            np.save("policy1_C" + str(DONATION_BOX_CAPACITY) + "_" + what + ".npy", policy1)
    else:
        what = "_" + str(weights[1])

        policy0 = np.load("policy0_C" + str(DONATION_BOX_CAPACITY) + what + ".npy")
        policy1 = np.load("policy1_C" + str(DONATION_BOX_CAPACITY) + what + ".npy")

        Q0 = np.load(+ "Q0.npy")
        print(Q0[15459], policy0[15459])
        how_many = 50
        for i in range(how_many):
            random_pos_index = random.randint(0, len(agent_positions) - 1)

            agents_position = [[3, 3], [3, 0]]  # agent_positions[random_pos_index]

            env = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3,
                           fullState=False, tabularState=tabularRL, agent_pos=agents_position)
            evaluation(env, tabularRL, we_render=False, policies=[policy0, policy1], how_much=400, it=i)

        from graphics import create_graphic

        create_graphic(how_many=how_many, how_much=400, pool_limit=DONATION_BOX_CAPACITY, agent_threshold=SURVIVAL_THRESHOLD)
