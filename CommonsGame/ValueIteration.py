import numpy as np
import gym

from CommonsGame.new_utils import *
from CommonsGame.ValueIterationAuxiliar import probsV_calculator, create_model

DONATE = 8
TAKE_DONATION = 9


def sweep_preparations(state, agent, model):
    original_state = model[state][6][6][:21]
    if original_state[0] == 0:  # we skip states that actually do not exist
        return False, [False], False

    there_is_an_apple = 0
    if DONATION_BOX_CAPACITY == 1:  # adaptations for the particular case in which the pool only has one apple
        if original_state[-1] == 1 or original_state[-1] == 2:
            return False, [False], False

    if original_state[-1] == 5:
        original_state[-1] = DONATION_BOX_CAPACITY

    for element in original_state:
        if element == 64:
            there_is_an_apple += 1

    if agent == 0:
        probs = [0.7, 0.3]
    else:
        probs = [1.0]

    num_apples2 = 1

    if check_common_pool(original_state) >= 2:
        num_apples2 = SURVIVAL_THRESHOLD
        if agent == 1:
            probs = [0.9, 0.1]
    if check_common_pool(original_state) == DONATION_BOX_CAPACITY:
        num_apples2 = SURVIVAL_THRESHOLD + 1

    state_no_learn = new_state(agent, original_state, True, forced_agent_apples=num_apples2)
    state_no_learn2 = new_state(agent, original_state, True, forced_agent_apples=SURVIVAL_THRESHOLD + 1)

    states_next = [state_no_learn, state_no_learn2]

    return original_state, probs, states_next



def sweep_Q_function_with_model(model, agent, Q, V, action_space, mode, discount_factor, weights, equilibrium=[]):
    """
    Calculates the value of applying each action to all states.

    :param model: the environment of the Markov Decision Process
    :param state: the current state
    :param V: value function to see the value of the next state V(s')
    :param discount_factor: discount factor considered, a real number
    :return: Q-table, V-table, delta to indicate how much V has changed from the previous sweep
    """

    state_count = 0
    delta = 0

    target_policies = list()

    if len(equilibrium) > 0:
        target_policies = equilibrium
    else:
        target_policies = reference_policy



    for state in range(len_state_space):

        state_count += 1

        if state_count % 1000 == 0:
            print("States swept :", state_count, "/", len_state_space + 1)

        original_state, probs, states_now = sweep_preparations(state, agent, model)

        if not probs[0]:  # occurs when the state where are now does not actually exist
            continue

        for action in action_space:

            V_prima = 0

            for i in range(len(probs)):
                actions = [int(target_policies[0][states_now[0]]), int(target_policies[1][states_now[i]])]


                if agent == 1:
                    if actions[0] == 8:
                        actions[0] = 6  # to create a more accurate model, Agent 0 barely donates due to a lack of opportunities

                actions[agent] = action

                # Apply actions to model
                output = model[state][actions[0]][actions[1]]

                # We start decomposing the different elements of the output
                state_prima = output[:21]


                rewards = [0, 0]
                rewards[0] = output[21:23]
                rewards[1] = output[23:25]

                if DONATION_BOX_CAPACITY == 1:  # only for the case when the COMMON_POOL has 1 apple of capacity
                    if state_prima[-1] > 1:
                        state_prima[-1] = 1
                        if rewards[agent][1] > 0 and original_state[-1] == 0:
                            rewards[agent][1] = 0

                reward = rewards[agent]


                V_prima += probs[i]*probsV_calculator(mode, agent, actions, Q[agent], V[agent], original_state, state_prima, True, action_space)


            Q[agent][state][action] = reward + discount_factor * V_prima


        if mode == "lex":
            V_ind, V_eth, _ = lexicographic_Qs(action_space, Q[agent][state][action_space])

            delta = max(delta, np.abs(
                scalarisation_function([V_ind, V_eth], [1.0, 10.0]) - scalarisation_function(V[agent][state],
                                                                                             [1.0, 10.0])))

            V[agent][state] = np.array([V_ind, V_eth])


        elif mode == "scalarisation":
            index = np.argmax(scalarised_Qs(len(action_space), Q[agent][state][action_space], weights))
            best_action_value = Q[agent][state][action_space[index]]

            delta = max(delta, np.abs(
                scalarisation_function(best_action_value, weights) - scalarisation_function(V[agent][state],
                                                                                            weights)))

            V[agent][state] = best_action_value
        else:
            print("Fatal error. Mode not recognised.")
            return -1


    return Q, V, delta




def value_iteration(agent, mode="scalarisation", discount_factor=0.8, weights=[1.0, 0.0], num_iterations=20, tabularRL = True, equilibrium=[]):
    """
    Adapted for VI

    :param environment: the environment already configured
    :param tabularRL: boolean to know if you will be using tabular RL or deep RL
    :return:
    """

    #try:
    #    model = np.load("model_SMALL.npy")
    #except:
    print("There was no model of the environment found. Proceeding to create one.")
    print("This while take a while (approx 50 minutes).")
    create_model()
    model = np.load("model_SMALL.npy")

    environment = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3, fullState=False, tabularState=tabularRL)
    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)

    Q_functions = np.zeros((number_of_agents, len_state_space, environment.action_space.n, number_of_objectives))
    V_functions = np.zeros((number_of_agents, len_state_space, number_of_objectives))

    iterations = 0

    while iterations < num_iterations:
        print("Iteration : ", iterations, " / ", num_iterations)
        iterations += 1

        Q_functions, V_functions, delta = sweep_Q_function_with_model(model, agent, Q_functions, V_functions, action_space, mode, discount_factor, weights)
        print("Delta : ", delta)
    return Q_functions, V_functions


if __name__ == '__main__':
    tabularRL = True
    training_now = True
    mode = "scalarisation"
    policy_folder = "policies/"

    ethical_weight = 2.6
    weights = [1.0, ethical_weight]

    if training_now:

        environment = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3, fullState=False, tabularState=tabularRL)

        total_action_space = [i for i in range(environment.action_space.n)]
        action_space = new_action_space(total_action_space, environment)

        for learner in [0, 1]:

            Q_functions, V_functions = value_iteration(learner, mode=mode, weights=weights)

            policy = policy_creator(Q_functions[learner], action_space, mode=mode, weights=weights)
            np.save(policy_folder + "policy" + str(learner) +"_C" + str(DONATION_BOX_CAPACITY) + "_" + str(weights[1]) + ".npy", policy)

    if not training_now:

        what = "_"+str(weights[1])

        policy0 = np.load(policy_folder +"policy0_C" + str(DONATION_BOX_CAPACITY) + what + ".npy")
        policy1 = np.load(policy_folder +"policy1_C" + str(DONATION_BOX_CAPACITY) + what + ".npy")

        env = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3, fullState=False, tabularState=tabularRL, agent_pos=[[3, 0], [3, 3]])

        evaluation(env, tabularRL, we_render=True, policies=[policy0, policy1], how_much=400)
