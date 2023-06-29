import numpy as np
import gym

from CommonsGame.new_utils import *

DONATE = 8
TAKE_DONATION = 9

def probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=-1, apples_in_ground=0, forced_pool=-1):
    """
    AUXILIAR FUNCTION

    Computes the probability of moving from the current state to another state where
    at least there is one apple in ground. It also computes its associated V(s').

    If there are apples_in_ground, of course the probability is 100 %

    """


    next_state = new_state(agent, state_prima, tabularRL, forced_agent_apples, forced_pool=forced_pool)
    where_apples, _, _, where_agents_OLD, old_pos_agents, new_pos_agents = apples_in_ground

    # where_agents_0 is always considering the next position of the learning agent
    # where_agents_1 is always considering the next position of the non-learning agent
    where_agents_0 = check_agents_where_apples(state_prima, agent)
    where_agents_1 = check_agents_where_apples(state_prima, abs(agent - 1))
    wheres_agents = [where_agents_0, where_agents_1, where_agents_OLD]

    there_are_apples_in_ground = where_apples[0] or where_apples[1] or where_apples[2] # True or False

    w_1 = 0.05
    w_2 = 0.05
    w_3 = 0.05
    if there_are_apples_in_ground:
        return V[next_state]
    else:
        next_state_1 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[not wheres_agents[0][0] and not wheres_agents[1][0] and not wheres_agents[2][0], False, False], forced_pool=forced_pool)
        next_state_2 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[False, not wheres_agents[0][1] and not wheres_agents[1][1] and not wheres_agents[2][1], False], forced_pool=forced_pool)
        next_state_3 = new_state(agent, state_prima,  tabularRL, forced_agent_apples, forced_grass=[False, False, not wheres_agents[0][2] and not wheres_agents[1][2] and not wheres_agents[2][2]], forced_pool=forced_pool)

        return (1.0-w_1-w_2-w_3) * V[next_state] + w_1 * V[next_state_1] + w_2 * V[next_state_2] + w_3 * V[next_state_3]


def probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool, apples_in_ground, agent_gains=True, forcing_apples=False, original_state=-1):

    """
    AUXILIAR FUNCTION

    Auxiliar for the following method. This one is only used when in the transition,
    the real number of apples of the agent has changed.

    agent_gains = True if the real number of apples has increased, False if it has decreased

    Notice that the if the number of apples has not changed, this method should not even be called.

    """
    if agent_gains:
        next_agent_apples = SURVIVAL_THRESHOLD
        p = 1.0 / (SURVIVAL_THRESHOLD - 1.0)
    else:
        next_agent_apples = 0
        p = 1.0

    if forcing_apples:
        p *= 0.5

    probs = [1-p, p]
    agent_apples = [SURVIVAL_THRESHOLD - 1, next_agent_apples]

    probsV_total = 0

    for i in range(len(probs)):
        new_V = probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=agent_apples[i],
                                          apples_in_ground=apples_in_ground, forced_pool=forced_pool)

        addition = probs[i]*new_V
        probsV_total += addition


    return probsV_total


def probsV_on_agent_apples(mode, agent, action, Q_ag, V, original_state, state_prima, tabularRL, action_space, forced_pool_apples, apples_in_ground, forcing_next_apples=False):
    """
    AUXILIAR FUNCTION

    Computes the probability of moving from a state where the agent has several apples (but neither 0 nor TOO_MANY)
    from either a state where it has 0 apples (if the agent decides to donate)
    or a state where it has SURVIVAL_THRESHOLD (if the agent collects an apple)

    and in either case it computes the corresponding V(s')

    If the agent is in a different apple_state, of course the computation is trivial.

    """
    checks_agent_before = check_agent_apples_state(agent, original_state)

    if forcing_next_apples:

        V_A = probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=checks_agent_before,
                                          apples_in_ground=apples_in_ground, forced_pool=forced_pool_apples)

        if checks_agent_before != SURVIVAL_THRESHOLD - 1:
            V_B = probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_agent_apples=checks_agent_before+1,
                                             apples_in_ground=apples_in_ground, forced_pool=forced_pool_apples)
        else:
            V_B = probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground, agent_gains=True, forcing_apples=True, original_state=original_state)
        return 0.5* V_A + 0.5 * V_B


    checks_agent_after = check_agent_apples_state(agent, state_prima)
    if checks_agent_before == 0:
        if action == TAKE_DONATION:
            if checks_agent_after == 1:

                new_original = new_state(agent, original_state, True)
                apples_in_pool = check_common_pool(original_state)
                new_prima = new_state(agent, state_prima, True)


                best_next_action = np.argmax(scalarised_Qs(len(action_space), Q_ag[new_prima][action_space], [0.0, 1.0]))
                best_next_action2 = -1
                if apples_in_pool == 2:
                    new_prima2 = new_state(agent, state_prima, True, forced_pool=2)
                    best_next_action2 = np.argmax(scalarised_Qs(len(action_space), Q_ag[new_prima2][action_space], [0.0, 1.0]))

                if action_space[best_next_action] == DONATE or action_space[best_next_action2] == DONATE and mode == "lex":
                    retcon_V = probsV_on_apples_in_ground(agent, V, original_state, tabularRL,
                                                                          forced_agent_apples=checks_agent_before,
                                                                          forced_pool=forced_pool_apples,
                                                                          apples_in_ground=apples_in_ground)

                    return [-2.0, 0.0] + 0.8 * np.array([retcon_V[0], 0.0])
    elif 0 < checks_agent_before < SURVIVAL_THRESHOLD:  # If agent was in state s = few apples for itself

        if checks_agent_before != checks_agent_after:

            if checks_agent_before < checks_agent_after:  # +1 in apples
                agent_gains = True


                if action == TAKE_DONATION and mode == "lex":

                    new_original = new_state(agent, original_state, True)
                    new_prima = new_state(agent, state_prima, True, forced_agent_apples=SURVIVAL_THRESHOLD - 1)
                    best_next_action = np.argmax(scalarised_Qs(len(action_space), Q_ag[new_prima][action_space], [0.0, 1.0]))

                    if action_space[best_next_action] == DONATE:
                        retcon_V = probsV_on_apples_in_ground(agent, V, original_state, tabularRL, forced_agent_apples=checks_agent_before, forced_pool=forced_pool_apples,
                                                      apples_in_ground=apples_in_ground)

                        return [-2.0, 0.0] + 0.8*np.array([retcon_V[0], 0.0])

            else:                                         # -1 in apples
                agent_gains = False

                if action == DONATE:

                    new_prima = new_state(agent, state_prima, True, forced_agent_apples=SURVIVAL_THRESHOLD - 1)

                    retcon_V = probsV_on_apples_in_ground(agent, V, original_state, tabularRL, forced_agent_apples=checks_agent_before, forced_pool=forced_pool_apples,
                                                  apples_in_ground=apples_in_ground)

                    return [-1.0, 0.0] + 0.8*np.array([retcon_V[0], 0.0])

            return probsV_on_agent_apples_auxiliar(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground, agent_gains=agent_gains, original_state=original_state)

    elif checks_agent_before >= SURVIVAL_THRESHOLD:
        if action == TAKE_DONATION:
            if checks_agent_before < checks_agent_after:
                new_prima = new_state(agent, state_prima, True)

                if mode == "lex":

                    return [0.0, 0.7] + 0.8*np.array([probsV_on_apples_in_ground(agent, V, original_state, tabularRL, forced_agent_apples=checks_agent_before, forced_pool=forced_pool_apples,
                                                              apples_in_ground=apples_in_ground)[0], -1.0])

    return probsV_on_apples_in_ground(agent, V, state_prima, tabularRL, forced_pool=forced_pool_apples, apples_in_ground=apples_in_ground)


def probsV_apples_ground_and_agents(mode, agent, action, Q_ag, V, original_state, state_prima, tabularRL, action_space, forced_pool_apples=-1, forcing_next_apples=False):
    """
    AUXILIAR FUNCTION

    Fusion of the two previous methods. This one actually checks whether we are in a no-apples-in-ground state
    and then redirects to the corresponding method that needs to be applied.

    """
    checks1 = check_apples_state(original_state)
    checks2 = check_agents_where_apples(state_prima, agent)
    checks3 = check_agents_where_apples(original_state, abs(agent - 1))
    checks4 = check_agents_where_apples(original_state, agent)
    checks5 = check_agents_positions(original_state)
    checks6 = check_agents_positions(state_prima)

    return probsV_on_agent_apples(mode, agent, action, Q_ag, V, original_state, state_prima, tabularRL, action_space, forced_pool_apples=forced_pool_apples, apples_in_ground=[checks1, checks2, checks3, checks4, checks5, checks6], forcing_next_apples=forcing_next_apples)


def probsV_calculator(mode, agent, actions, Q_ag, V, original_state, state_prima, tabularRL, action_space):
    """

    We include in the previous method the checking of whether the common pool is in the state 2
    (i.e., more than 1 apple, but less than 24 apples) and we also check if the agents
    have decided to donate to the common pool.

    We compute the probability of the next state and its associated V(s').

    :return: the value of the next state V(s')
    """
    action = actions[agent]

    if DONATION_BOX_HAS_LIMIT:
        apples_in_pool = check_common_pool(original_state)

        if apples_in_pool == 2:
            if action == 8:
                # There are between 24 and two apples in the pool and the agent donated an apple
                # So 95% chance that the next state is the same
                p = 1.0 / (common_pool_states[-1] - 2.0)

                probs = [1 - p, p]
                pool_apples = [2, common_pool_states[-1]]

                probsV_total = 0

                for i in range(len(probs)):
                    probsV_total += probs[i] * probsV_apples_ground_and_agents(mode, agent, actions[agent], Q_ag, V, original_state, state_prima, tabularRL, action_space,
                                                                               forced_pool_apples=pool_apples[i])

                return probsV_total
            elif action == 9:
                # There are between 24 and two apples in the pool and the agent took an apple from the pool
                # So 95% chance that the next state is the same
                p = 1.0 / (common_pool_states[-1] - 2.0)

                probs = [1 - p, p]
                pool_apples = [2, 1]

                probsV_total = 0

                for i in range(len(probs)):
                    probsV_total += probs[i] * probsV_apples_ground_and_agents(mode, agent, actions[agent], Q_ag, V, original_state, state_prima, tabularRL, action_space,
                                                                               forced_pool_apples=pool_apples[i])

                return probsV_total

    both_agents_took_when_pool_1 = check_random_reward(original_state, actions)

    return probsV_apples_ground_and_agents(mode, agent, actions[agent], Q_ag, V, original_state, state_prima, tabularRL, action_space, forcing_next_apples=both_agents_took_when_pool_1)



def create_model():
    """

    Creates an abstract model of the Ethical Gathering Game to make computations with Value Iteration
    faster.

    :return: a table in format .npy that for each state-action pair returns the next state and
             the reward that each agent receives
    """

    state_count = 0
    environment = gym.make('CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map, visualRadius=3,
                           fullState=False, tabularState=True)

    total_action_space = [i for i in range(environment.action_space.n)]
    action_space = new_action_space(total_action_space, environment)
    env_size = environment.mapWidth*(environment.mapHeight-2)

    model = np.zeros((len_state_space, environment.action_space.n, environment.action_space.n, env_size + 2*number_of_agents + 1))
    total_states = len(agent_positions)*len(apple_states)*len(agent_num_apples)*len(common_pool_states)
    print("States in total: ", total_states)
    for ag_pos in agent_positions:

        for ap_state in apple_states:


            if check_redundant_states(positions_with_apples, ag_pos, ap_state):
                continue

            for n_apples in agent_num_apples:


                    for c_state in common_pool_states:




                        state_count += 1

                        if state_count % 100 == 0:
                            print(state_count, " / ", total_states)


                        for action in action_space:

                            for action2 in action_space:
                                env = gym.make('CommonsGame:CommonsGame-v0', numAgents=number_of_agents, mapSketch=current_map,
                                               visualRadius=3, fullState=False, tabularState=True, agent_pos=ag_pos)
                                original_state = env.reset(num_apples=[n_apples], common_pool=c_state, apples_yes_or_not=ap_state)

                                those = check_apples_state(original_state[0])

                                if ap_state[0] != those[0] or ap_state[1] != those[1] or ap_state[2] != those[2]:
                                    print("------------------")
                                    print()
                                    print("Fatal error: you need to set CREATING_MODEL = True in the constants.py file")
                                    print()
                                    print("------------------")
                                    return -1

                                state = new_state(0, original_state[0], True)
                                actions = [action, action2]

                                nObservations, nRewards, _, _ = env.step(actions)

                                for i in range(len(positions_with_apples)):
                                    if nObservations[0][(positions_with_apples[i][0]-2)*(environment.mapHeight-2)+positions_with_apples[i][1]] == 64 and not ap_state[i]:
                                        nObservations[0][(positions_with_apples[i][0]-2)*(environment.mapHeight-2)+positions_with_apples[i][1]] = 32

                                # Donation box state
                                model[state][action][action2][-5] = nObservations[0][-1]

                                # Agents rewards
                                for i in range(2):
                                    model[state][action][action2][-4+i] = nRewards[0][i] # rewards agent 1
                                    model[state][action][action2][-2+i] = nRewards[1][i] # rewards agent 2

    print("huuhh??")
    np.save("model_SMALL.npy", model)

