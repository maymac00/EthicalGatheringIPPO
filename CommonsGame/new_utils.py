import numpy as np
import itertools
from CommonsGame.constants import SURVIVAL_THRESHOLD, DONATION_BOX_HAS_LIMIT, DONATION_BOX_CAPACITY, MAP_NAME, smallMap, tinyMap

policy_folder = "CommonsGame/policies/"
policy_NULL = np.load(policy_folder+"policy_NULL.npy")
try:
    policy_0 = np.load(policy_folder+"ref_policy0_C5.npy")
    policy_1 = np.load(policy_folder+"ref_policy1_C5.npy")
except:
    policy_0 = policy_NULL
    policy_1 = policy_NULL

reference_policy = [policy_0, policy_1]

number_of_agents = 2
number_of_objectives = 2

only_ethical_matters = [0.0, 1.0]
only_individual_matters = [1.0, 0.0]

agent_positions = list()

# for tinyMap

if MAP_NAME == "tiny":
   current_map = tinyMap
elif MAP_NAME == "small":
    current_map = smallMap

agent_x_position = len(current_map) - 2
agent_y_position = len(current_map[0])

# (agent_x_position, agent_y_position)

for x in range(2, 2 + agent_x_position):
    for y in range(agent_y_position):
        agent_positions.append([x, y])


two_agent_positions = list()

for ag_pos in agent_positions:
    for ag_pos2 in agent_positions:
        two_agent_positions.append([ag_pos, ag_pos2])


if number_of_agents == 2:
    agent_positions = two_agent_positions

# for tinyMap
if MAP_NAME == "tiny":
    positions_with_apples = [[2, 1], [2, 2], [3, 1]]
else:
    positions_with_apples = [[3, 2], [3, 3], [4, 2]]
apple_indices = []
for pos in positions_with_apples:
    apple_indices.append(agent_y_position*(pos[0]-2)+pos[1])

# print(apple_indices)
agent_num_apples = [0, SURVIVAL_THRESHOLD - 1, SURVIVAL_THRESHOLD, SURVIVAL_THRESHOLD + 1]
if DONATION_BOX_HAS_LIMIT:
    common_pool_states = [0, 1, 2, DONATION_BOX_CAPACITY]
else:
    common_pool_states = [0, 1, 2]

apple_possibilities = [False, True]
apple_states = [list(i) for i in itertools.product(apple_possibilities, repeat=3)]


n_agent_cells = len(agent_positions) if number_of_agents == 1 else len(agent_positions)
n_apples = len(agent_num_apples)
apples_in_ground = len(apple_states)
common_pool_max = len(common_pool_states)

len_state_space = n_agent_cells*n_apples*apples_in_ground*common_pool_max + 1  # You need to change this!!! This is provisional and only works for tinyMap


def new_action_space(action_space, env):
    """
        Modify this method if you want to limit the action space of the agents. This method
        is specially important if you are using Tabular RL

    :param action_space: a list of integers
    :return: a new list of integers, smaller or equal
    """

    action_space.remove(env.SHOOT)
    action_space.remove(env.TURN_CLOCKWISE)
    action_space.remove(env.TURN_COUNTERCLOCKWISE)


    return action_space


def new_state(agent, state, tabularRL, forced_agent_apples=-1, forced_grass=[False for _ in range(100)], forced_ag_pos=[], forced_pool=-1):
    """
    Modify the state (if you are using Tabular RL) to simplify it so it can be useful to the agent
    Ideally you will be able to create a map from every state to a different integer number
    :param agent: an integer to know which agent it is
    :param state: a list of integers
    :param tabularRL: boolean to know if you are using tabularRL or deep RL
    :return: a new list of integers, smaller or equal
    """
    #
    if tabularRL:

        if len(state) == 0:
            return len_state_space
        else:
            # Obtain the agent's position:

            agents_x = list()
            agents_y = list()

            if len(forced_ag_pos) > 0:

                for ag in range(number_of_agents):
                    agents_x.append(forced_ag_pos[ag][0] - 2)
                    agents_y.append(forced_ag_pos[ag][1])
            else:
                for ag in range(number_of_agents):
                    agents_x.append(state[-1 - 4 * number_of_agents + 4 * ag])
                    agents_y.append(state[-4 * number_of_agents + 4 * ag])

            position = 0

            for ag in range(number_of_agents):
                position += (agents_x[ag] + agent_x_position*agents_y[ag])*(agent_x_position*agent_y_position)**ag   # we encode them as a scalar, there are 16 different positions

            # Obtain the agent's amount of apples, but we only consider four possible values
            if forced_agent_apples > -1:
                agent_temp_apples = forced_agent_apples
            else:
                agent_temp_apples = state[1 - 4 * number_of_agents + 4 * agent]
            if agent_temp_apples == 0:
                agent_apples = 0
            elif agent_temp_apples < SURVIVAL_THRESHOLD:
                agent_apples = 1
            elif agent_temp_apples == SURVIVAL_THRESHOLD:
                agent_apples = 2
            else:
                agent_apples = 3

            if forced_pool >= 0:

                if forced_pool < common_pool_states[-1] or not DONATION_BOX_HAS_LIMIT:
                    common_pool_apples = min(forced_pool, 2)
                else:
                    common_pool_apples = 3
            else:
                real_common_pool = state[-1]

                if real_common_pool < common_pool_states[-1] or not DONATION_BOX_HAS_LIMIT:
                    common_pool_apples = min(real_common_pool, 2)

                else:
                    common_pool_apples = 3

            apple_state_list = []
            for i in range(len(apple_indices)):
                if forced_grass[i]:
                    apple_state_list.append(1)
                else:
                    apple_state_list.append(int(state[apple_indices[i]] == 64))

            where_apples = 0
            for i in range(len(apple_indices)):
                where_apples += apple_state_list[i]*2**i

            position_and_apples = position + n_agent_cells*(agent_apples + n_apples*(where_apples + apples_in_ground*common_pool_apples))

            return int(position_and_apples)
    else:
        return state


def check_agents_where_apples(state, agent):

    ags_pos = check_agents_positions(state)
    apple_state_i = []

    if ags_pos[0][0] == ags_pos[1][0] and ags_pos[0][1] == ags_pos[1][1]:
        for i in range(len(apple_indices)):
            apple_state_i.append(int(state[apple_indices[i]] > 64))
    else:
        for i in range(len(apple_indices)):
            apple_state_i.append(int(state[apple_indices[i]] == 64 + 1 + agent))

    return apple_state_i


def get_interesting_data(state):

    agents_apples = np.zeros(number_of_agents)
    agents_donated_apples = np.zeros(number_of_agents)
    for ag in range(number_of_agents):
        agents_apples[ag] = state[1 - 4 * number_of_agents + 4 * ag]
        agents_donated_apples[ag] = state[1 - 4 * number_of_agents + 4 * ag + 1]

    common_pool_apples = state[-1]

    return agents_apples, agents_donated_apples, common_pool_apples


def check_redundant_states(positions_with_apples, agents_positions, ap_state):

    for n in range(len(positions_with_apples)):
        for ag_pos in agents_positions:
            if ag_pos == positions_with_apples[n]:
                if ap_state[n]:
                    return True

    return False

def check_agents_positions(state):
    ag_0_pos = state[-1 - 4 * number_of_agents + 4 * 0: +1 - 4 * number_of_agents + 4 * 0]
    ag_1_pos = state[-1 - 4 * number_of_agents + 4 * 1: +1 - 4 * number_of_agents + 4 * 1]

    return ag_0_pos, ag_1_pos

def check_apples_state(state):
    apple_state_i = []
    for i in range(len(apple_indices)):
        apple_state_i.append(state[apple_indices[i]] == 64)

    return apple_state_i

def check_what_in_apples_state(state):
    apple_state_i = []
    for i in range(len(apple_indices)):
        apple_state_i.append(state[apple_indices[i]])

    return apple_state_i


def check_agent_apples_state(agent, state):
    # Obtain the agent's real amount of apples
    return state[1 - 4 * number_of_agents + 4 * agent]


def check_common_pool(state):
    return state[-1]


def check_random_reward(state, actions):

    everyone_took_donation = True
    single_apple_in_pool = False

    if check_common_pool(state) == 1:
        single_apple_in_pool = True

    for action in actions:
        if action == 9:
            everyone_took_donation *= True
        else:
            everyone_took_donation *= False

    if everyone_took_donation and single_apple_in_pool:
        return True
    else:
        return False

def scalarisation_function(values, w):
    """
    Scalarises the value of a state using a linear scalarisation function

    :param values: the different components V_0(s), ..., V_n(s) of the value of the state
    :param w:  the weight vector of the scalarisation function
    :return:  V(s), the scalarised value of the state
    """

    f = 0
    for objective in range(len(values)):
        f += w[objective]*values[objective]

    return f


def scalarised_Qs(len_action_space, Q_state, w):
    """
    Scalarises the value of each Q(s,a) for a given state using a linear scalarisation function

    :param Q_state: the different Q(s,a) for the state s, each with several components
    :param w: the weight vector of the scalarisation function
    :return: the scalarised value of each Q(s,a)
    """

    scalarised_Q = np.zeros(len_action_space)
    for action in range(len(Q_state)):
        scalarised_Q[action] = scalarisation_function(Q_state[action], w)

    return scalarised_Q


def lexicographic_Qs(action_space, Q_state):

    chosen_action = -1

    best_ethical_Q = np.max(scalarised_Qs(len(action_space), Q_state, only_ethical_matters))
    best_individual_Q = -np.inf

    for action in range(len(action_space)):
        q_Individual = scalarisation_function(Q_state[action], only_individual_matters)
        q_Ethical = scalarisation_function(Q_state[action], only_ethical_matters)
        if best_ethical_Q - q_Ethical <= 0.00001:
            if q_Individual > best_individual_Q:
                best_individual_Q = q_Individual
                chosen_action = action

    #print(chosen_action, best_individual_Q, best_ethical_Q, Q_state)
    #print("----")

    return best_individual_Q , best_ethical_Q, action_space[chosen_action]


def evaluation(env, tabularRL, policies=0, we_render=True, it=0, how_much=400, randomness=False):
    initial_state = env.reset(num_apples=[0, 0], common_pool=0)

    ags_apples_evolution = [[], []]
    ags_donation_evolution = [[], []]
    pool_evolution = []
    next_random = False

    states = list()
    for ag in range(number_of_agents):
        states.append(new_state(ag, initial_state[ag], tabularRL))

    if policies == 0:
        policies = list()
        policies.append(policy_0)
        policies.append(policy_1)

    previous_agent_apples = [0, 0]
    previous_state = states
    num_modifier_policy = 0
    for t in range(how_much):

        if we_render:
            env.render()

        actions = [policies[0][states[0]], policies[1][states[1]]]

        if randomness:
            if next_random:
                actions = [np.random.randint(0, 4), np.random.randint(0, 4)]
                next_random = False

        nObservations, rewards, nDone, _ = env.step(actions)


        states = list()
        for ag in range(number_of_agents):
            states.append(new_state(ag, nObservations[ag], tabularRL))

        if randomness:
            # Little helper in case that agents get stuck. There are many states and maybe agents miss training on some.
            # In case that this function is used a lot, probably you need to train your agents more.

            if states[0] == previous_state[0] and states[1] == previous_state[1]:
                apples_available = check_what_in_apples_state(nObservations[0])
                if apples_available[0] > 63 and apples_available[1] > 63 and apples_available[2] > 63:
                    next_random = True
                    print(nObservations[0])
                    num_modifier_policy += 1
        previous_state = states

        agents_apples, agents_donated_apples, common_pool_apples = get_interesting_data(nObservations[0])

        for ag in range(number_of_agents):
            ags_apples_evolution[ag].append(agents_apples[ag])

            if len(ags_donation_evolution[ag]) > 0:
                donation_appendable = ags_donation_evolution[ag][-1]
            else:
                donation_appendable = 0

            if previous_agent_apples[ag] > agents_apples[ag]:
                donation_appendable += 1
            elif actions[ag] == 9 and previous_agent_apples[ag] < agents_apples[ag]:
                donation_appendable += -1
            ags_donation_evolution[ag].append(donation_appendable)
        pool_evolution.append(min(DONATION_BOX_CAPACITY, common_pool_apples))

        previous_agent_apples = agents_apples


    for ag in range(number_of_agents):
        np.save("apple_evo"+str(ag)+"i"+str(it)+".npy", ags_apples_evolution[ag])
        np.save("apple_donation" + str(ag) + "i" + str(it) + ".npy", ags_donation_evolution[ag])
    np.save("pool_evoi"+str(it)+".npy", pool_evolution)

    print("Ohhh", num_modifier_policy)
    return True




def policy_creator(Q_function, action_space, mode="scalarisation", weights=[1.0 , 0.0]):

    policy = np.zeros(len_state_space)

    for state in range(len_state_space):

        if mode == "scalarisation":

            index = np.argmax(scalarised_Qs(len(action_space), Q_function[state][action_space], weights))

            policy[state] = action_space[index]
        elif mode == "lex":
            _, _, policy[state] = lexicographic_Qs(action_space, Q_function[state][action_space])
        else:
            policy[state] = 6  # env.STAY


    return policy




