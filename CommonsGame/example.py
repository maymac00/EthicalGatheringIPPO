import argparse
import time

import numpy as np
import gym
from CommonsGame.constants import tinyMap, smallMap, bigMap
import CommonsGame.constants as constants
import numpy as np


# parameters
n_agents = 3
survival_threshold = 15
time_steps = 500
render = True
simulations = 1

chars = [a for a in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']

bigMap = np.array(bigMap)
bigMap = bigMap[:, :16]
map2use = bigMap
map_empty = np.array(map2use)

apple_positions = np.array(np.where(map_empty == '@')).T
agent_positions = []
available_spots = np.argwhere(np.logical_and(map_empty != '@', map_empty != '='))

# Mantenim el ratio (generacio de pomes per step/ numero d'agents)
constants.REGENERATION_PROBABILITY = 0.005
constants.DONATION_BOX_CAPACITY = 10
constants.SURVIVAL_THRESHOLD = survival_threshold

MOVE_UP = 0
MOVE_DOWN = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3

TURN_CLOCKWISE = 4
TURN_COUNTERCLOCKWISE = 5
STAY = 6
SHOOT = 7
DONATE = 8
TAKE_DONATION = 9


def greedy_agent(gridmap, apples, db, char, random_move_prob=0.1):
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_agent_and_apples(gridmap):
        agent_pos = None
        apple_positions = []

        for i in range(gridmap.shape[0]):
            for j in range(gridmap.shape[1]):
                if gridmap[i, j] == char:
                    agent_pos = (i, j)
                elif gridmap[i, j] == '@':
                    apple_positions.append((i, j))

        return agent_pos, apple_positions

    def find_closest_agent(gridmap, agent_position):
        min_distance = np.infty
        neighbor = None
        for i in range(gridmap.shape[0]):
            for j in range(gridmap.shape[1]):
                if gridmap[i, j] in chars and gridmap[i, j] != char:
                    dist = manhattan_distance([i, j], agent_position)
                    if dist < min_distance:
                        min_distance = dist
                        neighbor = [i, j]
        return neighbor

    def find_closest_apple(agent_pos, apple_positions):
        closest_apple = None
        min_distance = float('inf')

        for apple_pos in apple_positions:
            distance = manhattan_distance(agent_pos, apple_pos)
            if distance < min_distance:
                min_distance = distance
                closest_apple = apple_pos

        return closest_apple

    def move_agent(gridmap, agent_pos, closest_apple):
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        best_move = None
        min_distance = float('inf')

        for move in moves:
            new_pos = (agent_pos[0] + move[0], agent_pos[1] + move[1])
            if 0 <= new_pos[0] < gridmap.shape[0] and 0 <= new_pos[1] < gridmap.shape[1]:
                distance = manhattan_distance(new_pos, closest_apple)
                if distance < min_distance:
                    min_distance = distance
                    best_move = move

        # make random move with a small probability
        if np.random.rand() < random_move_prob:
            best_move = moves[np.random.choice(len(moves))]
        return best_move

    def direction_to_escape(A, B):
        xA, yA = A
        xB, yB = B

        dx, dy = xA - xB, yA - yB

        if abs(dx) > abs(dy):  # Choose horizontal direction
            return (1, 0) if dx > 0 else (-1, 0)
        else:  # Choose vertical direction
            return (0, 1) if dy > 0 else (0, -1)

    def direction_to_escape_walls(gridmap, agent_position):
        min_distance = np.infty
        wall = None
        for i in range(gridmap.shape[0]):
            for j in range(gridmap.shape[1]):
                if gridmap[i, j] == "=":
                    dist = manhattan_distance([i, j], agent_position)
                    if dist < min_distance:
                        min_distance = dist
                        wall = [i, j]
        if wall is None:
            return None
        return direction_to_escape(agent_position, wall)

    def move2action(move):
        if move == (0, 1):
            return MOVE_RIGHT
        elif move == (1, 0):
            return MOVE_DOWN
        elif move == (0, -1):
            return MOVE_LEFT
        elif move == (-1, 0):
            return MOVE_UP
        return STAY

    if apples > constants.SURVIVAL_THRESHOLD and db < constants.DONATION_BOX_CAPACITY:
        return DONATE
    if apples < constants.SURVIVAL_THRESHOLD and db > 0:
        return TAKE_DONATION

    agent_pos, apple_positions = find_agent_and_apples(gridmap)

    # Agent over agent
    if agent_pos is None:
        # Agent not found
        return np.random.choice([0, 1, 2, 3])

    # no apples.
    if len(apple_positions) == 0:
        # if 0 apples on sight, run away from neighbor if possible
        neighbor = find_closest_agent(gridmap, agent_pos)
        if neighbor is None:
            # Random or avoid walls
            dir_walls = direction_to_escape_walls(gridmap, agent_pos)
            return move2action(dir_walls) if dir_walls is not None else np.random.choice([0, 1, 2, 3])
        else:
            # Run away
            return move2action(direction_to_escape(agent_pos, neighbor))

    closest_apple = find_closest_apple(agent_pos, apple_positions)
    move = move_agent(gridmap, agent_pos, closest_apple)
    return move2action(move)


for ag in range(n_agents):
    agent_positions.append(available_spots[np.random.choice(available_spots.shape[0])])

env = gym.make('CommonsGame-v0', numAgents=n_agents, visualRadius=3, mapSketch=map2use,
               fullState=True, agent_pos=agent_positions[:n_agents], inequality_mode="loss")
initial_state = env.reset()

# print(initial_state[0])


apple_history = np.zeros((simulations, time_steps, n_agents))
donation_box_history = np.zeros((simulations, time_steps))
rewards = np.zeros((simulations, time_steps, n_agents))
for sim in range(simulations):
    nInfo = {}
    nInfo["n"] = [0] * n_agents
    nInfo["donationBox"] = 0
    for t in range(time_steps):
        # print("--Time step", t, "--")
        # nActions = np.random.randint(low=0, high=env.action_space.n, size=(n_agents,)).tolist()
        nActions = np.zeros((n_agents,))
        # nActions = [6, 6, 6]
        for i in range(n_agents):
            nActions[i] = greedy_agent(env.getBoard(partiall_observability=True, ag=i), nInfo["n"][i],
                                       nInfo["donationBox"], chars[i])
            num_donation = list(nActions).count(TAKE_DONATION)
            if num_donation > nInfo["donationBox"]:
                nActions[np.random.choice(np.where(nActions == TAKE_DONATION)[0])] = STAY

        nObservations, nRewards, nDone, nInfo = env.step(nActions)
        rewards[sim, t] = nRewards
        apple_history[sim, t] = np.array(nInfo['n'])
        donation_box_history[sim, t] = nInfo["donationBox"]
        if sim == simulations - 1:
            if render:
                env.render()
                time.sleep(0.01)
    if sim < simulations - 1:
        env.reset()

print("Mean reward per episode: ", rewards.sum(axis=1).mean(0))
apple_history = np.swapaxes(apple_history, 0, 1)
apple_history = np.swapaxes(apple_history, 1, 2)
donation_box_history = np.swapaxes(donation_box_history, 0, 1)

from experiment_utils import graphical_evaluation_fig

args = argparse.Namespace()
config_data = {
    "n_agents": n_agents,
    "max_steps": 500
}
for key, value in config_data.items():
    setattr(args, key, value)
graphical_evaluation_fig(apple_history, donation_box_history, args, False, type="median")
