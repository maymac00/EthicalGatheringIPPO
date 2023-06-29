import time
import numpy as np
import gym
from CommonsGame.constants import tinyMap, smallMap, bigMap

import warnings
warnings.filterwarnings("ignore")

numAgents = 3

env = gym.make('CommonsGame-v0', numAgents=numAgents, visualRadius=2, mapSketch=bigMap,
               fullState=False, agent_pos=[[2, 0], [3, 4], [2, 2]])
initial_state = env.reset(num_apples=[8, 3, 5], common_pool=2, apples_yes_or_not=[True, True, True, True])

print(env.observation_space)

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


print(initial_state[0])


for t in range(100):
    print("--Time step", t, "--")
    nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    # nActions = [6, 6, 6]
    for i in range(numAgents):
        if nActions[i] == 7:
            nActions[i] = 6

    nObservations, nRewards, nDone, nInfo = env.step(nActions)

    env.render()
    time.sleep(0.5)

common_apples = 0
for n, agent in enumerate(env.get_agents()):
    print("Agent")
    print("Agent " + str(n) + " possessions : " + str(agent.has_apples))
    print("Agent " + str(n) + " donations : " + str(agent.donated_apples))
    print("Agent " + str(n) + "'s efficiency : " + str(agent.efficiency))
    common_apples += agent.donated_apples

    print("--")

print("Total common apples : ", common_apples)
