import random

import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from pycolab import ascii_art
from CommonsGame.utils import buildMap, ObservationToArrayWithRGB
from CommonsGame.objects import PlayerSprite, AppleDrape, SightDrape, ShotDrape
import CommonsGame.constants as constants
import importlib


class CommonsGame(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

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

    def __init__(self, n_agents, map_size=None, tabularState=True, agent_pos=[],
                 we=0.0, normalized_obs=True, inequality_mode="tie_break", max_steps=500, apple_regen=0.05,
                 past_actions_memory=0, visual_radius=2, partial_observability=False, donation_capacity=10,
                 survival_threshold=10):
        super(CommonsGame, self).__init__()
        self.fullState = not partial_observability
        self.max_steps = max_steps
        self.step_count = 0
        # Setup spaces
        self.action_space = spaces.Discrete(10)
        obHeight = obWidth = visual_radius * 2 + 1
        # Setup game
        self.numAgents = n_agents
        self.sightRadius = visual_radius
        self.agentChars = agentChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[0:n_agents]

        self.tabularState = tabularState
        self.common_pool = True
        self.agent_pos = agent_pos
        self.weight_vector = np.array([1, we])
        self.normalized_obs = normalized_obs
        self.currentStateInfo = {"apples": np.zeros(n_agents), "donationBox": 0}
        self.inequality_mode = inequality_mode
        # if tabularState:
        #    fullState = True
        #    self.fullState = True

        constants.DONATION_BOX_CAPACITY = donation_capacity
        constants.SURVIVAL_THRESHOLD = survival_threshold

        # Load map
        if map_size == 'tiny':
            self.map2use = constants.tinyMap
        elif map_size == 'small':
            self.map2use = constants.smallMap
        elif map_size == 'medium':
            bigMap = np.array(constants.bigMap)
            self.map2use = bigMap[:, :16]
        else:
            self.map2use = constants.bigMap

        self.mapHeight = len(self.map2use)
        self.mapWidth = len(self.map2use[0])

        # Untracked parameters
        constants.REGENERATION_PROBABILITY = apple_regen
        self.inequality_mode = inequality_mode
        self.past_actions_memory = past_actions_memory
        self.visual_radius = visual_radius
        self.partial_observability = partial_observability
        pass

        if self.fullState:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.mapHeight + 2, self.mapWidth + 2, 3),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(obHeight, obWidth, 3), dtype=np.uint8)
        self.numPadPixels = numPadPixels = visual_radius - 1

        self.gameField = buildMap(self.map2use, numPadPixels=numPadPixels, agentChars=agentChars,
                                  mandatory_initial_position=self.agent_pos)

        self.state = None
        self.sick_probabilities = np.random.choice(100, n_agents)
        self.efficiency_probabilites = np.random.randint(1, 5, n_agents)

        # Pycolab related setup:
        self._game = self.buildGame()
        # Awful workaround to make serializing work
        self._game.backdrop.__dict__['_p_a_l_e_t_t_e'] = self._game.backdrop.__dict__['_p_a_l_e_t_t_e'].__dict__
        colourMap = dict([(a, (999 - 4 * i, 0, 4 * i)) for i, a in enumerate(agentChars)]  # Agents
                         + [('=', (705, 705, 705))]  # Steel Impassable wall
                         + [(' ', (0, 0, 0))]  # Black background
                         + [('@', (0, 999, 0))]  # Green Apples
                         + [('.', (750, 750, 0))]  # Yellow beam
                         + [('-', (0, 0, 0))])  # Grey scope
        self.obToImage = ObservationToArrayWithRGB(colour_mapping=colourMap)
        # importlib.reload(constants)

    def seed(self, chosen_seed):
        np.random.seed(chosen_seed)

    def buildGame(self, apples_yes_or_not=[True, True, True]):
        agentsOrder = list(self.agentChars)
        random.shuffle(agentsOrder)

        return ascii_art.ascii_art_to_game(
            self.gameField,
            what_lies_beneath=' ',
            sprites=dict(
                [(a, ascii_art.Partial(PlayerSprite, self.agentChars, self.sightRadius, self.agent_pos,
                                       inequality_mode=self.inequality_mode)) for a in
                 self.agentChars]),
            drapes={'@': ascii_art.Partial(AppleDrape, self.agentChars, self.numPadPixels, apples_yes_or_not),
                    '-': ascii_art.Partial(SightDrape, self.agentChars, self.numPadPixels),
                    '.': ascii_art.Partial(ShotDrape, self.agentChars, self.numPadPixels)},
            # update_schedule=['.'] + agentsOrder + ['-'] + ['@'],
            update_schedule=['.'] + agentsOrder + ['-'] + ['@'],
            z_order=['-'] + ['@'] + agentsOrder + ['.']
        )

    def step(self, nActions):
        nInfo = {}
        self.state, nRewards, _ = self._game.play(nActions)

        scalar_rewards = np.zeros(self.numAgents)

        for i in range(self.numAgents):
            scalar_rewards[i] = np.dot(nRewards[i], self.weight_vector)

        nObservations, done = self.getObservation()
        nDone = [done] * self.numAgents
        nInfo['n'] = self.currentStateInfo["apples"]
        nInfo['donationBox'] = self.currentStateInfo["donationBox"]
        nInfo['survival'] = all(
            [self.currentStateInfo["apples"][i] >= constants.SURVIVAL_THRESHOLD for i in range(self.numAgents)])
        nInfo['donationBox_full'] = self.currentStateInfo["donationBox"] >= constants.DONATION_BOX_CAPACITY
        return nObservations, scalar_rewards, nDone, nInfo

    def reset(self, num_apples=[0], common_pool=0, apples_yes_or_not=[True, True, True]):
        # Reset the state of the environment to an initial state
        self._game = self.buildGame(apples_yes_or_not)
        ags = [self._game.things[c] for c in self.agentChars]

        for i, a in enumerate(ags):
            a.set_sickness(self.sick_probabilities[i])
            # a.set_efficiency(self.efficiency_probabilites[i]) # TODO: Not random efficiency
            if len(num_apples) == 1:
                a.set_init_apples(num_apples[0])  # all agents have the same amount of apples, why not?
            else:
                a.set_init_apples(num_apples[i])
        self._game.things['@'].common_pool = common_pool
        self.step_count = 0
        self.state, _, _ = self._game.its_showtime()
        nObservations, _ = self.getObservation()
        # Awful workaround to make serializing work
        self._game.backdrop.__dict__['_p_a_l_e_t_t_e'] = self._game.backdrop.__dict__['_p_a_l_e_t_t_e'].__dict__
        return nObservations

    def getBoard(self, partiall_observability=False, ag=0):
        if not partiall_observability:
            matrix = self.state.board.view('c').astype('str')[3 + self.numPadPixels:]
            # Remove rows where all elements are '='
            matrix = matrix[~np.all(matrix == '=', axis=1)]
            # Remove columns where all elements are '='
            matrix = matrix[:, ~np.all(matrix == '=', axis=0)]
            return matrix
        else:
            current_board = self.getBoard()
            size = self.sightRadius*2+1
            view = []
            ag = self.agentChars[ag]
            location = np.where(current_board == ag)
            # if not found, skip
            if len(location[0]) == 0:
                return np.zeros((size, size))
            i, j = location[0][0], location[1][0]
            for dx in range(-self.sightRadius, self.sightRadius + 1):
                for dy in range(-self.sightRadius, self.sightRadius + 1):
                    # Ensure the new position is within the matrix
                    if 0 <= i + dx < current_board.shape[0] and 0 <= j + dy < current_board.shape[1]:
                        view.append(current_board[i + dx, j + dy])
                    else:
                        # append pading
                        view.append('=')
            view = np.array(view).reshape((size, size))
            return view

    def getPlotText(self):
        ags = [self._game.things[c] for c in self.agentChars]
        plot_text = ""
        for i, agent in enumerate(ags):
            plot_text += "Agent " + str(i) + ": " + str(agent.has_apples) + ", "
        plot_text += "Donation box: " + str(self._game.things['@'].common_pool)
        return plot_text

    def render(self, mode='human', close=False, masked=False):
        # Render the environment to the screen
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]

        if masked:
            current_board = self.getBoard()
            unmasked_board = board.copy()
            mask = np.zeros((current_board.shape[0], current_board.shape[1]))
            for ag in self.agentChars:
                location = np.where(current_board == ag)
                # if not found, skip
                if len(location[0]) == 0:
                    continue
                i, j = location[0][0], location[1][0]
                i, j = location[0][0], location[1][0]
                for dx in range(-self.sightRadius, self.sightRadius + 1):
                    for dy in range(-self.sightRadius, self.sightRadius + 1):
                        # Ensure the new position is within the matrix
                        if 0 <= i + dx < current_board.shape[0] and 0 <= j + dy < current_board.shape[1]:
                            mask[i + dx, j + dy] = 1

            no_see = np.argwhere(mask == 0)
            # from board put no_see to gray
            for i in range(len(no_see)):
                board[no_see[i][0] + 3, no_see[i][1] + 1, :] = [128, 128, 128]
            # plot both masked and unmasked as subplots
            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(unmasked_board)
            plt.axis("off")
            plot_text = self.getPlotText()
            plt.title(plot_text)
            plt.subplot(1, 2, 2)
            plt.imshow(board)
            plt.axis("off")
            plt.title("Masked")
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()
        else:
            plt.figure(1)
            plt.imshow(board)
            plt.axis("off")

            plot_text = self.getPlotText()
            plt.title(plot_text)
            plt.show(block=False)
            # plt.show()
            plt.pause(0.05)
            plt.clf()

    def originalGetObservation(self):
        done = not (np.logical_or.reduce(self.state.layers['@'], axis=None))
        ags = [self._game.things[c] for c in self.agentChars]
        obs = []
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        for a in ags:
            if a.visible or a.timeout == 25:
                if self.fullState:
                    ob = np.copy(board)
                    if a.visible:
                        ob[a.position[0], a.position[1], :] = [0, 0, 255]
                    ob = ob[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                         self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
                else:
                    ob = np.copy(board[
                                 a.position[0] - self.sightRadius:a.position[0] + self.sightRadius + 1,
                                 a.position[1] - self.sightRadius:a.position[1] + self.sightRadius + 1, :])
                    if a.visible:
                        ob[self.sightRadius, self.sightRadius, :] = [0, 0, 255]
                ob = ob / 255.0
            else:
                ob = None
            obs.append(ob)
        return obs, done

    def normalizeObs(self, new_ob, a, common_apples):
        # Normalize
        new_ob = new_ob.astype(np.float32)
        new_ob[new_ob == 61] = 1
        new_ob[new_ob == 32] = 0
        new_ob[new_ob == 45] = 0
        new_ob[new_ob == 64] = 0.33
        new_ob[new_ob > 64] = 0.66

        new_ob = np.append(new_ob, [a.has_apples, common_apples])

        if new_ob[-2] == 0:
            new_ob[-2] = 0
        elif new_ob[-2] < constants.SURVIVAL_THRESHOLD:
            new_ob[-2] = 1
        elif new_ob[-2] == constants.SURVIVAL_THRESHOLD:
            new_ob[-2] = 2
        elif new_ob[-2] > constants.SURVIVAL_THRESHOLD:
            new_ob[-2] = 3

        # common_pool_states = list(range(self.n_agents + 1)) + [constants.DONATION_BOX_CAPACITY]
        common_pool_states = [0, 1, 2, constants.DONATION_BOX_CAPACITY]
        if new_ob[-1] < common_pool_states[-1]:
            new_ob[-1] = min(new_ob[-1], 2)

        else:
            new_ob[-1] = 3
        new_ob[-1] /= 3
        new_ob[-2] /= 3
        return new_ob

    def getObservation(self):
        # Solo si nos interesa la sostenibilidad el entorno llega a un estado final cuando deja de haber manzanas
        # en el suelo. Otherwise, no hay estados finales.

        if constants.SUSTAINABILITY_MATTERS:
            done = not (np.logical_or.reduce(self.state.layers['@'][self.sightRadius + 2:, :], axis=None))
        else:
            done = False

        if self.step_count >= self.max_steps:
            done = True

        ags = [self._game.things[c] for c in self.agentChars]
        obs = []

        addedRowsForAgentInfo = 2

        new_state = self.state.board[self.sightRadius + addedRowsForAgentInfo:-self.sightRadius,
                    self.sightRadius:-self.sightRadius]
        common_apples = self._game.things['@'].common_pool
        self.currentStateInfo["donationBox"] = common_apples
        self.currentStateInfo["apples"] = np.array([a.has_apples for a in ags])
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        for a in ags:
            if not self.tabularState:
                if a.visible or a.timeout == 25:
                    if self.fullState:
                        ob = np.copy(board)
                        if a.visible:
                            ob[a.position[0], a.position[1], :] = [0, 0, 255]
                        ob = ob[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                             self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
                    else:
                        ob = np.copy(board[
                                     a.position[0] - self.sightRadius:a.position[0] + self.sightRadius + 1,
                                     a.position[1] - self.sightRadius:a.position[1] + self.sightRadius + 1, :])
                        if a.visible:
                            ob[self.sightRadius, self.sightRadius, :] = [0, 0, 255]
                    ob = ob / 255.0
                else:
                    ob = None
                obs.append(ob)

            if not self.fullState:
                new_new_state = np.copy(self.state.board[addedRowsForAgentInfo:, :])

                for i in range(addedRowsForAgentInfo):
                    for j in range(len(new_new_state[i])):
                        if new_new_state[i][j] == 64:
                            new_new_state[i][j] = 61

                new_ob = np.copy(new_new_state[
                                 a.position[0] - self.sightRadius - addedRowsForAgentInfo:a.position[
                                                                                              0] + self.sightRadius + 1 - addedRowsForAgentInfo,
                                 a.position[1] - self.sightRadius:a.position[1] + self.sightRadius + 1])
                if self.normalized_obs:
                    new_ob = self.normalizeObs(new_ob, a, common_apples)
                else:
                    new_ob = np.append(new_ob, [a.has_apples, common_apples])

            else:
                new_ob = np.copy(new_state)
                if self.normalized_obs:
                    new_ob = self.normalizeObs(new_ob, a, common_apples)
                else:
                    new_ob = np.append(new_ob, [a.has_apples, common_apples])

            # new_state = np.append(new_state, [a.position[0] - self.sightRadius - addedRowsForAgentInfo,
            #                                  a.position[1] - self.sightRadius, a.has_apples, a.donated_apples])

            if not self.tabularState:
                obs.append(ob)
            else:

                obs.append(new_ob)
        new_state = np.append(new_state, [common_apples])
        # print("State : ", new_state)

        if not self.tabularState and self.fullState:
            for a in ags:
                if a.visible or a.timeout == constants.TIMEOUT_FRAMES:
                    obs.append(new_state)
                else:
                    obs.append([])
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        return obs, done

    def get_agents(self):
        return [self._game.things[c] for c in self.agentChars]
