import numpy as np
from pycolab.rendering import ObservationToArray


def buildMap(mapSketch, numPadPixels, agentChars, mandatory_initial_position=[]):
    """

    :param mapSketch:
    :param numPadPixels:
    :param agentChars:
    :return:
    """
    numAgents = len(agentChars)
    gameMap = np.array(mapSketch)

    def padWith(vector, padWidth, iaxis, kwargs):
        del iaxis
        padValue = kwargs.get('padder', ' ')
        vector[:padWidth[0]] = padValue
        vector[-padWidth[1]:] = padValue
        return vector

    # Put agents
    nonFilledSpots = np.argwhere(np.logical_and(gameMap != '@', gameMap != '='))
    selectedSpots = np.random.choice(nonFilledSpots.shape[0], size=(numAgents,), replace=False)
    agentsCoords = nonFilledSpots[selectedSpots, :]

    for idx, coord in enumerate(agentsCoords):
        if len(mandatory_initial_position) == 1:
            coord = mandatory_initial_position[idx]
        else:
            if len(mandatory_initial_position) > idx and not isinstance(mandatory_initial_position[idx], int):
                coord = mandatory_initial_position[idx]

        gameMap[coord[0], coord[1]] = agentChars[idx]

    # Put walls
    gameMap = np.pad(gameMap, numPadPixels + 1, padWith, padder='=')

    gameMap = [''.join(row.tolist()) for row in gameMap]
    return gameMap


class ObservationToArrayWithRGB(object):
    def __init__(self, colour_mapping):
        self._colour_mapping = colour_mapping
        # Rendering functions for the `board` representation and `RGB` values.
        self._renderers = {
            'RGB': ObservationToArray(value_mapping=colour_mapping)
        }

    def __call__(self, observation):
        # Perform observation rendering for agent and for video recording.
        result = {}
        for key, renderer in self._renderers.items():
            result[key] = renderer(observation)
        # Convert to [0, 255] RGB values.
        result['RGB'] = (result['RGB'] / 999.0 * 255.0).astype(np.uint8)
        return result
