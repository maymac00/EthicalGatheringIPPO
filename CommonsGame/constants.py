# Training logic altering
CREATING_MODEL = False
MAP_NAME = "tiny"

# Environment logic altering!
TIMEOUT_FRAMES = 25
SURVIVAL_THRESHOLD = 10
DONATION_BOX_HAS_LIMIT = True
DONATION_BOX_CAPACITY = 5
AGENTS_CAN_GET_SICK = False
AGENTS_HAVE_DIFFERENT_EFFICIENCY = True
SUSTAINABILITY_MATTERS = False  # If False, apples ALWAYS regenerate
REGENERATION_PROBABILITY = 0.05  # Only matters if SUSTAINABILITY does not matter
respawnProbs = [0.01, 0.05, 0.1]

# Positive rewards
DONATION_REWARD = 0.7
TOOK_DONATION_REWARD = 0.0
APPLE_GATHERING_REWARD = 1.0
DID_NOTHING_BECAUSE_MANY_APPLES_REWARD = 0.0  # related with sustainability probably

# Negative rewards
TOO_MANY_APPLES_PUNISHMENT = -1.0  # related with sustainability probably
SHOOTING_PUNISHMENT = -0.0
HUNGER = -1.0
LOST_APPLE = -1.0

bigMap = [
    list('======================================'),
    list('======================================'),
    list('                                      '),
    list('             @      @@@@@       @     '),
    list('         @   @@         @@@    @  @   '),
    list('      @ @@@  @@@    @    @ @@ @@@@    '),
    list('  @  @@@ @    @  @ @@@  @  @   @ @    '),
    list(' @@@  @ @    @  @@@ @  @@@        @   '),
    list('  @ @  @@@  @@@  @ @    @ @@   @@ @@  '),
    list('   @ @  @@@    @ @  @@@    @@@  @     '),
    list('    @@@  @      @@@  @    @@@@        '),
    list('     @       @  @ @@@    @  @         '),
    list(' @  @@@  @  @  @@@ @    @@@@          '),
    list('     @ @   @@@  @ @      @ @@   @     '),
    list('      @@@   @ @  @@@      @@   @@@    '),
    list('  @    @     @@@  @             @     '),
    list('              @                       '),
    list('                                      ')
]

smallMap = [
    list('====='),
    list('====='),
    list('     '),
    list('  @@ '),
    list('  @  '),
    list('     ')]

tinyMap = [
    list('===='),
    list('===='),
    list(' @@ '),
    list(' @  '),
    list('    ')]
