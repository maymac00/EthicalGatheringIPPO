import copy
import sys

import numpy as np
from matplotlib import pyplot as plt

import CommonsGame

sys.path.append('../..')
import warnings

warnings.filterwarnings("ignore", category=Warning)
import time

import config
from agent import SoftmaxActor, Critic, ACTIONS
from utils.misc import *
from IPPO import IPPO
import CommonsGame.constants as constants
from experiment_utils import graphical_evaluation_fig
from ActionSelection import *
from scipy.stats import iqr


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    return {k: th.as_tensor([d], dtype=astype).to(device) for k, d in zip(agents, data)}


def run(folder, args, render, plot, pause=0.01, masked=False, times=1, type="median", verbose=True):
    ppo = IPPO(args)
    result = ppo.play_from_folder(folder, render, plot, masked=masked, pause=pause, times=times, type=type, verbose=verbose)
    return result


def beautiful_run(folder, args):
    from utils.display import GatheringDisplay
    run_name = f"{args.env}__{args.tag}__{args.seed}__{int(time.time())}__{np.random.randint(0, 100)}"
    if args.map_size == 'tiny':
        map2use = constants.tinyMap
    elif args.map_size == 'small':
        map2use = constants.smallMap
        run_name += '_small'
    elif args.map_size == 'medium':
        bigMap = np.array(constants.bigMap)
        map2use = bigMap[:, :16]
        run_name += '_medium'
    else:
        map2use = constants.bigMap
    stats = {
        "convergence": False,
    }

    display = GatheringDisplay()

    # Environment setup

    n_agents = args.n_agents
    # Generate random positions for the agents in the map
    map_empty = np.array(map2use)
    apple_positions = np.array(np.where(map_empty == '@')).T
    agent_positions = []
    available_spots = np.argwhere(np.logical_and(map_empty != '@', map_empty != '='))

    for ag in range(n_agents):
        agent_positions.append(available_spots[np.random.choice(available_spots.shape[0])])

    weight = np.array([1, args.we])

    if hasattr(args, "visual_radius"):
        env = gym.make(args.env, numAgents=n_agents, visualRadius=args.visual_radius, mapSketch=map2use,
                       fullState=(not args.partial_observability), agent_pos=agent_positions[:n_agents],
                       weight_vector=weight)
    else:
        env = gym.make(args.env, numAgents=n_agents, visualRadius=2, mapSketch=map2use,
                       fullState=True, agent_pos=agent_positions[:n_agents], weight_vector=weight)
    # if args.norm_obs: env = gym.wrappers.NormalizeObservation(env)

    device = set_torch(args.n_cpus, args.cuda)

    # Using pettinzoo standard interface with dictionaries
    agents = range(n_agents)  # List of agents' idx i.e., [0, 1]
    o_size = len(env.reset()[0])
    a_size = 7  # Removed the unused actions

    # Actor-critic setup
    actor, critic = {}, {}
    for k in agents:
        # Load the actor and critic from the saved model
        actor[k] = SoftmaxActor(o_size, a_size, args.h_size).to(device)
        actor[k].load_state_dict(th.load(folder + f"/actor_{k}.pth"))
        critic[k] = Critic(o_size, args.h_size).to(device)
        critic[k].load_state_dict(th.load(folder + f"/critic_{k}.pth"))
    observation_ = env.reset()
    observation = _array_to_dict_tensor(agents, observation_, device)

    action, logprob = [{k: 0 for k in agents} for _ in range(2)]
    env_action, ep_reward = [np.zeros(n_agents) for _ in range(2)]

    apple_history = np.empty((args.max_steps, n_agents))
    donation_box_history = np.empty(args.max_steps)

    for step in range(args.max_steps):
        with th.no_grad():
            # Environment reset

            apple_history[step] = env.currentStateInfo["apples"]
            donation_box_history[step] = env.currentStateInfo["donationBox"]

            for k in agents:
                (
                    env_action[k],
                    action[k],
                    logprob[k],
                    _,
                ) = actor[k].get_action(observation[k])

            # env_action = [np.random.choice(ACTIONS) for _ in range(n_agents)]

            observation_, reward, done, info = env.step(env_action)
            # print(reward)
            # Consider the metrics of the first agent, probably want an average of the two
            ep_reward += reward

            display.draw_grid(env.getBoard(), env.getPlotText())
            time.sleep(0.2)

            reward = _array_to_dict_tensor(agents, reward, device)
            done = _array_to_dict_tensor(agents, done, device)

            observation = _array_to_dict_tensor(agents, observation_, device)

    stats["apple_history"] = apple_history
    stats["donation_box_history"] = donation_box_history

    fig = graphical_evaluation_fig(apple_history, donation_box_history, args, ckpt=False, type="median")
    fig.savefig(folder + "/apples_per_step.png")
    plt.show()


def run_all(folder, times=30):
    # folder = "Gathering_data/2.6_2500_30000_0"
    # args = config.args_from_json(folder)
    # print("Seed: ", args.seed)
    # print("Entropy: ", args.ent_coef)
    render = False
    plot = False

    global_args = None #config.args_from_json(os.path.join(folder, os.listdir(folder)[0]))
    # global stats
    global_stats = {}
    global_stats["reward_per_agent"] = []
    global_stats["convergence"] = []
    global_stats["survival"] = []
    global_stats["donation_box_full"] = []
    global_stats["seed"] = []
    global_stats["greedy"] = []

    seeds = 0
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Get the arguments from the json file
        try:
            args = config.args_from_json(subfolder_path)
            if global_args is None:
                global_args = copy.deepcopy(args)
        except FileNotFoundError:
            continue
        run_data = run(subfolder_path, args, render, plot, verbose=False, times=times)

        seeds += 1
        global_stats["reward_per_agent"].append(np.array([r["reward_per_agent"] for r in run_data]))
        global_stats["convergence"].append(np.array([r["convergence"] for r in run_data]))
        global_stats["survival"].append(np.array([r["survive"] for r in run_data]))
        global_stats["donation_box_full"].append(np.array([r["donation_box_full"] for r in run_data]))
        global_stats["seed"].append(args.seed)
        global_stats["greedy"].append(np.array([r["greedy"] for r in run_data]))

    # Convert lists on global_stats to numpy arrays
    for k in global_stats:
        global_stats[k] = np.array(global_stats[k])

    # Sort global stats per seed
    idx = np.argsort(global_stats["seed"])
    for k in global_stats:
        global_stats[k] = global_stats[k][idx]

    # Print descriptive statistics per seed
    for s in range(seeds):
        print(f"Seed {s+1}:")
        print(f"Reward per agent mean: {global_stats['reward_per_agent'][s].mean(axis=0)}")
        print(f"Reward per agent std: {global_stats['reward_per_agent'][s].std(axis=0)}")
        print(f"Reward per agent median: {np.median(global_stats['reward_per_agent'][s], axis=0)}")
        print(f"Reward per agent iqr: {iqr(global_stats['reward_per_agent'][s], axis=0)}")
        print(f"Convergence: {global_stats['convergence'][s].mean(axis=0)}")
        print(f"Survival: {global_stats['survival'][s].mean(axis=0)}")
        print(f"Donation box full: {global_stats['donation_box_full'][s].mean(axis=0)}")
        print()

    # Print descriptive statistics over all seeds
    print("All seeds:")
    print(f"Reward per agent mean: {global_stats['reward_per_agent'].mean(axis=0).mean(axis=0)}")
    print(f"Reward per agent std: {global_stats['reward_per_agent'].std(axis=0).mean(axis=0)}")
    print(f"Reward per agent median: {np.median(global_stats['reward_per_agent'], axis=0).mean(axis=0)}")
    print(f"Reward per agent iqr: {iqr(global_stats['reward_per_agent'], axis=0).mean(axis=0)}")
    print(f"Convergence: {global_stats['convergence'].mean(axis=0).mean(axis=0)}")
    print(f"Survival: {global_stats['survival'].mean(axis=0).mean(axis=0)}")
    print(f"Donation box full: {global_stats['donation_box_full'].mean(axis=0).mean(axis=0)}")
    print()

    # Save each of the stats in a npy file
    for k in global_stats:
        np.save(os.path.join(folder, k), global_stats[k])


if __name__ == "__main__":
    folder = "Gathering_data/medium_premium_long/2.6_2500_100000_1"
    # folder = "Gathering_data/medium_map_first_tries/2.6_2500_50000_1_(1)"
    args = config.args_from_json(folder)
    render = False
    plot = True
    args.apple_regen = 0.006
    # Action-Selection
    SoftmaxActor.action_selection = bottom_filter
    # beautiful_run(folder, args)
    # run(folder, args, render, plot, masked=True, times=30, type="median")
    run_all("Gathering_data/medium_with_repeat", 10)
