import json
import os
from typing import List, Type
import copy

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch as th
from scipy.stats import iqr

import CommonsGame.constants as constants

AGENTS_COLORS = ["blue", "orange", "magenta", "cyan", "brown", "lime", "pink", "yellow"]


def _array_to_dict_tensor(agents: List[int], data, device: th.device, astype: Type = th.float32):
    return {k: th.as_tensor([d], dtype=astype).to(device) for k, d in zip(agents, data)}


def graphical_evaluation_fig(apple_history, donation_box_history, args, ckpt=True, type="median"):
    font = {'size': 18}
    matplotlib.rc('font', **font)

    if len(apple_history.shape) == 2:
        apple_history = np.expand_dims(apple_history, axis=2)
    if len(donation_box_history.shape) == 1:
        donation_box_history = np.expand_dims(donation_box_history, axis=1)
    if ckpt:
        type = "mean"

    fig = plt.figure()
    plt.title("Apples collected")
    plt.xlabel("Step")
    plt.ylabel("Nº Apples")

    # Plot constants
    plt.axhline(y=constants.SURVIVAL_THRESHOLD, color='r', linestyle='-')
    plt.axhline(y=constants.DONATION_BOX_CAPACITY, color='black', linestyle='-')

    # Plot median
    if type == "median":
        apple_history_iqr = iqr(apple_history, axis=2)
        donation_box_history_iqr = iqr(donation_box_history, axis=1)

        median_apple_history = np.median(apple_history, axis=2)
        median_donation_box_history = np.median(donation_box_history, axis=1)

        for ag in range(args.n_agents):
            plt.plot(median_apple_history[:, ag], color=AGENTS_COLORS[ag])
        plt.plot(median_donation_box_history, color='g')

        # Plot iqr
        for ag in range(args.n_agents):
            plt.fill_between(np.arange(args.max_steps), median_apple_history[:, ag] - apple_history_iqr[:, ag],
                             median_apple_history[:, ag] + apple_history_iqr[:, ag], alpha=0.3, color=AGENTS_COLORS[ag])
        # Plot iqr
        plt.fill_between(np.arange(args.max_steps), median_donation_box_history - donation_box_history_iqr,
                         median_donation_box_history + donation_box_history_iqr, alpha=0.3, color='g')

    # Plot mean
    elif type == "mean":
        mean_apple_history = np.mean(apple_history, axis=2)
        mean_donation_box_history = np.mean(donation_box_history, axis=1)

        for ag in range(args.n_agents):
            plt.plot(mean_apple_history[:, ag], color=AGENTS_COLORS[ag])
        plt.plot(mean_donation_box_history, color='g')

        # Plot std
        std_apple_history = np.std(apple_history, axis=2)
        std_donation_box_history = np.std(donation_box_history, axis=1)

        for ag in range(args.n_agents):
            plt.fill_between(np.arange(args.max_steps), mean_apple_history[:, ag] - std_apple_history[:, ag],
                             mean_apple_history[:, ag] + std_apple_history[:, ag], alpha=0.3, color=AGENTS_COLORS[ag])

        plt.fill_between(np.arange(args.max_steps), mean_donation_box_history - std_donation_box_history,
                         mean_donation_box_history + std_donation_box_history, alpha=0.3, color='g')



    elif type == "both":
        plt.figure()
        plt.title("Apples collected")
        plt.xlabel("Step")
        plt.ylabel("Nº Apples")

        # Plot constants
        plt.axhline(y=constants.SURVIVAL_THRESHOLD, color='r', linestyle='-')
        plt.axhline(y=constants.DONATION_BOX_CAPACITY, color='black', linestyle='-')
        apple_history_iqr = iqr(apple_history, axis=2)
        donation_box_history_iqr = iqr(donation_box_history, axis=1)

        median_apple_history = np.median(apple_history, axis=2)
        median_donation_box_history = np.median(donation_box_history, axis=1)

        for ag in range(args.n_agents):
            plt.plot(median_apple_history[:, ag], color=AGENTS_COLORS[ag])
        plt.plot(median_donation_box_history, color='g')

        # Plot iqr
        for ag in range(args.n_agents):
            plt.fill_between(np.arange(args.max_steps), median_apple_history[:, ag] - apple_history_iqr[:, ag],
                             median_apple_history[:, ag] + apple_history_iqr[:, ag], alpha=0.3, color=AGENTS_COLORS[ag])
        # Plot iqr
        plt.fill_between(np.arange(args.max_steps), median_donation_box_history - donation_box_history_iqr,
                         median_donation_box_history + donation_box_history_iqr, alpha=0.3, color='g')

        # build legend
        lg = plt.legend(
            ["Survival threshold", "Donation box capacity", *[f"Agent {i}" for i in range(args.n_agents)],
             "Donation box"])

        plt.show()
        plt.figure()
        plt.title("Apples collected")
        plt.xlabel("Step")
        plt.ylabel("Nº Apples")

        # Plot constants
        plt.axhline(y=constants.SURVIVAL_THRESHOLD, color='r', linestyle='-')
        plt.axhline(y=constants.DONATION_BOX_CAPACITY, color='black', linestyle='-')
        mean_apple_history = np.mean(apple_history, axis=2)
        mean_donation_box_history = np.mean(donation_box_history, axis=1)

        for ag in range(args.n_agents):
            plt.plot(mean_apple_history[:, ag], color=AGENTS_COLORS[ag])
        plt.plot(mean_donation_box_history, color='g')

        # Plot std
        std_apple_history = np.std(apple_history, axis=2)
        std_donation_box_history = np.std(donation_box_history, axis=1)

        for ag in range(args.n_agents):
            plt.fill_between(np.arange(args.max_steps), mean_apple_history[:, ag] - std_apple_history[:, ag],
                             mean_apple_history[:, ag] + std_apple_history[:, ag], alpha=0.3, color=AGENTS_COLORS[ag])

        plt.fill_between(np.arange(args.max_steps), mean_donation_box_history - std_donation_box_history,
                         mean_donation_box_history + std_donation_box_history, alpha=0.3, color='g')
        # build legend
        lg = plt.legend(
            ["Survival threshold", "Donation box capacity", *[f"Agent {i}" for i in range(args.n_agents)],
             "Donation box"])

        plt.show()
    else:

        for r in range(apple_history.shape[2]):
            # Plot apples of each agent
            for ag in range(args.n_agents):
                plt.plot(apple_history[:, ag, r], color=AGENTS_COLORS[ag], linestyle='--', alpha=0.2)
            # Plot donation box
            plt.plot(donation_box_history[:, r], color='green', linestyle='--', alpha=0.2)

        mean_apple_history = np.mean(apple_history, axis=2)
        mean_donation_box_history = np.mean(donation_box_history, axis=1)
        for ag in range(args.n_agents):
            plt.plot(mean_apple_history[:, ag], color=AGENTS_COLORS[ag], linestyle='-')
        plt.plot(mean_donation_box_history, color='green', linestyle='-')

    # build legend
    lg = plt.legend(
        ["Survival threshold", "Donation box capacity", *[f"Agent {i}" for i in range(args.n_agents)], "Donation box"])

    for lh in lg.legendHandles:
        lh.set_alpha(1)

    if ckpt:
        plt.savefig(
            f"{args.save_dir}/{args.tag}_{args.we}_{args.n_steps}_{args.tot_steps // args.max_steps}_{args.seed}_ckpt.png")
        print(
            f"Saved checkpoint figure in {args.save_dir}/{args.tag}_{args.we}_{args.n_steps}_{args.tot_steps // args.max_steps}_{args.seed}_ckpt.png")
        plt.close(fig)
    else:
        return fig


def save_experiment_data(actors, critics, config, rewards_per_agent, rewards_q):
    # Create new folder in to save the model using args.we, args.n_steps, args.tot_steps as name
    folder = f"{config.save_dir}/{config.tag}/{config.we}_{config.n_steps}_{config.tot_steps // config.max_steps}_{config.seed}"

    # Check if folder's config file is the same as the current config
    def diff_config(path):
        if os.path.exists(path):
            with open(path + "/config.json", "r") as f:
                old_config = json.load(f)
            if old_config != vars(config):
                return True
            return False
        return False

    num = 1
    _folder = copy.copy(folder)
    while diff_config(_folder):
        # append a number to the folder name
        _folder = folder + "_(" + str(num) + ")"
        num += 1
    folder = _folder
    print(f"Saving model in {folder}")
    os.makedirs(folder)

    # Save the model
    for k in range(config.n_agents):
        th.save(actors[k].state_dict(), folder + f"/actor_{k}.pth")
        th.save(critics[k].state_dict(), folder + f"/critic_{k}.pth")

    # Export rewards to csv
    np.array(rewards_per_agent).tofile(f'{folder}/reward_per_agent.csv', sep=',')
    np.array(rewards_q).tofile(f'{folder}/reward_q.csv', sep=',')

    # Save the args as a json file
    with open(folder + "/config.json", "w") as f:
        json.dump(vars(config), f, indent=4)
    return folder
