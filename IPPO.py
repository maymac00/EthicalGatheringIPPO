import copy
import config
import time
import CommonsGame.constants as constants
from utils.misc import *
from CommonsGame.envs import CommonsGame
from collections import deque
import torch as th
import torch.nn as nn
import torch.optim as optim
from agent import SoftmaxActor, Critic, ACTIONS
from utils.memory import Buffer


def _array_to_dict_tensor(agents: List[int], data: Array, device: th.device, astype: Type = th.float32) -> Dict:
    return {k: th.as_tensor([d], dtype=astype).to(device) for k, d in zip(agents, data)}


class IPPO:
    def __init__(self, args, run_name=None):

        self.args = args
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = f"{args.env}__{args.tag}__{args.seed}__{int(time.time())}__{np.random.randint(0, 100)}"
        self.eval_mode = False

        # Action-Space
        self.o_size = None
        self.a_size = 7

        # Attributes
        self.map2use = None
        self.summary_w, self.wandb_path = None, None
        self.agents = range(args.n_agents)
        self.metrics = {
            'global_step': 0,
            'ep_count': 0,
            'start_time': time.time(),
            'reward_q': [],
            'reward_per_agent': [],
            'avg_reward': [],
        }
        self.n_updates = None

        #   Actor-Critic
        self.buffer = None
        self.c_optim = None
        self.a_optim = None
        self.critic = {}
        self.actor = {}
        self.past_actions_memory = {}

        #   Last run
        self.last_run = {}

        #   Torch init
        self.device = set_torch(self.args.n_cpus, self.args.cuda)
        self.env = None

        # Init processes
        self.transform_params()

    def transform_params(self):
        # Modify env constants
        constants.DONATION_BOX_CAPACITY = self.args.donation_capacity
        constants.SURVIVAL_THRESHOLD = self.args.survival_threshold

        # Load map
        if self.args.map_size == 'tiny':
            self.map2use = constants.tinyMap
        elif self.args.map_size == 'small':
            self.map2use = constants.smallMap
        elif self.args.map_size == 'medium':
            bigMap = np.array(constants.bigMap)
            self.map2use = bigMap[:, :16]
        else:
            self.map2use = constants.bigMap

        # Untracked parameters
        constants.REGENERATION_PROBABILITY = (self.args.apple_regen) if hasattr(self.args, "apple_regen") else 0.05
        self.args.inequality_mode = self.args.inequality_mode if hasattr(self.args, "inequality_mode") else "tie_break"
        self.args.past_actions_memory = self.args.past_actions_memory if hasattr(self.args,
                                                                                 "past_actions_memory") else 0
        self.args.visual_radius = self.args.visual_radius if hasattr(self.args, "visual_radius") else 2
        self.args.partial_observability = self.args.partial_observability if hasattr(self.args,
                                                                                     "partial_observability") else False

        pass

    def environment_setup(self):
        # Environment setup  
        n_agents = self.args.n_agents
        # Generate random positions for the agents in the map
        map_empty = np.array(self.map2use)
        apple_positions = np.array(np.where(map_empty == '@')).T
        agent_positions = []
        available_spots = np.argwhere(np.logical_and(map_empty != '@', map_empty != '='))

        for ag in range(n_agents):
            agent_positions.append(available_spots[np.random.choice(available_spots.shape[0])])
        weight = [1.0, self.args.we]
        self.env = gym.make(self.args.env, numAgents=n_agents, visualRadius=self.args.visual_radius,
                            mapSketch=self.map2use,
                            fullState=(not self.args.partial_observability), agent_pos=agent_positions[:n_agents],
                            weight_vector=weight, inequality_mode=self.args.inequality_mode)

        self.o_size = len(self.env.reset()[0]) + self.args.past_actions_memory

    def environment_reset(self):
        initial_memory = [ACTIONS.index(CommonsGame.STAY) / len(ACTIONS) for i in range(self.args.past_actions_memory)]
        non_tensor_observation = self.env.reset()
        if self.args.past_actions_memory > 0:
            for k in self.agents:
                self.past_actions_memory[k] = deque(initial_memory, maxlen=self.args.past_actions_memory)
                non_tensor_observation[k] = np.append(non_tensor_observation[k], self.past_actions_memory[k])
        observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)
        return observation

    def train(self, init_global_step=0, load_from_checkpoint=None):
        self.environment_setup()
        # set seed for training
        set_seeds(args.seed, self.args.th_deterministic)
        self.summary_w, self.wandb_path = init_loggers(self.run_name, self.args)
        # State and action spaces
        self.o_size = len(self.env.reset()[0]) + self.args.past_actions_memory
        self.a_size = 7  # Removed the unused actions
        print(f"Observation space: {self.o_size}, Action space: {self.a_size}")

        # Init actor-critic setup
        self.actor, self.critic, self.a_optim, self.c_optim, self.buffer = {}, {}, {}, {}, {}
        self.past_actions_memory = {}
        initial_memory = [ACTIONS.index(CommonsGame.STAY) / len(ACTIONS) for i in range(self.args.past_actions_memory)]

        if load_from_checkpoint is not None:
            self._load_models_from_files(load_from_checkpoint)
            for k in self.agents:
                self.past_actions_memory[k] = deque(initial_memory, maxlen=self.args.past_actions_memory)
                self.a_optim[k] = optim.Adam(list(self.actor[k].parameters()), lr=self.args.actor_lr, eps=1e-5)
                self.c_optim[k] = optim.Adam(list(self.critic[k].parameters()), lr=self.args.critic_lr, eps=1e-5)
                self.buffer[k] = Buffer(self.o_size, self.args.n_steps, self.args.max_steps, self.args.gamma,
                                        self.args.gae_lambda, self.device)
        else:
            for k in self.agents:
                self.actor[k] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size).to(self.device)
                self.a_optim[k] = optim.Adam(list(self.actor[k].parameters()), lr=self.args.actor_lr, eps=1e-5)
                self.critic[k] = Critic(self.o_size, self.args.h_size).to(self.device)
                self.c_optim[k] = optim.Adam(list(self.critic[k].parameters()), lr=self.args.critic_lr, eps=1e-5)
                self.past_actions_memory[k] = deque(initial_memory, maxlen=self.args.past_actions_memory)
                self.buffer[k] = Buffer(self.o_size, self.args.n_steps, self.args.max_steps, self.args.gamma,
                                        self.args.gae_lambda, self.device)

        # Reset Training metrics
        self.metrics = {
            'global_step': init_global_step,
            'ep_count': init_global_step/args.max_steps,
            'start_time': time.time(),
            'reward_q': [],
            'reward_per_agent': [],
            'avg_reward': [],
        }

        # Training loop
        self.n_updates = self.args.tot_steps // self.args.batch_size
        for update in range(1, self.n_updates + 1):
            if self.args.anneal_lr:
                frac = 1.0 - (update - 1.0) / self.n_updates
                for a_opt, c_opt in zip(self.a_optim.values(), self.c_optim.values()):
                    a_opt.param_groups[0]["lr"] = frac * self.args.actor_lr
                    c_opt.param_groups[0]["lr"] = frac * self.args.critic_lr
                if self.args.tb_log: self.summary_w.add_scalar('Training/Actor LR', a_opt.param_groups[0]["lr"],
                                                               self.metrics["global_step"])
                if self.args.tb_log: self.summary_w.add_scalar('Training/Critic LR', c_opt.param_groups[0]["lr"],

                                                               self.metrics["global_step"])
            self._sim()
            self._update()

            # Callbacks
            if update % 20 == 0:
                from experiment_utils import graphical_evaluation_fig

                graphical_evaluation_fig(self.last_run["apple_history"], self.last_run["donation_box_history"],
                                         self.args, ckpt=True, type="mean")
            if update % 2000 == 0:
                try:
                    from experiment_utils import save_experiment_data
                    folder = save_experiment_data(self.actor, self.critic, self.args, self.metrics["reward_q"],
                                                  self.metrics["reward_per_agent"])
                except:
                    pass
            sps = int(self.metrics["global_step"] / (time.time() - self.metrics["start_time"]))
            if self.args.tb_log: self.summary_w.add_scalar('Training/SPS', sps, self.metrics["global_step"])
            # if args.wandb_log: wandb.log({'SPS': sps})

            if self.metrics["global_step"] > self.args.tot_steps: break
            if self.metrics["global_step"] > self.args.early_stop: break

        self._end_training()

    def _sim(self, render=False, masked=False, pause=0.01):
        if self.eval_mode:
            self.args.tb_log = False

        observation = self.environment_reset()
        self.last_run = {
            "greedy": 0,
            "reward_per_agent": []
        }
        info = {
            "donationBox": 0,
            "n": [0] * self.args.n_agents,
            "donationBox_full": False,
        }

        action, logprob, s_value = [{k: 0 for k in self.agents} for _ in range(3)]
        env_action, ep_reward = [np.zeros(self.args.n_agents) for _ in range(2)]
        job_done = False
        apple_history = []
        donation_box_history = []
        for step in range(self.args.n_steps):
            self.metrics["global_step"] += 1  # * args.n_envs

            if render:
                self.env.render(masked=masked)
                time.sleep(pause)

            with th.no_grad():
                for k in self.agents:
                    (
                        env_action[k],
                        action[k],
                        logprob[k],
                        _,
                    ) = self.actor[k].get_action(observation[k])
                    if self.args.past_actions_memory > 0:
                        self.past_actions_memory[k].appendleft(float(action[k] / len(ACTIONS)))
                    if not self.eval_mode:
                        s_value[k] = self.critic[k](observation[k])

            non_tensor_observation, reward, done, info = self.env.step(env_action)
            if self.args.past_actions_memory > 0:
                for k in self.agents:
                    non_tensor_observation[k] = np.append(non_tensor_observation[k], self.past_actions_memory[k])
            apple_history.append(info['n'])
            donation_box_history.append(info['donationBox'])

            if self.metrics["global_step"] % self.args.max_steps == 0:
                done = [True] * self.args.n_agents

            # Consider the metrics of the first agent, probably want an average of the two
            ep_reward += reward
            if self.eval_mode and any(reward < -1):
                # print("Negative reward agent " + str(np.argmin(reward)), " step ", step)
                self.last_run["greedy"] += 1

            if info['survival'] and info['donationBox_full'] and not job_done:
                job_done = True
                if self.args.tb_log: self.summary_w.add_scalar('Training/Job_Done',
                                                               (step - np.floor(
                                                                   step / self.args.max_steps) * self.args.max_steps),
                                                               self.metrics["global_step"])

            reward = _array_to_dict_tensor(self.agents, reward, self.device)
            done = _array_to_dict_tensor(self.agents, done, self.device)
            if not self.eval_mode:
                for k in self.agents:
                    self.buffer[k].store(
                        observation[k],
                        action[k],
                        logprob[k],
                        reward[k],
                        s_value[k],
                        done[k]
                    )

            observation = _array_to_dict_tensor(self.agents, non_tensor_observation, self.device)

            # End of sim
            if all(list(done.values())):
                self.last_run["apple_history"] = np.array(apple_history)
                self.last_run["donation_box_history"] = np.array(donation_box_history)

                self.metrics["ep_count"] += 1
                self.metrics["reward_per_agent"].append(ep_reward)
                self.last_run["reward_per_agent"] = ep_reward
                self.metrics["reward_q"].append(np.mean(ep_reward))

                if self.eval_mode:
                    self.last_run["survive"] = info["survival"]
                    self.last_run["donation_box_full"] = info["donationBox_full"]
                    # Weak convergence
                    self.last_run["convergence"] = True if info["survival"] and not self.last_run["greedy"] \
                                                           and self.last_run["donation_box_full"] else False
                    return

                if not job_done:
                    if self.args.tb_log: self.summary_w.add_scalar('Training/Job_Done',
                                                                   (step - np.floor(
                                                                       step / self.args.max_steps) * self.args.max_steps),
                                                                   self.metrics["global_step"])
                job_done = False
                if step != self.args.n_steps - 1:
                    apple_history = []
                    donation_box_history = []
                else:
                    apple_history = np.array(apple_history).reshape(self.args.max_steps, self.args.n_agents, 1)
                    donation_box_history = np.array(donation_box_history).reshape(self.args.max_steps, 1)
                self.metrics["avg_reward"].append(np.mean(self.metrics["reward_q"]))
                record = {
                    'Training/Global_Step': self.metrics["global_step"],
                    'Training/Avg_Reward': np.mean(self.metrics["reward_q"]),
                }
                # add agent's reward to the record
                for k in self.agents:
                    record['Agent_' + str(k) + '/Reward'] = ep_reward[k]

                if self.args.tb_log:
                    # Add record to tensorboard
                    for k, v in record.items():
                        self.summary_w.add_scalar(k, v, self.metrics["global_step"])

                # if args.wandb_log: wandb.log(record)

                if self.args.verbose:
                    print(f"E: {self.metrics['ep_count']},\n\t "
                          f"Reward per agent: {ep_reward},\n\t "
                          f"Avg_Reward for all episodes: {record['Training/Avg_Reward']},\n\t "
                          f"Global_Step: {self.metrics['global_step']},\n\t "
                          )

                ep_reward = np.zeros(self.args.n_agents)
                # Reset environment
                observation = self.environment_reset()

    def _update(self):
        with th.no_grad():
            for k in self.agents:
                value_ = self.critic[k](self.environment_reset()[k])
                self.buffer[k].compute_mc(value_.reshape(-1))

        # Optimize the policy and value networks
        for k in self.agents:
            b = self.buffer[k].sample()
            self.buffer[k].clear()
            # Actor optimization
            for epoch in range(self.args.n_epochs):
                _, _, logprob, entropy = self.actor[k].get_action(b['observations'], b['actions'])
                entropy_loss = entropy.mean()

                logratio = logprob - b['logprobs']
                ratio = logratio.exp()

                mb_advantages = b['advantages']
                if self.args.norm_adv: mb_advantages = normalize(mb_advantages)

                actor_loss = mb_advantages * ratio

                actor_clip_loss = mb_advantages * th.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip)
                actor_loss = th.min(actor_loss, actor_clip_loss).mean()
                if self.args.tb_log: self.summary_w.add_scalar('Agent_' + str(k) + "/" + 'Actor loss', actor_loss,
                                                               (self.metrics[
                                                                    "global_step"] / self.args.max_steps) * self.args.n_epochs + epoch)

                actor_loss = -actor_loss - self.args.ent_coef * entropy_loss
                # if args.tb_log: summary_w.add_scalar('Agent_' + str(k) + "/" + 'Actor loss', actor_loss,
                # (global_step/args.max_steps)*args.n_epochs+epoch)
                self.a_optim[k].zero_grad(True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor[k].parameters(), self.args.max_grad_norm)
                self.a_optim[k].step()

                # Not using this rn
                """
                with th.no_grad():  # Early break from updates
                    if args.target_kl is not None:
                        approx_kl = ((ratio - 1) - logratio).mean()
                        if approx_kl > args.target_kl:
                            break"""

            # Critic optimization
            for epoch in range(self.args.n_epochs * self.args.critic_times):
                values = self.critic[k](b['observations']).squeeze()

                critic_loss = 0.5 * ((values - b['returns']) ** 2).mean()

                if self.args.tb_log: self.summary_w.add_scalar('Agent_' + str(k) + "/" + 'Critic loss', critic_loss,
                                                               (self.metrics[
                                                                    "global_step"] / self.args.max_steps) * self.args.n_epochs * self.args.critic_times + epoch)

                critic_loss = critic_loss * self.args.v_coef

                self.c_optim[k].zero_grad(True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic[k].parameters(), self.args.max_grad_norm)
                self.c_optim[k].step()

    def _end_training(self):
        try:
            from experiment_utils import save_experiment_data

            folder = save_experiment_data(self.actor, self.critic, self.args, self.metrics["reward_q"],
                                          self.metrics["reward_per_agent"])
            if self.args.tb_log: self.summary_w.add_text("Model Save Dir", folder)
            if self.args.tb_log: self.summary_w.close()
            '''
            if args.wandb_log: 
                wandb.finish()
                if args.wandb_mode == 'offline':
                    import subprocess
                    subprocess.run(['wandb', 'sync', wandb_path])  
            '''
            self.env.close()
        except:
            pass

    def _load_models_from_files(self, folder):
        not_found = []
        initial_memory = [ACTIONS.index(CommonsGame.STAY) / len(ACTIONS) for i in range(self.args.past_actions_memory)]
        for k in self.agents:
            # Load the actor and critic from the saved model
            try:
                # self.actor[k] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size, eval=self.eval_mode).to(self.device)
                self.actor[k] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size, eval=self.eval_mode).to(self.device)
                self.actor[k].load_state_dict(th.load(folder + f"/actor_{k}.pth"))
                self.critic[k] = Critic(self.o_size, self.args.h_size).to(self.device)
                self.critic[k].load_state_dict(th.load(folder + f"/critic_{k}.pth"))
                self.past_actions_memory[k] = deque(initial_memory, maxlen=self.args.past_actions_memory)
            except FileNotFoundError as e:
                not_found.append(k)

        if len(not_found) != 0:
            # Manage extra agents, copy the actor of other agents
            if len(not_found) == 2:
                # copy one efficient agent and one inefficient agent
                self.actor[not_found[0]] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size,
                                                        eval=self.eval_mode).to(
                    self.device)
                self.actor[not_found[0]].load_state_dict(th.load(folder + f"/actor_0.pth"))
                self.actor[not_found[1]] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size,
                                                        eval=self.eval_mode).to(
                    self.device)
                self.actor[not_found[1]].load_state_dict(th.load(folder + f"/actor_2.pth"))
            elif len(not_found) == 1:
                # add one inefficient agent
                self.actor[not_found[0]] = SoftmaxActor(self.o_size, self.a_size, self.args.h_size,
                                                        eval=self.eval_mode).to(
                    self.device)
                self.actor[not_found[0]].load_state_dict(th.load(folder + f"/actor_0.pth"))
            else:
                raise Exception("No agent found")

    def play_from_folder(self, folder, render, plot, pause=0.01, masked=False, times=1, type="median", verbose=True):
        self.eval_mode = True
        self.environment_setup()

        # State and action spaces
        self.o_size = len(self.env.reset()[0]) + self.args.past_actions_memory
        self.a_size = 7  # Removed the unused actions

        # Init actor-critic setup
        self._load_models_from_files(folder)

        results = []
        for t in range(times):
            self._sim(render, masked=masked, pause=pause)
            results.append(copy.deepcopy(self.last_run))
        if verbose:
            print("Survival rate: ", sum([run["survive"] for run in results]) / times)
            print("Convergence rate: ", sum([run["convergence"] for run in results]) / times)
            print("Mean return per agent: ", sum([run["reward_per_agent"] for run in results]) / times)
            print("Mean unethical actions commited per run: ", sum([run["greedy"] for run in results]) / times)
        if plot:
            import matplotlib.pyplot as plt
            from experiment_utils import graphical_evaluation_fig
            if times == 1:
                self.last_run["apple_history"] = np.expand_dims(self.last_run["apple_history"], axis=2)
                self.last_run["donation_box_history"] = np.expand_dims(self.last_run["donation_box_history"], axis=1)
                fig = graphical_evaluation_fig(self.last_run["apple_history"], self.last_run["donation_box_history"],
                                               self.args, ckpt=False, type="mean")
                plt.show()
            else:
                runs_apple_history = np.array([run["apple_history"] for run in results])
                runs_donation_box = np.array([run["donation_box_history"] for run in results])

                runs_apple_history = np.swapaxes(runs_apple_history, 0, 1)
                runs_apple_history = np.swapaxes(runs_apple_history, 1, 2)
                # print("Number of outliers: ", cont)

                runs_donation_box = np.array(runs_donation_box)
                runs_donation_box = np.swapaxes(runs_donation_box, 0, 1)

                fig = graphical_evaluation_fig(runs_apple_history, runs_donation_box, self.args, ckpt=False, type=type)

                plt.show()
                fig.savefig(folder + "/mean_return_per_step.png")

        self.eval_mode = False
        return results


if __name__ == "__main__":
    args = config.parse_args()
    # Print all the arguments, so they are visible in the log
    print(args)
    ppo = IPPO(args)
    ppo.train()
