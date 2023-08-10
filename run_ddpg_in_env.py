import time
import datetime
import os

import numpy as np
import torch

from config import GlobalConfig
from env.env_interface import EnvironmentManager
from agent.ddpg import Agent
from agent.action import Action

#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from types import SimpleNamespace as SN

from agent.components.transforms import OneHot
from agent.components.episode_buffer import ReplayBuffer
from agent.learners import REGISTRY as le_REGISTRY
from agent.runners import REGISTRY as r_REGISTRY
from agent.controllers import REGISTRY as mac_REGISTRY
from agent.utils.logging import get_logger

from args_utils import get_args
from agent.DCDRL.agent import Agent as Agent_DC
from agent.DCDRL.env_utils import init_pos, lower_ver, get_space, all_zeros, get_normalized_state, get_counter, step


def select_action_random(global_config: GlobalConfig):
    return np.random.uniform(0, 1, global_config.agent_config.action_config.dimension)


class AgentOperator:
    def __init__(self, global_config: GlobalConfig, json_name):
        # init the global config
        self.global_config = global_config
        self.json_name = json_name

        self.output_network_message = self.global_config.control_config.output_network_config
        self.output_action_message = self.global_config.control_config.output_action_config

        self.easy_output_mode = self.global_config.control_config.easy_output_mode
        self.easy_output_cycle = self.global_config.control_config.easy_output_cycle
        self.print_control = True

        # init the train config
        self.train_config = self.global_config.train_config
        self.action_config = self.global_config.agent_config.action_config

        # set the random seed
        np.random.seed(self.train_config.seed)
        torch.manual_seed(self.train_config.seed)

        # init the env manager
        self.env_manager = EnvironmentManager(self.global_config)

        # init the DDPG agent
        if torch.cuda.is_available() and self.train_config.algorithm != 'dc':
            self.device = torch.device('cuda', index=self.train_config.gpu_index)
        else:
            self.device = torch.device('cpu')
        # self.device = torch.device('cpu')
        self.ddpg_agent = Agent(self.device, self.global_config)

        # init the tensorboard writer
        self.exp_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if self.train_config.tensorboard:
            dir_name = 'runs_new/' + self.json_name + "/" + self.train_config.algorithm \
                       + '_s_' + str(self.train_config.seed) \
                       + '_t_' + self.exp_time
            self.writer = SummaryWriter(log_dir=dir_name)

        # save the init time
        self.init_time = time.time()

        # init the parameter
        self.ue_num = self.global_config.env_config.hexagon_network_config.user_equipment_num
        self.cur_epsilon = self.ddpg_agent.epsilon_max

        # save the real step count
        self.step_real_count = 0

        self.env_manager.step_real_count = self.step_real_count
        self.env_manager.writer = self.writer

        # new add attributes, some of them might be deleted later
        self.mac = None
        self.batch_size = None

    def select_action(self, state, agent_id):
        action = self.ddpg_agent.policy_online_network_list[agent_id](state).detach().cpu().numpy()
        return np.clip(action, 0, self.ddpg_agent.action_limit)

    def select_action_with_noise(self, state, agent_id):
        action = (self.ddpg_agent.policy_online_network_list[agent_id](state)).detach().cpu().numpy()
        action += self.ddpg_agent.action_noise * np.random.randn(self.ddpg_agent.action_dim)
        # self.ddpg_agent.action_noise *= self.ddpg_agent.action_noise_decay
        return np.clip(action, 0.01, self.ddpg_agent.action_limit)

    def run_all_episode(self):
        args = get_args(self.global_config)
        args.device = "cuda" if args.use_cuda and self.train_config.algorithm != 'dc' else "cpu"
        # self.device = args.device = "cpu"

        logger = get_logger()

        # Init runner so we can get env info
        runner = r_REGISTRY['episode'](args=args, logger=logger, global_config=self.global_config)

        # get state base message
        ue_state = None
        vshape = 0
        for ue_id in range(self.ue_num):
            ue_state = self.env_manager.get_state_per_user_equipment(ue_id)
            vshape += ue_state.get_state_array().shape[0]

        # get state real message
        # ue_centralized_state_list = self.get_centralized_state_list(ue_state_list)
        # Default/Base scheme
        scheme = {
            "state": {"vshape": vshape},
            "obs": {"vshape": ue_state.get_state_array().shape[0], "group": "agents"},
            "actions": {"vshape": (self.global_config.agent_config.action_config.dimension,), "group": "agents",
                        "dtype": torch.float32},
            "avail_actions": {"vshape": (self.global_config.agent_config.action_config.dimension,),
                              "group": "agents", "dtype": torch.int},
            "reward": {"vshape": (1,)},
            "terminated": {"vshape": (1,), "dtype": torch.uint8},
        }
        groups = {
            "agents": self.ue_num
        }
        if not args.actions_dtype == np.float32:
            preprocess = {
                "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
            }
        else:
            preprocess = {}
        args.n_agents = self.ue_num
        args.n_actions = self.global_config.agent_config.action_config.dimension
        args.state_shape = vshape
        self.batch_size = args.batch_size

        buffer = ReplayBuffer(scheme, groups, self.global_config.agent_config.torch_config.buffer_size,
                              self.global_config.agent_config.torch_config.max_seq_length + 1,
                              preprocess=preprocess,
                              device="cpu")  # might add cpu or gpu

        # Setup multiagent controller here
        mac = mac_REGISTRY["basic_mac"](buffer.scheme, groups, args)
        self.mac = mac

        # Give runner the scheme
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

        # Learner
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

        if args.use_cuda:
            learner.cuda()

        transfer_path = os.path.join(args.env_transfer_path,
                                     '{}_env{}_seed_{}'.format(self.train_config.algorithm, self.train_config.env_idx,
                                                               self.train_config.seed))

        # for DC-DRL algorithm
        if self.train_config.algorithm == 'dc':
            state_dim = self.global_config.agent_config.state_config.dimension
            action_dim = self.global_config.agent_config.action_config.dimension
            observation_space, action_space = get_space(state_dim, action_dim)
            self.agent_dc = Agent_DC(env_name=self.train_config.algorithm, obs_space=observation_space,
                                     act_space=action_space,
                                     device=self.device, writer=self.writer, mode=None, gamma=args.gamma,
                                     batch_size=args.batch_size,
                                     tau=0.001, replay_size=10000, replay_start_size=100,
                                     n_step=True, ou_noise=True, param_noise=False)

        if args.checkpoint_path != "":

            timesteps = []
            timestep_to_load = 0

            if not os.path.isdir(args.checkpoint_path):
                # logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
                print("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
                return

            # Go through all files in args.checkpoint_path
            for name in os.listdir(args.checkpoint_path):
                full_name = os.path.join(args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            model_path = args.checkpoint_path

            # logger.console_logger.info("Loading model from {}".format(model_path))
            print("Loading model from {}".format(model_path))
            learner.load_models(model_path)
            if self.train_config.algorithm == 'maddpg' or self.train_config.algorithm == 'ddpg':
                for idx in range(self.ue_num):
                    self.ddpg_agent.policy_online_network_list[idx].load_state_dict(
                        torch.load("{}/policy_{}.th".format(model_path, str(idx%20)), map_location=lambda storage, loc: storage)
                    )
                    self.ddpg_agent.policy_target_network_list[idx].load_state_dict(
                        torch.load("{}/policy_{}.th".format(model_path, str(idx%20)), map_location=lambda storage, loc: storage)
                    )
                    self.ddpg_agent.critic_online_network_list[idx].load_state_dict(
                        torch.load("{}/critic_ddpg_{}.th".format(model_path, str(idx%20)), map_location=lambda storage, loc: storage)
                    )
                    self.ddpg_agent.critic_target_network_list[idx].load_state_dict(
                        torch.load("{}/critic_ddpg_{}.th".format(model_path, str(idx%20)), map_location=lambda storage, loc: storage)
                    )
                    if self.train_config.algorithm == 'ddpg':
                        break
            elif self.train_config.algorithm == 'dc':
                self.agent_dc.load_model(model_path)
            runner.t_env = timestep_to_load

            if args.evaluate:
                for episode_count in range(args.test_nepisode):
                    print(episode_count)
                    if self.train_config.algorithm == 'maddpg':
                        test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = self.run_one_episode(
                            test_mode=True)
                    elif self.train_config.algorithm == 'ddpg':
                        test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = self.run_one_episode_ddpg(
                            test_mode=True)
                    elif self.train_config.algorithm == 'dc':
                        test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = self.run_one_episode_dc(
                            test_mode=True)
                    else:
                        test_episode_batch, test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = runner.run(
                            self.env_manager, test_mode=True)
                    self.writer.add_scalar("test_1_episode_message/1_step_num", test_step_real, episode_count + 1)
                    if len(test_reward_all):
                        self.writer.add_scalar("test_1_episode_message/2_average_reward",
                                               np.mean(np.array(test_reward_all)),
                                               episode_count + 1)
                    if len(test_cost_all):
                        self.writer.add_scalar("test_1_episode_message/5_average_cost",
                                               np.mean(np.array(test_cost_all)),
                                               episode_count + 1)
                    if len(test_local_baseline_all):
                        self.writer.add_scalar("test_1_episode_message/6_local_cost_baseline",
                                               np.mean(np.array(test_local_baseline_all)),
                                               episode_count + 1)
                    if len(test_offload_num_all):
                        self.writer.add_scalar("test_1_episode_message/7_offload_num_all",
                                               np.mean(np.array(test_offload_num_all)),
                                               episode_count + 1)
                print("run_test_finish!")
                return

        # start training
        episode = 0
        last_test_T = -args.test_interval - 1
        last_log_T = 0
        model_save_time = 0
        opt_min_cost = 99999

        start_time = time.time()
        last_time = start_time

        for episode_count in range(int(self.train_config.episode_num)):
            if self.easy_output_mode:
                if (episode_count + 1) % self.easy_output_cycle == 0:
                    self.print_control = True
                else:
                    self.print_control = False
            else:
                self.print_control = True

            # run one episode
            print("---------------------------------------------")
            print("episode_count:", episode_count)
            # if episode_count == 5:
            #     self.global_config.env_config.hexagon_network_config.user_equipment_num += 10
            # if episode_count == 205:
            #     exit()
            if self.print_control:
                print("Episode {:0>6d}".format(episode_count + 1), ": ", end="")
            if self.train_config.algorithm == 'maddpg':
                step_real, reward_all, cost_all, local_baseline_all, offload_num_all = self.run_one_episode()
            elif self.train_config.algorithm == 'ddpg':
                step_real, reward_all, cost_all, local_baseline_all, offload_num_all = self.run_one_episode_ddpg()
            elif self.train_config.algorithm == 'dc':
                step_real, reward_all, cost_all, local_baseline_all, offload_num_all = self.run_one_episode_dc()
            else:
                episode_batch, step_real, reward_all, cost_all, local_baseline_all, offload_num_all = \
                    runner.run(self.env_manager, test_mode=False)
                buffer.insert_episode_batch(episode_batch)

                if buffer.can_sample(args.batch_size):
                    episode_sample = buffer.sample(args.batch_size)

                    # Truncate batch to only filled timesteps
                    max_ep_t = episode_sample.max_t_filled()
                    episode_sample = episode_sample[:, :max_ep_t]

                    if episode_sample.device != args.device:
                        episode_sample.to(args.device)

                    learner.train(episode_sample, runner.t_env, episode)
            # step_real, reward_all, cost_all = self.run_one_episode()

            # Execute test runs once in a while
            n_test_runs = max(1, 0)  # args.test_nepisode // runner.batch_size
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0 or (
                    self.train_config.algorithm != 'mix' and episode_count % 3000 == 0):

                # logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
                print("\ntest_model:t_env: {} / {}".format(runner.t_env, args.t_max * self.train_config.step_num))
                # logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                #     time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
                last_time = time.time()
                test_episode_batch = None
                test_step_real = None
                test_reward_all = None
                test_cost_all = None
                test_local_baseline_all = None
                test_offload_num_all = None

                last_test_T = runner.t_env
                for _ in range(n_test_runs):
                    if self.train_config.algorithm == 'maddpg':
                        print("MADDPG Running!")
                        test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = self.run_one_episode(
                            test_mode=True)
                    elif self.train_config.algorithm == 'ddpg':
                        print("DDPG Running!")
                        test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = self.run_one_episode_ddpg(
                            test_mode=True)
                    elif self.train_config.algorithm == 'dc':
                        print("DC-DRL Running!")
                        test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = self.run_one_episode_dc(
                            test_mode=True)
                    else:
                        print("DEAT Running!")
                        test_episode_batch, test_step_real, test_reward_all, test_cost_all, test_local_baseline_all, test_offload_num_all = runner.run(
                            self.env_manager, test_mode=True)
                self.writer.add_scalar("test_1_episode_message/1_step_num", test_step_real, episode_count + 1)
                if len(test_reward_all):
                    self.writer.add_scalar("test_1_episode_message/2_average_reward",
                                           np.mean(np.array(test_reward_all)),
                                           episode_count + 1)
                if len(test_cost_all):
                    self.writer.add_scalar("test_1_episode_message/5_average_cost", np.mean(np.array(test_cost_all)),
                                           episode_count + 1)
                if len(test_local_baseline_all):
                    self.writer.add_scalar("test_1_episode_message/6_local_cost_baseline",
                                           np.mean(np.array(test_local_baseline_all)),
                                           episode_count + 1)
                if len(test_offload_num_all):
                    self.writer.add_scalar("test_1_episode_message/7_offload_num_all",
                                           np.mean(np.array(test_offload_num_all)),
                                           episode_count + 1)

            if args.save_model and ((opt_min_cost - np.mean(np.array(cost_all))) > 1e-5 or episode_count % 3000 == 0):
                opt_min_cost = np.mean(np.array(cost_all))
                model_save_cost_name = np.mean(np.array(cost_all))
                save_path = os.path.join(args.local_results_path,
                                         "models_" + self.exp_time + "_" + str(self.train_config.algorithm),
                                         str(episode_count) + 'cost_' + str(model_save_cost_name))
                # I delete args.unique_token,
                # "results/models/{}".format(unique_token)
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(transfer_path, exist_ok=True)
                # logger.console_logger.info("Saving models to {}".format(save_path))
                print("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)
                learner.save_models(transfer_path)

                if self.train_config.algorithm == 'maddpg' or self.train_config.algorithm == 'ddpg':
                    for idx in range(len(self.ddpg_agent.policy_online_network_list)):
                        torch.save(self.ddpg_agent.policy_online_network_list[idx].state_dict(), "{}/policy_{}.th".format(save_path, str(idx)))
                        torch.save(self.ddpg_agent.critic_online_network_list[idx].state_dict(), "{}/critic_ddpg_{}.th".format(save_path, str(idx)))
                elif self.train_config.algorithm == 'dc':
                    self.agent_dc.save_model(save_path)

            assert episode == episode_count
            episode += args.batch_size_run

            if (runner.t_env - last_log_T) >= args.log_interval:
                # logger.log_stat("episode", episode, runner.t_env)
                # logger.print_recent_stats()
                print("episode", episode, runner.t_env)
                last_log_T = runner.t_env

            if not self.output_network_message:
                return

            self.writer.add_scalar("1_episode_message/1_step_num", step_real, episode_count + 1)
            if len(reward_all):
                self.writer.add_scalar("1_episode_message/2_average_reward", np.mean(np.array(reward_all)),
                                       episode_count + 1)
            if learner.loss or learner.critic_loss:
                self.writer.add_scalar("1_episode_message/3_episode_q_learner_loss",
                                       learner.critic_loss.item(), episode_count + 1)
            if learner.actor_loss:
                self.writer.add_scalar("1_episode_message/4_average_policy_loss",
                                       learner.actor_loss.item(), episode_count + 1)
                self.ddpg_agent.temp_policy_loss_list = []
            if len(cost_all):
                self.writer.add_scalar("1_episode_message/5_average_cost", np.mean(np.array(cost_all)),
                                       episode_count + 1)
            if len(local_baseline_all):
                self.writer.add_scalar("1_episode_message/6_local_cost_baseline", np.mean(np.array(local_baseline_all)),
                                       episode_count + 1)
            if len(offload_num_all):
                self.writer.add_scalar("1_episode_message/7_offload_num_all", np.mean(np.array(offload_num_all)),
                                       episode_count + 1)
            print("---------------------------------------------")

        print("Finished Training")

    def run_one_episode(self, test_mode=False):
        self.ue_num = self.global_config.env_config.hexagon_network_config.user_equipment_num
        self.writer = self.env_manager.writer
        # reset the env
        self.env_manager.reset_environment_interface()
        episode_return = 0
        self.t_step_count = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        user_equipment_list = self.env_manager.hexagon_network.user_equipment_list

        # init the data structure to save episode data
        step_num_this_episode = 0
        reward_all_this_episode = []
        cost_all_this_episode = []
        local_cost_baseline_all_this_episode = []
        offload_num_all_this_episode = []

        ue_queue_time_now = np.zeros(self.ue_num)

        for step_count in range(self.train_config.step_num):

            self.step_real_count += 1
            self.env_manager.step_real_count += 1
            step_num_this_episode += 1

            # get state base message
            ue_state_list = []
            ue_state_list_ddpg = []
            ue_state_array = np.array([])
            ue_state_channel_gain_array = np.array([])
            ue_action_list = []

            computing_resources_ratio_list = []
            ES_computing_ratio = 0

            for ue_id in range(self.ue_num):
                ue_state = self.env_manager.get_state_per_user_equipment(ue_id)

                ue_state_list.append(ue_state)
                ue_state_list_ddpg.append(self.env_manager.get_state_per_user_equipment(ue_id).get_state_list())
                ue_state_array = np.append(ue_state_array, ue_state.get_state_array())
                # ue_state_channel_gain_array = np.append(ue_state_channel_gain_array, ue_state.channel_gain_to_all)
                ue_action_array = np.ones(self.global_config.agent_config.action_config.dimension)
                ue_action_list.append(ue_action_array.tolist())

            self.env_manager.ue_state_channel_gain_array = ue_state_channel_gain_array.reshape(self.ue_num, -1)
            # get state real message
            # ue_centralized_state_list = self.get_centralized_state_list(ue_state_list)

            # get action real message
            ue_action_list = []
            action_offload_mask = np.zeros((self.ue_num, int(self.action_config.dimension / 2)))
            action_offload_mask = action_offload_mask[:, 0]

            for ue_id in range(self.ue_num):
                if np.random.rand() < self.cur_epsilon and test_mode==False:
                    self.cur_epsilon *= self.ddpg_agent.epsilon_decay
                    ue_action_array = select_action_random(self.global_config)
                else:
                    ue_action_array = \
                        self.select_action_with_noise(torch.tensor(ue_state_list[ue_id].get_state_array(),
                                                                   device=self.device, dtype=torch.float32), ue_id)
                partial_offloading = ue_action_array[0]
                transmit_power = ue_action_array[1]
                ue_action = Action([partial_offloading, transmit_power], self.global_config)
                ue_action_list.append(ue_action)
                action_offload_mask[ue_id] = \
                    np.where(partial_offloading >= self.action_config.threshold_to_offload, 1, 0)

            # set all ue_normalized_transmitting_power, local zero
            ue_normalized_transmitting_power_array = np.zeros_like(action_offload_mask)
            for ue_id in range(self.ue_num):
                if ue_action_list[ue_id].get_whether_offload():
                    ue_normalized_transmitting_power_array[ue_id] = \
                        ue_action_list[ue_id].normalized_transmitting_power
            self.env_manager.ue_normalized_transmitting_power_array = ue_normalized_transmitting_power_array

            # init some list to save data before run one step
            ue_reward_list = []
            ue_next_state_list = []
            local_cost_baseline_list = []
            # ue_done_list = []
            ue_done_all = False
            ue_cost_list = []
            offload_count = 0
            for ue_id in range(self.ue_num):
                ue_queue_time_now[ue_id] = ue_state_list[ue_id].user_equipment_queue_time_now
                if ue_action_list[ue_id].get_whether_offload():
                    offload_count += 1

            base_station_queue_time_now = ue_state_list[0].base_station_queue_time_now
            self.env_manager.time_matrix = np.zeros_like(action_offload_mask)
            self.env_manager.energy_matrix = np.zeros_like(action_offload_mask)

            # run one step
            for ue_id in range(self.ue_num):
                # print(self.env_manager.step(ue_state_list[user_equipment_id], ue_action_list[user_equipment_id]))
                ue_reward, ue_next_state, ue_done, cost, queue_time_now, base_station_queue_time_now, local_cost_baseline = \
                    self.env_manager.step(ue_state_list[ue_id], ue_action_list[ue_id], offload_count,
                                          ue_queue_time_now, base_station_queue_time_now, action_offload_mask)
                ue_reward_list.append(ue_reward)
                ue_next_state_list.append(ue_next_state)
                # ue_done_list.append(ue_done)
                ue_done_all = ue_done_all or ue_done
                ue_cost_list.append(cost)
                local_cost_baseline_list.append(local_cost_baseline)
            self.env_manager.update_base_station_queue_time(self.ue_num, base_station_queue_time_now)
            reward = np.mean(ue_reward_list)
            temp_reward = reward
            # print("offload_count:", offload_count)
            cost_avg = np.mean(ue_cost_list)
            cost_baseline_avg = np.mean(local_cost_baseline_list)
            reward += (cost_baseline_avg - cost_avg) * 500
            ue_reward_list = [x + ((cost_baseline_avg - cost_avg) * 500) for x in ue_reward_list]
            # print(
            #     "cost:{}, baseline:{}, origin_r:{} reward:{}".format(cost_avg, cost_baseline_avg, temp_reward, reward))

            if step_count == self.train_config.step_num - 1:
                ue_done_all = True

            # get next state real message
            # ue_centralized_next_state_list = self.get_centralized_state_list(ue_next_state_list)

            # add to writer and debug, add to memory and train
            for ue_id in range(self.ue_num):
                if test_mode:
                    break
                if not self.output_action_message:
                    continue
                self.writer.add_scalar(
                    '2_action_message/ue_id_' + str(ue_id) + '_1_whether_offload',
                    ue_action_list[ue_id].normalized_whether_offload,
                    self.step_real_count)
                self.writer.add_scalar(
                    '2_action_message/ue_id_' + str(ue_id) + '_2_normalized_transmitting_power',
                    ue_action_list[ue_id].normalized_transmitting_power,
                    self.step_real_count)

            if test_mode:
                for ue_id in range(self.ue_num):
                    if not self.output_action_message:
                        continue
                    self.writer.add_scalar(
                        'test_2_action_message/ue_id_' + str(ue_id) + '_1_whether_offload',
                        ue_action_list[ue_id].normalized_whether_offload,
                        self.step_real_count)
                    self.writer.add_scalar(
                        '2_action_message/ue_id_' + str(ue_id) + '_2_tx_power',
                        ue_action_list[ue_id].normalized_transmitting_power,
                        self.step_real_count)
            # Add experience to replay buffer
            self.ddpg_agent.experience_pool.push([ue_state_list[ue_id].get_state_list()
                                                  for ue_id in range(self.ue_num)],
                                                 [ue_action_list[ue_id].get_action_list()
                                                  for ue_id in range(self.ue_num)],
                                                 [ue_next_state_list[ue_id].get_state_list()
                                                  for ue_id in range(self.ue_num)],
                                                 ue_reward_list,
                                                 ue_done_all)

            if self.ddpg_agent.experience_pool.whether_full():
                if self.print_control:
                    print("t", end="")
                self.ddpg_agent.train_model()

            assert step_count == self.t_step_count

            self.t_step_count += 1

            # save the cost and the reward
            reward_all_this_episode.append(ue_reward_list)
            cost_all_this_episode.append(ue_cost_list)
            local_cost_baseline_all_this_episode.append(local_cost_baseline_list)
            offload_num_all_this_episode.append(offload_count)

            if bool(self.env_manager.reason_to_finish_this_episode):
                break
            else:
                self.env_manager.update_the_hexagon_network()

        if self.print_control:
            print(step_num_this_episode, end=" ")
        if step_num_this_episode != 50:
            if self.global_config.debug_config.whether_output_finish_episode_reason:
                if self.print_control:
                    print(self.env_manager.reason_to_finish_this_episode)
        else:
            if self.print_control:
                print()

        return step_num_this_episode, reward_all_this_episode, cost_all_this_episode, \
               local_cost_baseline_all_this_episode, offload_num_all_this_episode

    def get_centralized_state_list(self, ue_state_list: list):
        # get state plus message
        # ue_sorted_state_list = sorted(ue_state_list, key=lambda state: state.judge_offload, reverse=True)
        ue_state_plus_array = np.zeros((self.ue_num, 5))
        # for user_equipment_id in range(self.user_equipment_num):
        #     plus_message_num = 0
        #     for ue_sorted_state in ue_sorted_state_list:
        #         if user_equipment_id != ue_sorted_state.user_equipment_id and plus_message_num < 5:
        #             ue_state_plus_array[user_equipment_id][plus_message_num] = ue_sorted_state.judge_offload
        #             plus_message_num += 1

        # get state real message
        ue_centralized_state_list = []
        for user_equipment_id in range(self.ue_num):
            ue_centralized_state = ue_state_list[user_equipment_id].get_state_list().copy()
            # ue_centralized_state.extend(ue_state_plus_array[user_equipment_id].tolist())
            ue_centralized_state_list.extend(ue_centralized_state)

        return ue_centralized_state_list

    def run_one_episode_ddpg(self, test_mode=False):
        self.ue_num = self.global_config.env_config.hexagon_network_config.user_equipment_num
        self.writer = self.env_manager.writer
        # reset the env
        self.env_manager.reset_environment_interface()
        episode_return = 0
        self.t_step_count = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        user_equipment_list = self.env_manager.hexagon_network.user_equipment_list

        # init the data structure to save episode data
        step_num_this_episode = 0
        reward_all_this_episode = []
        cost_all_this_episode = []
        local_cost_baseline_all_this_episode = []
        offload_num_all_this_episode = []

        ue_queue_time_now = np.zeros(self.ue_num)

        for step_count in range(self.train_config.step_num):

            self.step_real_count += 1
            self.env_manager.step_real_count += 1
            step_num_this_episode += 1

            # get state base message
            ue_state_list = []
            ue_state_list_ddpg = []
            ue_state_array = np.array([])
            ue_state_channel_gain_array = np.array([])
            ue_action_list = []

            computing_resources_ratio_list = []
            ES_computing_ratio = 0

            for ue_id in range(self.ue_num):
                ue_state = self.env_manager.get_state_per_user_equipment(ue_id)

                ue_state_list.append(ue_state)
                ue_state_list_ddpg.append(self.env_manager.get_state_per_user_equipment(ue_id).get_state_list())
                ue_state_array = np.append(ue_state_array, ue_state.get_state_array())
                ue_action_array = np.ones(self.global_config.agent_config.action_config.dimension)
                ue_action_list.append(ue_action_array.tolist())

            self.env_manager.ue_state_channel_gain_array = ue_state_channel_gain_array.reshape(self.ue_num, -1)
            # get state real message
            ue_centralized_state_list = self.get_centralized_state_list(ue_state_list)

            # get action real message
            ue_action_list = []
            action_offload_mask = np.zeros((self.ue_num, int(self.action_config.dimension / 2 / self.ue_num)))
            action_offload_mask = action_offload_mask[:, 0]

            if np.random.rand() < self.cur_epsilon and test_mode==False:
                self.cur_epsilon *= self.ddpg_agent.epsilon_decay
                ue_action_array = select_action_random(self.global_config)
            else:
                ue_action_array = \
                    self.select_action_with_noise(torch.tensor(np.array(ue_centralized_state_list),
                                                               device=self.device, dtype=torch.float32), 0)
            ue_action_array_shaped = ue_action_array.reshape(self.ue_num, -1)

            for ue_id in range(self.ue_num):
                partial_offloading = ue_action_array_shaped[ue_id][0]
                transmit_power = ue_action_array_shaped[ue_id][1]
                ue_action = Action([partial_offloading, transmit_power], self.global_config)
                ue_action_list.append(ue_action)
                action_offload_mask[ue_id] = \
                    np.where(partial_offloading >= self.action_config.threshold_to_offload, 1, 0)

            # set all ue_normalized_transmitting_power, local zero
            ue_normalized_transmitting_power_array = np.zeros(self.ue_num)
            for ue_id in range(self.ue_num):
                if ue_action_list[ue_id].get_whether_offload():
                    ue_normalized_transmitting_power_array[ue_id] = \
                        ue_action_list[ue_id].normalized_transmitting_power
            self.env_manager.ue_normalized_transmitting_power_array = ue_normalized_transmitting_power_array

            # init some list to save data before run one step
            ue_reward_list = []
            ue_next_state_list = []
            local_cost_baseline_list = []
            # ue_done_list = []
            ue_done_all = False
            ue_cost_list = []
            offload_count = 0

            for ue_id in range(self.ue_num):
                ue_queue_time_now[ue_id] = ue_state_list[ue_id].user_equipment_queue_time_now
                if ue_action_list[ue_id].get_whether_offload():
                    offload_count += 1

            base_station_queue_time_now = ue_state_list[0].base_station_queue_time_now
            self.env_manager.time_matrix = np.zeros_like(action_offload_mask)
            self.env_manager.energy_matrix = np.zeros_like(action_offload_mask)

            # run one step
            for ue_id in range(self.ue_num):
                # print(self.env_manager.step(ue_state_list[user_equipment_id], ue_action_list[user_equipment_id]))
                ue_reward, ue_next_state, ue_done, cost, queue_time_now, base_station_queue_time_now, local_cost_baseline = \
                    self.env_manager.step(ue_state_list[ue_id], ue_action_list[ue_id], offload_count,
                                          ue_queue_time_now, base_station_queue_time_now, action_offload_mask)
                ue_reward_list.append(ue_reward)
                ue_next_state_list.append(ue_next_state)
                ue_done_all = ue_done_all or ue_done
                ue_cost_list.append(cost)
                local_cost_baseline_list.append(local_cost_baseline)
            self.env_manager.update_base_station_queue_time(self.ue_num, base_station_queue_time_now)
            reward = np.mean(ue_reward_list)
            temp_reward = reward
            cost_avg = np.mean(ue_cost_list)
            cost_baseline_avg = np.mean(local_cost_baseline_list)
            reward += (cost_baseline_avg - cost_avg) * 500
            ue_reward_list = [x + ((cost_baseline_avg - cost_avg) * 500) for x in ue_reward_list]
            # print(
            #     "cost:{}, baseline:{}, origin_r:{} reward:{}".format(cost_avg, cost_baseline_avg, temp_reward, reward))

            if step_count == self.train_config.step_num - 1:
                ue_done_all = True

            # get next state real message
            # ue_centralized_next_state_list = self.get_centralized_state_list(ue_next_state_list)

            # add to writer and debug, add to memory and train
            for ue_id in range(self.ue_num):
                if test_mode:
                    break
                if not self.output_action_message:
                    continue
                self.writer.add_scalar(
                    '2_action_message/ue_id_' + str(ue_id) + '_1_whether_offload',
                    ue_action_list[ue_id].normalized_whether_offload,
                    self.step_real_count)
                self.writer.add_scalar(
                    '2_action_message/ue_id_' + str(ue_id) + '_2_normalized_transmitting_power',
                    ue_action_list[ue_id].normalized_transmitting_power,
                    self.step_real_count)

            if test_mode:
                for ue_id in range(self.ue_num):
                    if not self.output_action_message:
                        continue
                    self.writer.add_scalar(
                        'test_2_action_message/ue_id_' + str(ue_id) + '_1_whether_offload',
                        ue_action_list[ue_id].normalized_whether_offload,
                        self.step_real_count)
                    self.writer.add_scalar(
                        'test_2_action_message/ue_id_' + str(ue_id) + '_2_normalized_transmitting_power',
                        ue_action_list[ue_id].normalized_transmitting_power,
                        self.step_real_count)

            # Add experience to replay buffer
            self.ddpg_agent.experience_pool.push([ue_state_list[ue_id].get_state_list()
                                                  for ue_id in range(self.ue_num)],
                                                 [ue_action_list[ue_id].get_action_list()
                                                  for ue_id in range(self.ue_num)],
                                                 [ue_next_state_list[ue_id].get_state_list()
                                                  for ue_id in range(self.ue_num)],
                                                 reward,
                                                 ue_done_all)

            if self.ddpg_agent.experience_pool.whether_full():
                if self.print_control:
                    print("t", end="")
                self.ddpg_agent.train_model()
                '''self.ddpg_agent.help_train_model(ue_reward_list.index(max(ue_reward_list)),
                                                 ue_reward_list.index(min(ue_reward_list)))'''
                # self.ddpg_agent.help_train_model(ue_cost_list.index(min(ue_cost_list)),
                #                                  ue_cost_list.index(max(ue_cost_list)))

            assert step_count == self.t_step_count

            self.t_step_count += 1

            # save the cost and the reward
            reward_all_this_episode.append(ue_reward_list)
            cost_all_this_episode.append(ue_cost_list)
            local_cost_baseline_all_this_episode.append(local_cost_baseline_list)
            offload_num_all_this_episode.append(offload_count)

            if bool(self.env_manager.reason_to_finish_this_episode):
                break
            else:
                self.env_manager.update_the_hexagon_network()

        if self.print_control:
            print(step_num_this_episode, end=" ")
        if step_num_this_episode != 50:
            if self.global_config.debug_config.whether_output_finish_episode_reason:
                if self.print_control:
                    print(self.env_manager.reason_to_finish_this_episode)
        else:
            if self.print_control:
                print()

        return step_num_this_episode, reward_all_this_episode, cost_all_this_episode, \
               local_cost_baseline_all_this_episode, offload_num_all_this_episode

    def run_one_episode_dc(self, test_mode=False):
        self.ue_num = self.global_config.env_config.hexagon_network_config.user_equipment_num
        self.writer = self.env_manager.writer
        # reset the env
        self.env_manager.reset_environment_interface()
        episode_return = 0
        self.t_step_count = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        user_equipment_list = self.env_manager.hexagon_network.user_equipment_list

        # init the data structure to save episode data
        step_num_this_episode = 0
        reward_all_this_episode = []
        cost_all_this_episode = []
        local_cost_baseline_all_this_episode = []
        offload_num_all_this_episode = []

        ue_queue_time_now = np.zeros(self.ue_num)

        for step_count in range(self.train_config.step_num):

            self.step_real_count += 1
            self.env_manager.step_real_count += 1
            step_num_this_episode += 1

            # get state base message
            ue_state_list = []
            ue_state_list_ddpg = []
            ue_state_array = np.array([])
            ue_state_channel_gain_array = np.array([])
            ue_action_list = []

            computing_resources_ratio_list = []
            ES_computing_ratio = 0

            for ue_id in range(self.ue_num):
                ue_state = self.env_manager.get_state_per_user_equipment(ue_id)

                ue_state_list.append(ue_state)
                ue_state_list_ddpg.append(self.env_manager.get_state_per_user_equipment(ue_id).get_state_list())
                ue_state_array = np.append(ue_state_array, ue_state.get_state_array())
                ue_action_array = np.ones(self.global_config.agent_config.action_config.dimension)
                ue_action_list.append(ue_action_array.tolist())

            self.env_manager.ue_state_channel_gain_array = ue_state_channel_gain_array.reshape(self.ue_num, -1)
            # get state real message
            ue_centralized_state_list = self.get_centralized_state_list(ue_state_list)

            # get action real message
            ue_action_list = []
            action_offload_mask = np.zeros((self.ue_num, int(self.action_config.dimension / 2 / self.ue_num)))
            action_offload_mask = action_offload_mask[:, 0]

            if np.random.rand() < self.cur_epsilon and test_mode==False:
                self.cur_epsilon *= self.ddpg_agent.epsilon_decay
                ue_action_array = select_action_random(self.global_config)
            else:
                ue_action_array = \
                    self.agent_dc.select_action(torch.tensor(np.array(ue_centralized_state_list),
                                                             device=self.device, dtype=torch.float32))
            ue_action_array_shaped = ue_action_array.reshape(self.ue_num, -1)

            for ue_id in range(self.ue_num):
                partial_offloading = ue_action_array_shaped[ue_id][0].item()
                transmit_power = (ue_action_array_shaped[ue_id][1].item() + 1) / 2
                ue_action = Action([partial_offloading, transmit_power], self.global_config)
                ue_action_list.append(ue_action)
                action_offload_mask[ue_id] = \
                    np.where(partial_offloading >= self.action_config.threshold_to_offload, 1, 0)

            # set all ue_normalized_transmitting_power, local zero
            ue_normalized_transmitting_power_array = np.zeros(self.ue_num)
            for ue_id in range(self.ue_num):
                if ue_action_list[ue_id].get_whether_offload():
                    ue_normalized_transmitting_power_array[ue_id] = \
                        ue_action_list[ue_id].normalized_transmitting_power
            self.env_manager.ue_normalized_transmitting_power_array = ue_normalized_transmitting_power_array

            # init some list to save data before run one step
            ue_reward_list = []
            ue_next_state_list = []
            local_cost_baseline_list = []
            # ue_done_list = []
            ue_done_all = False
            ue_cost_list = []
            offload_count = 0

            for ue_id in range(self.ue_num):
                ue_queue_time_now[ue_id] = ue_state_list[ue_id].user_equipment_queue_time_now
                if ue_action_list[ue_id].get_whether_offload():
                    offload_count += 1

            base_station_queue_time_now = ue_state_list[0].base_station_queue_time_now
            self.env_manager.time_matrix = np.zeros_like(action_offload_mask)
            self.env_manager.energy_matrix = np.zeros_like(action_offload_mask)

            # run one step
            for ue_id in range(self.ue_num):
                # print(self.env_manager.step(ue_state_list[user_equipment_id], ue_action_list[user_equipment_id]))
                ue_reward, ue_next_state, ue_done, cost, queue_time_now, base_station_queue_time_now, local_cost_baseline = \
                    self.env_manager.step(ue_state_list[ue_id], ue_action_list[ue_id], offload_count,
                                          ue_queue_time_now, base_station_queue_time_now, action_offload_mask)
                ue_reward_list.append(ue_reward)
                ue_next_state_list.append(ue_next_state)
                # ue_done_list.append(ue_done)
                ue_done_all = ue_done_all or ue_done
                ue_cost_list.append(cost)
                local_cost_baseline_list.append(local_cost_baseline)
            self.env_manager.update_base_station_queue_time(self.ue_num, base_station_queue_time_now)
            reward = np.mean(ue_reward_list)
            temp_reward = reward
            # print("offload_count:", offload_count)
            cost_avg = np.mean(ue_cost_list)
            cost_baseline_avg = np.mean(local_cost_baseline_list)
            reward += (cost_baseline_avg - cost_avg) * 500
            ue_reward_list = [x + ((cost_baseline_avg - cost_avg) * 500) for x in ue_reward_list]
            # print(
            #     "cost:{}, baseline:{}, origin_r:{} reward:{}".format(cost_avg, cost_baseline_avg, temp_reward, reward))

            if step_count == self.train_config.step_num - 1:
                ue_done_all = True

            # get next state real message
            # ue_centralized_next_state_list = self.get_centralized_state_list(ue_next_state_list)

            # add to writer and debug, add to memory and train
            for ue_id in range(self.ue_num):
                if test_mode:
                    break
                if not self.output_action_message:
                    continue
                self.writer.add_scalar(
                    '2_action_message/ue_id_' + str(ue_id) + '_1_whether_offload',
                    ue_action_list[ue_id].normalized_whether_offload,
                    self.step_real_count)
                self.writer.add_scalar(
                    '2_action_message/ue_id_' + str(ue_id) + '_2_normalized_transmitting_power',
                    ue_action_list[ue_id].normalized_transmitting_power,
                    self.step_real_count)

            if test_mode:
                for ue_id in range(self.ue_num):
                    if not self.output_action_message:
                        continue
                    self.writer.add_scalar(
                        'test_2_action_message/ue_id_' + str(ue_id) + '_1_whether_offload',
                        ue_action_list[ue_id].normalized_whether_offload,
                        self.step_real_count)
                    self.writer.add_scalar(
                        'test_2_action_message/ue_id_' + str(ue_id) + '_2_normalized_transmitting_power',
                        ue_action_list[ue_id].normalized_transmitting_power,
                        self.step_real_count)

            state_list = torch.tensor(
                np.array([ue_state_list[ue_id].get_state_list() for ue_id in range(self.ue_num)]).reshape(-1,
                    self.global_config.agent_config.state_config.dimension)).float().to(self.device)
            action_list = torch.tensor(
                np.array([ue_action_list[ue_id].get_action_list() for ue_id in range(self.ue_num)]).reshape(-1,
                    self.global_config.agent_config.action_config.dimension)).float().to(self.device)
            next_state_list = torch.tensor(
                np.array([ue_next_state_list[ue_id].get_state_list() for ue_id in range(self.ue_num)]).reshape(-1,
                    self.global_config.agent_config.state_config.dimension)).float().to(self.device)
            mask = torch.tensor([not ue_done_all]).float().to(self.device)
            reward = torch.tensor([reward]).float().to(self.device)

            self.agent_dc.adv_preceive(state_list, action_list, mask, next_state_list, reward)

            if len(self.agent_dc.replay_buffer) > self.agent_dc.replay_start_size:
                for _ in range(self.agent_dc.updates_per_step):
                    batch = self.agent_dc.replay_buffer.sample(self.agent_dc.batch_size)
                    # batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = self.agent_dc.adv_update(batch)
                    # if agent.writer:
                    #     agent.writer.add_scalar('loss/value', value_loss, agent.updates)
                    #     agent.writer.add_scalar('loss/policy', policy_loss, agent.updates)

            assert step_count == self.t_step_count

            self.t_step_count += 1

            # save the cost and the reward
            reward_all_this_episode.append(ue_reward_list)
            cost_all_this_episode.append(ue_cost_list)
            local_cost_baseline_all_this_episode.append(local_cost_baseline_list)
            offload_num_all_this_episode.append(offload_count)

            if bool(self.env_manager.reason_to_finish_this_episode):
                break
            else:
                self.env_manager.update_the_hexagon_network()

        if self.print_control:
            print(step_num_this_episode, end=" ")
        if step_num_this_episode != 50:
            if self.global_config.debug_config.whether_output_finish_episode_reason:
                if self.print_control:
                    print(self.env_manager.reason_to_finish_this_episode)
        else:
            if self.print_control:
                print()

        return step_num_this_episode, reward_all_this_episode, cost_all_this_episode, \
               local_cost_baseline_all_this_episode, offload_num_all_this_episode
