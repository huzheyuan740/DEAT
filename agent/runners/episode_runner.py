# from src.envs import REGISTRY as env_REGISTRY # 星际争霸2环境的访问器 不需要
import time
import datetime
import os
from tensorboardX import SummaryWriter
from functools import partial
from agent.components.episode_buffer import EpisodeBatch
import numpy as np
import torch
from config import GlobalConfig
from env.env_interface import EnvironmentManager
from agent.ddpg import Agent
from agent.action import Action


class EpisodeRunner:

    def __init__(self, args, logger, global_config):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = 50
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

        # add for run_one_episode in run_ddpg_in_env.py
        self.global_config = global_config
        self.train_config = self.global_config.train_config
        self.env_manager = None
        self.output_network_message = self.global_config.control_config.output_network_config
        self.output_action_message = self.global_config.control_config.output_action_config
        self.easy_output_mode = self.global_config.control_config.easy_output_mode
        self.easy_output_cycle = self.global_config.control_config.easy_output_cycle
        self.print_control = True

        # init the parameter
        self.ue_num = self.global_config.env_config.hexagon_network_config.user_equipment_num
        # self.cur_epsilon = self.ddpg_agent.epsilon_max

        # save the real step count
        self.step_real_count = 0
        self.writer = None

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        print("Finish all episode")
        # self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env_manager.reset_environment_interface()
        self.t = 0

    def run(self, env_manager, test_mode=False):
        self.ue_num = self.global_config.env_config.hexagon_network_config.user_equipment_num
        self.env_manager = env_manager
        self.writer = self.env_manager.writer
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        # init the data structure to save episode data
        step_num_this_episode = 0
        reward_all_this_episode = []
        cost_all_this_episode = []
        local_cost_baseline_all_this_episode = []
        offload_num_all_this_episode = []

        last_ue_state_array = None
        last_ue_action_list = []
        ue_queue_time_now = np.zeros(self.ue_num)
        base_station_queue_time_now = 0

        for step_count in range(self.train_config.step_num):

            self.step_real_count += 1
            self.env_manager.step_real_count += 1
            step_num_this_episode += 1

            # get state base message
            ue_state_list = []
            ue_state_array = np.array([])
            ue_action_list = []
            for ue_id in range(self.ue_num):
                ue_state = self.env_manager.get_state_per_user_equipment(ue_id)
                ue_state_list.append(ue_state)
                ue_state_array = np.append(ue_state_array, ue_state.get_state_array())
                random_index = np.random.randint(0, 2)
                ue_action_array = np.ones(self.global_config.agent_config.action_config.dimension)
                ue_action_list.append(ue_action_array.tolist())

            pre_transition_data = {
                "state": [ue_state_array],
                "avail_actions": [ue_action_list],
                "obs": ue_state_array.reshape(self.ue_num, -1)
            }
            self.batch.update(pre_transition_data, ts=step_count)

            actions = self.mac.select_actions(self.batch, t_ep=step_count, t_env=self.t_env, test_mode=test_mode)
            ue_action_list = []
            action_offload_mask = np.zeros(self.ue_num)

            for action_idx in range(self.ue_num):
                ue_action = \
                    Action(np.array([actions[0][action_idx][0].item(),
                                     (actions[0][action_idx][1].item() + 1) / 2]), self.global_config)
                ue_action_list.append(ue_action)
                action_offload_mask[action_idx] = np.where(actions[0][action_idx][0].item() > 0, 1, 0)

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
            for ue_id in range(self.ue_num):

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
            # print(
            #     "cost:{}, baseline:{}, origin_r:{} reward:{}".format(cost_avg, cost_baseline_avg, temp_reward, reward))


            if step_count == self.train_config.step_num - 1:
                ue_done_all = True

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
                    '2_action_message/ue_id_' + str(ue_id) + '_2_tx_power',
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

            episode_return += reward
            terminated = False
            if ue_done_all:
                terminated = True
            else:
                terminated = False

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated,)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            assert step_count == self.t

            self.t += 1

            # save the cost and the reward
            reward_all_this_episode.append(reward)
            cost_all_this_episode.append(ue_cost_list)
            local_cost_baseline_all_this_episode.append(local_cost_baseline_list)
            offload_num_all_this_episode.append(offload_count)

            if bool(self.env_manager.reason_to_finish_this_episode) or terminated:
                last_ue_state_array = ue_state_array
                last_ue_action_list = ue_action_list
                break
            else:
                self.env_manager.update_the_hexagon_network()

        # get state base message
        ue_state_list = []
        ue_state_array = np.array([])
        ue_action_list = []
        for ue_id in range(self.ue_num):
            ue_state = self.env_manager.get_state_per_user_equipment(ue_id)
            ue_state_list.append(ue_state)
            ue_state_array = np.append(ue_state_array, ue_state.get_state_array())
            random_index = np.random.randint(0, 2)
            ue_action_array = np.ones(self.global_config.agent_config.action_config.dimension)
            ue_action_list.append(ue_action_array.tolist())

        last_data = {
            "state": [ue_state_array],
            "avail_actions": [ue_action_list],
            "obs": ue_state_array.reshape(self.ue_num, -1)
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t

        return self.batch, step_num_this_episode, reward_all_this_episode, cost_all_this_episode, \
               local_cost_baseline_all_this_episode, offload_num_all_this_episode

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
        stats.clear()
