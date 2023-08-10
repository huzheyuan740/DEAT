import copy

import numpy as np
import torch.optim as optim

from agent.common.buffers import *
from agent.common.networks import *
from agent.common.utils import *
from config import GlobalConfig


class Agent(object):

    def __init__(self, device, global_config: GlobalConfig):
        self.device = device
        self.agent_config = global_config.agent_config
        self.train_config = global_config.train_config

        self.user_equipment_num = global_config.env_config.hexagon_network_config.user_equipment_num

        # load all state config
        self.state_dim = self.agent_config.state_config.dimension

        # load all action config
        self.action_dim = self.agent_config.action_config.dimension
        self.action_noise = self.agent_config.action_config.action_noise
        self.action_noise_decay = self.agent_config.action_config.action_noise_decay
        self.threshold_to_offload = self.agent_config.action_config.threshold_to_offload

        # load all torch config
        self.gamma = self.agent_config.torch_config.gamma
        self.hidden_sizes = self.agent_config.torch_config.hidden_sizes
        self.buffer_size = self.agent_config.torch_config.buffer_size
        self.batch_size = self.agent_config.torch_config.batch_size
        self.policy_learning_rate = self.agent_config.torch_config.policy_learning_rate
        self.critic_learning_rate = self.agent_config.torch_config.critic_learning_rate
        self.policy_gradient_clip = self.agent_config.torch_config.policy_gradient_clip
        self.critic_gradient_clip = self.agent_config.torch_config.critic_gradient_clip
        self.epsilon_max = self.agent_config.torch_config.epsilon_max
        self.epsilon_decay = self.agent_config.torch_config.epsilon_decay
        self.action_limit = self.agent_config.torch_config.action_limit

        self.temp_policy_loss_list = []
        self.temp_critic_loss_list = []

        torch.set_default_dtype(torch.float)
        ue_num_for_build_list = self.user_equipment_num

        if self.train_config.algorithm == 'ddpg':
            ue_num_for_build_list = 1

        self.policy_online_network_list = [PolicyNetwork(self.state_dim, self.action_dim, (128, 128)).to(self.device)
                                           for _ in range(ue_num_for_build_list)]
        self.critic_online_network_list = [CriticNetwork(self.state_dim, self.action_dim,
                                                         (1024, 512, 256, 128), ue_num_for_build_list).to(self.device)
                                           for _ in range(ue_num_for_build_list)]

        self.policy_target_network_list = copy.deepcopy(self.policy_online_network_list)
        self.critic_target_network_list = copy.deepcopy(self.critic_online_network_list)

        self.policy_optimizer_list = [optim.Adam(x.parameters(), lr=self.policy_learning_rate)
                                      for x in self.policy_online_network_list]
        self.critic_optimizer_list = [optim.Adam(x.parameters(), lr=self.critic_learning_rate)
                                      for x in self.critic_online_network_list]

        self.experience_pool = ExperiencePool(self.buffer_size)

    def help_train_model(self, source_network_index, target_network_index):
        source_policy_network = self.policy_online_network_list[source_network_index]
        source_critic_network = self.critic_online_network_list[source_network_index]
        target_policy_network = self.policy_online_network_list[target_network_index]
        target_critic_network = self.critic_online_network_list[target_network_index]

        soft_target_update(source_policy_network, target_policy_network)
        soft_target_update(source_critic_network, target_critic_network)

    def train_model(self):
        self.temp_policy_loss_list = []
        self.temp_critic_loss_list = []

        sample_experience = self.experience_pool.sample(self.batch_size)
        sample_experience = Experience(*zip(*sample_experience))

        curr_state = torch.tensor(np.array(sample_experience.state), dtype=torch.float32, device=self.device)
        curr_action = torch.tensor(np.array(sample_experience.action), dtype=torch.float32, device=self.device)
        next_state = torch.tensor(np.array(sample_experience.next_state), dtype=torch.float32, device=self.device)
        reward = torch.tensor(np.array(sample_experience.reward), dtype=torch.float32, device=self.device)
        done = torch.tensor(np.array(sample_experience.done), dtype=torch.float32, device=self.device)

        agent_num = self.user_equipment_num
        if self.train_config.algorithm == 'ddpg':
            agent_num = 1

        if self.train_config.algorithm == 'ddpg':
            sample_all_curr_state = torch.reshape(curr_state, (self.batch_size, self.state_dim))
            sample_all_curr_action = torch.reshape(curr_action, (self.batch_size, self.action_dim))
            sample_all_next_state = torch.reshape(next_state, (self.batch_size, self.state_dim))
            sample_all_reward = torch.reshape(reward, (self.batch_size, 1))
        else:  # MADDPG
            sample_all_curr_state = torch.reshape(curr_state, (self.batch_size, self.state_dim * agent_num))
            sample_all_curr_action = torch.reshape(curr_action, (self.batch_size, self.action_dim * agent_num))
            sample_all_next_state = torch.reshape(next_state, (self.batch_size, self.state_dim * agent_num))
            sample_all_reward = torch.reshape(reward, (self.batch_size, 1 * agent_num))
        sample_all_done = torch.squeeze(done)

        if self.train_config.algorithm == 'ddpg':
            self.critic_optimizer_list[0].zero_grad()
            # Q_online(s, a)
            online_q_curr = self.critic_online_network_list[0](sample_all_curr_state, sample_all_curr_action)
            online_q_curr = torch.squeeze(online_q_curr)
            # Q_target(s', A_target(s'))
            target_all_action_next = [self.policy_target_network_list[0](next_state.reshape(-1, self.state_dim))]
            target_all_action_next = torch.stack(target_all_action_next)
            target_all_action_next = target_all_action_next.transpose(0, 1).contiguous()
            target_all_action_next = \
                torch.reshape(target_all_action_next, (self.batch_size, self.action_dim * agent_num))
            target_q_next = self.critic_target_network_list[0](sample_all_next_state, target_all_action_next)
            target_q_next = torch.squeeze(target_q_next)
            # Q_backup
            backup_q_curr = sample_all_reward[:, 0] + self.gamma * (1 - sample_all_done) * target_q_next
            backup_q_curr = backup_q_curr.to(self.device)
            critic_loss = F.mse_loss(online_q_curr, torch.detach(backup_q_curr))
            critic_loss.backward()
            self.critic_optimizer_list[0].step()

            self.policy_optimizer_list[0].zero_grad()
            # Q_online(s, A_online(s))
            sample_one_curr_state = curr_state.reshape(-1, self.state_dim)
            policy_one_curr_action = self.policy_online_network_list[0](sample_one_curr_state)
            temp_action = torch.clone(curr_action.reshape(-1, self.action_dim))
            temp_action = policy_one_curr_action
            temp_action = torch.reshape(temp_action, (self.batch_size, self.action_dim * agent_num))
            policy_loss = \
                -torch.mean((self.critic_online_network_list[0])(sample_all_curr_state, temp_action))
            # policy_loss = torch.tensor(policy_loss, dtype=torch.float32, device=self.device)
            policy_loss.backward()
            self.policy_optimizer_list[0].step()

            self.temp_critic_loss_list.append(critic_loss.item())
            self.temp_policy_loss_list.append(policy_loss.item())

            soft_target_update(self.critic_online_network_list[0],
                               self.critic_target_network_list[0])
            soft_target_update(self.policy_online_network_list[0],
                               self.policy_target_network_list[0])
        else:  # MADDPG
            for agent_count in range(agent_num):
                self.critic_optimizer_list[agent_count].zero_grad()
                # Q_online(s, a)
                online_q_curr = self.critic_online_network_list[agent_count](sample_all_curr_state, sample_all_curr_action)
                online_q_curr = torch.squeeze(online_q_curr)
                # Q_target(s', A_target(s'))
                target_all_action_next = [self.policy_target_network_list[i](next_state[:, i, :]) for i in range(agent_num)]
                target_all_action_next = torch.stack(target_all_action_next)
                target_all_action_next = target_all_action_next.transpose(0, 1).contiguous()
                target_all_action_next = \
                    torch.reshape(target_all_action_next, (self.batch_size, self.action_dim * agent_num))
                target_q_next = self.critic_target_network_list[agent_count](sample_all_next_state, target_all_action_next)
                target_q_next = torch.squeeze(target_q_next)
                # Q_backup
                backup_q_curr = sample_all_reward[:, agent_count] + self.gamma * (1 - sample_all_done) * target_q_next
                backup_q_curr = backup_q_curr.to(self.device)
                critic_loss = F.mse_loss(online_q_curr, torch.detach(backup_q_curr))
                critic_loss.backward()
                self.critic_optimizer_list[agent_count].step()

                self.policy_optimizer_list[agent_count].zero_grad()
                # Q_online(s, A_online(s))
                sample_one_curr_state = curr_state[:, agent_count, :]
                policy_one_curr_action = self.policy_online_network_list[agent_count](sample_one_curr_state)
                temp_action = torch.clone(curr_action)
                temp_action[:, agent_count, :] = policy_one_curr_action
                temp_action = torch.reshape(temp_action, (self.batch_size, self.action_dim * agent_num))
                policy_loss = \
                    -torch.mean((self.critic_online_network_list[agent_count])(sample_all_curr_state, temp_action))
                # policy_loss = torch.tensor(policy_loss, dtype=torch.float32, device=self.device)
                policy_loss.backward()
                self.policy_optimizer_list[agent_count].step()

                self.temp_critic_loss_list.append(critic_loss.item())
                self.temp_policy_loss_list.append(policy_loss.item())

                soft_target_update(self.critic_online_network_list[agent_count],
                                   self.critic_target_network_list[agent_count])
                soft_target_update(self.policy_online_network_list[agent_count],
                                   self.policy_target_network_list[agent_count])
