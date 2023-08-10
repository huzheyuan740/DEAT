import time
import datetime
import os

import numpy as np
import torch

from config import GlobalConfig
from env.env_interface import EnvironmentManager
from agent.ddpg import Agent
from agent.action import Action

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from types import SimpleNamespace as SN

from agent.components.transforms import OneHot
from agent.components.episode_buffer import ReplayBuffer
from agent.learners import REGISTRY as le_REGISTRY
from agent.runners import REGISTRY as r_REGISTRY
from agent.controllers import REGISTRY as mac_REGISTRY
from agent.utils.logging import get_logger


def get_args(global_config):
    train_config = global_config.train_config
    temp_args = {'runner': 'episode', 'mac': 'basic_mac', 'env': 'sc2',
                 'env_args': {'continuing_episode': False, 'difficulty': '7', 'game_version': None,
                              'map_name': '8m',
                              'move_amount': 2, 'obs_all_health': True, 'obs_instead_of_state': False,
                              'obs_last_action': False, 'obs_own_health': True, 'obs_pathing_grid': False,
                              'obs_terrain_height': False, 'obs_timestep_number': False, 'reward_death_value': 10,
                              'reward_defeat': 0, 'reward_negative_scale': 0.5, 'reward_only_positive': True,
                              'reward_scale': True, 'reward_scale_rate': 20, 'reward_sparse': False,
                              'reward_win': 200,
                              'replay_dir': '', 'replay_prefix': '', 'state_last_action': True,
                              'state_timestep_number': False, 'step_mul': 8, 'seed': 681210387,
                              'heuristic_ai': False,
                              'heuristic_rest': False, 'debug': False}, 'batch_size_run': 1, 'test_nepisode': 100,
                 'test_interval': 5000, 'test_greedy': True, 'log_interval': 10000, 'runner_log_interval': 10000,
                 'learner_log_interval': 10000, 't_max': train_config.episode_num, 'use_cuda': True,
                 'buffer_cpu_only': True,
                 'use_tensorboard': True, 'save_model': True, 'save_model_interval': 1000,
                 'checkpoint_path': '',
                 # /home/huzheyuan/Edge/sim2real_ciritc/results/models_2022-01-10-10-53-36/349516
                 'evaluate': True, 'load_step': 0, 'save_replay': False, 'local_results_path': 'results',
                 'env_transfer_path': 'transfer_results',
                 'gamma': 0.98,
                 'batch_size': global_config.agent_config.torch_config.batch_size,
                 'buffer_size': global_config.agent_config.torch_config.buffer_size,
                 'lr': 0.0005, 'critic_lr': 0.0005, 'optim_alpha': 0.99,
                 'optim_eps': 1e-05, 'grad_norm_clip': 10, 'agent': 'trans',  # updet trans rnn
                 'rnn_hidden_dim': 64,
                 'obs_agent_id': False,
                 'obs_last_action': False, 'token_dim': 9, 'emb': 32, 'heads': 3, 'depth': 2,
                 'ally_num': global_config.env_config.hexagon_network_config.user_equipment_num,
                 'enemy_num': 0,
                 'repeat_id': 1, 'label': 'default_label', 'action_selector': 'epsilon_greedy',
                 'epsilon_start': 1.0,
                 'epsilon_finish': 0.1,
                 'epsilon_anneal_time': global_config.train_config.episode_num - int(15000),
                 'target_update_interval': 100,
                 'agent_output_type': 'q', 'learner': 'facmac_learner', 'double_q': True, 'mixer': 'vdn',
                 # 'name': 'vdn',
                 'target_update_mode': 'soft', 'target_update_tau': 0.001,  # new add
                 'mixing_embed_dim': 64,
                 'hyper_initialization_nonzeros': 0,
                 'gated': False,
                 'skip_connections': False,
                 'verbose': False,
                 'actions_dtype': np.float32,
                 'seed': 681210387
                 }
    args = SN(**temp_args)

    return args
