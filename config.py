import numpy as np


class TrainConfig:
    def __init__(self, algorithm):
        self.algorithm = algorithm  # dc ddpg maddpg mix
        self.episode_num = int(25000)
        self.step_num = 10
        self.seed = 1  # both use for transfer_env
        self.env_idx = 0
        self.gpu_index = 0
        self.tensorboard = True


class BaseStationConfig:
    def __init__(self):
        self.base_station_computing_ability = 20.0 * 10 ** 9  # 5.0 * 10 ** 9
        self.base_station_computing_ability_max = 25.0 * 10 ** 9
        self.base_station_energy = 200.0
        self.base_station_height = 20.0
        self.queue_time = 0.0


class UserEquipmentConfig:
    def __init__(self):
        self.user_equipment_computing_ability = 1 * 10 ** 9
        self.user_equipment_energy = (10 ** -27) * ((1 * 10 ** 9) ** 2) * 500000 * 1000 * 10
        # self.user_equipment_energy = 200
        self.user_equipment_height = 1.8

        self.queue_time = 0.0

        self.max_transmitting_power_db = 38.0 - 30
        self.max_transmitting_power = 0.5 #np.power(10.0, self.max_transmitting_power_db / 10)

        self.user_equipment_idle_power = 100 * 10 ** (-3)
        self.energy_consumption_per_cpu_cycle = 10 ** -27


class TaskConfig:
    def __init__(self):
        self.task_data_size_min = 80000.0
        self.task_data_size_max = 520000.0
        self.task_data_size_now = 300000.0  # 400000.0
        self.task_computing_size_min = 800.0
        self.task_computing_size_max = 900.0
        self.task_tolerance_delay_min = 3000000 * 1000 / (1 * 10 ** 9)
        self.task_tolerance_delay_max = 4000000 * 1000 / (1 * 10 ** 9)


class MoveConfig:
    def __init__(self):
        self.speed_max = 5.0
        self.speed_min = -5.0
        self.accelerate_speed_max = 2.0
        self.accelerate_speed_min = -2.0
        self.direction_max = 360.0
        self.direction_min = 0.0
        self.accelerate_direction_max = 30.0
        self.accelerate_direction_min = -30.0
        self.move_time = 0.5
        self.whether_pass_hexagon = True

        self.min_distance_from_bs = 50


class ChannelGainConfig:
    def __init__(self):
        self.environment_correlation_length = 10.0
        self.carrier_frequency = 2.0
        self.speed_of_light = 3.0 * 10 ** 8
        self.log_normal_shadowing_standard_deviation = 10.0
        self.rayleigh_var = 1.0
        self.shadowing_dev = 10.0

        self.channel_gain_max = - 120.9 - 37.6 * np.log10(20)
        self.channel_gain_min = - 120.9 - 37.6 * np.log10(150)


class HexagonNetworkConfig:
    def __init__(self):
        self.hexagon_num = 1
        self.user_equipment_num = 25  # 20
        self.out_radius = 100.0  # 100.0
        self.time_max = 2  # 0.5
        self.base_station_config = BaseStationConfig()
        self.user_equipment_config = UserEquipmentConfig()
        self.task_config = TaskConfig()
        self.move_config = MoveConfig()
        self.channel_gain_config = ChannelGainConfig()


class ChannelConfig:
    def __init__(self):
        self.bandwidth = 8 * 10 ** 6  # 10 * 10 ** 6
        self.noise_power_db = -114.0 - 30
        self.noise_power = np.power(10.0, self.noise_power_db / 10)
        self.sinr_threshold = 10.0 ** (30.0 / 10.0)


class CostConfig:
    def __init__(self):
        self.time_cost_weight_if_local = 0.5
        self.time_cost_weight_if_offload = 0.5
        self.energy_cost_weight_if_local = 0.5
        self.energy_cost_weight_if_offload = 0.5


class RewardConfig:
    def __init__(self):
        self.penalty_transmitting_power_is_too_small = 1000
        self.penalty_ue_computing_ability_is_too_small = 1000
        self.penalty_bs_computing_ability_is_too_small = 1000

        self.penalty_ue_energy_exhaust_when_local = 100

        self.penalty_over_time_when_offload = 1000
        self.penalty_ue_energy_exhaust_when_offload = 100
        self.penalty_abs_bs_computing_ability = 100 / 10

        self.reward_weight_of_cost = 100
        self.reward_weight_if_better = 50

        # new add
        self.penalty_over_time_when_queue = 1000


class EnvInterfaceConfig:
    def __init__(self):
        self.channel_config = ChannelConfig()
        self.cost_config = CostConfig()
        self.reward_config = RewardConfig()


class EnvConfig:
    def __init__(self):
        self.hexagon_network_config = HexagonNetworkConfig()
        self.env_interface_config = EnvInterfaceConfig()


class StateConfig:
    def __init__(self, algorithm):
        self.hexagon_network_config = HexagonNetworkConfig()
        self.control_config = ControlConfig()
        self.train_config = TrainConfig(algorithm)
        self.ue_num = self.hexagon_network_config.user_equipment_num
        self.dimension = 9  # 20
        if self.train_config.algorithm == 'ddpg' or self.train_config.algorithm == 'dc':
            self.dimension *= self.ue_num


class ActionConfig:
    def __init__(self, algorithm):
        self.hexagon_network_config = HexagonNetworkConfig()
        self.control_config = ControlConfig()
        self.train_config = TrainConfig(algorithm)
        self.ue_num = self.hexagon_network_config.user_equipment_num
        self.dimension = 2
        if self.train_config.algorithm == 'ddpg' or self.train_config.algorithm == 'dc':
            self.dimension *= self.ue_num
        self.action_noise = 0.1  # np.random.uniform(0, 1, self.dimension)
        self.action_noise_decay = 0.995
        self.threshold_to_offload = 0.0  # 0.5
        if self.train_config.algorithm == 'ddpg' or self.train_config.algorithm == 'maddpg':
            self.threshold_to_offload = 0.5

class TorchConfig:
    def __init__(self, algorithm):
        self.gamma = 0.98
        self.hidden_sizes = (128, 128)
        self.buffer_size = int(4e3)  # int(4e3)
        self.max_seq_length = 50
        # self.buffer_size = int(64)
        self.batch_size = 32
        self.train_config = TrainConfig(algorithm)
        if self.train_config.algorithm == 'dc':
            self.batch_size = 64
        self.policy_learning_rate = 1e-4
        self.critic_learning_rate = 1e-3
        self.policy_gradient_clip = 0.5
        self.critic_gradient_clip = 1.0
        self.epsilon_max = 1.0
        self.epsilon_decay = 0.998
        self.action_limit = 1.0


class AgentConfig:
    def __init__(self, algorithm):
        self.state_config = StateConfig(algorithm)
        self.action_config = ActionConfig(algorithm)
        self.torch_config = TorchConfig(algorithm)


class DebugConfig:
    def __init__(self):
        self.whether_output_finish_episode_reason = 1
        self.whether_output_replay_buffer_message = 0


class ControlConfig:
    def __init__(self):
        self.save_runs = True
        self.save_save_model = True

        self.output_network_config = True
        self.output_action_config = True
        self.output_other_config = False

        self.easy_output_mode = True
        self.easy_output_cycle = 100


class GlobalConfig:
    def __init__(self, algorithm):
        self.train_config = TrainConfig(algorithm)
        self.env_config = EnvConfig()
        self.agent_config = AgentConfig(algorithm)
        self.debug_config = DebugConfig()
        self.control_config = ControlConfig()
