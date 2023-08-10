from config import GlobalConfig
import numpy as np


class Task:
    def __init__(self, global_config: GlobalConfig) -> None:
        self.task_config = global_config.env_config.hexagon_network_config.task_config

        self.task_data_size = np.random.normal(self.task_config.task_data_size_now, 33300) # np.random.uniform(self.task_config.task_data_size_min, self.task_config.task_data_size_max)
        self.task_computing_size = \
            np.random.uniform(self.task_config.task_computing_size_min, self.task_config.task_computing_size_max)
        self.task_tolerance_delay = \
            np.random.uniform(self.task_config.task_tolerance_delay_min, self.task_config.task_tolerance_delay_max)

    def get_task_triple_list(self):
        return [self.task_data_size, self.task_computing_size, self.task_tolerance_delay]

    def get_task_triple_array(self):
        return np.array([self.task_data_size, self.task_computing_size, self.task_tolerance_delay])
