from config import GlobalConfig
import numpy as np


class Action:
    def __init__(self, action_list, global_config: GlobalConfig):
        self.normalized_whether_offload = action_list[0]
        self.normalized_transmitting_power = action_list[1]
        # self.normalized_ue_computing_ability = action_list[2]
        # self.normalized_bs_computing_ability = action_list[3]

        self.global_config = global_config
        self.action_config = global_config.agent_config.action_config

    def get_whether_offload(self):
        if self.normalized_whether_offload > self.action_config.threshold_to_offload:
            return True
        else:
            return False

    def get_action_list(self):
        res_list = np.append(self.normalized_whether_offload, self.normalized_transmitting_power).tolist()
        return res_list

    def get_action_array(self):
        return np.array([self.normalized_whether_offload, self.normalized_transmitting_power])
