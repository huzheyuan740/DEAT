from env.env_object.user_equipment import UserEquipment
from env.env_object.base_station import BaseStation
from env.hexagon_network import HexagonNetwork


class State:
    def __init__(self, user_equipment: UserEquipment, base_station: BaseStation, hexagon_network: HexagonNetwork):
        self.user_equipment_id = user_equipment.user_equipment_id
        self.base_station_id = base_station.base_station_id

        # state message
        self.task_data_size = user_equipment.task.task_data_size
        self.task_computing_size = user_equipment.task.task_computing_size
        self.task_tolerance_delay = user_equipment.task.task_tolerance_delay
        self.energy_now = user_equipment.energy_now
        self.channel_gain_to_base_station = \
            user_equipment.channel_gain[user_equipment.belong_hexagon.base_station.base_station_id]

        # new add state
        self.user_equipment_computing_ability = user_equipment.computing_ability_max
        self.user_equipment_queue_time_now = user_equipment.queue_time_now
        self.base_station_computing_ability = base_station.computing_ability_now
        self.base_station_queue_time_now = base_station.queue_time_now

        # above 9 states

        self.task_average_calculation_frequency = \
            self.task_data_size * self.task_computing_size / self.task_tolerance_delay

        # state normalized message
        self.task_data_size = \
            (self.task_data_size -
             hexagon_network.hexagon_network_config.task_config.task_data_size_min) / \
            (hexagon_network.hexagon_network_config.task_config.task_data_size_max -
             hexagon_network.hexagon_network_config.task_config.task_data_size_min)
        self.task_computing_size = \
            (self.task_computing_size -
             hexagon_network.hexagon_network_config.task_config.task_computing_size_min) / \
            (hexagon_network.hexagon_network_config.task_config.task_computing_size_max -
             hexagon_network.hexagon_network_config.task_config.task_computing_size_min)
        self.task_tolerance_delay = \
            (self.task_tolerance_delay -
             hexagon_network.hexagon_network_config.task_config.task_tolerance_delay_min) / \
            (hexagon_network.hexagon_network_config.task_config.task_tolerance_delay_max -
             hexagon_network.hexagon_network_config.task_config.task_tolerance_delay_min)
        self.energy_now /= hexagon_network.hexagon_network_config.user_equipment_config.user_equipment_energy
        self.channel_gain_to_base_station = \
            (self.channel_gain_to_base_station -
             hexagon_network.hexagon_network_config.channel_gain_config.channel_gain_min) / \
            (hexagon_network.hexagon_network_config.channel_gain_config.channel_gain_max -
             hexagon_network.hexagon_network_config.channel_gain_config.channel_gain_min)

        self.user_equipment_computing_ability = \
            self.user_equipment_computing_ability / user_equipment.computing_ability_max
        self.user_equipment_queue_time_now = user_equipment.queue_time_now
        self.base_station_computing_ability = \
            self.base_station_computing_ability / base_station.computing_ability_max
        self.base_station_queue_time_now = base_station.queue_time_now

        # self.judge_offload = (self.channel_gain_to_base_station + 1 - self.energy_now) / 2

    def get_state_list(self):
        return [self.task_data_size, self.task_computing_size, self.task_tolerance_delay,
                self.energy_now, self.channel_gain_to_base_station, self.user_equipment_computing_ability,
                self.user_equipment_queue_time_now, self.base_station_computing_ability,
                self.base_station_queue_time_now]

    def get_state_array(self):
        import numpy as np
        return np.array([self.task_data_size, self.task_computing_size, self.task_tolerance_delay,
                         self.energy_now, self.channel_gain_to_base_station, self.user_equipment_computing_ability,
                         self.user_equipment_queue_time_now, self.base_station_computing_ability,
                         self.base_station_queue_time_now])
