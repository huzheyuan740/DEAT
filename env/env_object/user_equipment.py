from config import GlobalConfig
from env.env_object.coordinate import Coordinate
from env.env_object.coordinate_manager import CoordinateManager
from env.env_object.channel_gain_manager import ChannelGainManager
from env.env_object.task import Task


class UserEquipment:
    def __init__(self, coordinate: Coordinate, belong_hexagon_now, global_config: GlobalConfig,
                 hexagon_coordinate_list: list) -> None:
        self.user_equipment_id = 0

        self.coordinate = coordinate
        self.coordinate_before = Coordinate(coordinate.x_coordinate, coordinate.y_coordinate, coordinate.z_coordinate)
        self.belong_hexagon = belong_hexagon_now
        self.global_config = global_config
        self.user_equipment_config = global_config.env_config.hexagon_network_config.user_equipment_config
        self.hexagon_coordinate_list = hexagon_coordinate_list

        self.computing_ability_max = self.user_equipment_config.user_equipment_computing_ability
        self.computing_ability_now = self.computing_ability_max
        self.energy_max = self.user_equipment_config.user_equipment_energy
        self.energy_now = self.energy_max

        self.queue_time_min = self.user_equipment_config.queue_time
        self.queue_time_now = self.queue_time_min

        self.max_transmitting_power = self.user_equipment_config.max_transmitting_power

        self.coordinate_manager = CoordinateManager(self, self.global_config, self.hexagon_coordinate_list)
        self.channel_gain = None
        self.channel_gain_manager = ChannelGainManager(self, self.hexagon_coordinate_list, self.global_config)
        self.task = Task(self.global_config)

    def update_coordinate(self, hexagon_list: list):
        self.coordinate_manager.update_coordinate(hexagon_list)

    def update_channel_gain(self):
        self.channel_gain_manager.update_channel_gain()

    def update_task(self):
        self.task = Task(self.global_config)

    def update_ue_triple_message(self, hexagon_list: list):
        self.update_coordinate(hexagon_list)
        self.update_channel_gain()
        self.update_task()
