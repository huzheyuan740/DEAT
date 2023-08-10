from env.env_object.coordinate import Coordinate

from config import GlobalConfig


class BaseStation:
    def __init__(self, centre_coordinate: Coordinate, belong_hexagon, global_config: GlobalConfig) -> None:
        self.base_station_config = global_config.env_config.hexagon_network_config.base_station_config
        self.base_station_id = 0

        self.centre_coordinate = centre_coordinate
        self.belong_hexagon = belong_hexagon
        self.computing_ability_max = self.base_station_config.base_station_computing_ability_max
        self.computing_ability_now = self.base_station_config.base_station_computing_ability
        self.energy_max = self.base_station_config.base_station_energy
        self.energy_now = self.energy_max

        self.queue_time_min = self.base_station_config.queue_time
        self.queue_time_now = self.queue_time_min

        self.global_config = global_config
