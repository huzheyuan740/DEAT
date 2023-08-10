from env.env_object.base_station import BaseStation
from config import GlobalConfig
from env.env_object.user_equipment import UserEquipment
from env.env_object.utils import *


class Hexagon:
    def __init__(self, centre_coordinate: Coordinate, user_equipment_num: int, global_config: GlobalConfig,
                 hexagon_coordinate_list: list) -> None:
        self.hexagon_network_config = global_config.env_config.hexagon_network_config

        self.centre_coordinate = centre_coordinate
        self.hexagon_coordinate_list = hexagon_coordinate_list

        self.out_radius = self.hexagon_network_config.out_radius
        self.in_radius = self.out_radius / 2 * np.sqrt(3)

        self.vertex_coordinate_list = []
        for angle_60_count in range(6):
            self.vertex_coordinate_list.append(Coordinate(self.centre_coordinate.x_coordinate + self.out_radius *
                                                          np.cos(angle_60_count * np.pi / 3 + np.pi / 6),
                                                          self.centre_coordinate.y_coordinate + self.out_radius *
                                                          np.sin(angle_60_count * np.pi / 3 + np.pi / 6),
                                                          0))

        self.base_station = BaseStation(centre_coordinate, self, global_config)

        self.user_equipment_num = user_equipment_num
        self.user_equipment_list = []

        while True:
            random_x_coordinate = np.random.uniform(self.centre_coordinate.x_coordinate - self.in_radius,
                                                    self.centre_coordinate.x_coordinate + self.in_radius)
            random_y_coordinate = np.random.uniform(self.centre_coordinate.y_coordinate - self.out_radius,
                                                    self.centre_coordinate.y_coordinate + self.out_radius)
            temp_coordinate = Coordinate(random_x_coordinate, random_y_coordinate,
                                         self.hexagon_network_config.user_equipment_config.user_equipment_height)
            if (whether_coordinate_in_one_hexagon(temp_coordinate, self)
                    and (not whether_ue_too_close_to_bs(temp_coordinate, hexagon_coordinate_list,
                                                        self.hexagon_network_config.move_config.min_distance_from_bs))):
                self.user_equipment_list.append(UserEquipment(temp_coordinate, self, global_config,
                                                              self.hexagon_coordinate_list))
                if len(self.user_equipment_list) >= self.user_equipment_num:
                    break

    def __eq__(self, other):
        return self.centre_coordinate == other.centre_coordinate
