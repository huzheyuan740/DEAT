import numpy as np

from config import GlobalConfig
from env.env_object.coordinate import Coordinate
from env.env_object.hexagon import Hexagon


class HexagonNetwork:
    def __init__(self, global_config: GlobalConfig):
        self.hexagon_network_config = global_config.env_config.hexagon_network_config

        self.hexagon_num = self.hexagon_network_config.hexagon_num
        self.user_equipment_num = self.hexagon_network_config.user_equipment_num
        self.user_equipment_num_per_hexagon = self.user_equipment_num // self.hexagon_num

        self.hexagon_coordinate_list = [Coordinate(0.0, 0.0,
                                                   self.hexagon_network_config.base_station_config.base_station_height)]
        for coordinate in self.hexagon_coordinate_list:
            if len(self.hexagon_coordinate_list) >= self.hexagon_num:
                break
            for angle_60_count in range(6):
                temp_coordinate = Coordinate(coordinate.x_coordinate +
                                             np.sqrt(3) * self.hexagon_network_config.deployment_config.out_radius *
                                             np.cos(angle_60_count * np.pi / 3),
                                             coordinate.y_coordinate +
                                             np.sqrt(3) * self.hexagon_network_config.deployment_config.out_radius *
                                             np.sin(angle_60_count * np.pi / 3),
                                             self.hexagon_network_config.base_station_config.base_station_height)
                if temp_coordinate not in self.hexagon_coordinate_list:
                    self.hexagon_coordinate_list.append(temp_coordinate)
                    if len(self.hexagon_coordinate_list) >= self.hexagon_num:
                        break
            if len(self.hexagon_coordinate_list) >= self.hexagon_num:
                break

        self.hexagon_list = []
        for coordinate in self.hexagon_coordinate_list:
            self.hexagon_list.append(Hexagon(coordinate, self.user_equipment_num_per_hexagon, global_config,
                                             self.hexagon_coordinate_list))

        base_station_num = 0
        self.user_equipment_list = []
        user_equipment_num = 0
        for hexagon in self.hexagon_list:
            hexagon.base_station.base_station_id = base_station_num
            base_station_num += 1
            for user_equipment in hexagon.user_equipment_list:
                user_equipment.user_equipment_id = user_equipment_num
                self.user_equipment_list.append(user_equipment)
                user_equipment_num += 1

        self.min_x_coordinate = 0.0
        self.max_x_coordinate = 0.0
        self.min_y_coordinate = 0.0
        self.max_y_coordinate = 0.0

        for hexagon in self.hexagon_list:
            for coordinate in hexagon.vertex_coordinate_list:
                if coordinate.x_coordinate < self.min_x_coordinate:
                    self.min_x_coordinate = coordinate.x_coordinate
                if coordinate.x_coordinate > self.max_x_coordinate:
                    self.max_x_coordinate = coordinate.x_coordinate
                if coordinate.y_coordinate < self.min_y_coordinate:
                    self.min_y_coordinate = coordinate.y_coordinate
                if coordinate.y_coordinate > self.max_y_coordinate:
                    self.max_y_coordinate = coordinate.y_coordinate

    def print_hexagon_network(self, mode: int = 0, name: str = "", dpi: int = 0):
        import matplotlib.pyplot as plt
        for hexagon in self.hexagon_list:
            plt.plot(hexagon.centre_coordinate.x_coordinate, hexagon.centre_coordinate.y_coordinate, "g^")
            x_coordinate_list = []
            y_coordinate_list = []
            for vertex_coordinate in hexagon.vertex_coordinate_list:
                x_coordinate_list.append(vertex_coordinate.x_coordinate)
                y_coordinate_list.append(vertex_coordinate.y_coordinate)
            x_coordinate_list.append(x_coordinate_list[0])
            y_coordinate_list.append(y_coordinate_list[0])
            plt.plot(x_coordinate_list, y_coordinate_list, "k-")
            for user_equipment in hexagon.user_equipment_list:
                plt.plot(user_equipment.coordinate.x_coordinate,
                         user_equipment.coordinate.y_coordinate, "co")
        for user_equipment in self.user_equipment_list:
            temp_x_pair = [user_equipment.coordinate.x_coordinate,
                           user_equipment.belong_hexagon.base_station.centre_coordinate.x_coordinate]
            temp_y_pair = [user_equipment.coordinate.y_coordinate,
                           user_equipment.belong_hexagon.base_station.centre_coordinate.y_coordinate]
            plt.plot(temp_x_pair, temp_y_pair, "r:")
        if mode == 0:
            plt.show()
            plt.cla()
        elif mode == 1:
            if dpi == 0:
                plt.savefig(name)
            else:
                plt.savefig(name, dpi=dpi)
            plt.cla()

    def update_all_ue_triple_message(self):
        for user_equipment in self.user_equipment_list:
            user_equipment.update_ue_triple_message(self.hexagon_list)
