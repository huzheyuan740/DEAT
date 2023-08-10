import math

from config import GlobalConfig
from env.env_object.utils import *


class CoordinateManager:
    def __init__(self, belong_user_equipment, global_config: GlobalConfig,
                 hexagon_coordinate_list: list) -> None:
        self.belong_user_equipment = belong_user_equipment
        self.move_config = global_config.env_config.hexagon_network_config.move_config
        self.hexagon_coordinate_list = hexagon_coordinate_list

        self.init_speed_min = self.move_config.speed_min
        self.init_speed_max = self.move_config.speed_max
        self.speed_now = np.random.uniform(self.init_speed_min, self.init_speed_max)

        self.init_direction_min = self.move_config.direction_min
        self.init_direction_max = self.move_config.direction_max
        self.direction_now = np.random.uniform(self.init_direction_min, self.init_direction_max)

        self.init_accelerate_speed_min = self.move_config.accelerate_speed_min
        self.init_accelerate_speed_max = self.move_config.accelerate_speed_max
        self.accelerate_speed_now = np.random.uniform(self.init_accelerate_speed_min, self.init_accelerate_speed_max)

        self.init_accelerate_direction_min = self.move_config.accelerate_direction_min
        self.init_accelerate_direction_max = self.move_config.accelerate_direction_max
        self.accelerate_direction_now = np.random.uniform(self.init_accelerate_direction_min,
                                                          self.init_accelerate_direction_max)

        self.move_time = self.move_config.move_time

    def update_coordinate_without_check(self, coordinate: Coordinate) -> None:
        coordinate.x_coordinate += self.move_time * self.speed_now * math.cos(self.direction_now / 180 * math.pi)
        coordinate.y_coordinate += self.move_time * self.speed_now * math.sin(self.direction_now / 180 * math.pi)

        self.speed_now += self.accelerate_speed_now
        if self.speed_now > self.move_config.speed_max:
            self.speed_now = self.move_config.speed_max
        if self.speed_now < self.move_config.speed_min:
            self.speed_now = self.move_config.speed_min

        self.direction_now += self.accelerate_direction_now
        if self.direction_now > self.move_config.direction_max:
            self.direction_now = self.move_config.direction_max
        if self.direction_now < self.move_config.direction_min:
            self.direction_now = self.move_config.direction_min

        self.accelerate_speed_now = np.random.uniform(self.init_accelerate_speed_min, self.init_accelerate_speed_max)
        self.accelerate_direction_now = np.random.uniform(self.init_accelerate_direction_min,
                                                          self.init_accelerate_direction_max)

    def reset_direction(self):
        self.direction_now = np.random.uniform(self.init_direction_min, self.init_direction_max)

    def check_and_repair_coordinate(self, coordinate_before: Coordinate, coordinate: Coordinate,
                                    hexagon_list: list) -> bool:
        # check if too close to bs
        if whether_ue_too_close_to_bs(coordinate, self.hexagon_coordinate_list, self.move_config.min_distance_from_bs):
            coordinate.x_coordinate = coordinate_before.x_coordinate
            coordinate.y_coordinate = coordinate_before.y_coordinate
            coordinate.z_coordinate = coordinate_before.z_coordinate
            # return to coordinate before
            self.reset_direction()
            return False

        belong_hexagon = self.belong_user_equipment.belong_hexagon

        if not whether_coordinate_in_one_hexagon(coordinate, belong_hexagon):
            # went into another hexagon

            if not self.move_config.whether_pass_hexagon:
                coordinate.x_coordinate = coordinate_before.x_coordinate
                coordinate.y_coordinate = coordinate_before.y_coordinate
                coordinate.z_coordinate = coordinate_before.z_coordinate
                # return to coordinate before
                self.reset_direction()
                return False

            if whether_coordinate_in_hexagon_list(coordinate, hexagon_list):

                # in the space of the hexagon list
                belong_hexagon.user_equipment_list.remove(self.belong_user_equipment)
                belong_hexagon = judge_coordinate_in_which_hexagon(coordinate, hexagon_list)
                belong_hexagon.user_equipment_list.append(self.belong_user_equipment)
            else:
                coordinate.x_coordinate = coordinate_before.x_coordinate
                coordinate.y_coordinate = coordinate_before.y_coordinate
                coordinate.z_coordinate = coordinate_before.z_coordinate
                # return to coordinate before
                self.reset_direction()
                return False
                # not in the space of the hexagon list
                # self.replace_coordinate_with_uniform(self.coordinate)

        return True

    def update_coordinate(self, hexagon_list: list):
        coordinate_before: Coordinate = self.belong_user_equipment.coordinate_before
        coordinate: Coordinate = self.belong_user_equipment.coordinate

        while True:
            # save before get new coordinate
            coordinate_before.x_coordinate = coordinate.x_coordinate
            coordinate_before.y_coordinate = coordinate.y_coordinate
            coordinate_before.z_coordinate = coordinate.z_coordinate
            # update coordinate without check
            self.update_coordinate_without_check(coordinate)
            # check and repair the coordinate if invalid when it can be repair
            if self.check_and_repair_coordinate(coordinate_before, coordinate, hexagon_list):
                return
