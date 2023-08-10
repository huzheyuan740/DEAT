import numpy as np

from env.env_object.coordinate import Coordinate


def get_y_coordinate_for_compare(coordinate_1: Coordinate, coordinate_2: Coordinate,
                                 x_for_compare_inside: float) -> float:
    return (x_for_compare_inside - coordinate_1.x_coordinate) / \
           (coordinate_2.x_coordinate - coordinate_1.x_coordinate) * \
           (coordinate_2.y_coordinate - coordinate_1.y_coordinate) + \
           coordinate_1.y_coordinate


def whether_coordinate_in_one_hexagon(coordinate: Coordinate, hexagon) -> bool:
    return not (coordinate.x_coordinate < hexagon.centre_coordinate.x_coordinate - hexagon.in_radius
                or
                coordinate.x_coordinate > hexagon.centre_coordinate.x_coordinate + hexagon.in_radius
                or
                coordinate.y_coordinate < hexagon.centre_coordinate.y_coordinate - hexagon.out_radius
                or
                coordinate.y_coordinate > hexagon.centre_coordinate.y_coordinate + hexagon.out_radius
                or
                coordinate.y_coordinate > get_y_coordinate_for_compare(hexagon.vertex_coordinate_list[0],
                                                                       hexagon.vertex_coordinate_list[1],
                                                                       coordinate.x_coordinate)
                or
                coordinate.y_coordinate > get_y_coordinate_for_compare(hexagon.vertex_coordinate_list[1],
                                                                       hexagon.vertex_coordinate_list[2],
                                                                       coordinate.x_coordinate)
                or
                coordinate.y_coordinate < get_y_coordinate_for_compare(hexagon.vertex_coordinate_list[3],
                                                                       hexagon.vertex_coordinate_list[4],
                                                                       coordinate.x_coordinate)
                or
                coordinate.y_coordinate < get_y_coordinate_for_compare(hexagon.vertex_coordinate_list[4],
                                                                       hexagon.vertex_coordinate_list[5],
                                                                       coordinate.x_coordinate))


def whether_coordinate_in_hexagon_list(coordinate: Coordinate, hexagon_list: list) -> bool:
    for hexagon in hexagon_list:
        if whether_coordinate_in_one_hexagon(coordinate, hexagon):
            return True
    return False


def judge_coordinate_in_which_hexagon(coordinate: Coordinate, hexagon_list: list):
    for hexagon in hexagon_list:
        if whether_coordinate_in_one_hexagon(coordinate, hexagon):
            return hexagon


def get_distance_between_two_coordinate(coordinate1: Coordinate, coordinate2: Coordinate) -> float:
    return np.sqrt(np.square(coordinate1.x_coordinate - coordinate2.x_coordinate) +
                   np.square(coordinate1.y_coordinate - coordinate2.y_coordinate) +
                   np.square(coordinate1.z_coordinate - coordinate2.z_coordinate))


def from_distance_to_path_loss(distance: float):
    return - (128.1 + 37.6 * np.log10(0.001 * distance))


def whether_ue_too_close_to_bs(coordinate: Coordinate, hexagon_coordinate_list: list, min_distance: float):
    for hexagon_coordinate in hexagon_coordinate_list:
        if get_distance_between_two_coordinate(coordinate, hexagon_coordinate) < min_distance:
            return True
    return False
