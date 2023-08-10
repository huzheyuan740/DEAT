from config import GlobalConfig
from env.env_object.utils import *


class ChannelGainManager:
    def __init__(self, belong_user_equipment, hexagon_coordinate_list: list,
                 global_config: GlobalConfig):
        self.belong_user_equipment = belong_user_equipment
        self.hexagon_coordinate_list = hexagon_coordinate_list
        self.global_config = global_config

        # bind coordinate and coordinate_before with belong user equipment
        self.coordinate_before = self.belong_user_equipment.coordinate_before
        self.coordinate = self.belong_user_equipment.coordinate

        # get distance to all hexagon list
        self.distance_to_all_hexagon_list = []
        for hexagon_coordinate in self.hexagon_coordinate_list:
            self.distance_to_all_hexagon_list.append(
                get_distance_between_two_coordinate(self.coordinate, hexagon_coordinate))
        self.distance_to_all_hexagon_array = np.array(self.distance_to_all_hexagon_list)

        # get path loss to all hexagon list
        self.path_loss_in_db = - 120.9 - 37.6 * np.log10(self.distance_to_all_hexagon_array)

        self.belong_user_equipment.channel_gain = self.path_loss_in_db

    def update_channel_gain(self):

        # get distance to all hexagon coordinate
        self.distance_to_all_hexagon_list = []
        for hexagon_coordinate in self.hexagon_coordinate_list:
            self.distance_to_all_hexagon_list.append(get_distance_between_two_coordinate(self.coordinate,
                                                                                         hexagon_coordinate))
        self.distance_to_all_hexagon_array = np.array(self.distance_to_all_hexagon_list)

        # get path loss to all hexagon coordinate
        self.path_loss_in_db = - 120.9 - 37.6 * np.log10(self.distance_to_all_hexagon_array)

        self.belong_user_equipment.channel_gain = self.path_loss_in_db


'''class ChannelGainManager:
    def __init__(self, belong_user_equipment: UserEquipment, hexagon_coordinate_list: list,
                 global_config: GlobalConfig):
        self.belong_user_equipment = belong_user_equipment
        self.hexagon_coordinate_list = hexagon_coordinate_list
        self.hexagon_network_config = global_config.env_config.hexagon_network_config

        # bind coordinate and coordinate_before with belong user equipment
        self.coordinate_before = self.belong_user_equipment.coordinate_before
        self.coordinate = self.belong_user_equipment.coordinate

        # get distance to all hexagon list
        self.distance_to_all_hexagon_list = []
        for hexagon_coordinate in self.hexagon_coordinate_list:
            self.distance_to_all_hexagon_list.append(
                get_distance_between_two_coordinate(self.coordinate, hexagon_coordinate))
        self.distance_to_all_hexagon_array = np.array(self.distance_to_all_hexagon_list)

        # get path loss to all hexagon list, refer to nasir code
        self.path_loss_in_db = - (128.1 + 37.6 * np.log10(0.001 * self.distance_to_all_hexagon_array))

        # get correlation_for_large
        self.det_x = 0.0
        self.det_y = 0.0
        self.det_x_y = np.sqrt(self.det_x ** 2 + self.det_y ** 2)
        self.correlation_for_large = \
            np.exp(-self.det_x_y / self.hexagon_network_config.channel_gain_config.environment_correlation_length)

        # get shadowing
        self.shadowing_now = np.random.randn(self.hexagon_network_config.deployment_config.hexagon_num)
        self.shadowing_before = self.shadowing_now

        # get large-scale rayleigh fading component
        self.large_scale_rayleigh_fading_component_db = \
            np.add(self.shadowing_now * self.hexagon_network_config.channel_gain_config.shadowing_dev,
                   self.path_loss_in_db)
        self.large_scale_rayleigh_fading_component = \
            np.power(10.0, self.large_scale_rayleigh_fading_component_db / 10.0)

        # begin Small-scale rayleigh fading component
        self.final_speed = self.det_x_y / self.hexagon_network_config.move_config.move_time
        self.max_doppler_frequency = \
            self.final_speed * \
            self.hexagon_network_config.channel_gain_config.carrier_frequency / \
            self.hexagon_network_config.channel_gain_config.speed_of_light

        # get correlation_for_small
        from scipy import special
        self.correlation_for_small = special.j0(
            2.0 * np.pi * self.max_doppler_frequency * self.hexagon_network_config.move_config.move_time)

        self.small_scale_rayleigh_fading_component_now = \
            np.sqrt(2.0 / np.pi) * \
            np.add(self.hexagon_network_config.channel_gain_config.rayleigh_var *
                   np.random.randn(self.hexagon_network_config.deployment_config.hexagon_num),
                   self.hexagon_network_config.channel_gain_config.rayleigh_var *
                   np.random.randn(self.hexagon_network_config.deployment_config.hexagon_num) * 1j)
        self.small_scale_rayleigh_fading_component_before = self.small_scale_rayleigh_fading_component_now
        # small_scale_rayleigh_fading_component_before is not the same! k->n

        # Channel gain
        self.belong_user_equipment.channel_gain = \
            self.large_scale_rayleigh_fading_component * \
            np.square(np.abs(self.small_scale_rayleigh_fading_component_now))

    def update_channel_gain(self):

        # get distance to all hexagon coordinate
        self.distance_to_all_hexagon_list = []
        for hexagon_coordinate in self.hexagon_coordinate_list:
            self.distance_to_all_hexagon_list.append(get_distance_between_two_coordinate(self.coordinate,
                                                                                         hexagon_coordinate))
        self.distance_to_all_hexagon_array = np.array(self.distance_to_all_hexagon_list)

        # get path loss to all hexagon coordinate
        self.path_loss_in_db = - (128.1 + 37.6 * np.log10(0.001 * self.distance_to_all_hexagon_array))

        # get correlation for large
        self.det_x = self.coordinate.x_coordinate - self.coordinate_before.x_coordinate
        self.det_y = self.coordinate.y_coordinate - self.coordinate_before.y_coordinate
        self.det_x_y = np.sqrt(self.det_x ** 2 + self.det_y ** 2)
        self.correlation_for_large = \
            np.exp(-self.det_x_y / self.hexagon_network_config.channel_gain_config.environment_correlation_length)

        # get shadowing, refer to nasir code
        self.shadowing_before = self.shadowing_now
        self.shadowing_now = \
            self.shadowing_before * self.correlation_for_large + \
            np.sqrt(1.0 - np.square(self.correlation_for_large)) * \
            np.random.randn(self.hexagon_network_config.deployment_config.hexagon_num)

        # Large-scale rayleigh fading component
        self.large_scale_rayleigh_fading_component_db = \
            np.add(self.shadowing_now * self.hexagon_network_config.channel_gain_config.shadowing_dev,
                   self.path_loss_in_db)
        self.large_scale_rayleigh_fading_component = \
            np.power(10.0, self.large_scale_rayleigh_fading_component_db / 10.0)

        # begin Small-scale rayleigh fading component
        self.final_speed = self.det_x_y / self.hexagon_network_config.move_config.move_time
        self.max_doppler_frequency = \
            self.final_speed * \
            self.hexagon_network_config.channel_gain_config.carrier_frequency / \
            self.hexagon_network_config.channel_gain_config.speed_of_light

        # get correlation_for_small
        from scipy import special
        self.correlation_for_small = special.j0(
            2.0 * np.pi * self.max_doppler_frequency * self.hexagon_network_config.move_config.move_time)

        self.small_scale_rayleigh_fading_component_before = self.small_scale_rayleigh_fading_component_now
        self.small_scale_rayleigh_fading_component_now = \
            self.correlation_for_small * self.small_scale_rayleigh_fading_component_before + \
            np.sqrt(1 - np.square(self.correlation_for_small)) * \
            np.sqrt(2.0 / np.pi) * \
            (self.hexagon_network_config.channel_gain_config.rayleigh_var *
             np.random.randn(self.hexagon_network_config.deployment_config.hexagon_num) +
             self.hexagon_network_config.channel_gain_config.rayleigh_var *
             np.random.randn(self.hexagon_network_config.deployment_config.hexagon_num) * 1j)
        # small_scale_rayleigh_fading_component_before is not the same! k->n

        # Channel gain
        self.belong_user_equipment.channel_gain = \
            self.large_scale_rayleigh_fading_component * \
            np.square(np.abs(self.small_scale_rayleigh_fading_component_now))'''
