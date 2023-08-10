import copy

import numpy as np

from config import GlobalConfig
from env.hexagon_network import HexagonNetwork
from agent.state import State
from agent.action import Action

from env.episode_finish_reason import ReasonEnum


def whether_zero(num: float):
    if abs(num) < 1e-10:
        return True
    else:
        return False


class EnvironmentManager:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.env_config = global_config.env_config
        self.network_config = self.env_config.hexagon_network_config
        self.interface_config = self.env_config.env_interface_config
        self.writer = None
        self.output_other_message = self.global_config.control_config.output_other_config
        self.step_real_count = None

        self.hexagon_network = HexagonNetwork(self.global_config)
        self.next_hexagon_network = copy.deepcopy(self.hexagon_network)
        self.next_hexagon_network.update_all_ue_triple_message()

        self.ue_normalized_transmitting_power_array = None
        self.bs_normalized_computing_ability_array = None
        self.reason_to_finish_this_episode = []

    def reset_environment_interface(self):
        self.hexagon_network = HexagonNetwork(self.global_config)
        self.next_hexagon_network = copy.deepcopy(self.hexagon_network)
        self.next_hexagon_network.update_all_ue_triple_message()

        self.ue_normalized_transmitting_power_array = None
        self.bs_normalized_computing_ability_array = None
        self.reason_to_finish_this_episode = []

    def update_the_hexagon_network(self):
        self.hexagon_network = copy.deepcopy(self.next_hexagon_network)
        self.next_hexagon_network.update_all_ue_triple_message()

    def get_state_per_user_equipment(self, user_equipment_id):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        base_station = self.hexagon_network.user_equipment_list[user_equipment_id].belong_hexagon.base_station
        return State(user_equipment, base_station, self.hexagon_network)

    def get_next_state_per_user_equipment(self, user_equipment_id):
        user_equipment = self.next_hexagon_network.user_equipment_list[user_equipment_id]
        base_station = self.next_hexagon_network.user_equipment_list[user_equipment_id].belong_hexagon.base_station
        return State(user_equipment, base_station, self.next_hexagon_network)

    def get_channel_gain_list_to_one_base_station(self, base_station_id):
        return_list = []
        for user_equipment in self.hexagon_network.user_equipment_list:
            return_list.append(user_equipment.channel_gain[base_station_id])
        return return_list

    def get_max_uplink_spectral_efficiency(self, user_equipment_id, action_offload_mask):
        user_equipment_id_is_zero = False
        if not action_offload_mask[user_equipment_id]:
            user_equipment_id_is_zero = True
            action_offload_mask[user_equipment_id] = 1
        ue_transmitting_power = \
            np.ones(self.network_config.user_equipment_num) * \
            self.network_config.user_equipment_config.max_transmitting_power
        ue_transmitting_power *= action_offload_mask
        base_station_id = \
            self.hexagon_network.user_equipment_list[user_equipment_id].belong_hexagon.base_station.base_station_id
        channel_gain_to_base_station = self.get_channel_gain_list_to_one_base_station(base_station_id)

        tmp_1 = \
            channel_gain_to_base_station[user_equipment_id] * ue_transmitting_power[user_equipment_id]
        tmp_2 = \
            np.matmul(channel_gain_to_base_station, ue_transmitting_power) + \
            self.interface_config.channel_config.noise_power - \
            channel_gain_to_base_station[user_equipment_id] * ue_transmitting_power[user_equipment_id]

        # print("power??:", ue_transmitting_power)
        if whether_zero(tmp_2):
            tmp_2 -= 1e-8
        uplink_spectral_efficiency = \
            (np.log2(1 + np.minimum(self.interface_config.channel_config.sinr_threshold, tmp_1 / tmp_2)))
        if user_equipment_id_is_zero:
            action_offload_mask[user_equipment_id] = 0

        return uplink_spectral_efficiency

    def get_uplink_spectral_efficiency(self, user_equipment_id, action_offload_mask):
        ue_transmitting_power = \
            self.ue_normalized_transmitting_power_array * \
            self.network_config.user_equipment_config.max_transmitting_power
        ue_transmitting_power *= action_offload_mask
        base_station_id = \
            self.hexagon_network.user_equipment_list[user_equipment_id].belong_hexagon.base_station.base_station_id
        channel_gain_to_base_station = self.get_channel_gain_list_to_one_base_station(base_station_id)

        tmp_1 = \
            channel_gain_to_base_station[user_equipment_id] * ue_transmitting_power[user_equipment_id]
        tmp_2 = \
            np.matmul(channel_gain_to_base_station, ue_transmitting_power) + \
            self.interface_config.channel_config.noise_power - \
            channel_gain_to_base_station[user_equipment_id] * ue_transmitting_power[user_equipment_id]

        if whether_zero(tmp_2):
            tmp_2 -= 1e-8
        uplink_spectral_efficiency = \
            (np.log2(1 + np.minimum(self.interface_config.channel_config.sinr_threshold, tmp_1 / tmp_2)))

        return uplink_spectral_efficiency

    def get_max_real_bs_normalized_computing_ability(self, user_equipment_id, task_average_calculation_frequency):
        bs_normalized_computing_ability_array = copy.deepcopy(self.bs_normalized_computing_ability_array)
        bs_normalized_computing_ability_array[user_equipment_id] = task_average_calculation_frequency
        sum_bs_normalized_computing_ability_array = sum(bs_normalized_computing_ability_array)
        return bs_normalized_computing_ability_array[user_equipment_id] / sum_bs_normalized_computing_ability_array

    def get_real_bs_normalized_computing_ability(self, user_equipment_id):
        sum_bs_normalized_computing_ability_array = sum(self.bs_normalized_computing_ability_array)
        return self.bs_normalized_computing_ability_array[user_equipment_id] / sum_bs_normalized_computing_ability_array

    def get_min_normalized_ue_computing_ability(self, user_equipment_id):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size
        user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        min_ue_computing_ability = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            user_equipment_task_max_tolerance_delay

        min_normalized_ue_computing_ability = min_ue_computing_ability / user_equipment.computing_ability_max

        return min_normalized_ue_computing_ability

    def get_min_normalized_transmitting_power(self, user_equipment_id):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        # user_equipment_task_computing_size = user_equipment.task.task_computing_size
        user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        min_data_transmission_rate = user_equipment_task_data_size / user_equipment_task_max_tolerance_delay
        min_uplink_spectral_efficiency = min_data_transmission_rate / self.interface_config.channel_config.bandwidth

        base_station_id = \
            self.hexagon_network.user_equipment_list[user_equipment_id].belong_hexagon.base_station.base_station_id
        channel_gain_to_base_station = self.get_channel_gain_list_to_one_base_station(base_station_id)
        ue_transmitting_power = \
            self.ue_normalized_transmitting_power_array * \
            self.network_config.user_equipment_config.max_transmitting_power
        uplink_spectral_efficiency_tmp_2 = \
            np.matmul(channel_gain_to_base_station, ue_transmitting_power) + \
            self.interface_config.channel_config.noise_power - \
            channel_gain_to_base_station[user_equipment_id] * ue_transmitting_power[user_equipment_id]
        min_tmp1_tmp2 = np.power(2, min_uplink_spectral_efficiency) - 1
        min_uplink_spectral_efficiency_tmp_1 = min_tmp1_tmp2 * uplink_spectral_efficiency_tmp_2

        min_transmitting_power = min_uplink_spectral_efficiency_tmp_1 / channel_gain_to_base_station[user_equipment_id]
        min_normalized_transmitting_power = \
            min_transmitting_power / self.network_config.user_equipment_config.max_transmitting_power

        return min_normalized_transmitting_power

    def get_min_normalized_bs_computing_ability(self, user_equipment_id):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size
        user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        min_bs_computing_ability = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            user_equipment_task_max_tolerance_delay

        min_normalized_bs_computing_ability = \
            min_bs_computing_ability / self.network_config.base_station_config.base_station_computing_ability

        return min_normalized_bs_computing_ability

    def get_min_local_reward(self, user_equipment_id):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size

        user_equipment_allocated_computing_ability = \
            self.get_min_normalized_ue_computing_ability(user_equipment_id) * \
            self.network_config.user_equipment_config.user_equipment_computing_ability

        local_time = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            user_equipment_allocated_computing_ability

        local_energy = \
            self.network_config.user_equipment_config.energy_consumption_per_cpu_cycle * \
            (user_equipment_allocated_computing_ability ** 2) * \
            user_equipment_task_data_size * user_equipment_task_computing_size

        local_cost = (self.interface_config.cost_config.energy_cost_weight_if_local * local_energy +
                      self.interface_config.cost_config.time_cost_weight_if_local * local_time)

        local_reward = - local_cost * self.interface_config.reward_config.reward_weight_of_cost

        energy_now = user_equipment.energy_now
        if local_energy > energy_now:
            local_reward -= (local_energy - energy_now) * \
                            self.interface_config.reward_config.penalty_ue_energy_exhaust_when_local

        return local_reward

    def get_max_local_reward(self, user_equipment_id, ue_queue_time_now):

        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size
        # user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        user_equipment_allocated_computing_ability = \
            self.network_config.user_equipment_config.user_equipment_computing_ability

        local_time = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            user_equipment_allocated_computing_ability

        local_energy = \
            self.network_config.user_equipment_config.energy_consumption_per_cpu_cycle * \
            (user_equipment_allocated_computing_ability ** 2) * \
            user_equipment_task_data_size * user_equipment_task_computing_size

        local_cost = (self.interface_config.cost_config.energy_cost_weight_if_local * local_energy +
                      self.interface_config.cost_config.time_cost_weight_if_local * local_time)

        local_reward = - local_cost * self.interface_config.reward_config.reward_weight_of_cost

        energy_now = user_equipment.energy_now
        if local_energy > energy_now:
            temp_penalty = (local_energy - energy_now) * \
                           self.interface_config.reward_config.penalty_ue_energy_exhaust_when_local
            local_reward -= temp_penalty
            # self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.loc_energy_exhausted])

        if local_time + ue_queue_time_now[user_equipment_id] > user_equipment.task.task_tolerance_delay:
            temp_penalty = self.interface_config.reward_config.penalty_over_time_when_queue * \
                           (local_time + ue_queue_time_now[
                               user_equipment_id] - user_equipment.task.task_tolerance_delay)
            local_reward -= temp_penalty
            # self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.loc_queue_time_too_long])

        local_queue_time_now = max(0, ue_queue_time_now[user_equipment_id] + \
                                   local_time - user_equipment.task.task_tolerance_delay)

        return local_reward, local_cost

    def get_max_offload_reward(self, user_equipment_id,
                               normalized_transmitting_power, ue_queue_time_now,
                               base_station_queue_time_now,
                               offload_count,
                               action_offload_mask):

        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size
        user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        user_equipment_transmitting_power = \
            self.network_config.user_equipment_config.max_transmitting_power * normalized_transmitting_power
        base_station_allocated_computing_ability = \
            self.network_config.base_station_config.base_station_computing_ability

        user_equipment_idle_power = self.network_config.user_equipment_config.user_equipment_idle_power
        spectral_efficiency = self.get_max_uplink_spectral_efficiency(user_equipment_id, action_offload_mask)

        data_transmission_rate = spectral_efficiency * self.interface_config.channel_config.bandwidth
        offload_uplink_time = user_equipment_task_data_size / data_transmission_rate
        offload_uplink_energy = user_equipment_transmitting_power * offload_uplink_time

        offload_computing_time = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            base_station_allocated_computing_ability

        offload_idle_energy = user_equipment_idle_power * offload_computing_time

        offload_sum_time = offload_computing_time + offload_uplink_time
        offload_sum_energy = offload_uplink_energy + offload_idle_energy

        offload_cost = (self.interface_config.cost_config.energy_cost_weight_if_offload * offload_sum_energy +
                        self.interface_config.cost_config.time_cost_weight_if_offload * offload_sum_time)
        offload_reward = - offload_cost * self.interface_config.reward_config.reward_weight_of_cost

        if offload_sum_time + base_station_queue_time_now > user_equipment_task_max_tolerance_delay:
            temp_penalty = (offload_sum_time + base_station_queue_time_now - user_equipment_task_max_tolerance_delay) * \
                           self.interface_config.reward_config.penalty_over_time_when_offload
            offload_reward -= temp_penalty
            # self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.off_queue_time_too_long])

        off_queue_time_now = max(0, ue_queue_time_now[user_equipment_id] - user_equipment.task.task_tolerance_delay)

        user_equipment_energy_now = user_equipment.energy_now
        if offload_sum_energy > user_equipment_energy_now:
            temp_penalty = (offload_sum_energy - user_equipment_energy_now) * \
                           self.interface_config.reward_config.penalty_ue_energy_exhaust_when_offload
            offload_reward -= temp_penalty
            # self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.off_energy_exhausted])

        return offload_reward

    def get_real_local_reward(self, user_equipment_id, normalized_ue_computing_ability, ue_queue_time_now):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size
        # user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        user_equipment_allocated_computing_ability = \
            normalized_ue_computing_ability * self.network_config.user_equipment_config.user_equipment_computing_ability

        local_time = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            user_equipment_allocated_computing_ability

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/01_local_time',
                                   local_time, self.step_real_count)

        local_energy = \
            self.network_config.user_equipment_config.energy_consumption_per_cpu_cycle * \
            (user_equipment_allocated_computing_ability ** 2) * \
            user_equipment_task_data_size * user_equipment_task_computing_size

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/02_local_energy',
                                   local_energy, self.step_real_count)

        local_cost = (self.interface_config.cost_config.energy_cost_weight_if_local * local_energy +
                      self.interface_config.cost_config.time_cost_weight_if_local *
                      (local_time+ue_queue_time_now[user_equipment_id]))

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/03_local_cost',
                                   local_cost, self.step_real_count)

        local_reward = - local_cost * self.interface_config.reward_config.reward_weight_of_cost

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/04_local_reward_from_cost',
                                   local_reward, self.step_real_count)

        energy_now = user_equipment.energy_now
        if local_energy > energy_now:
            temp_penalty = (local_energy - energy_now) * \
                           self.interface_config.reward_config.penalty_ue_energy_exhaust_when_local
            if self.output_other_message:
                self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/05_local_energy_penalty',
                                       temp_penalty, self.step_real_count)
            local_reward -= temp_penalty
            self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.loc_energy_exhausted])
            print("ue_energy_penalty:", local_reward)

        if local_time + ue_queue_time_now[user_equipment_id] > user_equipment.task.task_tolerance_delay:
            temp_penalty = self.interface_config.reward_config.penalty_over_time_when_queue * \
                           (local_time + ue_queue_time_now[
                               user_equipment_id] - user_equipment.task.task_tolerance_delay)
            if self.output_other_message:
                self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/05_1_local_queue_penalty',
                                       temp_penalty, self.step_real_count)
            local_reward -= temp_penalty
            print("ue_queue_penalty:", local_reward)
            self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.loc_queue_time_too_long])

        local_queue_time_now = max(0, ue_queue_time_now[user_equipment_id] + \
                                   local_time - self.network_config.time_max)

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/06_local_reward_after_penalty',
                                   local_reward, self.step_real_count)

        '''self.writer.add_scalar('3_other_message/ue_id_' + str(user_equipment_id) + '_07_placeholder',
                               0, self.step_real_count)

        self.writer.add_scalar('3_other_message/ue_id_' + str(user_equipment_id) + '_08_placeholder',
                               0, self.step_real_count)'''

        return local_reward, local_energy, local_cost, local_queue_time_now

    def get_real_offload_reward(self, user_equipment_id, normalized_transmitting_power, ue_queue_time_now,
                                base_station_queue_time_now,
                                offload_count, action_offload_mask):
        user_equipment = self.hexagon_network.user_equipment_list[user_equipment_id]
        user_equipment_task_data_size = user_equipment.task.task_data_size
        user_equipment_task_computing_size = user_equipment.task.task_computing_size
        user_equipment_task_max_tolerance_delay = user_equipment.task.task_tolerance_delay

        user_equipment_transmitting_power = \
            self.network_config.user_equipment_config.max_transmitting_power * normalized_transmitting_power
        base_station_allocated_computing_ability = \
            self.network_config.base_station_config.base_station_computing_ability

        user_equipment_idle_power = self.network_config.user_equipment_config.user_equipment_idle_power
        spectral_efficiency = self.get_uplink_spectral_efficiency(user_equipment_id, action_offload_mask)
        data_transmission_rate = spectral_efficiency * self.interface_config.channel_config.bandwidth

        offload_uplink_time = user_equipment_task_data_size / data_transmission_rate
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/13_offload_uplink_time',
                                   offload_uplink_time, self.step_real_count)
        offload_uplink_energy = user_equipment_transmitting_power * offload_uplink_time
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/14_offload_uplink_energy',
                                   offload_uplink_energy, self.step_real_count)

        offload_computing_time = \
            user_equipment_task_data_size * user_equipment_task_computing_size / \
            base_station_allocated_computing_ability
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/15_offload_computing_time',
                                   offload_computing_time, self.step_real_count)
        offload_idle_energy = user_equipment_idle_power * offload_computing_time
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/16_offload_idle_energy',
                                   offload_idle_energy, self.step_real_count)

        offload_sum_time = offload_computing_time + offload_uplink_time
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/17_offload_sum_time',
                                   offload_sum_time, self.step_real_count)
        offload_sum_energy = offload_uplink_energy + offload_idle_energy
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/18_offload_sum_energy',
                                   offload_sum_energy, self.step_real_count)

        offload_cost = (self.interface_config.cost_config.energy_cost_weight_if_offload * offload_sum_energy +
                        self.interface_config.cost_config.time_cost_weight_if_offload * (offload_sum_time + 0))  # base_station_queue_time_now
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/19_offload_cost',
                                   offload_cost, self.step_real_count)
        offload_reward = - offload_cost * self.interface_config.reward_config.reward_weight_of_cost
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/20_offload_reward_from_cost',
                                   offload_reward, self.step_real_count)

        if offload_sum_time + base_station_queue_time_now > user_equipment_task_max_tolerance_delay:
            temp_penalty = (offload_sum_time + base_station_queue_time_now - user_equipment_task_max_tolerance_delay) * \
                           self.interface_config.reward_config.penalty_over_time_when_offload
            if self.output_other_message:
                self.writer.add_scalar(
                    '3_other_message_ue_id_' + str(user_equipment_id) + '/21_offload_queue_time_penalty',
                    temp_penalty, self.step_real_count)
            offload_reward -= temp_penalty
            print("bs_queue_penalty:", offload_reward)
            self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.off_queue_time_too_long])

        off_queue_time_now = max(0, ue_queue_time_now[user_equipment_id] - self.network_config.time_max)
        base_station_queue_time_now += offload_computing_time

        user_equipment_energy_now = user_equipment.energy_now
        if offload_sum_energy > user_equipment_energy_now:
            temp_penalty = (offload_sum_energy - user_equipment_energy_now) * \
                           self.interface_config.reward_config.penalty_ue_energy_exhaust_when_offload
            if self.output_other_message:
                self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/22_offload_energy_penalty',
                                       temp_penalty, self.step_real_count)
            offload_reward -= temp_penalty
            self.reason_to_finish_this_episode.append([user_equipment_id, ReasonEnum.off_energy_exhausted])
            print("bs_energy_penalty:", offload_reward)

        # offload_reward -= self.penalty_if_get_max_real_bs_computing_ability(user_equipment_id)
        if self.output_other_message:
            self.writer.add_scalar(
                '3_other_message_ue_id_' + str(user_equipment_id) + '/23_offload_reward_after_penalty',
                offload_reward, self.step_real_count)

        return offload_reward, offload_sum_energy, offload_cost, off_queue_time_now, base_station_queue_time_now

    def whether_error_this_id(self, user_equipment_id):
        for reason in self.reason_to_finish_this_episode:
            if reason[0] == user_equipment_id:
                print("reason:", reason)
                return True
        return False

    def update_base_station_queue_time(self, ue_num, base_station_queue_time_now):
        base_station_queue_time_now = max(0, base_station_queue_time_now - self.network_config.time_max)
        for user_equipment_id in range(ue_num):
            self.next_hexagon_network.user_equipment_list[
                user_equipment_id].belong_hexagon.base_station.queue_time_now = base_station_queue_time_now

    def step(self, user_equipment_state: State, user_equipment_action: Action, offload_count,
             ue_queue_time_now, base_station_queue_time_now, action_offload_mask):
        user_equipment_id = user_equipment_state.user_equipment_id

        action_too_small_flag = False
        action_too_small_penalty = 0
        user_equipment_action_max_transmitting_power = 1
        user_equipment_action_normalized_transmitting_power = \
            user_equipment_action.normalized_transmitting_power
        user_equipment_action_normalized_ue_computing_ability = 1

        reward_cmp1, local_cost_baseline = self.get_max_local_reward(user_equipment_id, ue_queue_time_now)
        if user_equipment_action.get_whether_offload():
            min_normalized_transmitting_power = self.get_min_normalized_transmitting_power(user_equipment_id)
            min_normalized_transmitting_power = 0.0
            # print("min_normalized_transmitting_power:{},real:{}".format(min_normalized_transmitting_power, user_equipment_action.normalized_transmitting_power))
            if user_equipment_action.normalized_transmitting_power <= min_normalized_transmitting_power:
                action_too_small_penalty -= \
                    (0.5 - min_normalized_transmitting_power) * \
                    self.interface_config.reward_config.penalty_transmitting_power_is_too_small

        if action_too_small_flag:
            if self.output_other_message:
                self.writer.add_scalar(
                    '3_other_message_ue_id_' + str(user_equipment_id) + '/25_action_too_small_penalty',
                    action_too_small_penalty, self.step_real_count)
            return action_too_small_penalty, self.get_next_state_per_user_equipment(user_equipment_id), 1, 0.5, 10, 10, local_cost_baseline

        if user_equipment_action.get_whether_offload():
            reward, energy, cost, queue_time_now, base_station_queue_time_now = self.get_real_offload_reward(
                user_equipment_id,
                user_equipment_action_normalized_transmitting_power,
                ue_queue_time_now,
                base_station_queue_time_now,
                offload_count,
                action_offload_mask)
        else:
            reward, energy, cost, queue_time_now = \
                self.get_real_local_reward(user_equipment_id,
                                           user_equipment_action_normalized_ue_computing_ability,
                                           ue_queue_time_now)

        reward += action_too_small_penalty
        done = 1
        if not self.whether_error_this_id(user_equipment_id):
            done = 0
            ue_queue_time_now[user_equipment_id] = queue_time_now
            self.next_hexagon_network.user_equipment_list[user_equipment_id].energy_now -= energy
            self.next_hexagon_network.user_equipment_list[user_equipment_id].queue_time_now = queue_time_now
            self.next_hexagon_network.user_equipment_list[
                user_equipment_id].belong_hexagon.base_station.queue_time_now = \
                base_station_queue_time_now

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/26_reward_before_cmp',
                                   reward, self.step_real_count)

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/28_reward_after_cmp',
                                   reward, self.step_real_count)

        return reward, self.get_next_state_per_user_equipment(
            user_equipment_id), done, cost, ue_queue_time_now, base_station_queue_time_now, local_cost_baseline

    def step_torch(self, user_equipment_state: State, user_equipment_action: Action, offload_count,
             ue_queue_time_now, base_station_queue_time_now, action_offload_mask):
        user_equipment_id = user_equipment_state.user_equipment_id

        action_too_small_flag = False
        action_too_small_penalty = 0
        user_equipment_action_max_transmitting_power = 1
        user_equipment_action_normalized_transmitting_power = \
            user_equipment_action.normalized_transmitting_power
        user_equipment_action_normalized_ue_computing_ability = 1

        reward_cmp1, local_cost_baseline = self.get_max_local_reward(user_equipment_id, ue_queue_time_now)
        if user_equipment_action.get_whether_offload():
            min_normalized_transmitting_power = self.get_min_normalized_transmitting_power(user_equipment_id)
            min_normalized_transmitting_power = 0.0
            # print("min_normalized_transmitting_power:{},real:{}".format(min_normalized_transmitting_power, user_equipment_action.normalized_transmitting_power))
            if user_equipment_action.normalized_transmitting_power <= min_normalized_transmitting_power:
                action_too_small_penalty -= \
                    (0.5 - min_normalized_transmitting_power) * \
                    self.interface_config.reward_config.penalty_transmitting_power_is_too_small

        if action_too_small_flag:
            if self.output_other_message:
                self.writer.add_scalar(
                    '3_other_message_ue_id_' + str(user_equipment_id) + '/25_action_too_small_penalty',
                    action_too_small_penalty, self.step_real_count)
            return action_too_small_penalty, self.get_next_state_per_user_equipment(user_equipment_id), 1, 0.5, 10, 10, local_cost_baseline

        if user_equipment_action.get_whether_offload():
            reward, energy, cost, queue_time_now, base_station_queue_time_now = self.get_real_offload_reward(
                user_equipment_id,
                user_equipment_action_normalized_transmitting_power,
                ue_queue_time_now,
                base_station_queue_time_now,
                offload_count,
                action_offload_mask)
        else:
            reward, energy, cost, queue_time_now = \
                self.get_real_local_reward(user_equipment_id,
                                           user_equipment_action_normalized_ue_computing_ability,
                                           ue_queue_time_now)

        reward += action_too_small_penalty
        done = 1
        if not self.whether_error_this_id(user_equipment_id):
            done = 0
            ue_queue_time_now[user_equipment_id] = queue_time_now
            self.next_hexagon_network.user_equipment_list[user_equipment_id].energy_now -= energy
            self.next_hexagon_network.user_equipment_list[user_equipment_id].queue_time_now = queue_time_now
            self.next_hexagon_network.user_equipment_list[
                user_equipment_id].belong_hexagon.base_station.queue_time_now = \
                base_station_queue_time_now

        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/26_reward_before_cmp',
                                   reward, self.step_real_count)
        if self.output_other_message:
            self.writer.add_scalar('3_other_message_ue_id_' + str(user_equipment_id) + '/28_reward_after_cmp',
                                   reward, self.step_real_count)

        return reward, self.get_next_state_per_user_equipment(
            user_equipment_id), done, cost, ue_queue_time_now, base_station_queue_time_now, local_cost_baseline
