from enum import Enum, auto


class ReasonEnum(Enum):
    transmitting_power_is_too_small = auto()
    ue_computing_ability_is_too_small = auto()
    bs_computing_ability_is_too_small = auto()

    loc_energy_exhausted = auto()

    off_time_over_delay = auto()
    off_energy_exhausted = auto()

    loc_queue_time_too_long = auto()
    off_queue_time_too_long = auto()
