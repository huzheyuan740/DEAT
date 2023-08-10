from env.hn_printer import HexagonNetworkPrinter
from config import GlobalConfig

global_config = GlobalConfig()
hexagon_network_printer = HexagonNetworkPrinter(global_config)
hexagon_network_printer.record_hexagon_network_move_way_to_gif()
