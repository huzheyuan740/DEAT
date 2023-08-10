from config import GlobalConfig
from env.hexagon_network import HexagonNetwork


class HexagonNetworkPrinter:
    def __init__(self, global_config: GlobalConfig):
        self.global_config = global_config
        self.hexagon_network = HexagonNetwork(global_config)

    def print_hexagon_network(self):
        # print the hexagon_network to the screen
        self.hexagon_network.print_hexagon_network()

    def record_hexagon_network_move_way_to_gif(self, fig_num: int = 100, fps: int = 10, dpi: int = 0):
        # save the figure of the hexagon_network and add all these figures to one gif for play
        # fig_num is the num of all the saved figures
        # fps is the figure played one second
        # dpi is the screen resolution
        name_str_list = []
        for i in range(fig_num):
            name_str_list.append(str(i) + ".jpg")
        temp_hexagon_network = HexagonNetwork(self.global_config)
        for i in range(fig_num):
            if dpi == 0:
                temp_hexagon_network.print_hexagon_network(1, name_str_list[i])
            else:
                temp_hexagon_network.print_hexagon_network(1, name_str_list[i], dpi=dpi)
            temp_hexagon_network.update_all_ue_triple_message()
        import imageio
        gif_images = []
        for name_str in name_str_list:
            gif_images.append(imageio.imread(name_str))
        imageio.mimsave("moving_user_equipment.gif", gif_images, fps=fps)
        import os
        for name_str in name_str_list:
            os.remove(name_str)
