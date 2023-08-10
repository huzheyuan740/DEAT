class Coordinate:
    def __init__(self, x_coordinate: float, y_coordinate: float, z_coordinate: float) -> None:
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.z_coordinate = z_coordinate

    def __eq__(self, other) -> bool:
        import math
        return (math.isclose(self.x_coordinate, other.x_coordinate, abs_tol=1e-2)
                and
                math.isclose(self.y_coordinate, other.y_coordinate, abs_tol=1e-2)
                and
                math.isclose(self.z_coordinate, other.z_coordinate, abs_tol=1e-2))

    def get_coordinate_triple_list(self):
        return [self.x_coordinate, self.y_coordinate, self.z_coordinate]

    def get_coordinate_triple_array(self):
        import numpy as np
        return np.array(self.x_coordinate, self.y_coordinate, self.z_coordinate)
