import numpy as np

def get_world_grid(uppr_left_corner_coord: tuple, grid_size: int):
    upper_x_coord, upper_y_coord = uppr_left_corner_coord
    x_coords = np.tile(np.array([upper_x_coord-25*i for i in range(grid_size)]),(grid_size,1)).T
    y_coords = np.tile(np.array([upper_y_coord-25*j for j in range(grid_size)]),(grid_size,1))
    world_points = np.stack((x_coords, y_coords),axis = 2 )
    return world_points


def grid_to_vec(world_points):
    (rows,col,_) = world_points.shape

    coords = np.zeros((rows*col,2))
    for k in range(rows):
        for l in range(col):
            index = k * col + l
            coords[index,0] = world_points[k,l,0]
            coords[index,1] = world_points[k,l,1]
    return coords


world_points = get_world_grid((250,250),4)

def calibration(camera_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """
    Fonction qui détermine les coefficiens de calibration de la caméra
    -------------------
    Input:
        camera_points: vecteur contenant les coordonnées des points sélectionnés dans le repère du robot
        world_points: vecteur contenant les coordonnées des points sélectionnés dans l'image
    """
    world_points = grid_to_vec(world_points)
    camera_points = grid_to_vec(camera_points)
    m_vector = camera_points[:,0]
    n_vector = camera_points[:,1]
    x_vector = world_points[:,0]
    y_vector = world_points[:,1]

    M = np.stack(arrays = (m_vector ** 2, m_vector * n_vector, n_vector ** 2, m_vector, n_vector, np.ones_like(m_vector)), axis =1 )
    R = np.stack(arrays = (x_vector, y_vector), axis = 1)
    U = np.linalg.inv(M.T @ M) @ M.T @ R
    return U


