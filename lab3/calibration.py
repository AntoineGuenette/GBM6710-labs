import numpy as np

def calibration(camera_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    """
    Fonction qui détermine les coefficiens de calibration de la caméra
    -------------------
    Input:
        camera_points: vecteur contenant les coordonnées des points sélectionnés dans le repère du robot
        world_points: vecteur contenant les coordonnées des points sélectionnés dans l'image
    """
    m_vector = camera_points[:,0]
    n_vector = camera_points[:,1]
    x_vector = world_points[:,0]
    y_vector = world_points[:,1]

    M = np.stack(arrays = (m_vector ** 2, m_vector * n_vector, n_vector ** 2, m_vector, n_vector, np.ones_like(m_vector)), axis =1 )
    R = np.stack(arrays = (x_vector, y_vector), axis = 1)
    U = np.linalg.inv(M.T @ M) @ M.T @ R
    return U


