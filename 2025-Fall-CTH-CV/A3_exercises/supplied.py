import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

def psphere(x):
    """ Normalization of projective points. Do a normalization on the projective
        points in x. Each column is considered to be a point in homogeneous coordinates.
         INPUT:
              x     - matrix in which each column is a point.
         OUTPUT:
              y     - result after normalization.
              alpha - depth
    """
    alpha = np.sqrt(np.sum(x**2, axis=0))       
    y = x / alpha                             
    return y, alpha
    
    
# Helper functions
def pflat(x):
    # Normalizes (m,n)-array x of n homogeneous points
    # to last coordinate 1.
    y = x / x[-1,:]
    return y

def camera_center_and_axis(P):
    # The camera center can be found by taking the null space of the camera matrix
    camera_center = pflat(sp.linalg.null_space(P))[:3]

    principal_axis = P[-1, :3]
    principal_axis = principal_axis / np.linalg.norm(principal_axis)

    return camera_center, principal_axis

def plot_camera(camera_matrix, scale, ax=None):
    if ax is None:
        ax = plt.axes(projection='3d')
    (camera_center, principal_direction) = camera_center_and_axis(camera_matrix)

    ax.scatter3D(camera_center[0], camera_center[1], camera_center[2], c='g')
    dir = principal_direction * scale
    ax.quiver(camera_center[0], camera_center[1], camera_center[2], dir[0], dir[1], dir[2], color='r')

def rital(lines, st='r-'):
    """ 
    Takes a nx3 matrix "lines" as input where each row represents the homogeneous coordinates of a 
    2D line. The function then plots the lines in the last opened figure.
    
    Args:
        lines (np.ndarray): A NumPy array of shape (n, 3) where each row represents the homogeneous 
            coordinates of a 2D line.
        st (str): Optional argument that controls the line style of the plot.
    """
    
    ax = plt.gca()
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    
    nn = lines.shape[0]
    for i in range(nn):
        line = lines[i]
        rikt,_ = psphere(np.array([[line[1]], [-line[0]], [0]]))
        punkter = np.cross(rikt.T, line).reshape(-1, 1)
        punkter = punkter / punkter[-1,:]
    
        x_coords = [punkter[0, 0] - 2000 * rikt[0, 0], punkter[0, 0] + 2000 * rikt[0, 0]]
        y_coords = [punkter[1, 0] - 2000 * rikt[1, 0], punkter[1, 0] + 2000 * rikt[1, 0]]
        ax.plot(x_coords, y_coords, st)
        
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    
