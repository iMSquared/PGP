import itertools
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from typing import Dict, List, Tuple, Union, Optional

from imm.pybullet_util.bullet_client import BulletClient



def generate_point_cloud_from_uid(bc:BulletClient, uid:int, num_points:int=300) -> \
        Tuple[o3d.geometry.PointCloud, np.ndarray] :
    '''
    Generate Open3D pointcloud from PyBullet URDF id.
    The poincloud has its origin at the base of the model.
    This function will return only the first link of the multi-rigidbody object.

    params
    - bc: Bullet clinet
    - uid: Object uid to generate point cloud
    - num_points: Number of points in the cloud

    returns
    - pcd: A point cloud
    - T_v2l: A transformation matrix from local visual frame(mesh design) to link frame(center of mass). 
    '''

    # Parse base link info
    base_link_info = bc.getVisualShapeData(uid)[0]  # [0] is base link

    # Generate the shape     
    mesh = o3d.io.read_triangle_mesh(base_link_info[4].decode('UTF-8'))
    pcd:o3d.geometry.PointCloud = mesh.sample_points_poisson_disk(number_of_points=num_points, init_factor=2)
    
    # Create the transformation matrix
    # local visual frame(mesh design) -> link frame(center of mass)
    v2l_pos = base_link_info[5]
    v2l_orn = np.reshape(
        bc.getMatrixFromQuaternion(base_link_info[6]), 
        (3, 3))
    v2l_scale = base_link_info[3]
    
    scaling_matrix = np.array([[v2l_scale[0], 0, 0, 0],
                               [0, v2l_scale[1], 0, 0],
                               [0, 0, v2l_scale[2], 0],
                               [0, 0, 0, 1]])

    translation_matrix = np.array([[1, 0, 0, v2l_pos[0]],
                                   [0, 1, 0, v2l_pos[1]],                                   
                                   [0, 0, 1, v2l_pos[2]],
                                   [0, 0, 0, 1]]) 
                                   
    rotation_matrix = np.array([[v2l_orn[0,0], v2l_orn[0,1], v2l_orn[0,2], 0],
                                [v2l_orn[1,0], v2l_orn[1,1], v2l_orn[1,2], 0],    
                                [v2l_orn[2,0], v2l_orn[2,1], v2l_orn[2,2], 0],
                                [0, 0, 0, 1]])

    # Transformation matrix (link -> visual)
    # NOTE(ssh): 
    #   In URDF, the origin tag works with the following order
    #   1. rpy 
    #   2. xyz
    #   3. scaling 
    T_v2l = np.matmul(translation_matrix, rotation_matrix)
    T_v2l = np.matmul(T_v2l, scaling_matrix)

    return pcd, T_v2l
    


def transform_point_cloud(pcd:o3d.geometry.PointCloud, T:np.ndarray):
    # Apply transform
    pcd.transform(T)
    


def merge_point_cloud(pcd_list: Tuple[o3d.geometry.PointCloud, ...]) -> o3d.geometry.PointCloud:
    '''
    Merge multiple point clouds in the list
    Reference: https://sungchenhsi.medium.com/adding-pointcloud-to-pointcloud-9bf035707273
    '''
    merged_pcd_numpy = np.empty((0, 3), dtype=float)
    for _pcd in pcd_list:
        _pcd_numpy = np.asarray(_pcd.points)                 # Use np.asarray to avoid meaningless copy
        merged_pcd_numpy = np.concatenate((merged_pcd_numpy, _pcd_numpy), axis=0)

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(merged_pcd_numpy)

    return merged_pcd



def remove_hidden_points(pcd: o3d.geometry.PointCloud, camera: np.array) -> Tuple[o3d.geometry.PointCloud, List[int]]:
    '''
    Remove Hidden Points from Open3D pointcloud
    Reference: http://www.open3d.org/docs/latest/tutorial/Basic/pointcloud.html

    Params:
    - pcd: Point cloud to process
    - camera: The location of the camera. Orientation is not considered.
    '''
    diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))

    radius = diameter * 100
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd = pcd.select_by_index(pt_map)
    
    return pcd, pt_map



def visualize_point_cloud(pcds:list, 
                          lower_lim=-0.25, 
                          upper_lim=0.25, 
                          save:bool=False, 
                          save_path:Optional[str]=None):
    '''
    Visualize the numpy point cloud
    '''

    if save:
        plt.switch_backend('Agg') # tkinter keeps crashing... :(

    colors = ["Red", "Blue", "Green", "tab:orange", "magenta", "tab:blue", "tab:purple", "tab:olive"]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim([lower_lim, upper_lim])
    ax.set_xlim([0.25, 0.75])
    ax.set_ylim([lower_lim, upper_lim])
    # ax.set_zlim([lower_lim, upper_lim])
    ax.set_zlim([0.2, 0.7])

    # Plot points
    for i, pcd in enumerate(pcds):
        ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=colors[i % len(colors)])

    if not save:
        plt.show()
        print(4)
    else:
        fig.savefig(save_path)
        plt.close(fig)



class DepthToPointcloudOpenGL:
    """I don't want to explain..."""

    def __init__(
            self,
            cx: float, 
            cy: float, 
            near_val: float, 
            far_val: float, 
            focal_length: float):

        # Camera intrinsics
        self.cx = cx
        self.cy = cy
        self.near_val = near_val
        self.far_val = far_val
        self.focal_length = focal_length


    def __call__(self, depth_2d: np.ndarray, 
                       mask_target_2d: np.ndarray) -> np.ndarray:
        """Convert depth image to Open3D point clouds.
        The depth value should follow openGL convention which is used in PyBullet.

        Args:
            depth_2d (np.ndarray): Depth image
            mask_target_2d (np.ndarray[bool]): Target pixels to convert to point cloud
        Returns:
            pointcloud (np.ndarray): Reprojected target point cloud
        """
        # Convert depth image to pcd in pixel unit
        # (x, y, z, class), y-up
        pcd = np.array([[
                    (self.cy - v, self.cx - u, depth_2d[v, u], mask_target_2d[v, u])
                for u in range(depth_2d.shape[1])]
            for v in range(depth_2d.shape[0])]).reshape(-1, 4)

        # Getting true depth from OpenGL style perspective matrix
        #   NOTE(ssh): 
        #   For the calculation of the reprojection, see the material below
        #   https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
        # Calculate z
        z_b = pcd[:,2]
        z_n = 2.0 * z_b - 1.0
        z = 2.0 * self.near_val * self.far_val / (self.far_val + self.near_val - z_n * (self.far_val - self.near_val))
        # Calculate x
        x = z * pcd[:,1] / self.focal_length
        # Calculate y
        y = z * pcd[:,0] / self.focal_length
        # Copy uid class label
        c = pcd[:,3]

        # Stack
        pcd = np.stack((x, y, z, c), axis=1)
        # Convert y-up to z-up
        pcd = pcd[:,[2, 0, 1, 3]]

        # Select target points only
        pcd = pcd[pcd[:,3]==True]
        pcd = pcd[:,:3]


        return pcd

