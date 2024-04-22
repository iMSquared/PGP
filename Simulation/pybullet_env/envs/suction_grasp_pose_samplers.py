import random
import numpy as np
import open3d as o3d
from typing import Tuple, Dict
from scipy.spatial.transform import Rotation as R
from utils.process_pointcloud import visualize_point_cloud
import time
import numpy.typing as npt

import alphashape
from shapely import geometry

from imm.pybullet_util.typing_extra import TranslationT, EulerT



class TopPointSamplingGraspSuction:
        
    R_SUCTION           = 0.01     # Suction cup radius
    ALPHA               = 100      # Alpha shape radius?
    ALIGNMENT_THRESHOLD = 0.95     # cosine(\theta) similarity between normal and grasp
    
    @classmethod
    def make_afford_boundary_and_filtered_points(cls, pcd_points: Tuple[TranslationT], 
                                                      pcd_normals: Tuple[TranslationT],
                                                      obj_pos: TranslationT,
                                                      obj_orn: EulerT,
                                                      T_v2b: npt.NDArray,
                                                      ) -> Tuple[object, npt.NDArray]:
        """Get afford boundary(2d alpha shape) and filtered points(from point cloud.)

        Args:
            pcd_points (Tuple[TranslationT])
            pcd_normals (Tuple[TranslationT])
            obj_pos (TranslationT): URDF base pos
            obj_orn (EulerT): URDF base orn
            T_v2b (npt.NDArray): Transformation form visual mesh to URDF base.

        Returns:
            afford_boundary (alphashape): Eroded boundary alphashape.
            filtered_points (npt.NDArray): Subset of pointcloud with z-up normals in world frame.
        """
    
        # Get transformation matrix
        pos = obj_pos
        orn = obj_orn
        r = R.from_euler('xyz', orn, degrees=False)
        rot_mat = r.as_matrix()
        trans_mat = np.array(
            [[1, 0, 0, pos[0]],
            [0, 1, 0, pos[1]],
            [0, 0, 1, pos[2]],
            [0, 0, 0, 1]])
        trans_mat[:3, :3] = rot_mat
        trans_mat = np.matmul(trans_mat, T_v2b) # Visual -> World


        # Project the object's pointcloud from the local to world frame.
        candidates = np.ones((len(pcd_points), 4))
        candidates[:,0:3] = pcd_points
        candidates = candidates@trans_mat.T
        candidates = candidates[:,0:3]
        pcd_normals = pcd_normals@rot_mat.T

        # Filter out points without normals pointing z-direction
        # (Normal should point z-up direction when transformed)
        filtered_idx = [i for i, normal in enumerate(pcd_normals) 
                        if normal.T@np.array([0, 0, 1]) > cls.ALIGNMENT_THRESHOLD]
        if len(filtered_idx) == 0:
            return None, None

        # Create contour of the filtered points using alphashape
        filtered_points = candidates[filtered_idx]
        points_2d = filtered_points[:,:2]           # Discard z.. without clustering.
        
        boundary = alphashape.alphashape(points_2d, cls.ALPHA)
        afford_boundary = boundary.buffer(-cls.R_SUCTION)
        
        return afford_boundary, filtered_points


    @classmethod
    def sample_grasp_in_boundary(cls, afford_boundary: object, 
                                      filtered_points: npt.NDArray) \
                                        -> Tuple[TranslationT, EulerT, int]:
        """Sample pick pose from the afford_boundary and filtered_points
        acquired from `make_afford_boundary_and_filtered_points()` 

        Args:
            afford_boundary (alphashape): Eroded alphashape
            filtered_points (npt.NDArray): Subset of pointcloud with z-up normals in world frame.

        Returns:
            TranslationT: Global position of sampled point 
            EulerT: Z-up vector
            int: Index of selected point in pcd.
        """
        # Selecting a point
        points_2d = filtered_points[:,:2]
        indices = list(range(len(points_2d)))
        random.shuffle(indices)

        # Handling multipolygon alphashape (multiple instance)
        if isinstance(afford_boundary, geometry.multipolygon.MultiPolygon):
            afford_boundary = random.choice(afford_boundary.geoms)
            print('multi polygon')

        # Check whether the sample point lies in the afforance boundary.
        idx = None
        for i in indices:
            p = points_2d[i]
            if afford_boundary.contains(geometry.Point(p[0], p[1])):
                idx = i
                break
        
        # 
        if idx is None:
            return None, None, None

        pos = filtered_points[idx]
        orn = (0., 0., 0.)   # Top grasp (Z-up vector) (Should not be numpy type)

        return pos, orn, idx




class PointSamplingGraspSuction:

    def __init__(self):
        pass

    def __call__(self, pcd_points: Tuple[TranslationT], 
                       pcd_normals: Tuple[TranslationT],
                       obj_pos: TranslationT,
                       obj_orn: EulerT) -> Tuple[TranslationT, EulerT]:
        '''
        Return grasp pose in SE(3) by following the process as below:
            1) Sample one surface point and obtain its normal vector - it is to be a target pose
            2) Calculate the rotation matrix (1) from default heading axis to the normal vector
            3) Obtain orthonormal vector to the normal vector which is parallel to the xy-plane
            4) Sample one orthonormal vector and its angle
            5) Calculate the rotation matrix (2) from default axis aligning to fingers to selected vector in 4)
            7) Calculate target orientation which rotates (1) and (2) sequentially

        Args:
            pcd_points (Tuple[TranslationT]): skip
            pcd_normals (Tuple[TranslationT]): skip
            obj_pos (TranslationT): Object's base position
            obj_orn (EulerT): Object's base orientation.

        Returns:
            pos (TranslationT): x y z
            orn (EulerT): yaw pitch roll
        '''
        # Select surface point and its normal vector
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=5))
        
        pos = obj_pos
        orn = obj_orn
        r = R.from_euler('zyx', orn, degrees=False)
        rot_mat = r.as_matrix()
        
        trans_mat = np.array(
            [[1, 0, 0, pos[0]],
            [0, 1, 0, pos[1]],
            [0, 0, 1, pos[2]],
            [0, 0, 0, 1]])
        trans_mat[:3, :3] = rot_mat
        
        # pcd_points = pcd['points']
        # pcd_points = np.asarray(pcd.points)
                
        # # Debugging - visualizing point cloud
        # visualize_point_cloud([pcd_points])
        
        center_point = [0, 0, 0, 1]
        center_point[0:3] = np.average(pcd_points, axis=0)
        center_point = np.matmul(center_point, trans_mat.T)
        center_point = center_point[0:3]
        
        rnd_idx = random.choice(range(len(pcd_points)))
        
        sampled_point = [0, 0, 0, 1]
        sampled_point[0:3] = pcd_points[rnd_idx]
        sampled_point = np.matmul(sampled_point, trans_mat.T)   # World frame
        sampled_point = sampled_point[0:3]
        
        
        # normal_vec = pcd.normals[rnd_idx]
        normal_vec = pcd_normals[rnd_idx]
        normal_vec = np.matmul(normal_vec, rot_mat.T)
        
        # Transform sampled point and its normal vector
        

        # Choose the normal vector toward object
        sampled_point_to_center_point_vec = center_point - sampled_point
        if np.dot(normal_vec, sampled_point_to_center_point_vec) < 0:
            normal_vec = - normal_vec

        z_coordinate = normal_vec / np.linalg.norm(normal_vec)

        z_axis = np.asarray([0, 0, 1])
        y_coordinate = np.cross(z_axis, normal_vec)
        y_coordinate /= np.linalg.norm(y_coordinate)

        x_coordinate = np.cross(y_coordinate, z_coordinate)
        x_coordinate /= np.linalg.norm(x_coordinate)
        
        # Target position
        pos = sampled_point

        #  Target orientation (two possible grasp)
        rot_mat = np.stack([x_coordinate, y_coordinate, z_coordinate], axis=-1)
        orn_q = self.rot_matrix_to_quaternion(rot_mat)
        orn = self.euler_from_quaternion(orn_q)

        return pos, orn


    @classmethod
    def rot_matrix_to_quaternion(cls, m):    # |NOTE(Jiyong)|: It can be replaced with rotation transform of Scipy
        t = np.matrix.trace(m)
        q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if(t > 0):
            t = np.sqrt(t + 1)
            q[3] = 0.5 * t
            t = 0.5/t
            q[0] = (m[2,1] - m[1,2]) * t
            q[1] = (m[0,2] - m[2,0]) * t
            q[2] = (m[1,0] - m[0,1]) * t

        else:
            i = 0
            if (m[1,1] > m[0,0]):
                i = 1
            if (m[2,2] > m[i,i]):
                i = 2
            j = (i+1)%3
            k = (j+1)%3

            t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
            q[i] = 0.5 * t
            t = 0.5 / t
            q[3] = (m[k,j] - m[j,k]) * t
            q[j] = (m[j,i] + m[i,j]) * t
            q[k] = (m[k,i] + m[i,k]) * t

        return q
    
    @classmethod
    def euler_from_quaternion(cls, orn_q):
            """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)

            Reference: https://automaticaddison.com/how-to-convert-a-quaternion-into-euler-angles-in-python/
            """
            t0 = +2.0 * (orn_q[3] * orn_q[0] + orn_q[1] * orn_q[2])
            t1 = +1.0 - 2.0 * (orn_q[0] * orn_q[0] + orn_q[1] * orn_q[1])
            roll_x = np.arctan2(t0, t1)
        
            t2 = +2.0 * (orn_q[3] * orn_q[1] - orn_q[2] * orn_q[0])
            t2 = +1.0 if t2 > +1.0 else t2
            t2 = -1.0 if t2 < -1.0 else t2
            pitch_y = np.arcsin(t2)
        
            t3 = +2.0 * (orn_q[3] * orn_q[2] + orn_q[0] * orn_q[1])
            t4 = +1.0 - 2.0 * (orn_q[1] * orn_q[1] + orn_q[2] * orn_q[2])
            yaw_z = np.arctan2(t3, t4)
        
            return roll_x, pitch_y, yaw_z # in radians
