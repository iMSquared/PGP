from heapq import merge
import os
import math
from typing import Dict
import numpy as np

import pybullet as pb
import pybullet_data
from pybullet_object_models import ycb_objects

from imm.pybullet_util.bullet_client import BulletClient
from utils.process_pointcloud import (
    generate_point_cloud_from_uid, 
    DepthToPointcloudOpenGL )
from utils.process_geometry import (
    matrix_base_to_world,
    matrix_camera_to_world,
    matrix_com_to_base )
from envs.robot import UR5


class BinpickEnv():

    def __init__(self, bc, config, shape_table=None, objects_urdf=None):

        self.bc = bc
        self.config = config
        self._shape_table = shape_table
        self._objects_urdf = objects_urdf
        env_params = config["env_params"]["binpick_env"]
        

        # Path to URDF
        pybullet_data_path = pybullet_data.getDataPath()
        project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])

        # Load URDFs
        self.plane_uid = self.bc.loadURDF(
            fileName        = os.path.join(pybullet_data_path, "plane.urdf"), 
            basePosition    = (0.0, 0.0, 0.0), 
            baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase    = True)
        self.cabinet_uid = self.bc.loadURDF(
            fileName        = os.path.join(urdf_path, env_params["cabinet"]["path"]),
            basePosition    = env_params["cabinet"]["pos"],
            baseOrientation = self.bc.getQuaternionFromEuler(env_params["cabinet"]["orn"]),
            useFixedBase    = True)
        # TODO: This will later be replaced with random initialization
        self.objects_uid = ([    
            self.bc.loadURDF(
                fileName        = os.path.join(ycb_objects.getDataPath(), env_params["objects"]["path"][i]),
                basePosition    = env_params["objects"]["pos"][i],
                baseOrientation = self.bc.getQuaternionFromEuler(env_params["objects"]["orn"][i]),
                useFixedBase    = False,
                globalScaling   = env_params["objects"]["scale"][i])
            for i in range(env_params["objects"]["num_objects"])
        ])
        # Adjust dynamics
        for uid in self.objects_uid:
            self.bc.changeDynamics(
                uid, 
                -1, 
                lateralFriction=1.6,
                rollingFriction=0.0004,
                spinningFriction=0.0004,
                restitution=0.2)

        # RGB-D camera config (intrinsic)
        camera_params = env_params["depth_camera"]

        self.width    = camera_params["width"]
        self.height   = camera_params["height"]
        self.fov      = camera_params["fov"]
        self.voxel_down_sample_size = camera_params["voxel_down_sample_size"]

        self.camera_view_matrix = self.bc.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition = camera_params["target_pos"],
            distance             = camera_params["distance"],
            roll                 = camera_params["roll"],
            pitch                = camera_params["pitch"],
            yaw                  = camera_params["yaw"],
            upAxisIndex          = 2)
        
        self.camera_proj_matrix = self.bc.computeProjectionMatrixFOV(
            fov     = camera_params["fov"],
            aspect  = float(self.width) / float(self.height),
            nearVal = camera_params["near_val"],
            farVal  = camera_params["far_val"])

        self.reprojection_function = DepthToPointcloudOpenGL(
            cx                     = self.width / 2.0,
            cy                     = self.height / 2.0,
            near_val               = camera_params["near_val"],
            far_val                = camera_params["far_val"],
            focal_length           = self.height / ( 2.0 * math.tan(math.radians(self.fov) / 2.0) ))

        # NOTE(ssh):For the calculatation of the focal length in pybullet, see the material below
        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet



    def reset(self, state): 
        pass



    # def get_measurement(self) -> Dict[int, object]:
    #     '''
    #     Capture a measurement the RGB-D camera in pybullet.

    #     Process includes...
    #     1. (Pre)Get view matrix
    #     2. (Pre)Get projection matrix from view matrix
    #     3. (Runtime)Get Image from proj matrix
    #     4. (Runtime)Convert values to np.array
    #     5. (Runtime)Convert and reproject depth image to open3d type pointcloud.
    #     '''
    #     # Get camera image from simulation
        
    #     # import time
    #     # t = time.time()
        
    #     (w, h, px, px_d, px_id) = self.bc.getCameraImage(
    #         width            = self.width,
    #         height           = self.height,
    #         viewMatrix       = self.camera_view_matrix,
    #         projectionMatrix = self.camera_proj_matrix,
    #         renderer         = self.bc.ER_BULLET_HARDWARE_OPENGL)


    #     # Reshape list into ndarray(image)
    #     rgb_array = np.array(px, dtype=np.uint8)
    #     rgb_array = rgb_array[:, :, :3]                 # remove alpha
    #     depth_array = np.array(px_d, dtype=np.float32)
    #     mask_array = np.array(px_id, dtype=np.uint8)

    #     # t2 = time.time()
    #     # print("getCameraImage():", t2 - t)

    #     # Reprojection and background removal
    #     measurement = self.reprojection_function(
    #         depth_2d     = depth_array, 
    #         mask_2d      = mask_array,
    #         segment_list = self.objects_uid)

    #     # Transform to world coordinate
    #     T_c2w = matrix_camera_to_world(self.camera_view_matrix)
    #     for key in measurement.keys():
    #         measurement[key].transform(T_c2w)
            
    #     # t3 = time.time()
    #     # print("Image2PointCloud:", t3 - t2)

    #     return measurement


    def get_pcd(self) -> Dict[int, object]:
        '''
        Acquire the current groundtruth state of the environment in point cloud.
        TODO(ssh): Support segmentation
        '''
        # Get groundtruth point clouds
        pcd_dict_groundtruth = {
                uid : self.get_pcd_from_uid(uid)
            for uid in self.objects_uid}

        return pcd_dict_groundtruth


    def get_pcd_from_uid(self, uid) -> object:
        '''
        A private member function for aquiring the groundtruth of a single object in environment.
        '''
        # Transformation process 
        # 1. Pointcloud is sampled from the local visual frame(mesh design). 
        # 2. T_v2l: local visual frame(mesh design) ->   link frame(center of mass).
        # 3. T_l2b: link frame(center of mass)      ->   base frame(urdf link origin) of the object.
        # 4. T_b2w: base frame(urdf link origin)    ->   world frame(pybullet origin)
        # 5. T_w2c: world frame(pybullet origin)    ->   camera frame(camera origin)

        # Get pcd and matrices
        pcd, T_v2l = generate_point_cloud_from_uid(self.bc, uid)
        T_l2b = matrix_com_to_base(self.bc, uid)
        T_b2w = matrix_base_to_world(self.bc, uid)

        # Matrix chain. Be aware that the later transform comes to the first argument.
        T_v2b = np.matmul(T_l2b, T_v2l)
        T_v2w = np.matmul(T_b2w, T_v2b)
        pcd.transform(T_v2w)

        return pcd
    
    
    @property
    def shape_table(self):
        return self._shape_table
    
    @shape_table.setter
    def shape_table(self, shape_table):
        self._shape_table = shape_table
        

    @property
    def objects_urdf(self):
        return self._objects_urdf
    
    @objects_urdf.setter
    def objects_urdf(self, objects_urdf):
        self._objects_urdf = objects_urdf