import os
import numpy as np
import pybullet_data
from typing import Dict

from utils.process_pointcloud import (
    generate_point_cloud_from_uid, 
    DepthToPointcloudOpenGL )
from utils.process_geometry import (
    matrix_base_to_world,
    matrix_camera_to_world,
    matrix_com_to_base )


class GraspTestEnv():

    def __init__(self, bc, robot, config, objects_urdf):
        self.bc = bc
        self.robot = robot
        self.config = config
        self._objects_urdf = objects_urdf
        self.env_params = config["env_params"]["grasp_test_env"]

        # Path to URDF
        pybullet_data_path = pybullet_data.getDataPath()
        # Load URDFs
        self.plane_uid = self.bc.loadURDF(
            fileName        = os.path.join(pybullet_data_path, "plane.urdf"), 
            basePosition    = (0.0, 0.0, 0.0), 
            baseOrientation = bc.getQuaternionFromEuler((0.0, 0.0, 0.0)),
            useFixedBase    = True)
        self.table_uid = self.bc.loadURDF(
            fileName        = os.path.join(pybullet_data_path, "table/table.urdf"), 
            basePosition    = (0.7, 0.0, -0.15), 
            baseOrientation= (0, 0, 0.7071, 0.7071),
            useFixedBase    = True)
        self.objects_uid = [self.bc.loadURDF(
                fileName        = objects_urdf,
                basePosition    = self.env_params["objects"]["pos"],
                baseOrientation = self.bc.getQuaternionFromEuler(self.env_params["objects"]["orn"]),
                useFixedBase    = False)]
        
        # Adjust dynamics
        for uid in self.objects_uid:
            self.bc.changeDynamics(
                uid, 
                -1, 
                lateralFriction=0.8,
                rollingFriction=0.0004,
                spinningFriction=0.0004,
                restitution=0.2)
            
    def reset(self):
        # Reset robot joints
        for i, idx in enumerate(self.robot.joint_indices_arm):
            self.bc.resetJointState(self.robot.uid, idx, self.config["robot_params"]["ur5"]["rest_pose"][self.robot.joint_indices_arm[i]])
        for idx in self.robot.joint_indices_arm:
            self.robot.last_pose[idx] = self.config["robot_params"]["ur5"]["rest_pose"][idx]
        
        self.bc.resetJointState(self.robot.uid, self.robot.joint_index_finger, self.config["robot_params"]["ur5"]["rest_pose"][self.robot.joint_index_finger])
        self.robot.last_pose[self.robot.joint_index_finger] = self.config["robot_params"]["ur5"]["rest_pose"][self.robot.joint_index_finger]
        
        target_position = -1.0 * self.robot.joint_gear_ratio_mimic * np.asarray(self.config["robot_params"]["ur5"]["rest_pose"][self.robot.joint_index_finger])
        for i, idx in enumerate(self.robot.joint_indices_finger_mimic):
            self.bc.resetJointState(self.robot.uid, idx, target_position[i])

        # Remove objects
        for obj_uid in self.objects_uid:
            self.bc.removeBody(obj_uid)

        # Reload objects
        self.objects_uid = [self.bc.loadURDF(
                fileName        = self.objects_urdf,
                basePosition    = self.env_params["objects"]["pos"],
                baseOrientation = self.bc.getQuaternionFromEuler(self.env_params["objects"]["orn"]),
                useFixedBase    = False)]


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
    def objects_urdf(self):
        return self._objects_urdf
    
    @objects_urdf.setter
    def objects_urdf(self, objects_urdf):
        self._objects_urdf = objects_urdf