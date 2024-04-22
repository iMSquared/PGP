import numpy as np
from typing import List

from imm.pybullet_util.typing_extra import TranslationT, QuaternionT
from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.common import get_joint_limits


class UR5RG2(object):
    def __init__(self, bc: BulletClient, pos: TranslationT, orn: QuaternionT):
        self.bc = bc
        self.robot_id: int = bc.loadURDF('./urdf/ur5_rg2.urdf', pos, orn,
                                         useFixedBase=True, flags=bc.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
        # Camera setup (based on the Intel D435 hardware datasheet)
        self.projectionMatrix = bc.computeProjectionMatrixFOV(fov=58, aspect=1.5, nearVal=0.28, farVal=5)
        self.image_renderer = bc.ER_BULLET_HARDWARE_OPENGL

        # Disable closed linkage collision (gripper's inner fingers)
        bc.setCollisionFilterPair(self.robot_id, self.robot_id, 9, 12, 0)
        bc.setCollisionFilterPair(self.robot_id, self.robot_id, 9, 15, 0)

        self.ee_id: int = 7  # NOTE(choi): End-effector joint ID for UR5 robot
        self.arm_joint_ids: List[int] = [1, 2, 3, 4, 5, 6]  # NOTE(choi): Hardcoded arm joint id for UR5 robot
        self.arm_joint_limits = get_joint_limits(bc, self.robot_id, self.arm_joint_ids)

        self.arm_rest_poses: List[float] = [0, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                                            -1.5707970583733368, 0]  # NOTE(choi): Manually set arm rest poses

        self.gripper_z_offset: float = 0.221
        self.gripper_link_ids: List[int] = [10, 11, 12, 13, 14, 15]
        self.gripper_link_sign: List[float] = [-1, -1, 1, 1, 1, -1]
        self.gripper_link_limit: List[float] = [0, 1.343904]
        self.gripper_x_line = None
        self.gripper_y_line = None
        self.gripper_z_line = None

        for i in range(len(self.gripper_link_ids)):
            if i != 0:
                c = bc.createConstraint(self.robot_id, self.gripper_link_ids[0],
                                        self.robot_id, self.gripper_link_ids[i],
                                        jointType=bc.JOINT_GEAR,
                                        jointAxis=[0, 1, 0],
                                        parentFramePosition=[0, 0, 0],
                                        childFramePosition=[0, 0, 0])
                gearRatio = -self.gripper_link_sign[0] * self.gripper_link_sign[i]
                bc.changeConstraint(c, gearRatio=gearRatio, maxForce=3, erp=1)

            gripper_link_limit = sorted([limit * self.gripper_link_sign[i] for limit in self.gripper_link_limit])
            print(gripper_link_limit)
            bc.changeDynamics(self.robot_id, self.gripper_link_ids[i],
                              jointLowerLimit=gripper_link_limit[0],
                              jointUpperLimit=gripper_link_limit[1])

            # Initially set the gripper as opened
            bc.resetJointState(self.robot_id, self.gripper_link_ids[i],
                               self.gripper_link_limit[1] * self.gripper_link_sign[i])

    def reset_arm_poses(self):
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_joint_ids):
            self.bc.resetJointState(self.robot_id, joint_id, rest_pose)

    def open_gripper(self):
        self.bc.setJointMotorControlArray(self.robot_id, self.gripper_link_ids, self.bc.POSITION_CONTROL,
                                          targetPositions=[i * self.gripper_link_limit[1] for i in
                                                           self.gripper_link_sign],
                                          positionGains=[1 for i in range(len(self.gripper_link_ids))])

    def close_gripper(self):
        # Releasing the position control, leaving a small friction
        self.bc.setJointMotorControlArray(self.robot_id, self.gripper_link_ids, self.bc.VELOCITY_CONTROL,
                                          forces=[0.1 for i in range(len(self.gripper_link_ids))])
        # Force/torque control
        self.bc.setJointMotorControl2(self.robot_id, self.gripper_link_ids[0], self.bc.TORQUE_CONTROL,
                                      force=-3 * self.gripper_link_sign[0])

    def draw_gripper_frame(self):
        scale: float = 0.1
        ee_state = np.array(self.bc.getLinkState(self.robot_id, 6))
        ee_rotation_matrix = np.array(self.bc.getMatrixFromQuaternion(ee_state[1])).reshape((3, 3))
        gripper_state = ee_state[0] + ee_rotation_matrix[:, 2] * self.gripper_z_offset
        self.gripper_x_line = self.bc.addUserDebugLine(lineFromXYZ=gripper_state,
                                                       lineToXYZ=gripper_state + ee_rotation_matrix[:, 0] * scale,
                                                       lineColorRGB=[1, 0, 0])
        self.gripper_y_line = self.bc.addUserDebugLine(lineFromXYZ=gripper_state,
                                                       lineToXYZ=gripper_state + ee_rotation_matrix[:, 1] * scale,
                                                       lineColorRGB=[0, 1, 0])
        self.gripper_z_line = self.bc.addUserDebugLine(lineFromXYZ=gripper_state,
                                                       lineToXYZ=gripper_state + ee_rotation_matrix[:, 2] * scale,
                                                       lineColorRGB=[0, 0, 1])

    def delete_gripper_frame(self):
        if self.gripper_x_line is not None:
            self.bc.removeUserDebugItem(self.gripper_x_line)
        if self.gripper_y_line is not None:
            self.bc.removeUserDebugItem(self.gripper_y_line)
        if self.gripper_z_line is not None:
            self.bc.removeUserDebugItem(self.gripper_z_line)

    def get_camera_from_world_frame(self):
        # Get on-module camera (Intel Realsense D435) frame expressed in world frame
        camera_joint_id = 26
        joint_info = self.bc.getJointInfo(self.robot_id, camera_joint_id)
        joint_pose_parent = joint_info[-3]
        joint_ori_parent = joint_info[-2]
        parent_link = joint_info[-1]
        link_info = self.bc.getLinkState(self.robot_id, parent_link)
        link_pose_world = link_info[0]
        link_ori_world = link_info[1]
        return [joint_pose_parent, joint_ori_parent, link_pose_world, link_ori_world]

    def get_image(self):
        _, _, pos, ori = self.get_camera_from_world_frame()

        rot_matrix = self.bc.getMatrixFromQuaternion(ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Initial vectors
        init_camera_vector = (1, 0, 0)  # z-axis
        init_up_vector = (0, 0, 1)  # y-axis

        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)

        view_matrix_gripper = self.bc.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
        img = self.bc.getCameraImage(640, 360, view_matrix_gripper, self.projectionMatrix, shadow=0,
                                     renderer=self.image_renderer,
                                     flags=self.bc.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)

        return img