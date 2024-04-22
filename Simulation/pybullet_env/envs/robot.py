import os
from typing import Tuple, Dict, List, Set, Union
import numpy as np
import numpy.typing as nptype

from abc import ABC, abstractmethod

import pybullet_data
from imm.pybullet_util.typing_extra import TranslationT, QuaternionT, Tuple3, EulerT

from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.common import (
    get_joint_limits, get_joint_positions,
    get_link_pose
)


class Robot(ABC):
    '''
    This is an abstract class that governs the 
    common properties between manipulation robots.

    NOTE(ssh): Pose arrays include the value for fixed joints. 
    However, the IK solver in PyBullet returns the array without fixed joints.
    Thus, you should be careful when controlling. 
    '''

    # Flags for arm control input format
    ARM_CONTROL_FORMAT_LAST_POSE = 0        # Format in Robot.last_pose
    ARM_CONTROL_FORMAT_IK_SOLUTION = 1      # Format in ik solution
    ARM_CONTROL_FORMAT_ARM_JOINT_VALUES = 2 # Format in values for joint_indices_arm

    
    def __init__(
            self, 
            bc: BulletClient, 
            uid: int,
            joint_index_last: int,
            joint_indices_arm: np.ndarray,
            link_index_endeffector_base: int,
            rest_pose: np.ndarray):
        
        # Pybullet properties
        self.bc: BulletClient = bc
        self.uid: int = uid

        # URDF properties
        self.joint_index_last: int                  = joint_index_last
        self.joint_indices_arm: np.ndarray          = joint_indices_arm
        self.link_index_endeffector_base: int       = link_index_endeffector_base
        self.joint_limits: np.ndarray               = get_joint_limits(self.bc, 
                                                                     self.uid,
                                                                     range(self.joint_index_last + 1))
        self.joint_range: np.ndarray                = self.joint_limits[1] - self.joint_limits[0]

        # Rest pose
        self.rest_pose: np.ndarray                  = rest_pose                  # include fixed joints
        # Last pose for control input
        self.last_pose: np.ndarray                  = np.copy(self.rest_pose)    # include fixed joints
        # Reset robot with init
        for i in self.joint_indices_arm:
            self.bc.resetJointState(self.uid, i, self.last_pose[i])

        # Reset all controls
        Robot.update_arm_control(self)



    def update_arm_control(self, values: nptype.NDArray = None,
                                 format = ARM_CONTROL_FORMAT_LAST_POSE):
        '''
        Update the control of the robot arm. Not the finger.
        If the parameters are not given, it will reinstate 
        the positional control from the last_pose buffer.

        Params:
        - values(optional): Position values for the joints.
        The format should either match with
            - `last_pose`
            - direct return value of pybullet ik solver.
            - contiguous values for Robot.joint_indices_arm.
        This function will ignore the finger control value, 
        but the full array should be given.
        - from_ik_solution(optional): Set this as true when putting ik solution directly.
        '''
        
        # If joint value is none, the robot stays the position
        if values is not None:
            if format == self.ARM_CONTROL_FORMAT_LAST_POSE:
                self.last_pose[self.joint_indices_arm] = values[self.joint_indices_arm]
            elif format == self.ARM_CONTROL_FORMAT_IK_SOLUTION:
                self.last_pose[self.joint_indices_arm] = values[:len(self.joint_indices_arm)]
            elif format == self.ARM_CONTROL_FORMAT_ARM_JOINT_VALUES:
                self.last_pose[self.joint_indices_arm] = values
            else:
                raise ValueError("Invalid flag")

        # Arm control
        position_gains = [1.0 for _ in self.joint_indices_arm]
        self.bc.setJointMotorControlArray(
            self.uid,
            self.joint_indices_arm,
            self.bc.POSITION_CONTROL,
            targetPositions = self.last_pose[self.joint_indices_arm],
            positionGains = position_gains)



    def get_arm_state(self) -> np.ndarray:
        '''
        Get the joint positions of arm. Not the `last_pose` buffer.
        Only support format in ARM_CONTROL_FORMAT_ARM_JOINT_VALUES
        '''
        positions = get_joint_positions(self.bc, self.uid, self.joint_indices_arm)
        
        return np.asarray(positions)



    def get_endeffector_pose(self) -> Tuple[TranslationT, QuaternionT]:
        '''
        Get the current pose of the end-effector
        '''

        pos, orn_q = get_link_pose(self.bc, self.uid, self.link_index_endeffector_base)
        
        return pos, orn_q





class UR5Suction(Robot):
    '''
    Wrapper class for UR5 with a suction gripper in PyBullet
    '''

    def __init__(self, bc: BulletClient,
                       config: dict):
        """
        Args:
            bc (BulletClient): PyBullet Client
            config (dict): Configuration file
        """
        # Path to URDF
        project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])

        # Init superclass
        robot_params = config["robot_params"]["ur5_suction"]
        super().__init__(
            bc = bc, 
            uid = bc.loadURDF(
                fileName        = os.path.join(urdf_path, robot_params["path"]),
                basePosition    = robot_params["pos"],
                baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                useFixedBase    = True),
            joint_index_last                    = robot_params["joint_index_last"],                     # Not used...?
            joint_indices_arm                   = np.array(robot_params["joint_indices_arm"]),          # For control
            link_index_endeffector_base         = robot_params["link_index_endeffector_base"],          # For robot uid. Not ee uid.
            rest_pose                           = np.array(robot_params["rest_pose"]))                  # asdf


        # Gripper
        gripper_params = robot_params["gripper"]
        self.link_indices_endeffector_tip = gripper_params["link_indices_endeffector_tip"]
        self.gripper_base_to_tip_stroke   = gripper_params["base_to_tip_stroke"]
        self.grasp_poke_backward          = gripper_params["grasp_poke_backward"]
        self.grasp_poke_criteria          = gripper_params["grasp_poke_criteria"]

        # Gripper flags
        self.contact_constraint = None
        self.activated = False

        # Erase the reactions at the tip of end-effector
        for idx in self.link_indices_endeffector_tip:
            self.bc.changeDynamics(self.uid, 
                                   idx,
                                   restitution = 0.0,
                                   contactStiffness = 0.0,
                                   contactDamping = 0.0)


    # From Ravens
    def activate(self, object_uids: List[int]):
        """Simulate suction using a rigid fixed constraint to contacted object.
        
        Args:
            object_uids: objects to manipulate in the environment.
        """

        # Activate when not activated yet
        if not self.activated:
            # Collision detection
            self.bc.performCollisionDetection()

            # Check contact at all links
            points = []
            for tip_idx in self.link_indices_endeffector_tip:
                pts = self.bc.getContactPoints(bodyA=self.uid, linkIndexA=tip_idx)
                points += pts
                # Grasp fail when some link has no contact.
                if len(pts)==0:
                    points = []
                    break

            # If contact exists, apply constraint.
            if len(points) != 0:
                # Check uniqueness
                is_unique = True
                target_uid_unique, target_contact_link_unique = points[0][2], points[0][4]
                for point in points:
                    if target_uid_unique != point[2] and target_contact_link_unique != point[2]:
                        is_unique = False

                # Apply constraint if unique.
                if target_uid_unique in object_uids and is_unique:
                    # Get relative transform
                    ee_body_pose = self.bc.getLinkState(self.uid, self.link_indices_endeffector_tip[0])
                    object_pose = self.bc.getBasePositionAndOrientation(target_uid_unique)

                    world_to_ee_body = self.bc.invertTransform(ee_body_pose[0], ee_body_pose[1])
                    object_to_ee_body = self.bc.multiplyTransforms(world_to_ee_body[0], world_to_ee_body[1],
                                                                    object_pose[0], object_pose[1])
                    # Apply constraint
                    self.contact_constraint = self.bc.createConstraint(
                        parentBodyUniqueId = self.uid,
                        parentLinkIndex    = self.link_indices_endeffector_tip[0],
                        childBodyUniqueId = target_uid_unique,
                        childLinkIndex = target_contact_link_unique,
                        jointType = self.bc.JOINT_FIXED,
                        jointAxis = (0, 0, 0),
                        parentFramePosition = object_to_ee_body[0],
                        parentFrameOrientation = object_to_ee_body[1],
                        childFramePosition = (0, 0, 0),
                        childFrameOrientation = (0, 0, 0))
        
        # Always mark as activated whether succeeded or not.
        self.activated = True
        
        # |FIXME(Jiyong)|: check it is possible to set (self.activated == True and self.contact_constraint is None)
        if self.contact_constraint is None:
            return None
        else:
            return self.bc.getConstraintInfo(self.contact_constraint)[2]


    # From Ravens
    def release(self):
        """Release gripper object, only applied if gripper is 'activated'.

        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.

        To handle deformables, simply remove constraints (i.e., anchors).
        Also reset any relevant variables, e.g., if releasing a rigid, we
        should reset init_grip values back to None, which will be re-assigned
        in any subsequent grasps.
        """
        if self.activated:
            self.activated = False
            # Release gripped rigit object (if any)
            if self.contact_constraint is not None:
                self.bc.removeConstraint(self.contact_constraint)
                self.contact_constraint = None


    # From Ravens
    def detect_contact(self) -> bool:
        """Detects a full contact with a grasped rigid object."""
        

        # Detect contact at grasped object if constraint exist.
        if self.activated and self.contact_constraint is not None:
            info = self.bc.getConstraintInfo(self.contact_constraint)
            uid, link = info[2], info[3]
            # Get all contact points between the suction and a rigid body.
            points = self.bc.getContactPoints(bodyA=uid, linkIndexA=link)   # Detected from +0.0005
        # Detect at cup otherwise.
        else:
            points = []
            # Suction gripper tip
            uid = self.uid
            for link in self.link_indices_endeffector_tip:
                pts = self.bc.getContactPoints(bodyA=uid, linkIndexA=link)
                points += pts
                # Grasp fail when some link has no contact.
                if len(pts)==0:
                    points = []
                    break

        if self.activated:
            points = [point for point in points if point[2] != self.uid]

        # We know if len(points) > 0, contact is made with SOME rigid item.
        if points:
            return True
        
        return False


    # From Ravens
    def check_grasp(self) -> int:
        """
        Check a grasp object in contact?" for picking success
        
        Returns:
            None (if not grasp) / object uid (if grasp)
        """
        suctioned_object = None
        if self.contact_constraint is not None:
            suctioned_object = self.bc.getConstraintInfo(self.contact_constraint)[2]
        return suctioned_object

    
    # For SE3 action space
    def get_target_ee_pose_from_se3(self, pos: TranslationT, orn: EulerT) -> Tuple[TranslationT, EulerT]:
        """This function returns the SE3 target ee pose(inward) given SE3 surface pose(outward).

        Args:
            pos (TranslationT): Sampled end effector tip pose (outward, surface normal)
            orn (EulerT):  Sampled end effector orn (outward, surface normal)

        Returns:
            Tuple[TranslationT, EulerT]: End effector base pose (inward)
        """
        # Get the tip-end coordinate in world frame. This flips the contact point normal inward.
        tip_end_target_pos_in_world, tip_end_target_orn_q_in_world \
            = self.bc.multiplyTransforms(
                pos, self.bc.getQuaternionFromEuler(orn),
                [0.0, 0.0, 0.0], self.bc.getQuaternionFromEuler([3.1416, 0.0, 0.0]))

        # Target in world frame -> tip-end frame -> ur5 ee base link frame
        ee_base_link_pos_in_tip_end_frame = [0.0, 0.0, -self.gripper_base_to_tip_stroke]   # NOTE(ssh): Okay... 12cm is appropriate
        ee_base_link_orn_q_in_tip_end_frame = self.bc.getQuaternionFromEuler([0.0, 0.0, 0.0])
        ee_base_link_target_pos_in_world, ee_base_link_target_orn_q_in_world \
            = self.bc.multiplyTransforms(tip_end_target_pos_in_world, tip_end_target_orn_q_in_world,
                                    ee_base_link_pos_in_tip_end_frame, ee_base_link_orn_q_in_tip_end_frame)

        return ee_base_link_target_pos_in_world, self.bc.getEulerFromQuaternion(ee_base_link_target_orn_q_in_world)


    # For primitive objects (SE2 action space)
    def get_target_ee_pose_from_se2(self, pos: TranslationT, yaw_rad: float) -> Tuple[TranslationT, EulerT]:
        """This function returns the SE3 version of target ee pose given SE2 pose.
        NOTE(ssh): This function helps preserving the roll and pitch consistency of target object!!!!!

        Args:
            pos (TranslationT): Sampled end effector pos from SE2 action space
            yaw_rad (float): Sampled end effector yaw from SE2 action space in radians (outward, surface normals)

        Returns:
            target_ee_base_link_pos (TranslationT): Target end effector position in SE3
            target_ee_base_link_orn (Tuple3): Target end effector euler orientation in SE3 
                (inward, ee_base, preserves the roll and pitch of target objects)
        """

        # NOTE(ssh): To preserve roll and pitch, I always apply constraint that gripper has x-up orientation.

        # Reference orientation in SE2
        frame_rot1 = [0.0, 1.5708, 0.0]
        frame_rot2 = [0.0, 0.0, 3.1416]
        _, se2_reference_ee_frame_orn_q = self.bc.multiplyTransforms([0, 0, 0], self.bc.getQuaternionFromEuler(frame_rot1),
                                                                     [0, 0, 0], self.bc.getQuaternionFromEuler(frame_rot2))
        yaw_rot = [0.0, 0.0, yaw_rad]

        # Target orientation is acquired by rotating the reference frame
        _, target_ee_base_link_orn_q = self.bc.multiplyTransforms([0, 0, 0], self.bc.getQuaternionFromEuler(yaw_rot),
                                                                  [0, 0, 0], se2_reference_ee_frame_orn_q)
                                                    
        
        # Target position is acquire by pushing back by the stroke of end effector in target frame.
        translation_in_target_frame = [0, 0, -self.gripper_base_to_tip_stroke]
        target_ee_base_link_pos, _ = self.bc.multiplyTransforms(pos, target_ee_base_link_orn_q,
                                                                      translation_in_target_frame, [0, 0, 0, 1])
        target_ee_base_link_orn = self.bc.getEulerFromQuaternion(target_ee_base_link_orn_q)



        return target_ee_base_link_pos, target_ee_base_link_orn


    # Reserved
    def get_overriden_pose_wrist_joint_zero_position(self, ee_base_link_target_pos: TranslationT, 
                                                           ee_base_link_target_orn_q: QuaternionT) -> Tuple[TranslationT, QuaternionT]:
        """Force reset wrist joint to 0 and recalculate the forward kinematics of UR5 EE Link
        
        Args:
            ee_base_link_target_pos (TranslationT): End effector position to override in world frame.
            ee_base_link_target_orn_q (QuaternionT): End effector orientation to override in world frame.

        Returns:
            ee_base_link_target_pos (TranslationT): Overriden end effector position in world frame.
            ee_base_link_target_orn_q (QuaternionT): Overriden end effector orientation in world frame.
        """

        # 1. Solve IK
        joint_position_list = np.array( self.bc.calculateInverseKinematics(self.uid, 
                                                                           self.link_index_endeffector_base, 
                                                                           ee_base_link_target_pos, 
                                                                           ee_base_link_target_orn_q, 
                                                                           maxNumIterations = 1000, 
                                                                           residualThreshold = 1e-6) )

        # 2. Overriding the ee base joint
        # Override
        joint_position_list[-1] = 0.0
        # Forward kinematics
        backup_joint_state_list = self.bc.getJointStates(self.uid, self.joint_indices_arm)
        for list_i, joint_idx in enumerate(self.joint_indices_arm):
            self.bc.resetJointState(self.uid, 
                                    joint_idx, 
                                    targetValue = joint_position_list[list_i])
        ur5_ee_link_info = self.bc.getLinkState(self.uid, self.joint_index_last)
        ee_base_link_target_pos = ur5_ee_link_info[4]
        ee_base_link_target_orn_q = ur5_ee_link_info[5]

        # 3. Restore
        for list_i, joint_idx in enumerate(self.joint_indices_arm):
            target_position = backup_joint_state_list[list_i][0]
            target_velocity = backup_joint_state_list[list_i][1]
            self.bc.resetJointState(self.uid,
                                    joint_idx,
                                    targetValue = target_position,
                                    targetVelocity = target_velocity)


        return ee_base_link_target_pos, ee_base_link_target_orn_q




class UR5(Robot):
    '''
    Wrapper class for UR5 robot instance in PyBullet.
    '''

    def __init__(
            self, 
            bc: BulletClient, 
            config: dict):

        # Path to URDF
        project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../")
        urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])

        # Init superclass
        robot_params = config["robot_params"]["ur5"]
        super().__init__(
            bc = bc,            # NOTE(ssh): `bc` will be class attribute in super.__init__().
            uid = bc.loadURDF(
                fileName        = os.path.join(urdf_path, robot_params["path"]),
                basePosition    = robot_params["pos"],
                baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                useFixedBase    = True),
            joint_index_last                    = robot_params["joint_index_last"],
            joint_indices_arm                   = np.array(robot_params["joint_indices_arm"]),
            link_index_endeffector_base         = robot_params["link_index_endeffector_base"],
            rest_pose                           = np.array(robot_params["rest_pose"]))


        # Robot dependent params
        gripper_params = robot_params["gripper"]
        self.joint_index_finger                  = gripper_params["joint_index_finger"],
        self.joint_value_finger_open             = gripper_params["joint_value_finger_open"],
        self.distance_finger_to_endeffector_base = gripper_params["distance_finger_to_endeffector_base"],
        self.joint_indices_finger_mimic          = np.array(gripper_params["joint_indices_finger_mimic"])
        self.joint_gear_ratio_mimic              = np.array(gripper_params["joint_gear_ratio_mimic"])


        # Reset all controls
        self.update_arm_control()
        self.update_finger_control()


        # Update dynamics
        self.bc.changeDynamics(
            self.uid, 
            self.joint_index_last-1, 
            lateralFriction=15.0,
            rollingFriction=0.01,
            spinningFriction=0.2,
            restitution=0.5)
        self.bc.changeDynamics(
            self.uid, 
            self.joint_index_last-4, 
            lateralFriction=15.0,
            rollingFriction=0.01,
            spinningFriction=0.2,
            restitution=0.5)



    def __create_mimic_constraints(self):
        '''
        Create mimic constraints for the grippers with closed-loop joints
        '''
        # Finger contsraint chain:
        #   right: 11->12->13 
        #   left: 11->8->9->10
        parent_joint_info = self.bc.getJointInfo(self.uid, self.joint_index_finger)
        parent_joint_id = parent_joint_info[0]
        parent_joint_frame_pos = parent_joint_info[14]

        for list_i, joint_i in enumerate(self.joint_indices_finger_mimic):

            child_joint_info = self.bc.getJointInfo(self.uid, joint_i)
            child_joint_id = child_joint_info[0]
            child_joint_axis = child_joint_info[13]
            child_joint_frame_pos = child_joint_info[14]
            constraint_id = self.bc.createConstraint(self.uid, parent_joint_id,
                                                self.uid, child_joint_id,
                                                jointType = self.bc.JOINT_GEAR,
                                                jointAxis = child_joint_axis,
                                                parentFramePosition = parent_joint_frame_pos,
                                                childFramePosition = child_joint_frame_pos)
            
            gear_ratio = self.joint_gear_ratio_mimic[list_i]
            self.bc.changeConstraint(constraint_id, gearRatio=gear_ratio, maxForce=10000, erp=1.0)



    def update_finger_control(self, finger_value=None):
        '''
        Set joint motor control value to last_pose, and refresh the mimic joints.
        '''
        # Default controlling method
        # If joint value is none, the robot stays the position
        if finger_value is not None:
            # Update last pose buffer
            self.last_pose[self.joint_index_finger] = finger_value
        
        # Finger control
        self.bc.setJointMotorControl2(
            self.uid,
            self.joint_index_finger,
            self.bc.POSITION_CONTROL,
            targetPosition = self.last_pose[self.joint_index_finger])    

        # Mimic finger controls
        finger_state = self.bc.getJointState(
            self.uid,
            self.joint_index_finger)
        target_position = -1.0 * self.joint_gear_ratio_mimic * np.asarray(finger_state[0])
        target_velocity = -1.0 * self.joint_gear_ratio_mimic * np.asarray(finger_state[1])
        self.bc.setJointMotorControlArray(
            self.uid,
            self.joint_indices_finger_mimic,
            self.bc.POSITION_CONTROL,
            targetPositions = target_position,
            targetVelocities = target_velocity,
            positionGains = np.full_like(self.joint_indices_finger_mimic, 1.2, dtype=np.float32),
            forces = np.full_like(self.joint_indices_finger_mimic, 50, dtype=np.float32))

        # Propagate the finger control value to all fingers
        self.last_pose[self.joint_indices_finger_mimic] = np.asarray(finger_state[0])



    def get_finger_state(self) -> float:
        '''
        Get the finger position. Not the `last_pose` buffer.
        '''
        finger_state = self.bc.getJointState(
            self.uid,
            self.joint_index_finger)[0]

        return finger_state