import math
import time
import copy
import numpy as np
import numpy.typing as npt

from contextlib import contextmanager
from typing import List, Tuple, Union

from imm.pybullet_util.typing_extra import TranslationT, QuaternionT
from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.common import (
    get_joint_positions,
    get_link_pose,
    get_relative_transform)
from imm.pybullet_util.collision import (
    ContactBasedCollision,
    GraspAffordance,
    LinkPair)
from imm.motion_planners.rrt_connect import birrt
from scipy.spatial.transform import Rotation as R

from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import UR5Suction


# For debugging
from scipy.spatial.transform import Rotation
from debug.debug import debug_data


@contextmanager
def imagine(bc: BulletClient) -> int:
    """Temporarily change the simulation, but restore the state afterwards.
    NOTE(ycho): Do not abuse this function! saveState() is quite expensive.
    """
    try:
        state_id = bc.saveState()
        yield state_id
    finally:
        bc.restoreState(state_id)
        bc.removeState(state_id)



def interpolate_trajectory(cur: List, 
                           goal: List, 
                           action_duration: float, 
                           control_dt: float) -> Tuple[npt.NDArray, ...]:
    '''
    This function returns linear-interpolated (dividing straight line)
    trajectory between current and goal pose.
    Acc, Jerk is not considered.
    '''
    # Interpolation steps
    steps = math.ceil(action_duration/control_dt)
    
    # Calculate difference
    delta = [ goal[i] - cur[i] for i in range(len(cur)) ]
    
    # Linear interpolation
    trajectory: Tuple[npt.NDArray, ...] = ([
        np.array([
            cur[j] + ( delta[j] * float(i)/float(steps) ) 
            for j in range(len(cur))
        ])
        for i in range(1, steps+1)
    ])

    return trajectory



def quaternion_distance(orn_q1, orn_q2):
    return 1 - np.dot(orn_q1, orn_q2)



class Manipulation():

    def __init__(self, bc    : BulletClient,
                       env   : BinpickEnvPrimitive, 
                       robot : UR5Suction,
                       config: dict): 
        
        self.config = config
        self.bc     = bc
        self.env    = env
        self.robot  = robot
        
        # Sim params
        sim_params = config["sim_params"]
        self.DEBUG_GET_DATA = self.config['project_params']['debug']['get_data']
        self.delay          = sim_params["delay"]
        self.control_dt     = 1. / sim_params["control_hz"]
        # Manipulation params
        manipulation_params = config["manipulation_params"]
        # For numerical IK solver
        self.ik_max_num_iterations = manipulation_params["inverse_kinematics"]["max_num_iterations"]
        self.ik_residual_threshold = manipulation_params["inverse_kinematics"]["residual_threshold"]
        self.rrt_trials            = manipulation_params["rrt_trials"]


        # Default allow pairs (allow pair is not commutative!!!!)
        self.allow_pair_ground = LinkPair(      # Robot and ground plane
            body_id_a = robot.uid,
            link_id_a = None,
            body_id_b = env.plane_uid,
            link_id_b = None)
        self.allow_pair_self = LinkPair(        # Robot self collision
            body_id_a = robot.uid,
            link_id_a = None,
            body_id_b = robot.uid,
            link_id_b = None)
        # Default RRT callbacks
        def distance_fn(q0: np.ndarray, q1: np.ndarray):
            return np.linalg.norm(np.subtract(q1, q0))
        def sample_fn():
            return np.random.uniform(self.robot.joint_limits[0]/2.0, self.robot.joint_limits[1]/2.0)
        def extend_fn(q0: np.ndarray, q1: np.ndarray, num_ext=2):
            dq = np.subtract(q1, q0)  # Nx6
            return q0 + np.linspace(0, 1, num=100)[:, None] * dq
        self.distance_fn = distance_fn
        self.sample_fn = sample_fn
        self.extend_fn = extend_fn
        self.default_collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = list(range(self.robot.joint_index_last+1)),
            # allowlist    = [self.allow_pair_ground, self.allow_pair_self],
            allowlist    = [self.allow_pair_ground],
            attachlist   = [],
            joint_limits = self.robot.joint_limits,
            tol          = {})
        # self.grasp_affordance_fn = GraspAffordance(
        #     bc        = self.bc,
        #     robot     = self.robot,
        #     allowlist = [self.allow_pair_ground, self.allow_pair_self],
        #     config    = self.config)



    def __move(self, traj):

        # Execute the trajectory
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('The collision free trajectory is founded')
        #     print('Start to approch to target pose')

        # DEBUG
        if self.DEBUG_GET_DATA:
            end_effector_pose_prv = get_link_pose(
                bc       = self.bc, 
                body_id  = self.robot.uid, 
                link_id  = self.robot.link_index_endeffector_base, 
                inertial = False)

        # Execute trajectory
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('Start arm control')
        #     check_time_arm = time.time()
            
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)
            
            # # DEBUG
            # if self.DEBUG_GET_DATA:
            #     end_effector_pose_cur = get_link_pose(
            #         bc       = self.bc, 
            #         body_id  = self.robot.uid, 
            #         link_id  = self.robot.link_index_endeffector_base, 
            #         inertial = False)
            #     self.bc.addUserDebugLine(
            #         end_effector_pose_prv[0],
            #         end_effector_pose_cur[0],
            #         lineColorRGB=(0.5,0.5,0.5),
            #         lineWidth=4,
            #         lifeTime=30.0)
            #     end_effector_pose_prv = end_effector_pose_cur
        
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('End arm control')
        #     check_time_arm = time.time() - check_time_arm
        #     debug_data['arm_control'].append(check_time_arm)

        # Wait until control completes
        self.wait(steps=240)
        
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('End to approch to target pose')


    def open(self, pos: TranslationT, 
                   orn_q: QuaternionT):
        """_summary_

        Args:
            pos (TranslationT): _description_
            orn_q (QuaternionT): _description_
        """
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('Start to open the fingers')
            
        self.robot.release()

        # Pull ee back        
        modified_pos_back, modified_orn_q_back, modified_pos, modified_orn_q = self.get_ee_pose_from_target_pose(pos, orn_q)
        joint_pos_src, joint_pos_dst = self.solve_ik_numerical(modified_pos_back, modified_orn_q_back)

        traj = interpolate_trajectory(list(joint_pos_src), list(joint_pos_dst), 0.2, self.control_dt)
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)

    
    def close(self, obj_uid: int, 
                    pos: TranslationT, 
                    orn_q: QuaternionT) -> Union[int, None]:
        """
        Args:
            obj_uid: uid of target object
            pos: position when grasping occurs
            orn_q: orientation when grasping occurs

        Returns:
            Union[int, None]: Returns obj_uid when grasp succeeded. None when failed.
        """
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('Start to close the fingers')
        
        # Approach for poke
        modified_pos_back, modified_orn_q_back, \
            modified_pos, modified_orn_q \
            = self.get_ee_pose_from_target_pose(pos, orn_q)
        joint_pos_src, joint_pos_dst = self.solve_ik_numerical(modified_pos, modified_orn_q)

        traj = interpolate_trajectory(list(joint_pos_src), list(joint_pos_dst), 0.2, self.control_dt)
        for joint_pos in traj:
            # Control
            self.robot.update_arm_control(joint_pos)
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)
                
        # Wait until control completes
        self.wait(steps=240)
                
        # Check contact after approaching
        if self.robot.detect_contact():
            grasp_obj_uid = self.robot.activate(self.env.object_uids)
        else:
            grasp_obj_uid = None
        
        # print(f"(In manip) endeffector contact?: {self.robot.detect_contact()}")
        # print(f"(In manip) endeffector grasped {grasp_obj_uid}")
        
        # return self.robot.check_grasp()   #|FIXME(Jiyong)|: Sometimes, check_grasp() is None, even if detect_contact() is True.
        return grasp_obj_uid
    

    def pick(self, obj_uid: int, 
                   pos: TranslationT, 
                   orn_q: QuaternionT, 
                   traj = None) -> Union[int, None]:
        """
        Implementation of the pick action
        
        Args:
            obj_uid: Target object id
            pos: Postion when PICK occurs
            orn_q: Orientation when PICK occurs
            traj: Collision-free trajectory for pose to reach before picking (backward as much as self.robot.poke_backward)

        Returns:
            holding_obj_uid (Union(int, None)): Return the uid of grasped object.
        """
        
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('pick() start')
        
        # Perform move
        self.__move(traj)
        
        # Close the gripper
        holding_obj_uid = self.close(obj_uid, pos, orn_q)

        if holding_obj_uid is None:
            # NOTE(ssh): Release again if grasp failed.
            self.open(pos, orn_q)
            return None
        else:
            return holding_obj_uid


    def place(self, obj_uid: int, 
                    pos: TranslationT, 
                    orn_q: QuaternionT, 
                    traj: List[npt.NDArray] = None) -> Union[int, None]:
        """
        Implementation of the place action

        Args:
            obj_uid (int): Target object uid
            pos (TranslationT): End effector position
            orn_q (QuaternionT): End effector orientation
            traj (List[npt.NDArray]): Collision-free trajectory for PLACE pose
        """
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('place() start')
            
        # Perform move
        self.__move(traj)

        # Open the gripper
        self.open(pos, orn_q)
        holding_obj_uid = None  # No object is being holden.

        # Wait until the control completes
        self.wait(steps=240)

        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('place() end - success')
        
        return holding_obj_uid
    
    
    def wait(self, steps=-1):

        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('wait() start')
        #     check_time = time.time()
            
        while steps != 0: 
            self.robot.update_arm_control()
            self.bc.stepSimulation()
            
            if self.delay:
                time.sleep(self.delay)
                
            steps -= 1
        
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('wait() end')
        #     check_time = time.time() - check_time
        #     debug_data['wait()'].append(check_time)


    def solve_ik_numerical(self, pos: TranslationT, 
                                 orn_q: QuaternionT) -> Tuple[ npt.NDArray, npt.NDArray ]:
        '''
        Solve the inverse kinematics of the robot given the finger center position.
        
        Args:
        - pos (TranslationT) : R^3 position
        - orn_q (QuaternionT): Quaternion
        
        Returns
        - joint_pos_src (npt.NDArray): Source position in buffer format
        - joint_pos_dst (npt.NDArray): Destination position in buffer format

        NOTE(ssh):
            Finger joint value will be preserved in destination position.
            The indices of the value match with `rest_pose`
            Return type is numpy because it requires advanced indexing. 
            Make sure to cast for other hashing.
        '''

        # NOTE: For patterning reason, calling private function inside another private function is highly not recommended...
        
        # Solve goal ik
        if orn_q is not None:            
            ik = self.bc.calculateInverseKinematics(
                bodyUniqueId         = self.robot.uid, 
                endEffectorLinkIndex = self.robot.link_index_endeffector_base, 
                targetPosition       = pos, 
                targetOrientation    = orn_q,
                lowerLimits          = self.robot.joint_limits[0],     # upper
                upperLimits          = self.robot.joint_limits[1],     # lower
                jointRanges          = self.robot.joint_range,
                restPoses            = self.robot.rest_pose[self.robot.joint_indices_arm],
                maxNumIterations     = self.ik_max_num_iterations,
                residualThreshold    = self.ik_residual_threshold)
        # else:
        #     ik = self.bc.calculateInverseKinematics(
        #         bodyUniqueId         = self.robot.uid, 
        #         endEffectorLinkIndex = self.robot.link_index_endeffector_base, 
        #         targetPosition       = pos,
        #         lowerLimits          = self.robot.joint_limits[0],     # upper
        #         upperLimits          = self.robot.joint_limits[1],     # lower
        #         jointRanges          = self.robot.joint_range,
        #         restPoses            = self.robot.rest_pose[self.robot.joint_indices_arm],
        #         maxNumIterations     = self.ik_max_num_iterations,
        #         residualThreshold    = self.ik_residual_threshold)

        # Set source and destination position of motion planner (entire joint)
        # NOTE(ssh): This is awkward... but necessary as ik index ignores fixed joints.
        joint_pos_src = np.array(get_joint_positions( self.bc, 
                                                      self.robot.uid,
                                                      range(0, self.robot.joint_index_last + 1) ))
        joint_pos_dst = np.copy(joint_pos_src)
        joint_pos_dst[self.robot.joint_indices_arm] = np.array(ik)[ : len(self.robot.joint_indices_arm)]

        return joint_pos_src, joint_pos_dst
    

    def _define_grasp_collision_fn(self, grasp_uid, debug=False):
        '''
        This function returns the collision function that allows the grasped object to
        1. collide with the robot body
        2. be attached at robot finger
        3. touch with the cabinet.
        '''
        # Attached object will be moved together when searching the path
        attach_pair = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=self.robot.joint_index_last-1,
            body_id_b=grasp_uid,
            link_id_b=-1)
        # Allow pair is not commutative
        allow_pair_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=grasp_uid,
            link_id_b=None)
        allow_pair_b = LinkPair(
            body_id_a=grasp_uid,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        # Touch pair is not commutative. One should define bidirectional pair.
        # NOTE(ssh): allow touch between the object and cabinet, but not the penetration.
        touch_pair_objcab_a = LinkPair(
            body_id_a=self.env.cabinet_uid, 
            link_id_a=None,
            body_id_b=grasp_uid,
            link_id_b=None)
        touch_pair_objcab_b = LinkPair(
            body_id_a=grasp_uid,
            link_id_a=None,
            body_id_b=self.env.cabinet_uid,
            link_id_b=None)
        touch_pair_objgoal_a = LinkPair(
            body_id_a=self.env.goal_table,
            link_id_a=None,
            body_id_b=grasp_uid,
            link_id_b=None)
        touch_pair_objgoal_b = LinkPair(
            body_id_a=grasp_uid,
            link_id_a=None,
            body_id_b=self.env.goal_table,
            link_id_b=None)
        # NOTE(ssh): allow touch between the robot and cabinet, but not the penetration.
        touch_pair_robot_a = LinkPair(
            body_id_a=self.robot.uid,
            link_id_a=None,
            body_id_b=self.env.cabinet_uid,
            link_id_b=None)
        touch_pair_robot_b = LinkPair(
            body_id_a=self.env.cabinet_uid,
            link_id_a=None,
            body_id_b=self.robot.uid,
            link_id_b=None)
        # # |NOTE(Jiyong)|: allow touch between the held object and the other objects
        # touch_pair_holding_list = []
        # for obj_uid in self.env.object_uids:
        #     if obj_uid != grasp_uid:
        #         touch_pair_holding_list.append(LinkPair(
        #             body_id_a=grasp_uid,
        #             link_id_a=None,
        #             body_id_b=obj_uid,
        #             link_id_b=None))
        #         touch_pair_holding_list.append(LinkPair(
        #             body_id_a=obj_uid,
        #             link_id_a=None,
        #             body_id_b=grasp_uid,
        #             link_id_b=None))
        # Collision function should be redefined at every plan
        collision_fn = ContactBasedCollision(
            bc           = self.bc,
            robot_id     = self.robot.uid,
            joint_ids    = list(range(self.robot.joint_index_last+1)),
            allowlist    = [self.allow_pair_ground, self.allow_pair_self, allow_pair_a, allow_pair_b],
            attachlist   = [attach_pair],
            touchlist    = [touch_pair_objcab_a, touch_pair_objcab_b, 
                            touch_pair_objgoal_a, touch_pair_objgoal_b,
                            touch_pair_robot_a,touch_pair_robot_b], # + touch_pair_holding_list,
            joint_limits = self.robot.joint_limits,
            tol          = {},
            touch_tol    = -0.002)   #TODO: config

        # Debug
        if debug:
            contact = np.array(self.bc.getContactPoints(bodyA=grasp_uid, bodyB=self.env.cabinet_uid), dtype=object)
            for c in contact:
                print(f"Cabinet contact point: {c[8]}")
            print(f"min: {np.min(contact[:,8])}")
            
            for uid in self.env.object_uids:
                contact2 = np.array(self.bc.getContactPoints(bodyA=grasp_uid, bodyB=uid), dtype=object)
                for c in contact2:
                    print(f"Object {uid} contact: {c[8]}")

        return collision_fn
    
    
    def get_ee_pose_from_target_pose(self, pos: TranslationT, orn_q: QuaternionT):
        """
        Adjust the target pose to the end-effector pose.
        It reflects adjustment as much as poke backward.

        Args:
            pos (TranslationT): Target position
            orn (QuaternionT): Target orientation, outward orientation (surface normal)
        
        Returns:
            modified_pos_back (TranslationT): Position to reach before picking (backward as much as self.robot.poke_backward, only for PICK)
            modified_orn_q_back (QuaternionT): Orientation to reach before picking (only for PICK)
            modified_pos (TranslationT)
            modified_orn_q (QuaternionT): Inward orientation (ee base)
        """
        
        orn = self.bc.getEulerFromQuaternion(orn_q)
        ee_base_link_target_pos, ee_base_link_target_orn_e = self.robot.get_target_ee_pose_from_se3(pos, orn)
        ee_base_link_target_orn_q = self.bc.getQuaternionFromEuler(ee_base_link_target_orn_e)

        modified_pos = ee_base_link_target_pos
        modified_orn_q = ee_base_link_target_orn_q
        
        poke_backward = self.robot.grasp_poke_backward
        ee_base_link_backward_pos, ee_base_link_backward_orn_q \
            = self.bc.multiplyTransforms(ee_base_link_target_pos, ee_base_link_target_orn_q,
                                            [0, 0, -poke_backward], [0, 0, 0, 1])
        
        modified_pos_back = ee_base_link_backward_pos
        modified_orn_q_back = ee_base_link_backward_orn_q
        
        return modified_pos_back, modified_orn_q_back, modified_pos, modified_orn_q
    

    def motion_plan(self, joint_pos_src: npt.NDArray, 
                          joint_pos_dst: npt.NDArray, 
                          holding_obj_uid: Union[int, None], 
                          use_interpolation: bool = False) -> Union[List[npt.NDArray], None]:
        """Get a motion plan trajectory.

        Args:
            joint_pos_src (npt.NDArray): Source joint position
            joint_pos_dst (npt.NDArray): Destination joint position
            holding_obj_uid (Union[int, None]): Uid of the holding object.
            use_interpolation (bool, Optional): Flag for force return linear interpolated trajectory. Defaults to False.

        Returns:
            Union[List[npt.NDArray], None]: Generated trajectory. None when no feasible trajectory is found.
        """

        # Compute motion plan
        if use_interpolation:
            trajectory = interpolate_trajectory(
                cur             = joint_pos_src, 
                goal            = joint_pos_dst, 
                action_duration = 0.5, 
                control_dt      = self.control_dt)
        else:
            # Try RRT
            # # ======================== Debug =========================
            # if self.DEBUG_GET_DATA:
            #     print('Start RRT')
            #     check_time_rrt = time.time()
            
            # Deactivate before planning?
            # self.robot.release()

            with imagine(self.bc):
                # Moving with grapsed object
                if holding_obj_uid is not None:
                    collision_fn = self._define_grasp_collision_fn(holding_obj_uid, debug=False)
                    # Get RRT path using constraints
                    trajectory = birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.distance_fn,
                        self.sample_fn,
                        self.extend_fn,
                        collision_fn,
                        max_solutions=1,
                        restarts=self.rrt_trials)
                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    trajectory = birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.distance_fn,
                        self.sample_fn,
                        self.extend_fn,
                        self.default_collision_fn,
                        max_solutions=1,
                        restarts=self.rrt_trials)
                                    
                # # ======================== Debug =========================
                # if self.DEBUG_GET_DATA:
                #     print('End RRT')
                #     check_time_rrt = time.time() - check_time_rrt
                #     debug_data['rrt_time'].append(check_time_rrt)

            # Regrasp after planning?
            # self.robot.activate([holding_obj_uid])

        return trajectory
    
