"""
POMDP modelling for the fetching problem
"""

import os
import random
import copy
import numpy as np
import pybullet as pb
from typing import Tuple, Dict, List

from pybullet_object_models import ycb_objects
from POMDP_framework import *
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env import BinpickEnv
from envs.robot import Robot, UR5, FrankaPanda
from imm.motion_planners.rrt_connect import birrt
from envs.manipulation import Manipulation, imagine, interpolate_trajectory
from envs.grasp_pose_sampler import PointSamplingGrasp
from observation.observation import ( get_pixelwise_distance, 
                                      get_hausdorff_distance_norm_pdf_value, )

# Debugging
import pdb
import time
import pickle
from debug import debug_data
# from debug_belief_update import debug_belief



class FetchingState(State):
    """
    State consist of robot state (joint) and object state (pose and  shape)
    """

    # |FIXME(Jiyong)|: think about how to express the shape of objects, instead of index of URDF 
    # |FIXME(Jiyong)|: need to append filtering invalid state
    def __init__(self, bc: BulletClient, env: Environment, robot: Tuple, object: Dict):
        """
        Args:
            robot_joint: Tuple
            object: Dict[str: Tuple[Tuple(position), Tuple(orientation), float(scale), bool(target)]] - str: name of URDF, Tuple: pose in SE(3)
        """
        self.robot = robot
        self.object = object
        self._uid = None        # dict{uid: shape} <- _set_simulation() of FetchingAgent
        self._img = None        # numpy.array <- FetchingObservationModel.sample()

    
    def __eq__(self, other):
        if self.robot != other.robot:
            return False
        else:
            for k, v in self.object.items():
                if k not in other.object:
                    return False
                else:
                    if v != other.object[k]:
                        return False
                    else:
                        continue
            return True
    
    # |NOTE(Jiyong)|: need to find better hash function
    def __hash__(self):
        return hash((self.robot, tuple(self.object)))
    
    @property
    def uid(self):
        return self._uid
    
    @uid.setter
    def uid(self, uid):
        self._uid = uid
        
    @property
    def img(self):
        return self._img
    
    @img.setter
    def img(self, img):
        self._img = img


# # ===================== Testing =============================

# class FetchingState(State):
#     """
#     State consist of robot state (joint) and object state (pose and  shape)
#     """

#     # |FIXME(Jiyong)|: think about how to express the shape of objects, instead of index of URDF 
#     # |FIXME(Jiyong)|: need to append filtering invalid state
#     def __init__(self, bc: BulletClient, env: Environment, robot: Tuple, object: Dict):
#         """
#         Args:
#             robot_joint: Tuple
#             object: Dict[str: Tuple[Tuple(position), Tuple(orientation), bool(target)] - str: name of URDF, Tuple: pose in SE(3)
#         """
#         self.robot = robot
#         self.object = object
#         self.uid = {}
#         for i, k in enumerate(object.keys()):   # |FIXME(Jiyong)|: would be wrong when the order and the number of objects is not matched
#             self.uid[k] = env.objects_uid[i]
    
#     def __eq__(self, other):
#         if self.robot != other.robot:
#             return False
#         else:
#             for k, v in self.object.items():
#                 if k not in other.object:
#                     return False
#                 else:
#                     if v != other.object[k]:
#                         return False
#                     else:
#                         continue
#             return True
    
#     # |NOTE(Jiyong)|: need to find better hash function
#     def __hash__(self):
#         return hash((self.robot, tuple(self.object)))

# # ===========================================================


class FetchingAction(Action):
    """
    There are three types of actions in SE(3):
        PICK(g)
        MOVE(p1)
        PLACE(p2)
    """

    # |FIXME(Jiyong)|: need to append filtering invalid action
    def __init__(self, type: str, uid: int, pos: Tuple[float], orn: Tuple[float], traj: List[np.ndarray] = None):
        self.type = type
        self.uid = uid
        self.pos = pos
        self.orn = orn
        self.traj = traj

    def __eq__(self, other):
        return (self.type == other.type) and (self.uid == other.uid) and (self.pos == other.pos) and (self.orn == other.orn)

    def __hash__(self):
        return hash((self.type, self.uid, self.pos, self.orn))


class FetchingObservation(Observation):
    """
    Observation is seqmented partial point cloud
    """

    # |FIXME(Jiyong)|: need to append filtering invalid observation
    def __init__(self, observation: np.ndarray, seg_mask: Dict[int, np.ndarray]):
        self.observation = observation  # depth image
        self.seg_mask = seg_mask        # segmentation mask of robot and objects
    
    def __eq__(self, other):
        for k, v in self.observation.items():
            if k not in other.observation:
                return False
            else:
                if v != other.observation[k]:
                    return False
                else:
                    continue
        return True

    def __hash__(self):
        return hash(tuple(tuple(obs.tolist()) for obs in self.observation))


class FetchingTransitionModel(TransitionModel):
    def __init__(self, bc: BulletClient, env: BinpickEnv, robot: Robot, config):
        self.config = config
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = Manipulation(bc, env, robot, config)
    
    def probability(self, next_state: FetchingState, state: FetchingState, action: FetchingAction):
        """
        determinisitic
        """
        return 1.0
    
    def sample(self, state: FetchingState, action: FetchingAction, execution: bool=False, *args):
        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('TransitionModel.sample() start')
            check_time = time.time()
        
        # fail to find an feasible action
        if action.traj is None:
            return state, None
        
        try:
            if action.type not in ("PICK", "MOVE", "PLACE"):
                raise ValueError
        except ValueError:
            print(f"{action.type} is not supported!")
        
        if action.type == "PICK":
            
            # To make the environment stable
            self.manip.wait(steps=240)
            
            if action.uid is None:
                # Not findind a feasible grasp pose
                return state, None
            
            obj_uid = action.uid
            pos = action.pos
            orn_q = self.bc.getQuaternionFromEuler(action.orn)
            traj = action.traj
            result, plan = self.manip.pick(obj_uid, pos, orn_q, traj)    # success/fail to find motion plan
            if result:
                action.traj = plan
                
        elif action.type == "MOVE":
            obj_uid = action.uid
            pos = action.pos
            orn_q = self.bc.getQuaternionFromEuler(action.orn)
            traj = action.traj
            result, plan = self.manip.move(obj_uid, pos, orn_q, traj)
            if result:
                action.traj = plan
                
        elif action.type == "PLACE":
            obj_uid = action.uid
            pos = action.pos
            orn_q = self.bc.getQuaternionFromEuler(action.orn)
            traj = action.traj
            result, plan = self.manip.place(obj_uid, pos, orn_q, traj)
            if result:
                action.traj = plan

        # Get next state of objects
        robot_state, object_state = get_state_info(self.bc, self.env, self.robot, self.config)

        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('TransitionModel.sample() end')
            check_time = time.time() - check_time
            debug_data['TransitionModel.sample()'].append(check_time)
            
        return FetchingState(self.bc, self.env, robot_state, object_state), result


class FetchingObservationModel(ObservationModel):
    def __init__(self, bc: BulletClient, env: BinpickEnv, robot, config):
        self.bc = bc
        self.env = env
        self.robot = robot
        self.config = config
        self.sigma = config["env_params"]["binpick_env"]["depth_camera"]["noise_std"]
        # self.distance_model = DistanceModel(self.bc, self.config)

    def get_sensor_observation(self, next_state: FetchingState, action: FetchingAction):
        """
        Return depth image with adding Gaussian noise
        """
        
        (w, h, px, px_d, px_id) = self.bc.getCameraImage(
            width            = self.env.width,
            height           = self.env.height,
            viewMatrix       = self.env.camera_view_matrix,
            projectionMatrix = self.env.camera_proj_matrix,
            renderer         = self.bc.ER_BULLET_HARDWARE_OPENGL)

        # Reshape list into ndarray(image)
        depth_array = np.array(px_d, dtype=np.float32)
        noise = self.sigma * np.random.randn(h, w)
        obs = depth_array + noise
        
        # segmentation
        mask_array = np.array(px_id, dtype=np.uint8)
        seg_mask = {}
        seg_mask[self.robot.uid] = np.where(mask_array==self.robot.uid, True, False)
        for uid in self.env.objects_uid:
            seg_mask[uid] = np.where(mask_array==uid, True, False)
        
        next_state.img = depth_array    # storing ground truth image        
        
        return FetchingObservation(obs, seg_mask)
    
    # def get_sensor_observation(self, next_state: FetchingState, action: FetchingAction):
    #     """
    #     Return segmented partial point cloud from depth image
    #     """
    #     return FetchingObservation(self.env.get_measurement())

    def sample(self, next_state: FetchingState, action: FetchingAction, *args):
        """
        Currently, observation model is same as capturing depth image (with segmentation) with adding Gaussian noise
        """
        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('ObservationModel.sample() start')
            check_time = time.time()
            
        observation = self.get_sensor_observation(next_state, action)
        
        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('ObservationModel.sample() end')
            check_time = time.time() - check_time
            debug_data['ObservationModel.sample()'].append(check_time)
            
        return observation
    
    def probability(self, observation: FetchingObservation, 
                          next_state: FetchingState, 
                          action: FetchingAction, 
                          segmentation=True, 
                          log=False, 
                          beta=0.1, 
                          m=5):
        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('ObservationModel.probability() start')
            check_time = time.time()
        

        # Get probability
        # prob = get_pixelwise_distance(self.bc, self.env,
        #                               observation, segmentation, log,
        #                               self.sigma, beta, m)
        prob = get_hausdorff_distance_norm_pdf_value(self.bc, self.env, self.robot,
                                      observation)

        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('ObservationModel.probability() end')
            check_time = time.time() - check_time
            debug_data['ObservationModel.probability()'].append(check_time)
        

        return prob


    
    # def probability(self, observation: FetchingObservation, next_state: FetchingState, action: FetchingAction):
    #     # ======================== Debug =========================
    #     if self.config['project_params']['debug']['get_data']:
    #         print('ObservationModel.probability() start')
    #         check_time = time.time()
        
    #     obs = observation.observation

    #     t = time.time()

    #     # |NOTE(Jiyong)|: get pcd from state
    #     state = self.env.get_pcd()

    #     t2 = time.time()

    #     avg_likelihood, _ = self.distance_model(obs, state)
        
    #     t3 = time.time()
        
    #     print("get_pcd():", t2 - t)
    #     print("cal_distance:", t3 - t2)
        
    #     # ======================== Debug =========================
    #     if self.config['project_params']['debug']['get_data']:
    #         print('ObservationModel.probability() end')
    #         check_time = time.time() - check_time
    #         debug_data['ObservationModel.probability()'].append(check_time)
            
    #     return avg_likelihood


class FetchingRewardModel(RewardModel):
    def __init__(self, bc: BulletClient, env: BinpickEnv, robot: Robot, config):
        self.bc = bc
        self.env = env
        self.robot = robot
        self.config = config

    def probability(self, reward: float, state: FetchingState, action: FetchingAction, next_state: FetchingState):
        """
        determinisitic
        """
        return 1.0
    
    def _check_termination(self, state: FetchingState):
        """
        terminate conditions:
            success condition: holding the target object in front of the robot without falling down the other objects
            fail condition: falling down any object
        """
        if self.is_fail(state):
            return "fail"
        elif self.is_success(state):
            return "success"
        else:
            return "continue"
    
    def is_success(self, state: FetchingState):
        for pos, orn, scale, target in state.object.values():
            if target:
                # |NOTE(Jiyong)|: x of the a target object is less than that of cabinet
                if pos[0] < self.config["env_params"]["binpick_env"]["cabinet"]["pos"][0] - 0.35:   #|NOTE(Jiyong)|: hardcoded of the goal pose
                    return True
            else:
                continue
        return False

    def is_fail(self, state: FetchingState):
        for pos, orn, scale, target in state.object.values():
            if pos[2] < self.config["env_params"]["binpick_env"]["cabinet"]["pos"][2] + 0.5:   # |NOTE(Jiyong)|: hardcoded of the height of the second floor of the cabinet
                return True
        return False

    def sample(self, next_state: FetchingState, action: FetchingAction, state: FetchingState, *args):
        """
        +100 for satisfying success condition -> terminate
        -100 for fail -> terminate
        -10 for action failed
            infeasible case:
                - not found collision free trajectory
                - fail to grasp with collision free trajectory
        -1 for each step
        """

        # check termination condition
        termination = self._check_termination(next_state)

        if termination == "success":
            self.robot.last_pose[self.robot.joint_index_finger] = self.robot.joint_value_finger_open
            
            if self.config['project_params']['debug']['get_data']:
                    print('======== Success =========')
            
            return 100., termination

        elif termination == "fail":
            self.robot.last_pose[self.robot.joint_index_finger] = self.robot.joint_value_finger_open
            
            if self.config['project_params']['debug']['get_data']:
                    print('======== Fail =========')
                    
                    debug_data['fail_case']['termination'] += 1
            
            return -100., termination

        else:
            # check whether finding motion planning or holding the object after pick
            if args:
                manipulation_result = args[0][0]
                if manipulation_result is None: # not found a feasible action
                    self.robot.last_pose[self.robot.joint_index_finger] = self.robot.joint_value_finger_open
                    return -100., "fail"
                elif not manipulation_result:  # not found collision free trajectory and missed held objecty during MOVE()
                    return -10., termination
                else:
                    return -1., termination
            else:
                # |NOTE(Jiyong)|: For getting reward without transition (line 17 in POMCPOW's psuedo code) and not satisfying termination condition. Think about this is proper reward. Because there are cases get -1 for infeasible action
                return -1., termination


class FetchingRolloutPolicyModel(RolloutPolicyModel):
    def __init__(self, bc: BulletClient, env: BinpickEnv, robot: Robot, config):
        super(FetchingRolloutPolicyModel, self).__init__()
        self.bc = bc
        self.env = env
        self.robot = robot
        self.config = config
        self.manip = Manipulation(bc, env, robot, config)
        self.grasp_pose_sampler = PointSamplingGrasp()
        self.num_filter_trials = config["pose_sampler_params"]["num_filter_trials"]

    def sample(self, history: Tuple, state: FetchingState, prior: List=None):
        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('PolicyModel.sample() start')
            check_time = time.time()
        
        # Check type prior
        if (prior is None) or (prior is not None and prior[0] is None):
            holding_obj = self.manip.check_holding_object()            
            if holding_obj is None:   # not holding an object
                type = "PICK"
            else:
                type = "PLACE"
        else:
            type = prior[0]
        
        if type == "PICK":
            if self.config['project_params']['debug']['get_data']:
                debug_data['num_action'][0] += 1
                print('Start to find a grasp pose')
                check_time_g = time.time()
                
            # Find filtered grasp pose
            obj_pcds = self.env.get_pcd()
            for i in range(self.num_filter_trials):
                
                # Check uid prior
                if (prior is None) or (prior is not None and prior[1] is None):
                    obj_uid = random.choice(self.env.objects_uid)
                else:
                    obj_uid = prior[1]                
                
                obstacle_list = copy.deepcopy(self.env.objects_uid)
                obstacle_list.remove(obj_uid)
                
                # Check pose prior
                if (prior is None) or (prior is not None and prior[2] is None):
                    obj_pcd = obj_pcds[obj_uid]
                    grasp_pos, grasp_orn, joint_pos_src, joint_pos_dst = self.manip.filter_valid_grasp_pose(
                        self.grasp_pose_sampler,
                        obj_uid,
                        [self.env.cabinet_uid] + obstacle_list,
                        obj_pcd
                    )
                    grasp_pose = (grasp_pos, grasp_orn)
                else:
                    grasp_pose = prior[2]
                    grasp_orn_q = self.bc.getQuaternionFromEuler(grasp_pose[1])
                    joint_pos_src, joint_pos_dst = self.manip._solve_ik(np.asarray(grasp_pose[0]), grasp_orn_q)

                if grasp_pose[0] is not None:
                    # filtering by motion plan
                    traj = self._motion_plan(joint_pos_src, joint_pos_dst) 
                    if traj is not None:
                        uid = obj_uid
                        pos = tuple(grasp_pose[0].tolist())
                        orn = self.bc.getEulerFromQuaternion(grasp_pose[1])
                        break
                    # fail to find collision free trajectory
                    else:
                        uid, pos, orn, traj = None, None, None, None
                # fail to find grasp pose
                else:
                    uid, pos, orn, traj = None, None, None, None
                    
            if (self.config['project_params']['debug']['get_data']) and (uid is None):
                print('Fail to find a grasp pose')
            
                debug_data['fail_case']['PICK'][0] += 1
            
            if self.config['project_params']['debug']['get_data']:
                print(f'End to find a grasp pose - {i} trials')
                check_time_g = time.time() - check_time_g
                debug_data['filter_grasp_pose_time'].append(check_time_g)

        elif type == "PLACE":   # |FIXME(Jiyong)|: hardcoded
            if self.config['project_params']['debug']['get_data']:
                debug_data['num_action'][2] += 1
            
            for i in range(self.num_filter_trials):
                # Check prior: take prior only if both uid and pose are exist
                if (prior is None) or (prior is not None and (prior[1] is None or prior[2] is None)):
                    # sample position
                    x = random.uniform(self.config["env_params"]["binpick_env"]["cabinet"]["pos"][0] - 0.3, self.config["env_params"]["binpick_env"]["cabinet"]["pos"][0] + 0.3)
                    y = random.uniform(self.config["env_params"]["binpick_env"]["cabinet"]["pos"][1] - 0.3, self.config["env_params"]["binpick_env"]["cabinet"]["pos"][1] + 0.3)
                    z = random.uniform(self.config["env_params"]["binpick_env"]["cabinet"]["pos"][2] + 0.54, self.config["env_params"]["binpick_env"]["cabinet"]["pos"][2] + 0.69)
                    pos = (x, y, z)

                    # Sample orientation
                    roll = random.uniform(-np.pi/2, np.pi/2)
                    pitch = random.uniform(np.pi/2, np.pi)
                    yaw = random.uniform(-np.pi/3, np.pi/3)
                    orn = (roll, pitch, yaw)
                    uid = holding_obj
                else:
                    pos = prior[2][0]
                    orn = prior[2][1]
                    uid = prior[1]
                    
                # filtering by motion plan
                orn_q = self.bc.getQuaternionFromEuler(np.asarray(orn))
                joint_pos_src, joint_pos_dst = self.manip._solve_ik(np.asarray(pos), orn_q)
                traj = self._motion_plan(joint_pos_src, joint_pos_dst) 
                if traj is not None:
                    break
                else:
                    uid, pos, orn, traj = None, None, None, None      

        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('PolicyModel.sample() end', type, pos, orn)
            check_time = time.time() - check_time
            debug_data['PolicyModel.sample()'].append(check_time)
            
        return FetchingAction(type, uid, pos, orn, traj)
    
    
    def _motion_plan(self, joint_pos_src, joint_pos_dst, use_interpolation=False):
        # Solve IK and get positions
        
        # t = time.time()
                
        # grasp_test_result["time_ik"].append(time.time() - t)

        # Check grasped object
        grasp_uid = self.manip.check_holding_object()

        # Compute motion plan
        # Force using interpolated trajectory (used when picking up)
        if use_interpolation:
            trajectory = interpolate_trajectory(
                cur             = joint_pos_src, 
                goal            = joint_pos_dst, 
                action_duration = 0.5, 
                control_dt      = self.control_dt)
        else:
            # Try RRT
            # ======================== Debug =========================
            if self.config['project_params']['debug']['get_data']:
                print('Start RRT')
                check_time_rrt = time.time()
            
            # t = time.time()
            
            with imagine(self.bc):
                # Moving with grapsed object
                if grasp_uid is not None:
                    collision_fn = self.manip._define_grasp_collision_fn(grasp_uid, debug=False)
                    # Get RRT path using constraints
                    trajectory = birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.manip.distance_fn,
                        self.manip.sample_fn,
                        self.manip.extend_fn,
                        collision_fn,
                        max_solutions=2,
                        restarts=self.manip.rrt_trials)
                # Moving without grasped object
                else:
                    # Get RRT path using default constraints
                    trajectory = birrt(
                        joint_pos_src,
                        joint_pos_dst,
                        self.manip.distance_fn,
                        self.manip.sample_fn,
                        self.manip.extend_fn,
                        self.manip.default_collision_fn,
                        max_solutions=2,
                        restarts=self.manip.rrt_trials)
                    
                # grasp_test_result["time_rrt"].append(time.time() - t)
                
                # ======================== Debug =========================
                if self.config['project_params']['debug']['get_data']:
                    print('End RRT')
                    check_time_rrt = time.time() - check_time_rrt
                    debug_data['rrt_time'].append(check_time_rrt)

                return trajectory


class FetchingAgent(Agent):
    def __init__(self, bc: BulletClient, env: BinpickEnv, robot: Robot, config, blackbox_model: BlackboxModel, policy_model: PolicyModel, rollout_policy_model: FetchingRolloutPolicyModel, value_model: ValueModel=None, init_belief: WeightedParticles=None, shape_table=None):
        super(FetchingAgent, self).__init__(blackbox_model, policy_model, rollout_policy_model, value_model, init_belief)
        self.config = config
        self.bc = bc
        self.env = env
        self.robot = robot

    def _set_simulation(self, state: FetchingState, reset: bool, simulator: str):
        """
        Reset robot joint configuration.
        Remove objects in current simulation.
        Reload objects according to state in simulation.
        """

        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('Agent._set_simulation() start')
            check_time = time.time()

        # Reset robot joints
        self.robot.last_pose[self.robot.joint_indices_arm] = state.robot[0]
        self.robot.last_pose[self.robot.joint_index_finger] = state.robot[1]
        for i, idx in enumerate(self.robot.joint_indices_arm):
            self.bc.resetJointState(self.robot.uid, idx, state.robot[0][i])
        self.bc.resetJointState(self.robot.uid, self.robot.joint_index_finger, state.robot[1])
        target_position = -1.0 * self.robot.joint_gear_ratio_mimic * np.asarray(state.robot[1])
        for i, idx in enumerate(self.robot.joint_indices_finger_mimic):
            self.bc.resetJointState(self.robot.uid, idx, target_position[i])
        self.robot.last_pose[self.robot.joint_indices_finger_mimic] = np.asarray(state.robot[1])

        # Remove objects
        for obj_uid in self.env.objects_uid:
            self.bc.removeBody(obj_uid)

        # Reload objects
        objects_uids = []
        uid_to_urdf = {}
        for shape, (pos, orn, scale, target) in state.object.items():
            uid = self.bc.loadURDF(
                fileName        = shape,
                basePosition    = pos,
                baseOrientation = self.bc.getQuaternionFromEuler(orn),
                useFixedBase    = False,
                globalScaling   = scale)
            objects_uids.append(uid)
            uid_to_urdf[uid] = shape
        
        # Store the uids to state and env
        state.uid = uid_to_urdf
        self.env.objects_uid = objects_uids
        self.env.objects_urdf = uid_to_urdf
        
        # Adjust dynamics
        for uid in objects_uids:
            self.bc.changeDynamics(
                uid, 
                -1, 
                lateralFriction=1.6,
                rollingFriction=0.0004,
                spinningFriction=0.0004,
                restitution=0.2)
        
        for _ in range(100): 
            self.robot.update_arm_control()
            self.robot.update_finger_control()
            self.bc.stepSimulation()
            if self.config["sim_params"]["delay"]:
                time.sleep(self.config["sim_params"]["delay"])
        
        # ======================== Debug =========================
        if self.config['project_params']['debug']['get_data']:
            print('Agent._set_simulation() end')
            check_time = time.time() - check_time
            debug_data['Agent._set_simulation()'].append(check_time)

        return

    # # ===================== Testing =============================

    # def _set_simulation(self, state: FetchingState, reset, simulator: str):
    #     """
    #     Reset robot joint configuration.
    #     Remove objects in current simulation.
    #     Reload objects according to state in simulation.
    #     """

    #     # ======================== Debug =========================
    #     if self.config['project_params']['debug']['get_data']:
    #         print('Agent._set_simulation() start')
    #         check_time = time.time()
        
    #     # Reset robot joints
    #     self.robot.last_pose[self.robot.joint_indices_arm] = state.robot[0]
    #     self.robot.last_pose[self.robot.joint_index_finger] = state.robot[1]
    #     for i, idx in enumerate(self.robot.joint_indices_arm):
    #         self.bc.resetJointState(self.robot.uid, idx, state.robot[0][i])
    #     self.bc.resetJointState(self.robot.uid, self.robot.joint_index_finger, state.robot[1])
    #     target_position = -1.0 * self.robot.joint_gear_ratio_mimic * np.asarray(state.robot[1])
    #     for i, idx in enumerate(self.robot.joint_indices_finger_mimic):
    #         self.bc.resetJointState(self.robot.uid, idx, target_position[i])
    #     self.robot.last_pose[self.robot.joint_indices_finger_mimic] = np.asarray(state.robot[1])

    #     # Remove objects
    #     for obj_uid in self.env.objects_uid:
    #         self.bc.removeBody(obj_uid)

    #     # Reload objects
    #     self.env.objects_uid = ([
    #         self.bc.loadURDF(
    #             fileName        = os.path.join(ycb_objects.getDataPath(), obj),
    #             basePosition    = pos,
    #             baseOrientation = self.bc.getQuaternionFromEuler(orn),
    #             useFixedBase    = False)
    #         for obj, (pos, orn, target) in state.object.items()
    #     ])
    
    #     for _ in range(100): 
    #         self.robot.update_arm_control()
    #         self.robot.update_finger_control()
    #         self.bc.stepSimulation()
    #         if self.config["sim_params"]["delay"]:
    #             time.sleep(self.config["sim_params"]["delay"])

    #     # ======================== Debug =========================
    #     if self.config['project_params']['debug']['get_data']:
    #         print('Agent._set_simulation() end')
    #         check_time = time.time() - check_time
    #         debug_data['Agent._set_simulation()'].append(check_time)
        
    #     return

    # # ===========================================================
    
    def update(self, real_action: FetchingAction, real_observation: FetchingObservation, real_reward: float):
        # Update history
        self._update_history(real_action, real_observation, real_reward)
        
        sim_id = pb.connect(pb.DIRECT)
        bc = BulletClient(sim_id)
        # bc.resetSimulation()
        
        # Sim params
        CONTROL_DT = 1. / self.config["sim_params"]["control_hz"]
        bc.setTimeStep(CONTROL_DT)
        bc.setGravity(0, 0, self.config["sim_params"]["gravity"])
        bc.resetDebugVisualizerCamera(
            cameraDistance       = self.config["sim_params"]["debug_camera"]["distance"], 
            cameraYaw            = self.config["sim_params"]["debug_camera"]["yaw"], 
            cameraPitch          = self.config["sim_params"]["debug_camera"]["pitch"], 
            cameraTargetPosition = self.config["sim_params"]["debug_camera"]["target_position"])
        bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING)
        
        # Simulation initialization
        # env
        binpick_env = BinpickEnv(bc, self.config)
        # robot
        if self.config["project_params"]["robot"] == "ur5":
            robot = UR5(bc, self.config)
        elif self.config["project_params"]["robot"] == "franka_panda":
            robot = FrankaPanda(bc, self.config)
        else:
            raise Exception("invalid robot")
        
        # Update particles
        particles = []
        weights = []
        cur_particle = self.belief.particles
        for p, v in cur_particle.items():
            # |FIXME(ssh)|: Reimplementation is required.
            #               Please refer to the `observation_test.py`
            raise NotImplementedError
            # self.imagine_state(p, reset=True, simulator=self.config["plan_params"]["simulator"])
            # next_p = self._blackbox_model._transition_model.sample(p, real_action)[0]  # Prediction
            # log_next_v = np.log(v) + np.log(self._blackbox_model._transition_model.probability(next_p, p, real_action))
            # log_next_v += self._blackbox_model._observation_model.probability(real_observation, next_p, real_action, log=True)
            # particles.append(next_p)
            # weights.append(log_next_v)

        # Normalize weight: use Exp-normalize trick for numerical stability
        weights = np.asarray(weights)
        max_weight = np.max(weights)
        weights = np.exp(weights - max_weight) / np.sum(np.exp(weights - max_weight))
        
        next_particles = {}
        for p, v in zip(particles, weights):
            next_particles[p] = v
            
            # debug_belief.append({
            #     "robot": p.robot,
            #     "object": p.object,
            #     "weight": v
            #     })
        

            for p, v in zip(particles, weights):
                next_particles[p] = v
                
        # set updated belief
        next_belief = WeightedParticles(next_particles)
        self.set_belief(next_belief)
        

def get_state_info(bc, env, robot, config):
    # ======================== Debug =========================
    if config['project_params']['debug']['get_data']:
        print('get_state_info() start')
        check_time = time.time()
    
    robot_arm_state = robot.get_arm_state()
    robot_arm_state = tuple(robot_arm_state.tolist())
    robot_finger_state = robot.get_finger_state()
    robot_state = (robot_arm_state, robot_finger_state)

    obj_state = {}
    for i, obj_uid in enumerate(env.objects_uid):
        if env.objects_urdf is None:    # For initial state
            obj_path = config['env_params']['binpick_env']['objects']['path'][i]
            shape = os.path.join(ycb_objects.getDataPath(), obj_path)
            scale = config['env_params']['binpick_env']['objects']['scale'][i]
            target = bool(obj_path == config['env_params']['binpick_env']['objects']['target'])
        else:    
            shape = env.objects_urdf[obj_uid]
            scale = config['env_params']['binpick_env']['objects']['scale'][i]
            target = bool(shape.split('/')[-3] == config["env_params"]["binpick_env"]["objects"]["target"].split('/')[-2])
        pos, orn_q = bc.getBasePositionAndOrientation(obj_uid)
        orn = bc.getEulerFromQuaternion(orn_q)
        obj_state[shape] = (pos, orn, scale, target)
        
    # ======================== Debug =========================
    if config['project_params']['debug']['get_data']:
        print('get_state_info() end')
        check_time = time.time() - check_time
        debug_data['get_state_info()'].append(check_time)
        
    return robot_state, obj_state


# # ===================== Testing =============================

# def get_state_info(bc, env, robot, config):
#     # ======================== Debug =========================
#     if config['project_params']['debug']['get_data']:
#         print('get_state_info() start')
#         check_time = time.time()
        
#     robot_arm_state = robot.get_arm_state()
#     robot_arm_state = tuple(robot_arm_state.tolist())
#     robot_finger_state = robot.get_finger_state()
#     robot_state = (robot_arm_state, robot_finger_state)

#     obj_state = {}
#     for i, obj_uid in enumerate(env.objects_uid):
#         key = config["env_params"]["binpick_env"]["objects"]["path"][i]
#         # key = bc.getBodyInfo(obj_uid)
#         if key == config["env_params"]["binpick_env"]["objects"]["target"]:
#             target = True
#         else:
#             target = False
#         pos, orn_q = bc.getBasePositionAndOrientation(obj_uid)
#         orn = bc.getEulerFromQuaternion(orn_q)
#         obj_state[key] = (pos, orn, target)
        
#     # ======================== Debug =========================
#     if config['project_params']['debug']['get_data']:
#         print('get_state_info() end')
#         check_time = time.time() - check_time
#         debug_data['get_state_info()'].append(check_time)
    
#     return robot_state, obj_state

# # ===========================================================


# |FIXME(Jiyong)|: temporaliy use
def make_shape_table(root_folder):
    shape_table = {}
    cls_indices = {}
    obj_cls_list = os.listdir(root_folder)
    for obj_cls in obj_cls_list:
        shape_list = os.listdir(os.path.join(root_folder, obj_cls))
        cls_indices[obj_cls] = list(range(len(shape_table), len(shape_table) + len(shape_list)))
        for shape in shape_list:
            shape_table[len(shape_table)] = os.path.join(root_folder, obj_cls, shape, "model.urdf")
        
    return shape_table, cls_indices


# |FIXME(Jiyong)|: temporaliy use -> move into Agent?
def make_belief(bc, env, robot, config, shape_table, cls_indices, num_particle):
    """
    This is temporal initial belief which has the uncertainty only for position.
    """
    import re
    from scipy.stats import truncnorm
    
    robot_state, object_state = get_state_info(bc, env, robot, config)
    
    fmt = re.compile("/model.urdf")
    init_belief = {}
    for _ in range(num_particle):
        objs_state = {}
        for i in range(config['env_params']['binpick_env']['objects']['num_objects']):
            obj_path = config['env_params']['binpick_env']['objects']['path'][i]
            try:
                m = fmt.search(obj_path)
                obj_cls = obj_path[:m.start()]
                
                if obj_cls not in cls_indices:
                    raise Exception(f"{obj_cls} is not in shape pool.")

                shape_idx = random.choice(cls_indices[obj_cls])
                shape = shape_table[shape_idx]
            
            except Exception as e:
                print(e)
            
            pos = config['env_params']['binpick_env']['objects']['pos'][i]
            pos[0] += truncnorm.rvs(-0.2, 0.2, loc=0, scale=0.05)
            pos[1] += truncnorm.rvs(-0.2, 0.2, loc=0, scale=0.05)
            pos = tuple(pos)
            
            orn = config['env_params']['binpick_env']['objects']['orn'][i]
            orn[2] += truncnorm.rvs(-0.2, 0.2, loc=0, scale=0.05)
            orn = tuple(orn)
            
            scale = config['env_params']['binpick_env']['objects']['scale'][i]
            
            target = bool(obj_path == config['env_params']['binpick_env']['objects']['target'])
            
            objs_state[shape] = (pos, orn, scale, target)
            
        particle = FetchingState(bc, env, robot_state, objs_state)
        init_belief[particle] = 1.0

    return WeightedParticles(init_belief)
