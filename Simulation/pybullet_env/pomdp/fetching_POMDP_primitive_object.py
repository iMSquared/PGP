
"""
POMDP modelling for the fetching problem
"""

import os
import random
import copy
import numpy as np
import numpy.typing as npt
import pybullet as pb
from pathlib import Path
from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, List, TypeVar, Union, Set, Optional
from dataclasses import dataclass, field



from pomdp.POMDP_framework import *
from imm.pybullet_util.bullet_client import BulletClient
from imm.motion_planners.rrt_connect import birrt
from imm.pybullet_util.typing_extra import TranslationT, EulerT, QuaternionT
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.global_object_id_table import Gid_T
from envs.robot import UR5Suction
from envs.manipulation import Manipulation
from observation.observation import reproject_observation_to_pointcloud, get_hausdorff_distance_norm_pdf_value
from utils.process_geometry import random_sample_array_from_config

# Debugging
import pdb
import time
import pickle
from debug.debug import debug_data
from debug.debug_belief_update import debug_belief


# Global flags
FetchingActionT = TypeVar("FetchingActionT", bound=str)
ACTION_PICK  : FetchingActionT = "PICK"
ACTION_PLACE : FetchingActionT = "PLACE"



@dataclass(eq=True, frozen=True)
class BulletObjectPose:
    """
    This class stores the current object state of the gid.
    NOTE(ssh): Make sure to not pass the List for pos or orn. It will raise unhashable error.
    """
    pos: TranslationT
    orn: EulerT            # yaw pitch roll



class FetchingState(State):
    """
    State consist of robot state (joint) and object states (pose, shape, target)
    NOTE(ssh): Make sure to pass frozenset. It will raise unhashable error.
    """

    def __init__(self, robot_state    : Tuple[float],                # Joint values of the robot arm
                       object_states  : Dict[Gid_T, BulletObjectPose], # Dictionary of object poses
                       holding_obj_gid: Union[Gid_T, None]):           # Identifier of the holding object
        """Constructor initializes member variables.
        
        Args:
            robot_state (Tuple[float]): Joint values of the robot arm
            object_states (Dict[Gid_T, BulletObjectPose]): Object states
            holding_obj_gid (Union[int, None]): GID of the holding object.
        """
        self.robot_state = robot_state
        self.object_states = object_states
        self.holding_obj_gid = holding_obj_gid


    def __eq__(self, other: "FetchingState"):
        # NOTE(ssh): This eq seems to be wrong.
        if self.robot_state != other.robot_state:
            return False
        else:
            for k, v in self.object_states.items():
                if k not in other.object_states:
                    return False
                else:
                    if v != other.object_states[k]:
                        return False
                    else:
                        continue
            return True


    def __hash__(self):
        # NOTE(ssh): This hash seems not reflecting enough information...
        # but just leaving as it works fine for now. 
        return hash((self.robot_state, tuple(self.object_states)))



class FetchingAction(Action):
    """
    There are two types of actions in SE(3):
        PICK(obj, g)
        PLACE(p)

    All fields are None when infeasible action.    
    NOTE(ssh): pos and orn are represented in object's SURFACE NORMAL. (Outward)
    aimed_gid does not guarantee that the object is in contact.
    """
    def __init__(self, type: FetchingActionT,
                       aimed_gid: Union[Gid_T, None],
                       pos: Union[TranslationT, None],
                       orn: Union[EulerT, None],
                       traj: Union[npt.NDArray, None],
                       delta_theta: Union[float, None]):
        """Simple constructor

        Args:
            type (FetchingActionT): ACTION_PICK or ACTION_PLACE
            target (Union[Gid_T, None],): None when infeasible action.
            pos (Union[TranslationT, None]): [xyz]
            orn (Union[EulerT, None]): [roll pitch yaw]
            traj (Union[npt.NDArray, None]): Motion trajectory
            delta_theta (Union[float, None]): PLACE only
        """
        self.type = type
        self.aimed_gid = aimed_gid
        self.pos = pos
        self.orn = orn
        self.traj = traj
        self.delta_theta = delta_theta


    def __eq__(self, other: "FetchingAction"):
        return (self.type == other.type) \
            and (self.aimed_gid == other.aimed_gid) \
            and (self.pos == other.pos) \
            and (self.orn == other.orn)


    def __hash__(self):
        return hash((self.type, self.aimed_gid, self.pos, self.orn))


    def __str__(self):
        """Simple tostring."""
        if self.is_feasible():
            pos_str = f"({self.pos[0]:.3f}, {self.pos[1]:.3f}, {self.pos[2]:.3f})"
            orn_str = f"({self.orn[0]:.3f}, {self.orn[1]:.3f}, {self.orn[2]:.3f})"
            return f"{self.type}, {self.aimed_gid}, {pos_str}, {orn_str}"
        else:
            return f"{self.type}, infeasible action"


    def __repr__(self):
        return self.__str__()


    def is_feasible(self):
        """Check whether generated action is feasible."""
        if self.aimed_gid is None \
            or self.pos is None \
            or self.orn is None \
            or self.traj is None:
            return False
        else:
            return True



class FetchingObservation(Observation):
    """
    Observation is segmented partial point cloud
    """
    def __init__(self, depth_image: npt.NDArray,
                       rgb_image: npt.NDArray,
                       seg_mask: Dict[Gid_T, npt.NDArray],
                       grasp_contact: Union[bool, None]):
        """Simple constructor

        Args:
            depth_image (npt.NDArray): Depth image
            rgb_image (npt.NDArray): RGB image
            seg_mask (Dict[Gid_T, npt.NDArray]): Segmentation mask of robot and objects.
            grasp_contact (Union[bool, None]): Robot pick evaluation
        """
        self.depth_image = depth_image
        self.rgb_image = rgb_image
        self.seg_mask = seg_mask
        self.grasp_contact = grasp_contact

    depth_image  : npt.NDArray                           # Depth image
    rgb_image    : npt.NDArray                           # RGB image
    seg_mask     : Dict[Gid_T, npt.NDArray]  # Segmentation mask of robot and objects.
    grasp_contact: Union[bool, None]                        # Robot pick evaluation


    def __eq__(self, other: "FetchingObservation"):
        """Seems not updated recently????"""
        for k, v in self.depth_image.items():
            if k not in other.depth_image:
                return False
            else:
                if v != other.depth_image[k]:
                    return False
                else:
                    continue
        return True


    def __hash__(self):
        return hash(tuple(tuple(obs.tolist()) for obs in self.depth_image))
    

    def __str__(self):
        """Simple tostring."""
        return f"grasp_contact: {self.grasp_contact}"


    def __repr__(self):
        return self.__str__()



class FetchingTransitionModel(TransitionModel):

    def __init__(self, bc: BulletClient, 
                       env: BinpickEnvPrimitive, 
                       robot: UR5Suction,
                       manip: Manipulation, 
                       config: Dict):
        """Initalize a pybullet simulation transition model.
        Be sure to pass Manipulation instance together."""
        self.config = config
        self.DEBUG_GET_DATA = config['project_params']['debug']['get_data']

        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
    

    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: BinpickEnvPrimitive, 
                                   robot: UR5Suction,
                                   manip: Manipulation):
        """Re-initalize a pybullet simulation transition model with new BulletClient.
        Be sure to pass Manipulation instance together.

        Args:
            bc (BulletClient): New bullet client
            env (BinpickEnvPrimitive): New simulation environment
            robot (UR5Suction): New robot instance
            manip (Manipulation): Manipulation instance of the current client.
        """
        self.bc    = bc
        self.env   = env
        self.robot = robot
        self.manip = manip


    def probability(self, next_state: FetchingState, 
                          state: FetchingState, 
                          action: FetchingAction) -> float:
        """
        determinisitic
        """
        return 1.0
    

    def sample(self, state: FetchingState, 
                     action: FetchingAction, 
                     execution: bool = False) -> FetchingState:
        """Sample the result of the action.
        See the definition of the policy model for detail of the `action`.

        Args:
            state (FetchingState): Current state.
            action (FetchingAction): Action to do.
            execution (bool, optional): Flag for execution stage. Defaults to False.

        Raises:
            ValueError: Invalid action type error.

        Returns:
            next_state (FetchingState): Transition result
        """
        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('TransitionModel.sample() start')
        #     check_time = time.time()
        
        # No feasible action is found
        #   - No RRT path is found
        #   - Reachable pick pose is not found.
        #   This will be filtered in the reward function.
        if not action.is_feasible():
            return state
        
        # Query uid
        obj_gid = action.aimed_gid
        obj_uid = self.env.gid_to_uid[obj_gid]
        
        # Do pick action
        if action.type == ACTION_PICK:
            holding_obj_uid: Union[int, None] \
                = self.manip.pick(
                    obj_uid = obj_uid, 
                    pos     = action.pos,
                    orn_q   = self.bc.getQuaternionFromEuler(action.orn),
                    traj    = action.traj)
        # Do place action
        elif action.type == ACTION_PLACE:
            holding_obj_uid: Union[int, None] \
                = self.manip.place(
                    obj_uid = obj_uid, 
                    pos     = action.pos, 
                    orn_q   = self.bc.getQuaternionFromEuler(action.orn),
                    traj    = action.traj)

        # Get next state of objects
        robot_state, new_object_states = capture_binpickenv_state(self.bc, self.env, self.robot, self.config)
        if holding_obj_uid is not None:
            holding_obj_gid = self.env.uid_to_gid[holding_obj_uid]
        else:
            holding_obj_gid = None
        next_state = FetchingState(robot_state, new_object_states, holding_obj_gid)

        # # ======================== Debug =========================
        # if self.DEBUG_GET_DATA:
        #     print('TransitionModel.sample() end')
        #     check_time = time.time() - check_time
        #     debug_data['TransitionModel.sample()'].append(check_time)
            
        return next_state



class FetchingObservationModel(ObservationModel):

    def __init__(self, bc: BulletClient, 
                       env: BinpickEnvPrimitive, 
                       robot: UR5Suction,
                       config: Dict):
        """Initalize an observation model."""
        self.config = config
        self.SIGMA          = config["env_params"]["binpick_env"]["depth_camera"]["noise_std"]
        self.DEBUG_GET_DATA = config['project_params']['debug']['get_data']
        
        self.bc = bc
        self.env = env
        self.robot = robot


    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: BinpickEnvPrimitive, 
                                   robot: UR5Suction):
        """Re-initalize an observation model with new BulletClient.

        Args:
            bc (BulletClient): New bullet client
            env (BinpickEnvPrimitive): New simulation environment
            robot (UR5Suction): New robot instance
        """
        self.bc = bc
        self.env = env
        self.robot = robot


    def get_sensor_observation(self, next_state: FetchingState, 
                                     action: FetchingAction) -> FetchingObservation:
        """
        Return depth image with simulated Gaussian noise
        """
        # 1. For current implementation, next_state is set outside of this function!
        # 2. Capture
        depth_array, rgb_array, seg_mask = self.env.capture_rgbd_image(self.SIGMA)
        grasp_contact = self.robot.detect_contact()
        # 3. Create observation instance
        observation = FetchingObservation(depth_image = depth_array, 
                                          rgb_image = rgb_array,
                                          seg_mask = seg_mask,
                                          grasp_contact = grasp_contact)

        return observation

    
    def sample(self, next_state: FetchingState, 
                     action: FetchingAction) -> FetchingObservation:
        """
        Currently, observation is same as capturing depth image 
        (with segmentation) with adding Gaussian noise

        Args:
            next_state (FetchingState)
            action (FetchingAction)

        Returns:
            FetchingObservation: Sampled observation.
        """
        # ======================== Debug =========================
        if self.DEBUG_GET_DATA:
            print('ObservationModel.sample() start')
            check_time = time.time()
            
        observation = self.get_sensor_observation(next_state, action)
        
        # ======================== Debug =========================
        if self.DEBUG_GET_DATA:
            print('ObservationModel.sample() end')
            check_time = time.time() - check_time
            debug_data['ObservationModel.sample()'].append(check_time)
            
        return observation


    def probability(self, observation: FetchingObservation, 
                          next_state: FetchingState, 
                          action: FetchingAction) -> float:
        """Get the observation probability of current environment

        Args:
            observation (FetchingObservation): Observation to evaluate
            next_state (FetchingState): Not used. Use agent.imagine_state() instead.
            action (FetchingAction): Not used.

        Returns:
            float: Unnormalized relative likelihood.
        """
        # ======================== Debug =========================
        if self.DEBUG_GET_DATA:
            print('ObservationModel.probability() start')
            check_time = time.time()

        # Get PCD(s)        
        state_observation = self.get_sensor_observation(next_state, action)

        # Reproject them into the 3D.
        state_obs_pcd \
            = reproject_observation_to_pointcloud(
                self.bc, self.env,
                state_observation.depth_image, state_observation.seg_mask)
        obs_pcd \
            = reproject_observation_to_pointcloud(
                self.bc, self.env,
                observation.depth_image, observation.seg_mask)

        # Get probability
        prob_i = get_hausdorff_distance_norm_pdf_value(state_obs_pcd, obs_pcd)    # P(I|S)
        prob_g = self.__grasp_probabiltiy(observation, next_state)                # P(G|S)
        prob = prob_i * prob_g                  # P(I,G|S) = P(I|S)*P(G|S), indenpendent.
        # prob = prob_i

        # ======================== Debug =========================
        if self.DEBUG_GET_DATA:
            print('ObservationModel.probability() end')
            check_time = time.time() - check_time
            debug_data['ObservationModel.probability()'].append(check_time)
        
        return prob
    

    def __grasp_probabiltiy(self, observation: FetchingObservation, 
                                  next_state: FetchingState) -> float:
        
        if observation.grasp_contact:
            if next_state.holding_obj_gid is not None:
                return 1.0
            else:
                return 0.0
        else:
            if next_state.holding_obj_gid is not None:
                return 0.0
            else:
                return 1.0

        



class FetchingRewardModel(RewardModel):


    def __init__(self, bc: BulletClient, 
                       env: BinpickEnvPrimitive, 
                       robot: UR5Suction, 
                       config: Dict):
        """Initalize a reward model."""
        # Some configurations...
        self.config = config
        self.GOAL_POSITION : Tuple[float] = self.config["env_params"]["binpick_env"]["goal"]["pos"]
        self.DEBUG_GET_DATA: bool         = self.config['project_params']['debug']['get_data']
        
        # Reward configuration
        guide_preference: bool = config["project_params"]["overridable"]["guide_preference"]
        value_or_pref = "value" if not guide_preference else "preference"
        self.REWARD_SUCCESS       : float = config["plan_params"]["reward"][value_or_pref]["success"]
        self.REWARD_FAIL          : float = config["plan_params"]["reward"][value_or_pref]["fail"]
        self.REWARD_INFEASIBLE    : float = config["plan_params"]["reward"][value_or_pref]["infeasible"]
        self.REWARD_TIMESTEP_PICK : float = config["plan_params"]["reward"][value_or_pref]["timestep_pick"]
        self.REWARD_TIMESTEP_PLACE: float = config["plan_params"]["reward"][value_or_pref]["timestep_place"]

        # Variables
        self.bc = bc
        self.env = env
        self.robot = robot


    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: BinpickEnvPrimitive, 
                                   robot: UR5Suction):
        """Re-initalize a reward model with new BulletClient.

        Args:
            bc (BulletClient): New bullet client
            env (BinpickEnvPrimitive): New simulation environment
            robot (UR5Suction): New robot instance
        """
        self.bc = bc
        self.env = env
        self.robot = robot


    def probability(self, reward: float, 
                          state: FetchingState, 
                          action: FetchingAction, 
                          next_state: FetchingState) -> float:
        """
        determinisitic
        """
        return 1.0


    def _check_termination(self, state: FetchingState) -> TerminationT:
        """
        Evaluate the termination of the given state.
        Terminate conditions:
            success condition: holding the target object in front of the robot without falling down the other objects
            fail condition: falling down any object

        Args:
            state (FetchingState): A state instance to evaluate
        
        Returns:
            TerminationT: Termination condition.
        """
        if self.is_fail(state):     # NOTE(ssh): Checking the failure first is safer.
            return TERMINATION_FAIL
        elif self.is_success(state):
            return TERMINATION_SUCCESS
        else:
            return TERMINATION_CONTINUE


    def is_success(self, state: FetchingState) -> bool:
        """Check only the success"""
        for obj_gid in state.object_states:
            obj_state = state.object_states[obj_gid]
            if self.env.gid_table[obj_gid].is_target \
                and (obj_state.pos[0] >= self.GOAL_POSITION[0] - 0.1) \
                and (obj_state.pos[0] <= self.GOAL_POSITION[0] + 0.1) \
                and (obj_state.pos[1] >= self.GOAL_POSITION[1] - 0.1) \
                and (obj_state.pos[1] <= self.GOAL_POSITION[1] + 0.1) \
                and (obj_state.pos[2] >= 0.2):   # |NOTE(Jiyong)|: hardcoded of the goal pose
                return True
        return False


    def is_fail(self, state: FetchingState) -> bool:
        """Check only the failure"""
        for obj_gid in state.object_states:
            obj_state = state.object_states[obj_gid]
            if obj_state.pos[2] < 0.2:   # |NOTE(Jiyong)|: hardcoded of the height of the second floor of the cabinet
                return True
            if (not self.env.gid_table[obj_gid].is_target) \
                and (obj_state.pos[0] >= self.GOAL_POSITION[0] - 0.1) \
                and (obj_state.pos[0] <= self.GOAL_POSITION[0] + 0.1) \
                and (obj_state.pos[1] >= self.GOAL_POSITION[1] - 0.1) \
                and (obj_state.pos[1] <= self.GOAL_POSITION[1] + 0.1):
                return True
        return False


    def sample(self, next_state: FetchingState, 
                     action: FetchingAction, 
                     state: FetchingState,) -> float:
        """Evaluate the reward of the `next_state`.
        +100 for satisfying success condition -> terminate
        -100 for fail -> terminate
        -1 for action failed
            infeasible case:
                - not found collision free trajectory
                - fail to grasp with collision free trajectory
        -1 for each step

        Arguments should match with TransitionModel.sample() returns.

        Args:
            next_state (FetchingState): A state to evaluate the reward
            action (FetchingAction): Action used
            state (FetchingState): Previous state

        Returns:
            reward (float): Reward value
            termination (TerminationT): Termination flag
        """
        # No feasible action is found
        if action.traj is None:
            print('======== Infeasible action sampled ========')
            return self.REWARD_INFEASIBLE, TERMINATION_FAIL

        # Check termination condition
        termination = self._check_termination(next_state)

        # Success
        if termination == TERMINATION_SUCCESS:            
            print('======== Success ========')
            return self.REWARD_SUCCESS, termination
        # Fail
        elif termination == TERMINATION_FAIL:
            print('======== Fail ========')
            return self.REWARD_FAIL, termination
        # Penalize all picks.
        elif termination == TERMINATION_CONTINUE and action.type == ACTION_PICK:
            return self.REWARD_TIMESTEP_PICK, termination
        # Trivial. Carray on.
        elif termination == TERMINATION_CONTINUE and action.type == ACTION_PLACE:
            return self.REWARD_TIMESTEP_PLACE, termination
        # Unhandled reward
        else:
            raise ValueError("Unhandled reward")


class FetchingAgent(Agent):

    def __init__(self, bc: BulletClient, 
                       sim_env: BinpickEnvPrimitive, 
                       robot: UR5Suction, 
                       manip: Manipulation,
                       config: Dict,
                       blackbox_model: BlackboxModel, 
                       policy_model: PolicyModel, 
                       rollout_policy_model: RolloutPolicyModel, 
                       value_model: ValueModel = None, 
                       init_belief: WeightedParticles = None,
                       init_observation: FetchingObservation = None,
                       goal_condition: List[float] = None):
        """Fetching agent keeps the blackbox and policy, value model.

        Args:
            bc (BulletClient): skipped
            sim_env (BinpickEnvPrimitive): skipped
            robot (UR5Suction): skipped
            config (Dict): skipped
            blackbox_model (BlackboxModel): Blackbox model instance
            policy_model (PolicyModel): Policy model instance
            rollout_policy_model (RolloutPolicyModel): Policy model instance
            value_model (ValueModel, optional): Value model instance. Defaults to None.
            init_belief (WeightedParticles, optional): Initial belief. Defaults to None.
            init_observation (FetchingObservation): Initial FetchingObservation instance
            goal_condition (List[float]): Goal condition. [rgbxyz]
        """
        super(FetchingAgent, self).__init__(blackbox_model, 
                                            policy_model, rollout_policy_model, 
                                            value_model, 
                                            init_belief, init_observation, 
                                            goal_condition)
        self.config = config
        self.DEBUG_GET_DATA  = self.config['project_params']['debug']['get_data']
        self.PLAN_NUM_SIMS   = config["plan_params"]["num_sims"]

        self.bc = bc
        self.env = sim_env
        self.robot = robot
        self.manip = manip

        # Some helpful typing of inherited variables.
        self.goal_condition: List[float]
        self.init_observation: FetchingObservation

    
    def set_new_bulletclient(self, bc: BulletClient, 
                                   sim_env: BinpickEnvPrimitive, 
                                   robot: UR5Suction):
        """Re-initalize a reward model with new BulletClient.

        Args:
            bc (BulletClient): New bullet client
            env (BinpickEnvPrimitive): New simulation environment
            robot (UR5Suction): New robot instance
        """
        self.bc = bc
        self.env = sim_env
        self.robot = robot


    def _set_simulation(self, state: FetchingState, 
                              reset: bool):
        """
        Reset robot joint configuration.
        Remove objects in current simulation.
        Reload objects according to state in simulation.
        """
        set_binpickenv_state(self.bc, self.env, self.robot, state)


    def update(self, real_action: FetchingAction,
                     real_observation: FetchingObservation, 
                     real_reward: float,
                     env: Optional[Environment] = None):
        """Belief update of the agent
        NOTE(ssh): Current reinvigoration scheme cheats from the groundtruth

        Args:
            real_action (FetchingAction)
            real_observation (FetchingObservation)
            real_reward (float)
            env (Optional[Environment): POMDP environment that contains the groundtruth.
        """
        # Update history
        self._update_history(real_action, real_observation, real_reward)
        
        # List to store all particles...
        resampled_particles = []

        # 0. Always reinvigorate using the initial belief generator
        if env is not None:
            # Resample 20% of particles 
            num_resample_particles = int(self.PLAN_NUM_SIMS * 0.2)
            new_particles = self.reinvigorate_particles_from_belief_generator(env, real_action, real_observation, num_resample_particles)
            # Aggregate all particles
            resampled_particles.extend(new_particles)

        # 1. Update particles
        particles = []
        weights = []
        cur_particle = self.belief.particles
        p: FetchingState
        for p in cur_particle:
            next_p, next_v = self.reinvigorate_from_single_particle(real_action, real_observation, real_reward, p)
            particles.append(next_p)
            weights.append(next_v)
            
        #   Normalizing weight 
        weights = np.asarray(weights)
        if np.sum(weights) != 0:
            weights = weights / np.sum(weights)
        else:
            # Handling division by zero.
            weights = np.full_like(weights, 1./len(weights))

        next_particles = {}
        for p, v in zip(particles, weights):
            if p not in next_particles.keys():
                next_particles[p] = v
            else:
                next_particles[p] += v
        
        # 2. Set updated belief
        next_belief = WeightedParticles(next_particles)
        # self.set_belief(next_belief)

        # 3. Optional (Resamping)
        num_resampling = self.PLAN_NUM_SIMS - len(resampled_particles)
        for i in range(num_resampling):
            resampled_particle = next_belief.sample()
            resampled_particles.append(resampled_particle)
        
        resampled_belief = UnweightedParticles(tuple(resampled_particles))
        self.set_belief(resampled_belief)


    def reinvigorate_particles_from_belief_generator(self, env: Environment,
                                                           real_action: FetchingAction,
                                                           real_observation: FetchingObservation,
                                                           num_resample_particles: int) -> List[FetchingState]:
        """Reinvigorate particles using initial belief generator.
        
        Args:
            env (Environment): Groundtruth to cheat from.
            real_action (FetchingAction)
            real_observation (FetchingObservation)
            num_resample_particles (int)
        
        Returns:
            List[FetchingState]: Regenerated particles
        """
        # Get robot / object states
        gt_state: FetchingState = env.state
        robot_state     = gt_state.robot_state
        object_states   = gt_state.object_states
        
        # Check visibility of gid object.
        all_objects_gid = [key for key in object_states]
        visible_objects_gid = []
        for key, value in real_observation.seg_mask.items():
            if value.sum()!=0:
                visible_objects_gid.append(key)
        invisible_objects_gid = sorted(set(all_objects_gid).difference(set(visible_objects_gid)))

        # Resampling
        resampled_particles = []
        for _ in range(num_resample_particles):
            # Until no collision / no visibility
            for _ in range(100):
                # Sampling a random candidate
                new_object_states: dict[Gid_T, BulletObjectPose] = dict()
                for obj_gid in all_objects_gid:
                    pos = object_states[obj_gid].pos
                    orn = object_states[obj_gid].orn
                    # Noising all objects.
                    # If visible, add noise on GT pose
                    if obj_gid in visible_objects_gid:
                        if obj_gid == 0:        # Noise target
                            mu, sigma = 0, 0.01
                            noise     = np.random.normal(mu, sigma, 3)
                            noise[2]  = 0.
                        else:                   # Denoise non-target
                            noise = np.asarray([0., 0., 0.])
                        new_pos         = pos + noise
                        new_orn         = orn
                        new_obj_state   = BulletObjectPose(tuple(new_pos), tuple(new_orn))
                    # If invisible, sample on occluding region
                    else:
                        random_pos = random_sample_array_from_config(
                            center      = pos,
                            half_ranges = self.env.RANDOM_INIT_TARGET_POS_HALF_RANGES)
                        new_pos = random_pos
                        new_orn = orn           # Cheat orn
                        new_obj_state = BulletObjectPose(tuple(new_pos), tuple(new_orn))
                    # Aggregate
                    new_object_states[obj_gid] = new_obj_state

                # Force reset
                if real_action.type == ACTION_PICK and real_observation.grasp_contact:
                    new_p = FetchingState(robot_state, new_object_states, real_action.aimed_gid)
                else:
                    new_p = FetchingState(robot_state, new_object_states, None)
                self.imagine_state(new_p, reset=True)

                # Reject sampled particle if collision occurs
                has_contact = self.env.check_objects_close(0.005)
                if has_contact:
                    continue
                
                # Reject sampled particle if any invisible object is visible
                _, _, seg_mask = self.env.capture_rgbd_image(sigma=0)
                check_invisiblity = True
                for gid in invisible_objects_gid:
                    visibility = np.array(np.sum(seg_mask[gid]), dtype=bool).item()
                    if visibility:
                        check_invisiblity = False
                if not check_invisiblity:
                    continue
                        
                # Check robot grasp. It should match with the real observation.
                if self.robot.detect_contact():
                    check_robot_grasp = True
                else:
                    check_robot_grasp = False
                if (check_robot_grasp == real_observation.grasp_contact):
                    break
                
            # Create particles
            resampled_particles.append(new_p)
        return resampled_particles


    def filter_unique_particles(self, resampled_particles: List[FetchingState]) \
                                        -> Tuple[List[FetchingState], float]:
        """Leave only unique particles

        Args:
            resampled_particles (List[FetchingState])

        Returns:
            List[FetchingState]: unique_particles
            int: number of unique particles
        """
        unique_particles = []
        for i in resampled_particles:
            unique = True
            # Check if the current particle is unique compared to the list of unique particles 
            for j in unique_particles:
                if i.__eq__(j):
                    unique = False
            if unique == True:
                unique_particles.append(i)
        return unique_particles, len(unique_particles)
    

    def reinvigorate_from_single_particle(self, real_action: FetchingAction,
                                                real_observation: FetchingObservation, 
                                                real_reward: float,
                                                particle: FetchingState) -> Union[Tuple[FetchingState, float], Tuple[None, None]]:
        """Scatter a particle with noise for particle reinvigoration

        Args:
            real_action (FetchingAction)
            real_observation (FetchingObservation)
            real_reward (float)
            particle (FetchingState)

        Returns:
            new_particle (Union[FetchingState, None])
            weight (Union[float, None])
        """
        # Only reinvigorate when pick
        if real_action.type == ACTION_PICK \
            and real_action.aimed_gid is not None \
            and self.env.gid_table[real_action.aimed_gid].is_target:
            robot_state, object_states = particle.robot_state, particle.object_states
            # Until no collision / no visibility
            for i in range(100): # preventing infinite loop.
                # Sampling a random candidate
                new_object_states: dict[Gid_T, BulletObjectPose] = dict()
                for obj_gid, obj_state in object_states.items():

                    # Noising only the target object
                    if self.env.gid_table[real_action.aimed_gid].is_target:
                        new_pos = list(obj_state.pos)
                        new_pos[0] += 0.01 * np.random.randn()
                        new_pos[1] += 0.01 * np.random.randn()
                        new_orn = list(obj_state.orn)
                        new_orn[2] += 0.05 * np.random.randn()
                        new_obj_state = BulletObjectPose(tuple(new_pos), tuple(new_orn))
                    else:
                        new_obj_state = copy.deepcopy(obj_state)
                    # Save the state
                    new_object_states[obj_gid] = new_obj_state
                
                # Force reset
                for obj_gid, obj_state in new_object_states.items():
                    uid = self.env.gid_to_uid[obj_gid]
                    self.bc.resetBasePositionAndOrientation(
                        uid, obj_state.pos, self.bc.getQuaternionFromEuler(obj_state.orn))
                # Check collision
                has_contact = self.env.check_objects_close()
                
                # End randomization when no collision exist
                if not has_contact:     
                    # Create particles
                    new_p = FetchingState(robot_state, new_object_states, None)
                    self.imagine_state(new_p, reset=True)
                    next_p = self._blackbox_model._transition_model.sample(new_p, real_action)  # Prediction
                    next_v = self._blackbox_model._transition_model.probability(next_p, new_p, real_action)
                    next_v = next_v * self._blackbox_model._observation_model.probability(real_observation, next_p, real_action)

                    return next_p, next_v
        
        # No need otherwise. or not find particle satisfying
        self.imagine_state(particle, reset=True)
        next_p = self._blackbox_model._transition_model.sample(particle, real_action)  # Prediction
        next_v = self._blackbox_model._transition_model.probability(next_p, particle, real_action)
        next_v = next_v * self._blackbox_model._observation_model.probability(real_observation, next_p, real_action)
        return next_p, next_v
        


def capture_binpickenv_state(bc: BulletClient, 
                             env: BinpickEnvPrimitive, 
                             robot: UR5Suction, 
                             config: Dict) -> Tuple[Tuple[float], 
                                                    Dict[Gid_T, BulletObjectPose]]: 
    """Capture the CURRENT state of the robot and environment 
    in the form of FetchingState compatible.

    Args:
        bc (BulletClient): Bullet client
        env (BinpickEnvPrimitive): Env instance
        robot (UR5Suction): Robot instance
        config (Dict): Config file

    Returns:
        robot_state (Tuple[float]): The values of the ARM joints that are not "FIXED". Not the endeffector.
        object_states (Dict[Gid_T, BulletObjectPose]]): Dictionary of bullet object states.
    """
    
    robot_arm_state = robot.get_arm_state()
    robot_arm_state = tuple(robot_arm_state.tolist())
    robot_state = robot_arm_state

    object_states: Dict[Gid_T, BulletObjectPose] = {}
    for obj_uid in env.object_uids:
        pos, orn_q = bc.getBasePositionAndOrientation(obj_uid)
        orn = bc.getEulerFromQuaternion(orn_q)
        obj_gid = env.uid_to_gid[obj_uid]
        obj_state = BulletObjectPose(pos = pos,
                                     orn = orn)
        object_states[obj_gid] = obj_state
 
    return robot_state, object_states


def set_binpickenv_state(bc: BulletClient, 
                         env: BinpickEnvPrimitive, 
                         robot: UR5Suction, 
                         state: FetchingState):
    """A function that set the BinpickEnv into a probpided state
    
    Reset robot joint configuration.
    Remove objects in current simulation.
    Reload objects according to state in simulation.
    """
    # Make sure to release the robot
    robot.release()
    
    # Remove objects
    # for obj_uid in env.object_uids:
    #     bc.removeBody(obj_uid)

    # Reset robot joints
    robot.last_pose[robot.joint_indices_arm] = state.robot_state
    for value_i, joint_i in enumerate(robot.joint_indices_arm):
        bc.resetJointState(robot.uid, joint_i, state.robot_state[value_i])

    # Reusing existing URDFs. 
    # NOTE(ssh): This is not compatible with shape uncertainty, but a lot faster.
    for obj_gid in state.object_states:
        obj_uid = env.gid_to_uid[obj_gid]
        obj_state = state.object_states[obj_gid]
        bc.resetBasePositionAndOrientation(bodyUniqueId = obj_uid, 
                                                posObj = obj_state.pos,
                                                ornObj = bc.getQuaternionFromEuler(obj_state.orn))

    # # Re-spawn objects (slow)
    # object_uids: List = []
    # gid_to_uid: dict[Gid_T, int] = {}
    # uid_to_gid: dict[int, Gid_T] = {}
    # for obj_gid in state.object_states:
    #     urdf_file_path = env.gid_table[obj_gid].shape_info.urdf_file_path
    #     obj_state = state.object_states[obj_gid]
    #     uid = bc.loadURDF(
    #         fileName        = urdf_file_path,
    #         basePosition    = obj_state.pos,
    #         baseOrientation = bc.getQuaternionFromEuler(obj_state.orn),
    #         useFixedBase    = False)
    #     gid_to_uid[obj_gid] = uid
    #     uid_to_gid[uid] = obj_gid
    #     object_uids.append(uid)
    # # Update the uid and gid map to env
    # env.object_uids = tuple(object_uids)
    # env.gid_to_uid = gid_to_uid
    # env.uid_to_gid = uid_to_gid
    # # Adjust dynamics
    # env.reset_dynamics()
    
    # Reactivate grasp
    if state.holding_obj_gid is not None:
        robot.activate(env.object_uids)
        # TODO(ssh): Double check and raise error if not in contact.
        # print(f"grasp success when reset?: {robot.check_grasp()}")
        # if robot.check_grasp() is None:
        #     print("Reset no grasp??")
        #     breakpoint()
        #     raise ValueError("This is insane...")




def make_gt_init_state(bc: BulletClient, 
                       sim_env: BinpickEnvPrimitive, 
                       robot: UR5Suction, 
                       config: Dict) -> FetchingState:
    """Inital ground truth state routine

    Args:
        bc (BulletClient): skipped
        sim_env (BinpickEnvPrimitive): skipped
        robot (UR5Suction): skipped
        config (Dict): skipped

    Returns:
        FetchingState: FetchingState of the initial ground truth state.
    """
    robot_state, object_states = capture_binpickenv_state(bc, sim_env, robot, config)
    gt_init_state = FetchingState(robot_state, object_states, None)

    return gt_init_state
    


def make_goal_condition(config) -> List[float]:
    GOAL_COLOR    = config["env_params"]["binpick_env"]["goal"]["color"]
    GOAL_POSITION = config["env_params"]["binpick_env"]["goal"]["pos"][0:2]
    
    return GOAL_COLOR + GOAL_POSITION



def get_initial_observation(sim_env: BinpickEnvPrimitive,
                            observation_model: FetchingObservationModel,
                            gt_init_state: FetchingState) -> FetchingObservation:
    """
    Initial observation routine

    Args: 
        sim_env (BinpickEnvPrimitive): skipped
        observation_model (FetchingObservationModel): Observation model to use.
        gt_init_state (FetchingState): Initial state to capture the observation from.
    
    Return:
        FetchingObservation
    """

    # Acquire initial FetchingObservation (assume the environment is reset outside.)
    init_observation = observation_model.get_sensor_observation(gt_init_state, None)

    return init_observation



def make_belief_random_problem(bc: BulletClient, 
                               sim_env: BinpickEnvPrimitive,
                               robot: UR5Suction,
                               config: Dict, 
                               num_particle: int,
                               check_full_occlusion: bool = True) -> UnweightedParticles:
    """asdf

    Args:
        bc (BulletClient)
        sim_env (BinpickEnvPrimitive)
        robot (UR5Suction)
        config (Dict)
        num_particle (int)
        check_full_occlusion (bool)

    Returns:
        UnweightedParticles
    """
    robot_state, object_states = capture_binpickenv_state(bc, sim_env, robot, config)
    particles = []
    
    # Creating N particles
    for _ in range(num_particle):
        # Until no collision / no visibility
        while True:
            # Sampling a random candidate
            new_object_states: dict[Gid_T, BulletObjectPose] = dict()
            for obj_gid, obj_state in object_states.items():
                obj_uid = sim_env.gid_to_uid[obj_gid]
                base_pos, base_orn_q = bc.getBasePositionAndOrientation(obj_uid)
                # Noising only the target object
                if sim_env.gid_table[obj_gid].is_target:
                    random_pos = random_sample_array_from_config(center      = sim_env.RANDOM_INIT_TARGET_POS_CENTER,
                                                                 half_ranges = sim_env.RANDOM_INIT_TARGET_POS_HALF_RANGES)
                    random_orn = random_sample_array_from_config(center      = sim_env.RANDOM_INIT_TARGET_ORN_CENTER,
                                                                 half_ranges = sim_env.RANDOM_INIT_TARGET_ORN_HALF_RANGES)
                    random_pos[2] = base_pos[2]
                    new_obj_state = BulletObjectPose(tuple(random_pos), tuple(random_orn))
                else:
                    new_obj_state = copy.deepcopy(obj_state)
                # Save the state
                new_object_states[obj_gid] = new_obj_state
            # Force reset
            for obj_gid, obj_state in new_object_states.items():
                uid = sim_env.gid_to_uid[obj_gid]
                bc.resetBasePositionAndOrientation(
                    uid, obj_state.pos, bc.getQuaternionFromEuler(obj_state.orn))
            # Check collision
            has_contact = sim_env.check_objects_close()
            # Check target object occlusion
            _, _, seg_mask = sim_env.capture_rgbd_image(sigma = 0)
            target_gid     = sim_env.gid_table.select_target_gid()
            is_target_in_obs: bool = np.array(np.sum(seg_mask[target_gid]), dtype=bool).item()
            # End randomization when no collision exist
            if check_full_occlusion:
                if not has_contact and not is_target_in_obs:
                    break
            else:
                if not has_contact:
                    break          
        # Create particles
        particles.append(FetchingState(robot_state, new_object_states, None))

    return UnweightedParticles(tuple(particles))



def draw_belief(bc: BulletClient, env: BinpickEnvPrimitive, belief: UnweightedParticles) -> List[int]:
    """Some temporary belief visualization function... Will soon be deprecated.

    Args:
        bc (BulletClient)
        env (BinpickEnvPrimitive)
        belief (WeightedParticles): Belief to visualize

    Returns:
        List[int]: List of uids newly spawned. Erase them all later.
    """
    # Iterating through the particles
    belief_uid_list = []
    p: FetchingState
    for p in belief.particles:
        # Spawn objects in a particle.
        for obj_gid in p.object_states:
            obj_state = p.object_states[obj_gid]
            # Spawn
            uid = bc.loadURDF( env.gid_table[obj_gid].shape_info.urdf_file_path, 
                            obj_state.pos,
                            bc.getQuaternionFromEuler(obj_state.orn) )
            # Coloring
            if env.gid_table[obj_gid].is_target:
                bc.changeVisualShape(uid, -1, rgbaColor=[0.8, 0.5, 0.8, 0.1])
            else:
                bc.changeVisualShape(uid, -1, rgbaColor=[0.2, 0.4, 0.4, 0.1])
            # Append
            belief_uid_list.append(uid)

    return belief_uid_list