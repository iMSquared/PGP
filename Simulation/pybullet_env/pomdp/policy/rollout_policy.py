import copy
import random
import numpy as np
import numpy.typing as npt
from typing import List, NamedTuple
import time
import open3d as o3d
from scipy.spatial.transform import Rotation


from imm.pybullet_util.bullet_client import BulletClient
from envs.global_object_id_table import Gid_T
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import Robot, UR5, UR5Suction
from envs.manipulation import Manipulation

from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import (FetchingState, FetchingAction, FetchingObservation, 
                                                   ACTION_PICK, ACTION_PLACE)
from pomdp.policy.random_pickplace import SampleRandomTopPickUncertainty, SampleRandomPlace


# Debugging
from debug.debug import debug_data



class FetchingRolloutPolicyModel(RolloutPolicyModel):

    def __init__(self, bc: BulletClient, 
                       env: BinpickEnvPrimitive, 
                       robot: Robot, 
                       manip: Manipulation,
                       config: Dict):
        """Initialize a random rollout policy model."""

        # configs
        self.config = config
        self.NUM_FILTER_TRIALS: int  = config["pose_sampler_params"]["num_filter_trials"]
        self.DEBUG_GET_DATA   : bool = config['project_params']['debug']['get_data']
        self.SHOW_GUI         : bool = config['project_params']['debug']['show_gui']
        # Bullet
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip
        # Samplers
        self.pick_action_sampler = SampleRandomTopPickUncertainty(
            NUM_FILTER_TRIALS  = self.NUM_FILTER_TRIALS )
        self.place_action_sampler = SampleRandomPlace(
            NUM_FILTER_TRIALS      = self.NUM_FILTER_TRIALS)
        

    def set_new_bulletclient(self, bc: BulletClient, 
                                   env: BinpickEnvPrimitive, 
                                   robot: Robot,
                                   manip: Manipulation):
        """Re-initalize a random policy model with new BulletClient.
        Be sure to pass Manipulation instance together.
        Args:
            bc (BulletClient): New bullet client
            env (BinpickEnvPrimitive): New simulation environment
            robot (Robot): New robot instance
            manip (Manipulation): Manipulation instance of the current client.
        """
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip


    def sample(self, init_observation: FetchingObservation,
                     history: Tuple[HistoryEntry], 
                     state: FetchingState, 
                     goal: List[float]) -> FetchingAction:
        """Random rollout policy model!
        Args:
            init_observation (FetchingObservation)
            history (Tuple[HistoryEntry])
            state (FetchingState): Current state to start rollout
            goal (List[float]): Goal passed from the agent.
        Returns:
            FetchingAction: Sampled next action
        """

        # Select primitive action.
        holding_obj_gid = state.holding_obj_gid
        if holding_obj_gid is None:   # Robot not holding an object.
            type = ACTION_PICK
        else:
            type = ACTION_PLACE
    

        # PICK action
        if type == ACTION_PICK:
            next_action = self.pick_action_sampler(self.bc, self.env, self.robot, self.manip, 
                                                   init_observation, history, state, goal)
        # PLACE action
        elif type == ACTION_PLACE:
            last_action = history[-1].action
            next_action = self.place_action_sampler(self.bc, self.env, self.robot, self.manip,
                                                    state, last_action, goal)


        return next_action