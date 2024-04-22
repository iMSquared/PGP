import os
import time
import random
import copy
import numpy as np
import numpy.typing as npt
import torch
from typing import List

from dataclasses import dataclass, replace, field
from simple_parsing import Serializable
from scipy.spatial.transform import Rotation

from imm.pybullet_util.bullet_client import BulletClient
from envs.global_object_id_table import Gid_T
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import Robot, UR5, UR5Suction
from envs.manipulation import Manipulation

from learning.utils import load_checkpoint_inference
from learning.dataset.common import FetchingDatasetConfig
from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.value import ValueNet, HistoryPlaceValueonly


from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import ( FetchingState, FetchingAction, FetchingObservation, 
                                                    ACTION_PICK, ACTION_PLACE )
from pomdp.common import process_history_v_value


class FetchingGuidedValuePreference(ValueModel):
    """A history-model implementation of value interface for POMCPOW"""

    @dataclass
    class Config(Serializable):

        # Experiment setup
        exp_name: str = "5.12_value_pref_May6th_3obj_depth8_3000_q=False_sim_dataset_dim256_batch1024_lr0.0001_norollout=False_augmentsuccess=True"
        weight  : str = "best.pth"
        temperature: float = 1.0

        # This parameter governs the model size
        dim_model_hidden: int = 256
        seq_length      : int = 8
        # Transformer params
        dim_gpt_hidden: int = dim_model_hidden
        dim_condition : int = dim_model_hidden
        gpt_config: GPT2FetchingConditioner.Config \
            = GPT2FetchingConditioner.Config(
                # Data type
                image_res           = 64,
                dim_obs_rgbd_ch     = 4,    # RGBD
                dim_obs_rgbd_encode = dim_model_hidden,
                dim_obs_grasp       = 1,
                dim_action_input    = 8,
                dim_goal_input      = 5,
                # Architecture
                dim_hidden          = dim_gpt_hidden,
                num_heads           = 2,
                dim_ffn             = dim_model_hidden,
                num_gpt_layers      = 2,
                dropout_rate        = 0.1,
                # Positional encoding
                max_len             = 100,
                seq_len             = seq_length,
                # Output
                dim_condition       = dim_condition)
        # Value params
        value_config: ValueNet.Config = ValueNet.Config(
            dim_condition = dim_model_hidden)

        # Dataset params
        dataset_config: FetchingDatasetConfig \
            = FetchingDatasetConfig(
                seq_length    = seq_length,
                image_res     = 64)



    def __init__(self, bc    : BulletClient, 
                       env   : BinpickEnvPrimitive, 
                       robot : Robot, 
                       manip : Manipulation,
                       config: Config): 
        """
        Args:
            bc (BulletClient): skipped
            env (BinpickEnvPrimitive): skippied
            robot (Robot): skipped
            manip (Manipulation): skipped
            config (Settings): skipped
        """
        super().__init__(is_v_model=True)
        # Config
        self.NUM_FILTER_TRIALS = config["pose_sampler_params"]["num_filter_trials"]
        self.EXP_LEARNING_DIR  = config["project_params"]["overridable"]["default_exp_learning_dir_path"]
        self.DEVICE            = config["project_params"]["overridable"]["inference_device"]

        # Exception
        if not (config["project_params"]["overridable"]["guide_q_value"] == False \
                and config["project_params"]["overridable"]["guide_preference"] == True):
            raise TypeError("Value model and configuration file mismatch.")

        # Initialize policy in fetching domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip

        # NN
        self.nn_config = self.Config()
        self.model = HistoryPlaceValueonly( 
            fetching_gpt_config = self.nn_config.gpt_config,
            value_config        = self.nn_config.value_config ).to(self.DEVICE)
        weight_path = os.path.join(self.EXP_LEARNING_DIR, self.nn_config.exp_name, self.nn_config.weight)
        load_checkpoint_inference(self.DEVICE, weight_path, self.model)



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



    def sample(self, init_observation: FetchingAction, 
                     history: Tuple[HistoryEntry], 
                     state: FetchingState, 
                     goal: List[float]) -> float:
        """Infer the value of the history using NN.

        Args:
            init_observation (FetchingAction)
            history (Tuple[HistoryEntry])
            state (FetchingState)
            goal (List[float])
        Returns:
            float: Estimated value
        """
        # Tensor 
        data = process_history_v_value(
            bc               = self.bc, 
            env              = self.env, 
            init_observation = init_observation,
            history          = history,
            goal_condition   = goal,
            dataset_config   = self.nn_config.dataset_config)
        
        # GPU
        init_observation_rgbd, init_observation_grasp, goal_token, \
            seq_action, seq_observation_rgbd, seq_observation_grasp, \
            action_mask, observation_mask, \
            input_trajectory_length, \
                = [d.to(self.DEVICE) for d in data]
        
        # Predict value with NN
        self.model.eval()
        with torch.no_grad():
            pred: torch.Tensor \
                = self.model.inference_value_only(
                    init_obs_rgbd   = init_observation_rgbd,
                    init_obs_grasp  = init_observation_grasp,
                    goal            = goal_token,
                    seq_action      = seq_action,
                    seq_obs_rgbd    = seq_observation_rgbd,
                    seq_obs_grasp   = seq_observation_grasp,
                    mask_seq_action = action_mask,
                    mask_seq_obs    = observation_mask).squeeze(0)   # Shape=(1, 1)->(1)
            value = pred.item() / self.nn_config.temperature

        return value