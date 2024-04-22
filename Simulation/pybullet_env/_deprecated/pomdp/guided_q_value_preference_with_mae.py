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
from learning.model.common.transformer import GPT2FetchingConditionerWithMAE
from learning.model.common.mae import MaskedAutoencoderViT
from learning.model.value import ValueNet, HistoryValueWithMAE


from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import ( FetchingState, FetchingAction, FetchingObservation, 
                                                    ACTION_PICK, ACTION_PLACE )
from pomdp.common import process_history_q_value


class FetchingGuidedQValuePreferenceWithMAE(ValueModel):
    """A history-model implementation of value interface for POMCPOW"""

    @dataclass
    class Config(Serializable):

        # Model
        exp       : str = ''
        model_name: str = "best.pth"
        mae_backone  = '4.20_value_mae_April17th_sim_dataset_batch128_lr0.0001_cnnTrue_emb128_p8_m0.75/best.pth'

        # This parameter governs the model size
        dim_model_hidden = 256
        seq_length = 6
        # Transformer params
        dim_gpt_hidden = dim_model_hidden
        dim_condition  = dim_model_hidden
        gpt_config: GPT2FetchingConditionerWithMAE.Config \
            = GPT2FetchingConditionerWithMAE.Config(
                # Data type
                image_res           = 64,
                dim_obs_rgbd_ch     = 4,    # RGBD
                dim_obs_rgbd_encode = dim_model_hidden,
                dim_obs_grasp       = 1,
                dim_action_input    = 8,
                dim_goal_input      = 5,
                # Architecture
                dim_hidden          = dim_gpt_hidden,
                num_heads           = 4,
                dim_ffn             = dim_model_hidden,
                num_gpt_layers      = 4,
                dropout_rate        = 0.1,
                # Positional encoding
                max_len             = 32,
                seq_len             = seq_length,
                # Output
                dim_condition       = dim_condition)
        # Checkpoint of the MAE backbone for image embedding
        mae_config: MaskedAutoencoderViT.Config\
            = MaskedAutoencoderViT.Config(
                img_size = 64,
                patch_size = 8,
                in_chans = 4,
                embed_dim = 128,
                depth = 4,
                num_heads = 8,
                decoder_embed_dim = 128,
                decoder_depth = 3,
                decoder_num_heads = 4,
                mlp_ratio = 4.,
                mask_ratio = 0.0,
                early_cnn = True,
                pred_reward = False
            )
        # Value params
        value_config: ValueNet.Config = ValueNet.Config(
            dim_condition = dim_model_hidden)
        
        # Dataset params
        dataset_config: FetchingDatasetConfig \
            = FetchingDatasetConfig(
                seq_length    = seq_length,
                image_res     = 64)

        # Inference
        device : str = "cuda:0"

        # One-hot encoding table
        action_type_encoding  : Dict \
            = field(default_factory = lambda: { "PICK": 0, "PLACE": 1 })
        action_target_encoding: Dict \
            = field(default_factory = lambda: { True: 1, False: 0 })



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
            config (Settings): skipped
        """
        super().__init__(is_v_model=False)
        # Config
        self.config = config
        self.EXP_LEARNING_DIR  = config["project_params"]["overridable"]["default_exp_learning_dir_path"]
        self.NUM_FILTER_TRIALS = config["pose_sampler_params"]["num_filter_trials"]
        
        # Initialize policy in fetching domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip

        # NN
        self.nn_config = self.Config()
        self.device = self.nn_config.device
        backbone_filename = os.path.join(self.EXP_LEARNING_DIR, self.nn_config.mae_backone)
        self.model = HistoryValueWithMAE(
            fetching_gpt_config = self.nn_config.gpt_config,
            value_config        = self.nn_config.value_config,
            backbone            = backbone_filename,
            backbone_config     = self.nn_config.mae_config).to(self.nn_config.device)
        # Load weights
        model_dir = os.path.join(self.EXP_LEARNING_DIR, self.nn_config.exp)
        model_name = self.nn_config.model_name
        filename = os.path.join(model_dir, model_name)
        load_checkpoint_inference(self.nn_config.device, filename, self.model)



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
        data = process_history_q_value(
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
                = [d.to(self.nn_config.device) for d in data]
                
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
            value = torch.exp(pred).item()
            # value = torch.exp(pred).item()

        
        return value