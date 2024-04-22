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
from learning.model.common.cvae import CVAE
from learning.model.policy import HistoryPlacePolicyonly


from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import ( FetchingState, FetchingAction, FetchingObservation, 
                                                    ACTION_PICK, ACTION_PLACE )
from pomdp.policy.random_pickplace import SampleRandomTopPickUncertainty
from pomdp.common import process_history_v_value


class FetchingGuidedPolicyPlace(PolicyModel):
    """A history-model implementation of policy interface for POMCPOW"""

    @dataclass
    class Config(Serializable):

        # Experiment setup
        exp_name: str = "5.06_policy_mse_May6th_3obj_depth8_3000_sim_dataset_dim256_beta0.25_batch64_lr0.0001"
        weight  : str = "best.pth"

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
        # CVAE head params
        dim_action_output: int = 3
        dim_cvae_embed   : int = 64
        dim_vae_latent   : int = dim_model_hidden
        cvae_config: CVAE.Config \
            = CVAE.Config(
                latent_size         = dim_vae_latent,
                dim_condition       = dim_condition,
                dim_output          = dim_action_output,  # Action output
                dim_embed           = dim_cvae_embed,
                encoder_layer_sizes = (dim_cvae_embed, dim_cvae_embed + dim_condition, dim_vae_latent, dim_vae_latent),
                decoder_layer_sizes = (dim_vae_latent, dim_vae_latent + dim_condition, dim_vae_latent, dim_action_output))

        # Dataset params
        dataset_config: FetchingDatasetConfig \
            = FetchingDatasetConfig(
                seq_length    = seq_length,
                image_res     = 64)



    def __init__(self, bc     : BulletClient, 
                       env: BinpickEnvPrimitive, 
                       robot  : Robot, 
                       manip  : Manipulation,
                       config : Dict):
        """
        Args:
            bc (BulletClient): skipped
            env (BinpickEnvPrimitive): skippied
            robot (Robot): skipped
            manip (Manipulation): skipped
            config (Settings): skipped
        """

        # Config
        self.config = config
        self.NUM_FILTER_TRIALS = config["pose_sampler_params"]["num_filter_trials"]
        self.EXP_LEARNING_DIR  = config["project_params"]["overridable"]["default_exp_learning_dir_path"]
        self.DEVICE            = config["project_params"]["overridable"]["inference_device"]

        # Initialize policy in fetching domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip

        # Pick sampler
        self.pick_action_sampler = SampleRandomTopPickUncertainty(
            NUM_FILTER_TRIALS  = self.NUM_FILTER_TRIALS )

        # NN
        self.nn_config = self.Config()
        self.model = HistoryPlacePolicyonly( 
            cvae_config = self.nn_config.cvae_config,
            fetching_gpt_config = self.nn_config.gpt_config ).to(self.DEVICE)
        # Load weights
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



    def sample(self, init_observation: FetchingObservation, 
                     history: Tuple[HistoryEntry], 
                     state: FetchingState, 
                     goal: List) -> FetchingAction:
        """Infer next_action using neural network

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
        # PLACE: Use guided policy
        elif type == ACTION_PLACE:
            next_action = self.infer_place_action(init_observation, history, state, goal)

        return next_action



    def infer_place_action(self, init_observation: FetchingAction, 
                                 history: Tuple[HistoryEntry], 
                                 state: FetchingState, 
                                 goal: List[float]) -> FetchingAction:
        """Infer place action

        Args:
            init_observation (FetchingAction)
            history (Tuple[HistoryEntry])
            state (FetchingState)
            goal (List[float])

        Returns:
            FetchingAction: next action
        """

        # Get previous
        prev_action: FetchingAction = history[-1].action

        # Find holding object.
        holding_obj_uid = self.env.gid_to_uid[state.holding_obj_gid]

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
        
        # Try until find a pose without collision
        for i in range(self.NUM_FILTER_TRIALS):

            # Recover joint state before action. 
            # This sampler may have modified the src pose in previous loop.
            for value_i, joint_i in enumerate(self.robot.joint_indices_arm):
                self.bc.resetJointState(self.robot.uid, joint_i, state.robot_state[value_i])

            # Predict (x, y, delta_theta) of place action with NN
            self.model.eval()
            with torch.no_grad():
                pred: torch.Tensor \
                    = self.model.inference(
                        init_obs_rgbd   = init_observation_rgbd,
                        init_obs_grasp  = init_observation_grasp,
                        goal            = goal_token,
                        seq_action      = seq_action,
                        seq_obs_rgbd    = seq_observation_rgbd,
                        seq_obs_grasp   = seq_observation_grasp,
                        mask_seq_action = action_mask,
                        mask_seq_obs    = observation_mask).squeeze(0)   # Shape=(1, 3)->(3)
            # Compose position
            x, y, delta_theta = pred.tolist()
            z = prev_action.pos[2] + 0.01   # Use z from the last action.
            place_pos = (x, y, z)
            # Compose orientation
            prev_orn = prev_action.orn
            prev_orn_q = self.bc.getQuaternionFromEuler(prev_orn)
            yaw_rot_in_world = np.array([0, 0, delta_theta])
            yaw_rot_in_world_q = self.bc.getQuaternionFromEuler(yaw_rot_in_world) 
            _, place_orn_q = self.bc.multiplyTransforms([0, 0, 0], yaw_rot_in_world_q,
                                                [0, 0, 0], prev_orn_q)
            place_orn = self.bc.getEulerFromQuaternion(place_orn_q)
            
            
            # Filtering by motion plan
            place_orn_q = self.bc.getQuaternionFromEuler(place_orn)                 # Outward orientation (surface normal)
            modified_pos_back, modified_orn_q_back, \
                modified_pos, modified_orn_q \
                = self.manip.get_ee_pose_from_target_pose(place_pos, place_orn_q)   # Inward orientation (ee base)
            joint_pos_src, joint_pos_dst \
                = self.manip.solve_ik_numerical(modified_pos, modified_orn_q)       # src is captured from current pose.
            
            
            # Try motion planning
            traj = self.manip.motion_plan(joint_pos_src, joint_pos_dst, holding_obj_uid) 
            if traj is not None:
                # Found some nice motion planning. Break here.
                # Double check: type, target, pos, orn, traj, delta_theta
                return FetchingAction(type        = ACTION_PLACE,
                                      aimed_gid   = prev_action.aimed_gid,
                                      pos         = tuple(place_pos),
                                      orn         = tuple(place_orn),
                                      traj        = traj,
                                      delta_theta = delta_theta)
            else:
                # Continue filtering until the iteration reaches the num_filter_trials.
                continue
            
        # If reached here, it means that no feasible place pose is found.
        # Returning infeasible action.
        return FetchingAction(ACTION_PLACE, None, None, None, None, None)