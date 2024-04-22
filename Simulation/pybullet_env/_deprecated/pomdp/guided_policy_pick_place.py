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
from learning.model.common.transformer import GPT2FetchingConditioner
from learning.model.common.cvae import CVAE
from learning.model.policy import HistoryPlacePolicyonly
from learning.dataset.common import ( format_next_action_label,
                                      tokenize_action, 
                                      tokenize_masked_rgbd,
                                      add_zero_pre_padding,
                                      FetchingPickPlaceDatasetConfig )
from data_generation.collect_data import convert_key_and_join_seg_mask, remove_observation_background

from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import ( FetchingState, FetchingAction, FetchingObservation, 
                                                    ACTION_PICK, ACTION_PLACE, capture_binpickenv_state )


class FetchingGuidedPolicyPickPlace(PolicyModel):
    """A history-model implementation of policy interface for POMCPOW"""

    @dataclass
    class Config(Serializable):

        # Experiment setup
        exp_name: str = "4.25_pick&place_policy_mse_April17th_sim_dataset_dim256_beta0.25_batch64_lr0.0001"
        weight  : str = "best.pth"

        # This parameter governs the model size
        dim_model_hidden: int = 256
        seq_length      : int = 6
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
        dim_cvae_embed   : int = 128
        dim_vae_latent   : int = dim_model_hidden
        cvae_config: CVAE.Config \
            = CVAE.Config(
                latent_size         = dim_vae_latent,
                dim_condition       = dim_condition,
                dim_output          = dim_action_output,  # Action output
                dim_embed           = dim_cvae_embed,
                encoder_layer_sizes = (dim_cvae_embed, dim_cvae_embed + dim_condition, dim_vae_latent),
                decoder_layer_sizes = (dim_vae_latent, dim_vae_latent + dim_condition, dim_action_output))

        # Dataset params
        dataset_config: FetchingPickPlaceDatasetConfig \
            = FetchingPickPlaceDatasetConfig(
                seq_length = seq_length,
                image_res  = 64,
                dim_action_input = gpt_config.dim_action_input)


        # Inference
        device : str = "cuda:0"

        # One-hot encoding table
        action_type_encoding  : Dict \
            = field(default_factory = lambda: { "PICK": 0, "PLACE": 1 })
        action_target_encoding: Dict \
            = field(default_factory = lambda: { True: 1, False: 0 })



    def __init__(self, bc     : BulletClient, 
                       env: BinpickEnvPrimitive, 
                       robot  : Robot, 
                       manip  : Manipulation,
                       config : Config):
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
        
        # Initialize policy in fetching domain
        self.bc = bc
        self.env = env
        self.robot = robot
        self.manip = manip

        # NN
        self.nn_config = self.Config()
        self.device = self.nn_config.device
        self.model = HistoryPlacePolicyonly( 
            cvae_config = self.nn_config.cvae_config,
            fetching_gpt_config = self.nn_config.gpt_config ).to(self.nn_config.device)
        # Load weights
        weight_path = os.path.join(self.EXP_LEARNING_DIR, self.nn_config.exp_name, self.nn_config.weight)
        load_checkpoint_inference(self.nn_config.device, weight_path, self.model)



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
        
        next_action = self.infer_place_action(init_observation, history, state, goal, type)

        return next_action



    def infer_place_action(self, init_observation: FetchingAction, 
                                 history: Tuple[HistoryEntry], 
                                 state: FetchingState, 
                                 goal: List[float],
                                 type) -> FetchingAction:
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
        if len(history) == 0:
            prev_action = None
        else:
            prev_action: FetchingAction = history[-1].action

        # Find holding object.
        if type == ACTION_PICK:
            holding_obj_uid = None
        else:
            holding_obj_uid = self.env.gid_to_uid[state.holding_obj_gid]
        
        particle_robot_state, particle_object_state = capture_binpickenv_state(self.bc, self.env, self.robot, self.config)

        # Tensor 
        data = self._process_history(
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
            if type == ACTION_PICK:
                z = 0.681
            else:
                z = prev_action.pos[2] + 0.01   # Use z from the last action.
            pos = (x, y, z)
            # Compose orientation
            if type == ACTION_PICK:
                orn = (0, 0, 0)
            else:
                prev_orn = prev_action.orn
                prev_orn_q = self.bc.getQuaternionFromEuler(prev_orn)
                yaw_rot_in_world = np.array([0, 0, delta_theta])
                yaw_rot_in_world_q = self.bc.getQuaternionFromEuler(yaw_rot_in_world) 
                _, orn_q = self.bc.multiplyTransforms([0, 0, 0], yaw_rot_in_world_q,
                                                    [0, 0, 0], prev_orn_q)
                orn = self.bc.getEulerFromQuaternion(orn_q)
            
            
            # Filtering by motion plan
            orn_q = self.bc.getQuaternionFromEuler(orn)                 # Outward orientation (surface normal)
            modified_pos_back, modified_orn_q_back, \
                modified_pos, modified_orn_q \
                = self.manip.get_ee_pose_from_target_pose(pos, orn_q)   # Inward orientation (ee base)
            joint_pos_src, joint_pos_dst \
                = self.manip.solve_ik_numerical(modified_pos, modified_orn_q)       # src is captured from current pose.
            
            
            # Try motion planning
            traj = self.manip.motion_plan(joint_pos_src, joint_pos_dst, holding_obj_uid) 
            if traj is not None:
                # Found some nice motion planning. Break here.
                # Double check: type, target, pos, orn, traj, delta_theta
                if type == ACTION_PICK:
                    aimed_gid = 0
                    min_dist = 10000
                    for gid, p in particle_object_state.items():
                        dist = (pos[0] - p.pos[0])**2 + (pos[1] - p.pos[1])**2
                        if dist < min_dist:
                            aimed_gid = gid
                    return FetchingAction(type=ACTION_PICK,
                                          aimed_gid=aimed_gid,
                                          pos=tuple(pos),
                                          orn=tuple(orn),
                                          traj=traj,
                                          delta_theta=delta_theta)
                else:
                    return FetchingAction(type        = ACTION_PLACE,
                                        aimed_gid   = prev_action.aimed_gid,
                                        pos         = tuple(pos),
                                        orn         = tuple(orn),
                                        traj        = traj,
                                        delta_theta = delta_theta)
            else:
                # Continue filtering until the iteration reaches the num_filter_trials.
                continue
            
        # If reached here, it means that no feasible place pose is found.
        # Returning infeasible action.
        return FetchingAction(type, None, None, None, None, None)
    
    
    def _process_history(self, bc: BulletClient,
                            env: BinpickEnvPrimitive,
                            init_observation: FetchingObservation,
                            history: Tuple[HistoryEntry], 
                            goal_condition: List[float],
                            dataset_config: FetchingPickPlaceDatasetConfig) -> Tuple[torch.Tensor, ...]:
        """Process history to the data for V and policy inference.
        See fetching_value_dataset.py and collect_data.py
        
        Args:
            bc (BulletClient)
            env (BinpickEnvPrimitive)
            init_observation (FetchingObservation): Initial observation
            history (Tuple): Tuple of current history ((a0, o0, r0)), (a1, o1, r1), ... )
            goal_condition (List): Goal condition, [r,g,b,x,y]
            dataset_config (FetchingDatasetConfig): Model configuration.

        Returns:    
            init_observation_rgbd  : shape = (1, 4, 64, 64)
            init_observation_grasp : shape = (1, 1)
            goal                   : shape = (1, rgbxy)
            seq_action             : shape = (1, input_trajectory_length, 8)
            seq_observation_rgbd   : shape = (1, input_trajectory_length, 4, 64, 64)
            seq_observation_grasp  : shape = (1, input_trajectory_length, 1)
            action_mask            : shape = (1, input_trajectory_length)
            observation_mask       : shape = (1, input_trajectory_length)
            input_trajectory_length: shape = (1, 1)
        """
        # Tokenizing (shared function with dataset)
        #   Initial observation
        init_seg_mask_converted = convert_key_and_join_seg_mask(env.gid_table, init_observation.seg_mask)
        init_masked_depth_image, init_masked_rgb_image \
            = remove_observation_background(init_observation.depth_image,
                                            init_observation.rgb_image, 
                                            init_seg_mask_converted)
        init_grasp_contact = init_observation.grasp_contact
        init_observation_rgbd = tokenize_masked_rgbd(init_masked_depth_image, init_masked_rgb_image)
        init_observation_grasp = np.expand_dims(init_grasp_contact, axis=-1)

        #   Goal condition
        goal = np.array(goal_condition)
        #   Process the trajectory from time 0 to input_trajectory_length-1
        seq_action            = []
        seq_observation_rgbd  = []
        seq_observation_grasp = []
        input_trajectory_length = len(history)
        for t, history_entry in enumerate(history):
            a: FetchingAction = history_entry.action
            o: FetchingObservation = history_entry.observation
            # Tokenize action            
            action_token = tokenize_action(
                action_type            = a.type,
                action_is_target_bool  = env.gid_table[a.aimed_gid].is_target if a.aimed_gid is not None \
                                    else None,
                action_pos             = a.pos,
                action_orn_e           = a.orn,
                action_dyaw            = a.delta_theta,           
                action_type_encoding   = dataset_config.action_type_encoding,
                action_target_encoding = dataset_config.action_target_encoding)
            # Tokenize observation
            seg_mask_converted = convert_key_and_join_seg_mask(env.gid_table, o.seg_mask)
            masked_depth_image, masked_rgb_image \
                = remove_observation_background(o.depth_image,
                                                o.rgb_image,
                                                seg_mask_converted)
            observation_rgbd = tokenize_masked_rgbd(masked_depth_image, masked_rgb_image)
            observation_grasp = np.expand_dims(o.grasp_contact, axis=-1)
            # Collect (V-Value)
            seq_action.append(action_token)
            seq_observation_rgbd.append(observation_rgbd)
            seq_observation_grasp.append(observation_grasp)
        # Gather
        if len(seq_action) !=0 :
            seq_action = np.stack(seq_action, axis=0)
            seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
            seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
        input_trajectory_length = np.asarray([input_trajectory_length])


        # Make tensor
        init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
        init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
        goal                    = torch.from_numpy(goal).float()
        if len(seq_action) !=0 :
            seq_action              = torch.from_numpy(seq_action).float()
            seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
            seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
        input_trajectory_length = torch.from_numpy(input_trajectory_length).int()

        # Time step... (input_trajectory_length idx starts from 0)
        num_paddings          = self.nn_config.dataset_config.seq_length - (input_trajectory_length.item())
        # Make padding and concatenate
        if len(seq_action) !=0 :
            seq_action            = add_zero_pre_padding(seq_variable = seq_action, 
                                                        fill_like    = seq_action[0], 
                                                        num_paddings = num_paddings)
            seq_observation_rgbd  = add_zero_pre_padding(seq_variable = seq_observation_rgbd, 
                                                        fill_like    = init_observation_rgbd, 
                                                        num_paddings = num_paddings)
            seq_observation_grasp = add_zero_pre_padding(seq_variable = seq_observation_grasp,
                                                        fill_like    = init_observation_grasp,
                                                        num_paddings = num_paddings)
        else:
            seq_action = torch.zeros(self.nn_config.dataset_config.seq_length, self.nn_config.dataset_config.dim_action_input)
            obs_rgbd_padding = torch.zeros_like(init_observation_rgbd)
            seq_observation_rgbd = obs_rgbd_padding.expand(self.nn_config.dataset_config.seq_length, *obs_rgbd_padding.shape)
            obs_grasp_padding = torch.zeros_like(init_observation_grasp)
            seq_observation_grasp = obs_grasp_padding.expand(self.nn_config.dataset_config.seq_length, *obs_grasp_padding.shape)

        # Create sequence mask
        action_mask      = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
        observation_mask = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)

        # Batchify
        init_observation_rgbd   = init_observation_rgbd.unsqueeze(0)
        init_observation_grasp  = init_observation_grasp.unsqueeze(0)
        goal                    = goal.unsqueeze(0)
        seq_action              = seq_action.unsqueeze(0)
        seq_observation_rgbd    = seq_observation_rgbd.unsqueeze(0)
        seq_observation_grasp   = seq_observation_grasp.unsqueeze(0)
        action_mask             = action_mask.unsqueeze(0)
        observation_mask        = observation_mask.unsqueeze(0)
        input_trajectory_length = input_trajectory_length.unsqueeze(0)


        return ( init_observation_rgbd, init_observation_grasp, goal,
                seq_action, seq_observation_rgbd, seq_observation_grasp, 
                action_mask, observation_mask, input_trajectory_length )