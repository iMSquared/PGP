
import numpy as np
import numpy.typing as npt
import torch
from typing import List


from imm.pybullet_util.bullet_client import BulletClient
from envs.global_object_id_table import Gid_T
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import Robot, UR5, UR5Suction
from envs.manipulation import Manipulation

from data_generation.collect_data import convert_key_and_join_seg_mask, remove_observation_background
from learning.dataset.common import FetchingDatasetConfig, add_zero_pre_padding, tokenize_action, tokenize_masked_rgbd
from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import ( FetchingState, FetchingAction, FetchingObservation, 
                                                    ACTION_PICK, ACTION_PLACE )




def process_history_v_value(bc: BulletClient,
                            env: BinpickEnvPrimitive,
                            init_observation: FetchingObservation,
                            history: Tuple[HistoryEntry], 
                            goal_condition: List[float],
                            dataset_config: FetchingDatasetConfig) -> Tuple[torch.Tensor, ...]:
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
    seq_action = np.stack(seq_action, axis=0)
    seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
    seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
    input_trajectory_length = np.asarray([input_trajectory_length])


    # Make tensor
    init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
    init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
    goal                    = torch.from_numpy(goal).float()
    seq_action              = torch.from_numpy(seq_action).float()
    seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
    seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
    input_trajectory_length = torch.from_numpy(input_trajectory_length).int()

    # Time step... (input_trajectory_length idx starts from 0)
    num_paddings          = dataset_config.seq_length - (input_trajectory_length.item())
    # Make padding and concatenate
    seq_action            = add_zero_pre_padding(seq_variable = seq_action, 
                                                 fill_like    = seq_action[0], 
                                                 num_paddings = num_paddings)
    seq_observation_rgbd  = add_zero_pre_padding(seq_variable = seq_observation_rgbd, 
                                                 fill_like    = init_observation_rgbd, 
                                                 num_paddings = num_paddings)
    seq_observation_grasp = add_zero_pre_padding(seq_variable = seq_observation_grasp,
                                                 fill_like    = init_observation_grasp,
                                                 num_paddings = num_paddings)
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


def process_history_q_value(bc: BulletClient,
                            env: BinpickEnvPrimitive,
                            init_observation: FetchingObservation,
                            history: Tuple[HistoryEntry], 
                            goal_condition: List[float],
                            dataset_config: FetchingDatasetConfig) -> Tuple[torch.Tensor, ...]:
    """Process history to the data for Q inference.
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
    init_grasp_contact = np.array(init_observation.grasp_contact)
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
        # Collect (Q-Value)
        if t < input_trajectory_length-1:
            seq_action.append(action_token)
            seq_observation_rgbd.append(observation_rgbd)
            seq_observation_grasp.append(observation_grasp)
        else:
            # Omit last observation
            seq_action.append(action_token)
    # Gather
    seq_action = np.stack(seq_action, axis=0)
    if len(seq_observation_grasp) != 0:
        seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
        seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
    else:
        # Handling zero sequence when training Q at time step 1.
        seq_observation_rgbd = np.zeros(shape=(0, *init_observation_rgbd.shape))
        seq_observation_grasp = np.zeros(shape=(0, *init_observation_grasp.shape))
    input_trajectory_length = np.asarray([input_trajectory_length])


    # Make tensor
    init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
    init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
    goal                    = torch.from_numpy(goal).float()
    seq_action              = torch.from_numpy(seq_action).float()
    seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
    seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
    input_trajectory_length = torch.from_numpy(input_trajectory_length).int()

     # Time step... (input_trajectory_length idx starts from 0)
    num_paddings          = dataset_config.seq_length - (input_trajectory_length.item())
    # Make padding and concatenate
    seq_action            = add_zero_pre_padding(seq_variable = seq_action, 
                                                 fill_like    = seq_action[0], 
                                                 num_paddings = num_paddings)
    seq_observation_rgbd  = add_zero_pre_padding(seq_variable = seq_observation_rgbd, 
                                                 fill_like    = init_observation_rgbd, 
                                                 num_paddings = num_paddings+1)     # +1 for Q!
    seq_observation_grasp = add_zero_pre_padding(seq_variable = seq_observation_grasp,
                                                 fill_like    = init_observation_grasp,
                                                 num_paddings = num_paddings+1)     # +1 for Q!
    # Create sequence mask
    action_mask      = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
    observation_mask = torch.cat((torch.ones(input_trajectory_length-1), torch.zeros(num_paddings+1)), dim=-1)  # +1 for Q!

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