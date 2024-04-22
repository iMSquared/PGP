import os
import json
import numpy as np
import numpy.typing as npt
import torch

from typing import List, Dict, Tuple, Union

import shutil

from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import random


from learning.dataset.common import ( read_trajectory_from_json_and_numpy,
                                      tokenize_action, 
                                      tokenize_masked_rgbd,
                                      add_zero_pre_padding,
                                      FetchingDatasetConfig )



class FetchingVValueDataset(Dataset):


    def __init__(self, config: FetchingDatasetConfig,
                       data_path_json: str,
                       data_path_npz: str,
                       annotations: List[str],
                       num_subdata: List[int],
                       no_rollout_input: bool = False):
        """V-Value dataset

        Args:
            config (FetchingDatasetConfig): Configuration class
            data_path_json (str): Json dir path
            data_path_npz (str): Numpy dir path
            annotations (List[str]): List of data file name
            num_subdata (List[int]): Number of subdata in trajectory
            no_rollout_input (bool): Do not sample input from rollout when True. Defaults to False
        """
        self.config = config
        self.data_path_json = data_path_json
        self.data_path_npz = data_path_npz
        self.filenames = annotations
        self.weights = (np.asarray(num_subdata)/np.sum(num_subdata)).tolist()
        self.no_rollout_input = no_rollout_input



    def __len__(self):
        return len(self.filenames)



    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            init_observation_grasp         (torch.Tensor): shape=(1)
            goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            seq_action                     (torch.Tensor): shape=(full_trajectory_length, A)
            seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            observation_mask               (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            input_trajectory_length        (torch.Tensor): shape=(1)
            full_trajectory_length         (torch.Tensor): shape=(1)
            future_discounted_reward_label (torch.Tensor): shape=(1) NORMALIZED
            success_or_fail                (torch.Tensor): shape=(1)
        """

        # Loading json and numpy
        json_file_name = os.path.join(self.data_path_json, f"{self.filenames[index]}.json")
        npz_file_name = os.path.join(self.data_path_npz, f"{self.filenames[index]}.npz")
        with open(json_file_name, "r") as f:
            json_data = json.load(f)
        npz_data = np.load(npz_file_name)

        # Read trajectory
        init_observation, goal_condition, termination, \
            trajectory_action, trajectory_observation, trajectory_reward,\
            trajectory_length, \
            num_real_exec, num_sim_exec, num_rollout_exec \
                = read_trajectory_from_json_and_numpy(json_data, npz_data)

        # Randomly select the time step.
        if trajectory_length > 1:
            if self.no_rollout_input:
                input_trajectory_length = np.random.randint(1, num_real_exec+num_sim_exec+1)
            else:
                input_trajectory_length = np.random.randint(1, trajectory_length+1) # +1 here!
        else:
            input_trajectory_length = 1
            

        # Tokenizing
        #   Initial observation
        init_observation_depth, init_observation_rgb, init_observation_grasp = init_observation
        init_observation_rgbd = tokenize_masked_rgbd(init_observation_depth, init_observation_rgb)
        init_observation_grasp = np.expand_dims(init_observation_grasp, axis=-1)

        #   Goal condition
        goal = np.array(goal_condition)
        #   Process the trajectory from time 0 to input_trajectory_length-1
        seq_action            = []
        seq_observation_rgbd  = []
        seq_observation_grasp = []
        for t, (action, observation) \
            in enumerate(zip(trajectory_action[0:input_trajectory_length], 
                             trajectory_observation[0:input_trajectory_length])):
            # Action data
            action_token = tokenize_action(
                action_type            = action[0],
                action_is_target_bool  = action[1],
                action_pos             = action[2],
                action_orn_e           = action[3],
                action_dyaw            = action[4],
                action_type_encoding   = self.config.action_type_encoding,
                action_target_encoding = self.config.action_target_encoding)
            # Observation data
            observation_depth_image, observation_rgb_image, observation_grasp = observation
            observation_rgbd = tokenize_masked_rgbd(observation_depth_image, observation_rgb_image)
            observation_grasp = np.expand_dims(observation_grasp, axis=-1)
            # Collect (V-Value)
            seq_action.append(action_token)
            seq_observation_rgbd.append(observation_rgbd)
            seq_observation_grasp.append(observation_grasp)
        # Gather
        seq_action = np.stack(seq_action, axis=0)
        seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
        seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
        input_trajectory_length = np.asarray([input_trajectory_length])
        full_trajectory_length  = np.asarray([trajectory_length])


        # Make label 
        trajectory_reward = list(trajectory_reward) + [[0]]   # 0 means the terminal reward
        future_discounted_reward_label \
            = get_future_discounted_reward(future_rewards = trajectory_reward[input_trajectory_length.item():], 
                                           normalize      = True)  
        success_or_fail = (json_data["termination"] == "success")
        
        
        # Tensor (make sure to float())
        init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
        init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
        goal                    = torch.from_numpy(goal).float()
        seq_action              = torch.from_numpy(seq_action).float()
        seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
        seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
        input_trajectory_length = torch.from_numpy(input_trajectory_length).long()
        full_trajectory_length  = torch.from_numpy(full_trajectory_length).long()
        future_discounted_reward_label = torch.from_numpy(future_discounted_reward_label).float()
        success_or_fail         = torch.Tensor([success_or_fail]).bool()

        # Time step... (input_trajectory_length idx starts from 0)
        num_paddings = self.config.seq_length - input_trajectory_length
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
        # Create sequence mask (all filled with 1 and padded with 0.)
        action_mask      = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
        observation_mask = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)


        return (init_observation_rgbd, init_observation_grasp, goal, 
                seq_action, seq_observation_rgbd, seq_observation_grasp, 
                action_mask, observation_mask,
                input_trajectory_length, full_trajectory_length, 
                future_discounted_reward_label, success_or_fail)





class FetchingQValueDataset(Dataset):


    def __init__(self, config: FetchingDatasetConfig,
                       data_path_json: str,
                       data_path_npz: str,
                       annotations: List[str],
                       num_subdata: List[int],
                       no_rollout_input: bool):
        """ Q-Value dataset.
        NOTE(ssh): 
            Q-value and V-value have slightly different timestep slicing.
            It is quite tricky.

        Args:
            config (FetchingDatasetConfig): Configuration class
            data_path_json (str): Json dir path
            data_path_npz (str): Numpy dir path
            annotations (List[str]): List of data file name
            num_subdata (List[int]): Number of subdata in trajectory
            no_rollout_input (bool): Exclude rollout trajectory to the input.
        """
        self.config = config
        self.data_path_json = data_path_json
        self.data_path_npz = data_path_npz
        self.filenames = annotations
        self.weights = (np.asarray(num_subdata)/np.sum(num_subdata)).tolist()
        self.no_rollout_input = no_rollout_input



    def __len__(self):
        return len(self.filenames)



    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            init_observation_grasp         (torch.Tensor): shape=(1)
            goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            seq_action                     (torch.Tensor): shape=(full_trajectory_length, A)
            seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads. 
            observation_mask               (torch.Tensor): shape=(full_trajectory_length-1) Filled with all ones and zeropads. 
            input_trajectory_length        (torch.Tensor): shape=(1)
            full_trajectory_length         (torch.Tensor): shape=(1)
            future_discounted_reward_label (torch.Tensor): shape=(1) NORMALIZED
            success_or_fail                (torch.Tensor): shape=(1)
        """

        # Loading json and numpy
        json_file_name = os.path.join(self.data_path_json, f"{self.filenames[index]}.json")
        npz_file_name = os.path.join(self.data_path_npz, f"{self.filenames[index]}.npz")
        with open(json_file_name, "r") as f:
            json_data = json.load(f)
        npz_data = np.load(npz_file_name)

        # Read trajectory
        init_observation, goal_condition, termination, \
            trajectory_action, trajectory_observation, trajectory_reward,\
            trajectory_length, \
            num_real_exec, num_sim_exec, num_rollout_exec \
                = read_trajectory_from_json_and_numpy(json_data, npz_data)

        # Randomly select the time step.
        # NOTE(ssh): 
        #   Q-Value can also be defined with a full trajectory
        #   as observation at T is not included in trajectory and it can vary.
        #   Thus, the max length can be a full trajectory_length.
        if trajectory_length > 1:
            if self.no_rollout_input:
                input_trajectory_length = np.random.randint(1, num_real_exec+num_sim_exec+1)
            else:
                input_trajectory_length = np.random.randint(1, trajectory_length+1) # +1 here!
        else:
            input_trajectory_length = 1
            

        # Tokenizing
        #   Initial observation
        init_observation_depth, init_observation_rgb, init_observation_grasp = init_observation
        init_observation_rgbd = tokenize_masked_rgbd(init_observation_depth, init_observation_rgb)
        init_observation_grasp = np.expand_dims(init_observation_grasp, axis=-1)

        #   Goal condition
        goal = np.array(goal_condition)
        #   Process the trajectory from time 0 to input_trajectory_length-1
        seq_action            = []
        seq_observation_rgbd  = []
        seq_observation_grasp = []
        for t, (action, observation) \
            in enumerate(zip(trajectory_action[0:input_trajectory_length], 
                             trajectory_observation[0:input_trajectory_length])):
            # Action data
            action_token = tokenize_action(
                action_type            = action[0],
                action_is_target_bool  = action[1],
                action_pos             = action[2],
                action_orn_e           = action[3],
                action_dyaw            = action[4],
                action_type_encoding   = self.config.action_type_encoding,
                action_target_encoding = self.config.action_target_encoding)
            # Observation data
            observation_depth_image, observation_rgb_image, observation_grasp = observation
            observation_rgbd = tokenize_masked_rgbd(observation_depth_image, observation_rgb_image)
            observation_grasp = np.expand_dims(observation_grasp, axis=-1)
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
        full_trajectory_length  = np.asarray([trajectory_length])


        # Make label 
        # NOTE(ssh): V and Q values are the expectation of the future rewards INCLUDING the immediate reward.
        # For Q value, process reward trajectory from time input_trajectory_length-1 to full length.
        future_discounted_reward_label \
            = get_future_discounted_reward(future_rewards = trajectory_reward[input_trajectory_length.item()-1:], 
                                           normalize      = True)            
        success_or_fail = (json_data["termination"] == "success")
        
        
        # Tensor (make sure to float())
        init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
        init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
        goal                    = torch.from_numpy(goal).float()
        seq_action              = torch.from_numpy(seq_action).float()
        seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
        seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
        input_trajectory_length = torch.from_numpy(input_trajectory_length).int()
        full_trajectory_length  = torch.from_numpy(full_trajectory_length)
        future_discounted_reward_label = torch.from_numpy(future_discounted_reward_label).float()
        success_or_fail         = torch.Tensor([success_or_fail]).bool()

        # Time step... (input_trajectory_length idx starts from 0)
        num_paddings = self.config.seq_length - input_trajectory_length
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
        action_mask      = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
        observation_mask = torch.cat((torch.ones(input_trajectory_length-1), torch.zeros(num_paddings+1)), dim=-1)  # +1 for Q!

        return (init_observation_rgbd, init_observation_grasp, goal, 
                seq_action, seq_observation_rgbd, seq_observation_grasp, 
                action_mask, observation_mask,
                input_trajectory_length, full_trajectory_length, 
                future_discounted_reward_label, success_or_fail)




class FetchingPreferenceDataset(Dataset):

    def __init__(self, success_dataset: Union[FetchingVValueDataset, FetchingQValueDataset],
                       comparison_dataset: Union[FetchingVValueDataset, FetchingQValueDataset],
                       augment_success: bool = False):
        """Preference dataset"""
        self.success_dataset = success_dataset
        self.comparison_dataset = comparison_dataset
        self.weights = self.success_dataset.weights
        self.augment_success = augment_success


    def __len__(self):
        """
        This returns only the length of success trajectory for dataloading purpose.
        The true dataset size is combinatorial (i.e. len(success_dataset)*len(comparison_dataset) )
        """
        return len(self.success_dataset)


    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            success_init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            success_init_observation_grasp         (torch.Tensor): shape=(1)
            success_goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            success_seq_action                     (torch.Tensor): shape=(full_trajectory_length, 8)
            success_seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            success_seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            success_action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads. 
            success_observation_mask               (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            success_input_trajectory_length        (torch.Tensor): shape=(1)
            success_full_trajectory_length         (torch.Tensor): shape=(1)
            success_future_discounted_reward_label (torch.Tensor): shape=(1)
            success_success_or_fail                (torch.Tensor): shape=(1)
            
            comparison_init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            comparison_init_observation_grasp         (torch.Tensor): shape=(1)
            comparison_goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            comparison_seq_action                     (torch.Tensor): shape=(full_trajectory_length, 5)
            comparison_seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            comparison_seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            comparison_action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads. 
            comparison_observation_mask               (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            comparison_input_trajectory_length        (torch.Tensor): shape=(1)
            comparison_full_trajectory_length         (torch.Tensor): shape=(1)
            comparison_future_discounted_reward_label (torch.Tensor): shape=(1)
            comparison_success_or_fail                (torch.Tensor): shape=(1)

            preference (torch.Tensor): shape=(2)
            is_equal   (torch.Tensor): shape=(1) Mask that indicates which indices are [0.5, 0.5].
        """

        # Success data
        success_init_observation_rgbd, success_init_observation_grasp, success_goal, \
            success_seq_action, success_seq_observation_rgbd, success_seq_observation_grasp, \
            success_action_mask, success_observation_mask, \
            success_input_trajectory_length, success_full_trajectory_length, \
            success_future_discounted_reward_label, success_success_or_fail \
                = self.success_dataset[index]

        # Sampling comparison data
        #   Draw from the success set by 50% if augment.
        select_from_success = random.choice([True, False])
        if self.augment_success and select_from_success:
                comparison_idx = random.randint(0, len(self.success_dataset)-1)  # Uniform
                comparison_init_observation_rgbd, comparison_init_observation_grasp, comparison_goal, \
                    comparison_seq_action, comparison_seq_observation_rgbd, comparison_seq_observation_grasp, \
                    comparison_action_mask, comparison_observation_mask, \
                    comparison_input_trajectory_length, comparison_full_trajectory_length, \
                    comparison_future_discounted_reward_label, comparison_success_or_fail \
                        = self.success_dataset[comparison_idx]
        #   Just draw from the fail set otherwise
        else: 
            comparison_idx = random.randint(0, len(self.comparison_dataset)-1)  # Uniform
            comparison_init_observation_rgbd, comparison_init_observation_grasp, comparison_goal, \
                comparison_seq_action, comparison_seq_observation_rgbd, comparison_seq_observation_grasp, \
                comparison_action_mask, comparison_observation_mask, \
                comparison_input_trajectory_length, comparison_full_trajectory_length, \
                comparison_future_discounted_reward_label, comparison_success_or_fail \
                    = self.comparison_dataset[comparison_idx]
            
        # Make preference label
        preference, is_equal = check_preference(success_success_or_fail, comparison_success_or_fail,
                                      success_full_trajectory_length, comparison_full_trajectory_length,
                                      success_input_trajectory_length, comparison_input_trajectory_length)

        return (success_init_observation_rgbd,
                success_init_observation_grasp,
                success_goal,
                success_seq_action,
                success_seq_observation_rgbd,
                success_seq_observation_grasp,
                success_action_mask,
                success_observation_mask,
                success_input_trajectory_length,
                success_full_trajectory_length,
                success_future_discounted_reward_label,
                success_success_or_fail,
                comparison_init_observation_rgbd,
                comparison_init_observation_grasp,
                comparison_goal,
                comparison_seq_action,
                comparison_seq_observation_rgbd,
                comparison_seq_observation_grasp,
                comparison_action_mask,
                comparison_observation_mask,
                comparison_input_trajectory_length,
                comparison_full_trajectory_length,
                comparison_future_discounted_reward_label,
                comparison_success_or_fail,
                preference,
                is_equal)
        
        
        
class FetchingStudentQValueDataset(Dataset):


    def __init__(self, config: FetchingDatasetConfig,
                       data_path_json: str,
                       data_path_npz: str,
                       annotations: List[str],
                       num_subdata: List[int]):
        """V-Value dataset

        Args:
            config (FetchingDatasetConfig): Configuration class
            data_path_json (str): Json dir path
            data_path_npz (str): Numpy dir path
            annotations (List[str]): List of data file name
            num_subdata (List[int]): Number of subdata in trajectory
        """
        self.config = config
        self.data_path_json = data_path_json
        self.data_path_npz = data_path_npz
        self.filenames = annotations
        self.weights = (np.asarray(num_subdata)/np.sum(num_subdata)).tolist()


    def __len__(self):
        return len(self.filenames)



    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            init_observation_grasp         (torch.Tensor): shape=(1)
            goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            seq_action                     (torch.Tensor): shape=(full_trajectory_length, A)
            seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            observation_mask               (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            input_trajectory_length        (torch.Tensor): shape=(1)
            full_trajectory_length         (torch.Tensor): shape=(1)
            future_discounted_reward_label (torch.Tensor): shape=(1) NORMALIZED
            success_or_fail                (torch.Tensor): shape=(1)
        """

        # Loading json and numpy
        json_file_name = os.path.join(self.data_path_json, f"{self.filenames[index]}.json")
        npz_file_name = os.path.join(self.data_path_npz, f"{self.filenames[index]}.npz")
        with open(json_file_name, "r") as f:
            json_data = json.load(f)
        npz_data = np.load(npz_file_name)

        # Initial observation
        init_observation  = (npz_data["init_observation_depth"], 
                             npz_data["init_observation_rgb"], 
                             npz_data["init_observation_grasp"])
        # Goal condition
        goal_condition    = json_data["goal_condition"]
        # Action trajectory
        num_real_exec     = len(json_data["exec_action"])
        num_sim_exec      = len(json_data["sim_action"])
        num_rollout_exec  = len(json_data["rollout_action"])
        trajectory_length = num_real_exec + num_sim_exec + num_rollout_exec
        trajectory_action = json_data["exec_action"]+json_data["sim_action"]+json_data["rollout_action"]
        # Observation trajectory
        trajectory_observation = []
        for i in range(num_real_exec):
            entry = (npz_data[f"exec_observation_{i}_depth"], 
                     npz_data[f"exec_observation_{i}_rgb"], 
                     npz_data[f"exec_observation_{i}_grasp"])
            trajectory_observation.append(entry)
        for i in range(num_sim_exec):
            entry = (npz_data[f"sim_observation_{i}_depth"], 
                     npz_data[f"sim_observation_{i}_rgb"], 
                     npz_data[f"sim_observation_{i}_grasp"])
            trajectory_observation.append(entry)
        for i in range(num_rollout_exec):
            entry = (npz_data[f"rollout_observation_{i}_depth"], 
                     npz_data[f"rollout_observation_{i}_rgb"], 
                     npz_data[f"rollout_observation_{i}_grasp"])
            trajectory_observation.append(entry)
        # Reward trajectory
        trajectory_reward = json_data["exec_reward"]+json_data["sim_reward"]+json_data["rollout_reward"]

        # Randomly select the time step.
        if trajectory_length > 1:
            input_trajectory_length = np.random.randint(1, trajectory_length+1)
        else:
            input_trajectory_length = 1
            

        # Tokenizing
        #   Initial observation
        init_observation_depth, init_observation_rgb, init_observation_grasp = init_observation
        init_observation_rgbd = tokenize_masked_rgbd(init_observation_depth, init_observation_rgb)
        init_observation_grasp = np.expand_dims(init_observation_grasp, axis=-1)

        #   Goal condition
        goal = np.array(goal_condition)
        #   Process the trajectory from time 0 to input_trajectory_length-1
        seq_action            = []
        seq_observation_rgbd  = []
        seq_observation_grasp = []
        for t, (action, observation) \
            in enumerate(zip(trajectory_action[0:input_trajectory_length-1], 
                             trajectory_observation[0:input_trajectory_length-1])):
            # Action data
            action_token = tokenize_action(
                action_type            = action[0],
                action_is_target_bool  = action[1],
                action_pos             = action[2],
                action_orn_e           = action[3],
                action_dyaw            = action[4],
                action_type_encoding   = self.config.action_type_encoding,
                action_target_encoding = self.config.action_target_encoding)
            # Observation data
            observation_depth_image, observation_rgb_image, observation_grasp = observation
            observation_rgbd = tokenize_masked_rgbd(observation_depth_image, observation_rgb_image)
            observation_grasp = np.expand_dims(observation_grasp, axis=-1)
            # Collect
            seq_action.append(action_token)
            seq_observation_rgbd.append(observation_rgbd)
            seq_observation_grasp.append(observation_grasp)
        
        # For last sampled time-step
        action = trajectory_action[input_trajectory_length-1]
        observation = trajectory_observation[input_trajectory_length-1]
        # Action data
        action_token = tokenize_action(
            action_type            = action[0],
            action_is_target_bool  = action[1],
            action_pos             = action[2],
            action_orn_e           = action[3],
            action_dyaw            = action[4],
            action_type_encoding   = self.config.action_type_encoding,
            action_target_encoding = self.config.action_target_encoding)
        # Observation data
        observation_depth_image, observation_rgb_image, observation_grasp = observation
        observation_rgbd = tokenize_masked_rgbd(observation_depth_image, observation_rgb_image)
        observation_grasp = np.expand_dims(observation_grasp, axis=-1)
        # Collect
        seq_action.append(action_token)
        seq_observation_rgbd_for_label = deepcopy(seq_observation_rgbd)
        seq_observation_rgbd_for_label.append(observation_rgbd)
        seq_observation_grasp_for_label = deepcopy(seq_observation_grasp)
        seq_observation_grasp_for_label.append(observation_grasp)
        
        # Gather
        seq_action = np.stack(seq_action, axis=0)
        if len(seq_observation_grasp) != 0:
            seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
            seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
        else:
            # Handling zero sequence when training Q at time step 1.
            seq_observation_rgbd = np.zeros(shape=(0, *init_observation_rgbd.shape))
            seq_observation_grasp = np.zeros(shape=(0, *init_observation_grasp.shape))
        seq_observation_rgbd_for_label  = np.stack(seq_observation_rgbd_for_label, axis=0)
        seq_observation_grasp_for_label = np.stack(seq_observation_grasp_for_label, axis=0)
        input_trajectory_length = np.asarray([input_trajectory_length])
        full_trajectory_length  = np.asarray([trajectory_length])
        
        # Tensor (make sure to float())
        init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
        init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
        goal                    = torch.from_numpy(goal).float()
        seq_action              = torch.from_numpy(seq_action).float()
        seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
        seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
        seq_observation_rgbd_for_label    = torch.from_numpy(seq_observation_rgbd_for_label).float()
        seq_observation_grasp_for_label   = torch.from_numpy(seq_observation_grasp_for_label).float()
        input_trajectory_length = torch.from_numpy(input_trajectory_length).long()
        full_trajectory_length  = torch.from_numpy(full_trajectory_length).long()

        # Time step... (input_trajectory_length idx starts from 0)
        num_paddings = self.config.seq_length - input_trajectory_length
        # Make padding and concatenate
        seq_action            = add_zero_pre_padding(seq_variable = seq_action, 
                                                     fill_like    = seq_action[0], 
                                                     num_paddings = num_paddings)
        seq_observation_rgbd  = add_zero_pre_padding(seq_variable = seq_observation_rgbd, 
                                                    fill_like    = init_observation_rgbd, 
                                                    num_paddings = num_paddings+1)     # +1 for Q
        seq_observation_grasp = add_zero_pre_padding(seq_variable = seq_observation_grasp,
                                                    fill_like    = init_observation_grasp,
                                                    num_paddings = num_paddings+1)     # +1 for Q
        seq_observation_rgbd_for_label  = add_zero_pre_padding(seq_variable = seq_observation_rgbd_for_label, 
                                                     fill_like    = init_observation_rgbd, 
                                                     num_paddings = num_paddings)
        seq_observation_grasp_for_label = add_zero_pre_padding(seq_variable = seq_observation_grasp_for_label,
                                                     fill_like    = init_observation_grasp,
                                                     num_paddings = num_paddings)
        # Create sequence mask (all filled with 1 and padded with 0.)
        action_mask      = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
        observation_mask = torch.cat((torch.ones(input_trajectory_length-1), torch.zeros(num_paddings+1)), dim=-1)
        observation_mask_for_label = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
        

        return (init_observation_rgbd, init_observation_grasp, goal, 
                seq_action, seq_observation_rgbd, seq_observation_grasp, seq_observation_rgbd_for_label, seq_observation_grasp_for_label,
                action_mask, observation_mask, observation_mask_for_label,
                input_trajectory_length, full_trajectory_length)



def check_preference(node1_success_or_fail: torch.Tensor,
                     node2_success_or_fail: torch.Tensor,
                     node1_full_trajectory_length: torch.Tensor,
                     node2_full_trajectory_length: torch.Tensor,
                     node1_input_trajectory_length: torch.Tensor,
                     node2_input_trajectory_length: torch.Tensor) -> Tuple[ torch.Tensor, torch.Tensor ] :

    """Get preference label

    Args:
        node1_success_or_fail (torch.Tensor)
        node2_success_or_fail (torch.Tensor)
        node1_full_trajectory_length (torch.Tensor)
        node2_full_trajectory_length (torch.Tensor)
        node1_input_trajectory_length (torch.Tensor)
        node2_input_trajectory_length (torch.Tensor)

    Returns:
        preference (torch.Tensor): [1, 0], [0, 1], [0.5, 0.5] 
        is_equal (torch.Tensor): [bool]
    """
    node1_success = node1_success_or_fail.item()
    node2_success = node2_success_or_fail.item()
    node1_steps_to_goal = node1_full_trajectory_length.item() - node1_input_trajectory_length.item()
    node2_steps_to_goal = node2_full_trajectory_length.item() - node2_input_trajectory_length.item()

    # Rule 1: Always prefer success
    if node1_success != node2_success:
        return torch.Tensor([node1_success, node2_success]), torch.Tensor([False]).bool()
    elif node1_success:
        # Rule 2: Prefer shorter trajectory
        if node1_steps_to_goal > node2_steps_to_goal:
            return torch.Tensor([0, 1]), torch.Tensor([False]).bool()
        elif node1_steps_to_goal < node2_steps_to_goal:
            return torch.Tensor([1, 0]), torch.Tensor([False]).bool()
        # Rule 3: Set equal otherwise.
        else:
            return torch.Tensor([0.5, 0.5]), torch.Tensor([True]).bool()
    else:
        raise ValueError("At least one node must be in successful trajectory")


def check_preference_count_picks_only(node1_success_or_fail: torch.Tensor,
                                      node2_success_or_fail: torch.Tensor,
                                      node1_num_future_picks: torch.Tensor,
                                      node2_num_future_picks: torch.Tensor,) -> Tuple[ torch.Tensor, torch.Tensor ] :
    """Get preference label

    Args:
        node1_success_or_fail (torch.Tensor)
        node2_success_or_fail (torch.Tensor)
        node1_num_future_picks (torch.Tensor)
        node2_num_future_picks (torch.Tensor)

    Returns:
        preference (torch.Tensor): [1, 0], [0, 1], [0.5, 0.5] 
        is_equal (torch.Tensor): [bool]
    """
    node1_success = node1_success_or_fail.item()
    node2_success = node2_success_or_fail.item()
    node1_steps_to_goal = node1_num_future_picks.item()
    node2_steps_to_goal = node2_num_future_picks.item()

    # Rule 1: Always prefer success
    if node1_success != node2_success:
        return torch.Tensor([node1_success, node2_success]), torch.Tensor([False]).bool()
    elif node1_success:
        # Rule 2: Prefer shorter trajectory
        if node1_steps_to_goal > node2_steps_to_goal:
            return torch.Tensor([0, 1]), torch.Tensor([False]).bool()
        elif node1_steps_to_goal < node2_steps_to_goal:
            return torch.Tensor([1, 0]), torch.Tensor([False]).bool()
        # Rule 3: Set equal otherwise.
        else:
            return torch.Tensor([0.5, 0.5]), torch.Tensor([True]).bool()
    else:
        raise ValueError("At least one node must be in successful trajectory") 




def get_future_discounted_reward(future_rewards: List[List[float]], 
                                 normalize: bool) -> np.ndarray:
    """
    Args:
        future_rewards (List[List[float]]): List of future rewards
        normalize (bool): Normalize the reward when True.
    """
    DISCOUNT_FACTOR = 1.0
    future_discounted_reward = 0

    for i, r in enumerate(future_rewards):
        future_discounted_reward += r[0] * (DISCOUNT_FACTOR ** i)
    
    if normalize:
        return np.array([future_discounted_reward/100.])
    else:
        return np.array([future_discounted_reward])
    


class FetchingSuccessFailDataset(Dataset):

    def __init__(self, success_dataset: Union[FetchingVValueDataset, FetchingQValueDataset],
                       comparison_dataset: Union[FetchingVValueDataset, FetchingQValueDataset]):
        """SuccessFail dataset"""
        self.success_dataset = success_dataset
        self.comparison_dataset = comparison_dataset
        self.weights = self.success_dataset.weights


    def __len__(self):
        """
        This returns only the length of success trajectory for dataloading purpose.
        The true dataset size is combinatorial (i.e. len(success_dataset)*len(comparison_dataset) )
        """
        return len(self.success_dataset)


    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            success_init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            success_init_observation_grasp         (torch.Tensor): shape=(1)
            success_goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            success_seq_action                     (torch.Tensor): shape=(full_trajectory_length, 8)
            success_seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            success_seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            success_action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads. 
            success_observation_mask               (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            success_input_trajectory_length        (torch.Tensor): shape=(1)
            success_full_trajectory_length         (torch.Tensor): shape=(1)
            success_future_discounted_reward_label (torch.Tensor): shape=(1)
            success_success_or_fail                (torch.Tensor): shape=(1)
            
            comparison_init_observation_rgbd          (torch.Tensor): shape=(C, H, W)
            comparison_init_observation_grasp         (torch.Tensor): shape=(1)
            comparison_goal                           (torch.Tensor): shape=(rgbxyz)=(6)
            comparison_seq_action                     (torch.Tensor): shape=(full_trajectory_length, 5)
            comparison_seq_observation_rgbd           (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            comparison_seq_observation_grasp          (torch.Tensor): shape=(full_trajectory_length, 1)
            comparison_action_mask                    (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads. 
            comparison_observation_mask               (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            comparison_input_trajectory_length        (torch.Tensor): shape=(1)
            comparison_full_trajectory_length         (torch.Tensor): shape=(1)
            comparison_future_discounted_reward_label (torch.Tensor): shape=(1)
            comparison_success_or_fail                (torch.Tensor): shape=(1)

            preference (torch.Tensor): shape=(2)
            is_equal   (torch.Tensor): shape=(1) Mask that indicates which indices are [0.5, 0.5].
        """

        # Success data
        success_init_observation_rgbd, success_init_observation_grasp, success_goal, \
            success_seq_action, success_seq_observation_rgbd, success_seq_observation_grasp, \
            success_action_mask, success_observation_mask, \
            success_input_trajectory_length, success_full_trajectory_length, \
            success_future_discounted_reward_label, success_success_or_fail \
                = self.success_dataset[index]

        # Sampling comparison data (Fail data)
        comparison_idx = random.randint(0, len(self.comparison_dataset)-1)  # Uniform
        # comparison_idx = random.choices(                                    # Weighted sampling is super slow.
        #     range(0, len(self.comparison_dataset)), 
        #     self.comparison_dataset.weights)[0]
        comparison_init_observation_rgbd, comparison_init_observation_grasp, comparison_goal, \
            comparison_seq_action, comparison_seq_observation_rgbd, comparison_seq_observation_grasp, \
            comparison_action_mask, comparison_observation_mask, \
            comparison_input_trajectory_length, comparison_full_trajectory_length, \
            comparison_future_discounted_reward_label, comparison_success_or_fail \
                = self.comparison_dataset[comparison_idx]

        # Make preference label
        preference, is_equal = check_preference(success_success_or_fail, comparison_success_or_fail,
                                      success_full_trajectory_length, comparison_full_trajectory_length,
                                      success_input_trajectory_length, comparison_input_trajectory_length)

        return (success_init_observation_rgbd,
                success_init_observation_grasp,
                success_goal,
                success_seq_action,
                success_seq_observation_rgbd,
                success_seq_observation_grasp,
                success_action_mask,
                success_observation_mask,
                success_input_trajectory_length,
                success_full_trajectory_length,
                success_future_discounted_reward_label,
                success_success_or_fail,
                comparison_init_observation_rgbd,
                comparison_init_observation_grasp,
                comparison_goal,
                comparison_seq_action,
                comparison_seq_observation_rgbd,
                comparison_seq_observation_grasp,
                comparison_action_mask,
                comparison_observation_mask,
                comparison_input_trajectory_length,
                comparison_full_trajectory_length,
                comparison_future_discounted_reward_label,
                comparison_success_or_fail,
                preference,
                is_equal)
    


def count_num_future_pick(trajectory_action: List, input_trajectory_length: int):
    """Count the number of future picks in the trajectory"""
    
    if input_trajectory_length == len(trajectory_action):
        return 0
    
    num_future_picks = 0
    for action in trajectory_action[input_trajectory_length:]:
        action_type = action[0]

        # Validation
        if not (action_type=="PICK" or action_type=="PLACE"):
            raise TypeError("Invalid action type")
        # Count num picks
        if action_type == "PICK":
            num_future_picks += 1
    
    return num_future_picks

    

        
