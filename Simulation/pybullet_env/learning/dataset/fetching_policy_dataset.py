import os
import json
import numpy as np
import numpy.typing as npt
import torch

from typing import List, Dict, Tuple, Union

import shutil

from dataclasses import dataclass, field
from simple_parsing import Serializable
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
from copy import deepcopy
from math import pi
import yaml

from learning.model.value import HistoryPlaceValueonly
from learning.dataset.common import ( format_next_action_label,
                                      tokenize_action, 
                                      tokenize_masked_rgbd,
                                      add_zero_pre_padding,
                                      FetchingDatasetConfig,
                                      FetchingPickPlaceDatasetConfig )


class FetchingPolicyDataset(Dataset):


    def __init__(self, config: FetchingDatasetConfig,
                       data_path_json: str,
                       data_path_npz: str,
                       filenames: List[str],
                       num_subdata: List[int]):
        """Fetching filename class
        Args:
            config (FetchingDatasetConfig): Configuration class
            data_path_json (str): Data path
            data_path_npz (str): Data path
            filenames (List): List of data file name
            num_subdata (List): List of subdata in each file
        """
        self.config = config
        self.data_path_json = data_path_json
        self.data_path_npz = data_path_npz
        self.filenames = filenames
        self.weights = (np.asarray(num_subdata)/np.sum(num_subdata)).tolist()


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            init_observation_rgbd   (torch.Tensor): shape=(C, H, W)
            init_observation_grasp  (torch.Tensor): shape=(1)
            goal                    (torch.Tensor): shape=(rgbxyz)=(6)
            seq_action              (torch.Tensor): shape=(full_trajectory_length, A)
            seq_observation_rgbd    (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            seq_observation_grasp   (torch.Tensor): shape=(full_trajectory_length, 1)
            action_mask             (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            observation_mask        (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            input_trajectory_length (torch.Tensor): shape=(1)
            full_trajectory_length  (torch.Tensor): shape=(1)
            next_action_label       (torch.Tensor): shape=(3)
            next_action_token       (torch.Tensor): Input to the Q-predictor. shape=(A)
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

        # Randomly select the time step. (action at time_step+1 will be the label)
        #   NOTE(ssh): Policy and value net have different time slicing scheme.
        if not validate_place_existance(trajectory_action):
            raise ValueError("PLACE action does not exist in the trajectory.")
        
        while True:
            i = np.random.randint(0, trajectory_length)
            # Parse action
            action = trajectory_action[i]
            action_type = action[0]
            action_pos = action[2]

            # Select place
            if action_type == "PLACE" and action_pos is not None:
                input_trajectory_length = i   # NOTE(ssh): Always double check index.
                break

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
        for action, observation \
            in zip(trajectory_action[0:input_trajectory_length], 
                   trajectory_observation[0:input_trajectory_length]):
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
        # Gather
        seq_action            = np.stack(seq_action, axis=0)
        seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
        seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
        input_trajectory_length = np.asarray([input_trajectory_length])
        full_trajectory_length  = np.asarray([trajectory_length])


        # Make label
        next_action = trajectory_action[input_trajectory_length.item()]
        next_action_label = format_next_action_label(next_action)
        next_action_token = tokenize_action(
                action_type            = next_action[0],
                action_is_target_bool  = next_action[1],
                action_pos             = next_action[2],
                action_orn_e           = next_action[3],
                action_dyaw            = next_action[4],
                action_type_encoding   = self.config.action_type_encoding,
                action_target_encoding = self.config.action_target_encoding)
        

        # Tensor (make sure to float())
        init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
        init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
        goal                    = torch.from_numpy(goal).float()
        seq_action              = torch.from_numpy(seq_action).float()
        seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
        seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
        input_trajectory_length = torch.from_numpy(input_trajectory_length).long()
        full_trajectory_length  = torch.from_numpy(full_trajectory_length).long()
        next_action_label       = torch.from_numpy(next_action_label).float()
        next_action_token       = torch.from_numpy(next_action_token).float()

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
                next_action_label, next_action_token)




class FetchingPreferencePolicyDataset(FetchingPolicyDataset):


    def __init__(self, config: FetchingDatasetConfig,
                       data_path_json: str,
                       data_path_npz: str,
                       filenames: List[str],
                       num_subdata: List[int],
                       sim_config: Dict,
                       temperature: float,
                       num_generate_action: int):
        """Dataset for policy preference.
        Args:
            config (Config): Configuration class
            data_path_json (str): Data path
            data_path_npz (str): Data path
            filenames (List): List of data file name
            num_subdata (List): List of subdata in each file
            sim_config (Dict): Config path to simuation (yaml file)
            evaluator (HistoryPlaceValueonly): Q function model for evaluating generated action
            temperature (float): Softmax temperature
            num_generate_action (int): Number of actions to be generated
        """
        super().__init__(config, data_path_json, data_path_npz, filenames, num_subdata)
        self.temperature = temperature
        self.num_gen_action = num_generate_action
        self.sim_config = sim_config


    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item
        Args:
            index (int): Data index
        Returns:
            init_observation_rgbd   (torch.Tensor): shape=(N, C, H, W)
            init_observation_grasp  (torch.Tensor): shape=(N, 1)
            goal                    (torch.Tensor): shape=(N, rgbxyz)=(N, 6)
            seq_action              (torch.Tensor): shape=(N, input_trajectory_length, A)
            seq_observation_rgbd    (torch.Tensor): shape=(N, input_trajectory_length, C, H, W)
            seq_observation_grasp   (torch.Tensor): shape=(N, input_trajectory_length, 1)            
            action_mask             (torch.Tensor): shape=(N, full_trajectory_length) Filled with all ones and zeropads.
            observation_mask        (torch.Tensor): shape=(N, full_trajectory_length) Filled with all ones and zeropads.
            input_trajectory_length (torch.Tensor): shape=(N, 1)
            full_trajectory_length  (torch.Tensor): shape=(N, 1)
            next_action_label       (torch.Tensor): shape=(N, 3)
            q_seq_action            (torch.Tensor): shape=(N, input_trajectory_length, A) next_action is added to the sequence.
            q_action_mask           (torch.Tensor): shape=(N, full_trajectory_length) next_action is added to the sequence.
        """

        # Get data
        init_obs_rgbd, init_obs_grasp, goal, \
            seq_action, seq_obs_rgbd, seq_obs_grasp, \
            action_mask, obs_mask, \
            input_trajectory_length, full_trajectory_length, \
            next_action_label, \
            next_action_token, \
                = super().__getitem__(index)    # V sequence


        # Sample random action
        # NOTE(dr): we will be using real data (next_action_label), too.
        # The total action_batch size is (num_gen_action+1)
        random_action_token, random_action_label \
            = self.sample_random_action(
                gt_next_action = next_action_token,
                num_action     = self.num_gen_action)
        next_action_token = torch.cat([next_action_token[None,], random_action_token], dim=0)    # unsqueeze -> expand -> concat
        next_action_label = torch.cat([next_action_label[None,], random_action_label], dim=0)

        # Expand all!
        init_obs_rgbd           = init_obs_rgbd.tile(dims=(self.num_gen_action+1, 1, 1, 1))
        init_obs_grasp          = init_obs_grasp.tile(dims=(self.num_gen_action+1, 1))
        goal                    = goal.tile(dims=(self.num_gen_action+1, 1)) 
        seq_action              = seq_action.tile(dims=(self.num_gen_action+1, 1, 1))
        seq_obs_rgbd            = seq_obs_rgbd.tile(dims=(self.num_gen_action+1, 1, 1, 1, 1))
        seq_obs_grasp           = seq_obs_grasp.tile(dims=(self.num_gen_action+1, 1, 1))
        action_mask             = action_mask.tile(dims=(self.num_gen_action+1, 1))
        obs_mask                = obs_mask.tile(dims=(self.num_gen_action+1, 1))
        # Also insert new action into the sequence for Q evalator
        q_seq_action = seq_action.clone()
        q_seq_action[:,input_trajectory_length,:] = next_action_token[:,None,:]
        q_action_mask = action_mask.clone()
        q_action_mask[:,input_trajectory_length] = 1
        # Expand length too. This should come after the Q action insertion.
        input_trajectory_length = input_trajectory_length.tile(dims=(self.num_gen_action+1, 1))
        full_trajectory_length  = full_trajectory_length.tile(dims=(self.num_gen_action+1, 1))

        return (init_obs_rgbd, init_obs_grasp, goal,
                seq_action, seq_obs_rgbd, seq_obs_grasp,
                action_mask, obs_mask,
                input_trajectory_length, full_trajectory_length,
                next_action_label, next_action_token,
                q_seq_action, q_action_mask)


    def sample_random_action(self, gt_next_action: torch.Tensor, 
                                   num_action: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate random action from the action space.
        Token (action_type, is_target, action_xy, action_dyaw)=(1,1,2,1)=5
    
        Args:
            gt_next_action (torch.Tensor): Next action token from groundtruth data.
            num_action (int, optional): Number of newly generated action.
        Returns:
            new_action_token (torch.Tensor):
            new_action_label (torch.Tensor):
        """
        # Read simulation config for random action generation
        cabinet_pos = self.sim_config["env_params"]["binpick_env"]["cabinet"]["pos"]
        goal_pos = self.sim_config["env_params"]["binpick_env"]["goal"]["pos"]
        CABINET_POS_X = cabinet_pos[0]
        CABINET_POS_Y = cabinet_pos[1]
        CABINET_GRASP_MARGIN_X = 0.1
        CABINET_GRASP_MARGIN_Y = 0.2
        GOAL_POS_X = goal_pos[0]
        GOAL_POS_Y = goal_pos[1]


        # Newly generated action token
        new_action_token = torch.zeros((num_action, 8))

        # Always guide PLACE action.
        new_action_token[:,:2] = torch.from_numpy(
            np.eye(len(self.config.action_type_encoding.keys()))[self.config.action_type_encoding["PLACE"]])
        # Targer or not is inherited from the previous PICK and groundtruth.
        new_action_token[:,2:5] = gt_next_action[2:5]
        # (Random) Sampling place position
        new_action_token[:,5] \
            = (torch.rand((num_action,))*2-1)*CABINET_GRASP_MARGIN_X + CABINET_POS_X
        new_action_token[:,6] \
            = (torch.rand((num_action,))*2-1)*CABINET_GRASP_MARGIN_Y + CABINET_POS_Y
        #          Override to goal pose
        cabinet_or_goal = torch.multinomial(torch.Tensor([0.8,0.2]), num_action, replacement=True).bool()
        new_action_token[cabinet_or_goal,5] = GOAL_POS_X
        new_action_token[cabinet_or_goal,6] = GOAL_POS_Y
        # (Random) Yaw sampling
        dyaw = (torch.rand((num_action,))-0.5)*pi
        new_action_token[:,7] = dyaw
        
        # Newly generated action label
        new_action_label = torch.zeros((num_action, 3))
        new_action_label[:,:2] = new_action_token[:,5:7]    # XY
        new_action_label[:,2] = new_action_token[:,7]       # DYAW

        return new_action_token, new_action_label
    
    
class FetchingPickPlacePolicyDataset(Dataset):


    def __init__(self, config: FetchingPickPlaceDatasetConfig,
                       data_path_json: str,
                       data_path_npz: str,
                       filenames: List[str],
                       num_subdata: List[int]):
        """Fetching filename class
        Args:
            config (FetchingDatasetConfig): Configuration class
            data_path_json (str): Data path
            data_path_npz (str): Data path
            filenames (List): List of data file name
            num_subdata (List): List of subdata in each file
        """
        self.config = config
        self.data_path_json = data_path_json
        self.data_path_npz = data_path_npz
        self.filenames = filenames
        self.weights = (np.asarray(num_subdata)/np.sum(num_subdata)).tolist()


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index) -> Tuple[torch.Tensor, ...]:
        """Get item

        Args:
            index (int): Data index

        Returns:
            init_observation_rgbd   (torch.Tensor): shape=(C, H, W)
            init_observation_grasp  (torch.Tensor): shape=(1)
            goal                    (torch.Tensor): shape=(rgbxyz)=(6)
            seq_action              (torch.Tensor): shape=(full_trajectory_length, A)
            seq_observation_rgbd    (torch.Tensor): shape=(full_trajectory_length, C, H, W)
            seq_observation_grasp   (torch.Tensor): shape=(full_trajectory_length, 1)
            action_mask             (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            observation_mask        (torch.Tensor): shape=(full_trajectory_length) Filled with all ones and zeropads.
            input_trajectory_length (torch.Tensor): shape=(1)
            full_trajectory_length  (torch.Tensor): shape=(1)
            next_action_label       (torch.Tensor): shape=(3)
            next_action_token       (torch.Tensor): Input to the Q-predictor. shape=(A)
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

        # Randomly select the time step. (action at time_step+1 will be the label)
        #   NOTE(ssh): Policy and value net have different time slicing scheme.
        i = np.random.randint(0, trajectory_length)
        # Parse action
        action = trajectory_action[i]
        action_type = action[0]
        input_trajectory_length = i   # NOTE(ssh): Always double check index.

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
        for action, observation \
            in zip(trajectory_action[0:input_trajectory_length], 
                   trajectory_observation[0:input_trajectory_length]):
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
        # Gather
        if len(seq_action) !=0 :
            seq_action            = np.stack(seq_action, axis=0)
            seq_observation_rgbd  = np.stack(seq_observation_rgbd, axis=0)
            seq_observation_grasp = np.stack(seq_observation_grasp, axis=0)
        input_trajectory_length = np.asarray([input_trajectory_length])
        full_trajectory_length  = np.asarray([trajectory_length])


        # Make label
        next_action = trajectory_action[input_trajectory_length.item()]
        next_action_label = format_next_action_label(next_action, only_place=False)
        next_action_token = tokenize_action(
                action_type            = next_action[0],
                action_is_target_bool  = next_action[1],
                action_pos             = next_action[2],
                action_orn_e           = next_action[3],
                action_dyaw            = next_action[4],
                action_type_encoding   = self.config.action_type_encoding,
                action_target_encoding = self.config.action_target_encoding)
        

        # Tensor (make sure to float())
        init_observation_rgbd   = torch.from_numpy(init_observation_rgbd).float()
        init_observation_grasp  = torch.from_numpy(init_observation_grasp).float()
        goal                    = torch.from_numpy(goal).float()
        if len(seq_action) !=0 :
            seq_action              = torch.from_numpy(seq_action).float()
            seq_observation_rgbd    = torch.from_numpy(seq_observation_rgbd).float()
            seq_observation_grasp   = torch.from_numpy(seq_observation_grasp).float()
        input_trajectory_length = torch.from_numpy(input_trajectory_length).long()
        full_trajectory_length  = torch.from_numpy(full_trajectory_length).long()
        next_action_label       = torch.from_numpy(next_action_label).float()
        next_action_token       = torch.from_numpy(next_action_token).float()

        # Time step... (input_trajectory_length idx starts from 0)
        num_paddings = self.config.seq_length - input_trajectory_length
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
            seq_action = torch.zeros(self.config.seq_length, self.config.dim_action_input)
            obs_rgbd_padding = torch.zeros_like(init_observation_rgbd)
            seq_observation_rgbd = obs_rgbd_padding.expand(self.config.seq_length, *obs_rgbd_padding.shape)
            obs_grasp_padding = torch.zeros_like(init_observation_grasp)
            seq_observation_grasp = obs_grasp_padding.expand(self.config.seq_length, *obs_grasp_padding.shape)

        # Create sequence mask (all filled with 1 and padded with 0.)
        action_mask      = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)
        observation_mask = torch.cat((torch.ones(input_trajectory_length), torch.zeros(num_paddings)), dim=-1)


        return (init_observation_rgbd, init_observation_grasp, goal, 
                seq_action, seq_observation_rgbd, seq_observation_grasp, 
                action_mask, observation_mask,
                input_trajectory_length, full_trajectory_length,
                next_action_label, next_action_token)
    




def validate_place_existance(trajectory_action: List) -> bool:
    """Validate existance of PLACE action in the trajectory
    
    Args:
        trajectory_action(List): Action trajectory
    
    Returns:
        bool: True if exist
    """
    is_place_exist = False
    for action in trajectory_action:
        action_type = action[0]
        action_pos = action[2]
        if action_type == "PLACE" and action_pos is not None:
            is_place_exist = True
    
    return is_place_exist
