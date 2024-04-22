import os
import json
import numpy as np
import numpy.typing as npt
import torch

from typing import List, Dict, Tuple, Union

from dataclasses import dataclass, field
from imm.pybullet_util.typing_extra import TranslationT, EulerT


@dataclass
class FetchingDatasetConfig:
    seq_length: int     # Trajectory length
    image_res : int
    # One-hot encoding table 
    action_type_encoding: Dict \
        = field(default_factory=lambda: { "PICK" : 0, 
                                          "PLACE": 1  })
    action_target_encoding: Dict \
        = field(default_factory=lambda: { False: 0,     # None for invalid action handling
                                          True : 1, 
                                          None : 2  })


@dataclass
class FetchingPickPlaceDatasetConfig:
    seq_length: int     # Trajectory length
    image_res : int
    dim_action_input: int
    # One-hot encoding table 
    action_type_encoding: Dict \
        = field(default_factory=lambda: { "PICK" : 0, 
                                          "PLACE": 1  })
    action_target_encoding: Dict \
        = field(default_factory=lambda: { False: 0,     # None for invalid action handling
                                          True : 1, 
                                          None : 2  })



def read_trajectory_from_json_and_numpy(json_data: Dict, 
                                         npz_data: Dict) \
                                            -> Tuple[ npt.NDArray, List, str,
                                                      List[npt.NDArray], List[npt.NDArray], List[npt.NDArray],
                                                      int, 
                                                      int, int, int ]:

    # Initial observation
    init_observation = (npz_data["init_observation_depth"], 
                        npz_data["init_observation_rgb"], 
                        npz_data["init_observation_grasp"])
    # Goal condition
    goal_condition   = json_data["goal_condition"]
    # Termination
    termination      = json_data["termination"]
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


    return init_observation, goal_condition, termination,\
            trajectory_action, trajectory_observation, trajectory_reward,\
            trajectory_length,\
            num_real_exec, num_sim_exec, num_rollout_exec



def tokenize_action(action_type           : str,
                    action_is_target_bool : Union[bool, None],
                    action_pos            : Union[TranslationT, None],
                    action_orn_e          : Union[EulerT, None],
                    action_dyaw           : Union[float, None],
                    action_type_encoding  : Dict[str, int],
                    action_target_encoding: Dict[Union[bool, None], int]) -> np.ndarray: 
    """Create action token from the action(list) data

    Args:
        action_type            (str)                         : "PICK" or "PLACE"
        action_is_target_bool  (Union[bool, None])           : Target or non-target. None when invalid action.
        action_pos             (Union[TranslationT, None])   : XYZ.
        action_orn_e           (Union[EulerT, None])         : Roll pitch yaw. None when invalid action.
        action_dyaw            (Union[float, None])          : Delta yaw. None when invalid action.
        action_type_encoding   (Dict[str, int])              : One-hot encoding table for "PICK" and "PLACE".
        action_target_encoding (Dict[Union[bool, None], int]): One-hot encoding table for is_target. None is the key when invalid action.

    Raises:
        TypeError: Raises type error if action primitive type is not str

    Returns:
        np.ndarray: token (action_type_onehot, is_target_onehot, xy, dyaw)=(2,3,2,1)=8
            shape=(8)
    """
    # Type validation
    if not isinstance(action_type, str):
        raise TypeError("Action primitive not string")

    # Action data
    action_type_onehot      = np.eye(len(action_type_encoding.keys()))[action_type_encoding[action_type]]
    action_is_target_onehot = np.eye(len(action_target_encoding.keys()))[action_target_encoding[action_is_target_bool]]
    action_xy               = np.array(action_pos[: 2]) if action_pos is not None else np.array([0,0])
    # xy
    if action_type == "PICK":
        action_dyaw = np.array([0])
    elif action_type == "PLACE":
        action_dyaw = np.array([action_dyaw]) if action_pos is not None else np.array([0])
    else:
        raise ValueError("Action type not `PICK` or `PLACE`")
    
    # Concat
    action_token = np.concatenate([action_type_onehot, action_is_target_onehot, action_xy, action_dyaw])
        
    return action_token



def tokenize_masked_rgbd(depth_image: npt.NDArray, 
                         rgb_image: npt.NDArray, 
                         normalize: bool = True) -> npt.NDArray:
    """Make input tensor from the masked rgbd image

    Args:
        depth_image (npt.NDArray): Depth image
        rgb_image (npt.NDArray): RGB image [0, 255]
        normalize (bool): Normalize pixel values within the range of [0, 1] when True. Defaults to True.

    Returns:
        npt.NDArray: tokenized RGB-D image, (4, W, H)
    """
    _depth_image = np.expand_dims(depth_image, -1).transpose(2, 0, 1)
    _rgb_image = rgb_image.transpose(2, 0, 1)
    rgbd_image = np.concatenate([_depth_image, _rgb_image], axis=0)

    if normalize:
        rgbd_image = rgbd_image/255.

    return rgbd_image



def format_next_action_label(action: Tuple[str, bool, TranslationT, EulerT, Union[float, None]], only_place: bool=True) -> np.ndarray:
    """Forming next action label

    Args:
        action (Tuple[str, bool, TranslationT, EulerT, Union[float, None]]): Raw action from data. Not class.

    Returns:
        np.ndarray: (x, y, dyaw). shape=(3)
    """

    # Action data
    action_type, action_target, action_pos, action_orn_e, action_dyaw = action
    
    # Validation
    if only_place and (action_type != "PLACE"):
        raise ValueError("Next action is not place. Double check dataloader.")

    # shape = (2+1=3)
    action_x_y  = np.array([action_pos[0], action_pos[1]])
    if not only_place and (action_type == "PICK"):
        action_dyaw = np.array([0])
    else:
        action_dyaw = np.array([action_dyaw])
    next_action = np.concatenate((action_x_y, action_dyaw), axis=0)
    
    return next_action



def add_zero_pre_padding(seq_variable: torch.Tensor, 
                         fill_like   : torch.Tensor,
                         num_paddings: int,
                         pre         : bool = False) -> torch.Tensor:
    """ Add zero pre-padding to the seq_variable by the num_paddings
    
    Args:
        seq_variable (torch.Tensor): sequence of variables to pad
        fill_like (torch.Tensor): The reference tensor to get the shape from.
        num_padding (int): number of zero paddings
        pre (bool): Pre-pad when true. |NOTE(ssh)|: False when using the transformer...?
    Returns:
        torch.Tensor: padded variable
    """
    variable_padding = torch.zeros_like(fill_like)
    seq_padding = variable_padding.expand(num_paddings, *variable_padding.shape)

    if pre:
        seq_variable = torch.cat((seq_padding, seq_variable), axis=0)
    else:
        seq_variable = torch.cat((seq_variable, seq_padding), axis=0)
        

    return seq_variable