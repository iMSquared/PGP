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



class FetchingImageDataset(Dataset):

    @dataclass
    class Config:
        seq_length   : int      # Trajectory length
        image_res    : int


    def __init__(self, config: Config,
                       data_path_json: str,
                       data_path_npz: str,
                       annotations: List[str],
                       num_subdata: List[int]):
        """Fetching filename class
        Args:
            config (Config): Configuration class
            data_path_json (str): Json dir path
            data_path_npz (str): Numpy dir path
            annotations (List): List of data file name
            nu
            
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
            seq_observation_rgbd (torch.Tensor): shape=(C, H, W)
        """

        # Loading json and numpy
        json_file_name = os.path.join(self.data_path_json, f"{self.filenames[index]}.json")
        npz_file_name = os.path.join(self.data_path_npz, f"{self.filenames[index]}.npz")
        with open(json_file_name, "r") as f:
            json_data = json.load(f)
        npz_data = np.load(npz_file_name)

        # Length of trajectory
        num_real_exec     = len(json_data["exec_action"])
        num_sim_exec      = len(json_data["sim_action"])
        num_rollout_exec  = len(json_data["rollout_action"])
        trajectory_length = num_real_exec + num_sim_exec + num_rollout_exec
        trajectory_steps_bound = [num_real_exec, num_real_exec+num_sim_exec, num_real_exec+num_sim_exec+num_rollout_exec]
        
        # Randomly select the time step.
        sampled_step = np.random.randint(0, trajectory_length+1)
        
        # Observation
        if sampled_step == 0:
            observation  = np.concatenate([np.expand_dims(npz_data["init_observation_depth"], axis=2), 
                                                          npz_data["init_observation_rgb"]], axis=2)
        elif sampled_step <= trajectory_steps_bound[0]:
            i = sampled_step - 1
            observation = np.concatenate([np.expand_dims(npz_data[f"exec_observation_{i}_depth"], axis=2),
                                                                        npz_data[f"exec_observation_{i}_rgb"]], axis=2)
        elif sampled_step <= trajectory_steps_bound[1]:
            i = sampled_step - trajectory_steps_bound[0] - 1
            observation = np.concatenate([np.expand_dims(npz_data[f"sim_observation_{i}_depth"], axis=2),
                                                                        npz_data[f"sim_observation_{i}_rgb"]], axis=2)
        elif sampled_step <= trajectory_steps_bound[2]:
            i = sampled_step - trajectory_steps_bound[1] - 1
            observation = np.concatenate([np.expand_dims(npz_data[f"rollout_observation_{i}_depth"], axis=2),
                                                                        npz_data[f"rollout_observation_{i}_rgb"]], axis=2)
        
        # Tensor (make sure to float())
        observation_rgbd   = torch.from_numpy(observation).float()
        # H X W X C -> C X H X W
        observation_rgbd = torch.permute(observation_rgbd, (2, 0, 1))

        return observation_rgbd
    

def normalize(imgs):
    imgs[:, 1:4, :, :] = imgs[:, 1:4, :, :] / 255
    min_depth = torch.min(imgs[:, 0, :, :])
    max_depth = torch.max(imgs[:, 0, :, :])
    imgs[:, 0, :, :] = (imgs[:, 0, :, :] - min_depth) / (max_depth - min_depth)
    
    return imgs, min_depth, max_depth

def denormaize(imgs, min_depth, max_depth):
    imgs[:, 1:4, :, :] = torch.clip(imgs[:, 1:4, :, :] * 255, min=0, max=255)
    imgs[:, 0, :, :] = torch.clip(imgs[:, 0, :, :] * (max_depth - min_depth) + min_depth, min=0)

    return imgs


if __name__ == '__main__':
    import pandas as pd
    from torch.utils.data import DataLoader, WeightedRandomSampler, BatchSampler
    
    dataname: str = "April17th"
    sim_or_exec: str = "sim_dataset"
    file_path_entire_annotation : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/entire.csv"
    data_path_dataset_json      : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_json"
    data_path_dataset_npz       : str = f"/home/sanghyeon/vessl/{dataname}/{sim_or_exec}/eval/dataset_numpy"
    batch_size = 2
    num_workers = 8

    # Dataset params
    dataset_config: FetchingImageDataset.Config = FetchingImageDataset.Config(
            seq_length    = 6,
            image_res     = 64)
    
    
    # Train/eval split
    entire_annots      = pd.read_csv(file_path_entire_annotation)["filename"].tolist()
    entire_num_subdata = pd.read_csv(file_path_entire_annotation)["num_subdata"].tolist()
    
    # Dataset
    dataset = FetchingImageDataset(
        config = dataset_config,
        data_path_json    = data_path_dataset_json,
        data_path_npz     = data_path_dataset_npz,
        annotations       = entire_annots,
        num_subdata       = entire_num_subdata)

    # Sampler
    sampler = BatchSampler(
        sampler = WeightedRandomSampler(
            weights     = dataset.weights, 
            num_samples = len(dataset), 
            replacement = True),
        batch_size = batch_size,
        drop_last = True)

    # Dataloader
    loader = DataLoader(
        dataset,
        batch_sampler = sampler,
        num_workers   = num_workers,
        pin_memory    = True)
    
    iterator = iter(dataset)
    data = next(iterator)
    print(data)