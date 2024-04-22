import os
import glob
import pickle
import random

import numpy as np
import torch as th
from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from dataclasses import dataclass, field
from copy import deepcopy

import matplotlib.pyplot as plt
import math


class DepthToPointcloudOpenGL:
    """I don't want to explain..."""

    def __init__(
            self,
            cx: float, 
            cy: float, 
            near_val: float, 
            far_val: float, 
            focal_length: float):

        # Camera intrinsics
        self.cx = cx
        self.cy = cy
        self.near_val = near_val
        self.far_val = far_val
        self.focal_length = focal_length


    def __call__(self, depth_2d: np.ndarray, 
                       mask_target_2d: np.ndarray) -> np.ndarray:
        """Convert depth image to Open3D point clouds.
        The depth value should follow openGL convention which is used in PyBullet.

        Args:
            depth_2d (np.ndarray): Depth image
            mask_target_2d (np.ndarray[bool]): Target pixels to convert to point cloud
        Returns:
            pointcloud (np.ndarray): Reprojected target point cloud
        """
        # Convert depth image to pcd in pixel unit
        # (x, y, z, class), y-up
        pcd = np.array([[
                    (self.cy - v, self.cx - u, depth_2d[v, u], mask_target_2d[v, u])
                for u in range(depth_2d.shape[1])]
            for v in range(depth_2d.shape[0])]).reshape(-1, 4)

        # Getting true depth from OpenGL style perspective matrix
        #   NOTE(ssh): 
        #   For the calculation of the reprojection, see the material below
        #   https://stackoverflow.com/questions/6652253/getting-the-true-z-value-from-the-depth-buffer
        # Calculate z
        z_b = pcd[:,2]
        z_n = 2.0 * z_b - 1.0
        z = 2.0 * self.near_val * self.far_val / (self.far_val + self.near_val - z_n * (self.far_val - self.near_val))
        # Calculate x
        x = z * pcd[:,1] / self.focal_length
        # Calculate y
        y = z * pcd[:,0] / self.focal_length
        # Copy uid class label
        c = pcd[:,3]

        # Stack
        pcd = np.stack((x, y, z, c), axis=1)
        # Convert y-up to z-up
        pcd = pcd[:,[2, 0, 1, 3]]

        # Select target points only
        pcd = pcd[pcd[:,3]==True]
        pcd = pcd[:,:3]


        return pcd



class ProcessObservationPolicyDataset(Dataset):

    @dataclass
    class Config:
        # Large
        # Use very small number for the observation based model.
        # Use some large number (e.g. 2048) for the belief based model.
        num_points = 256
        # NOTE(ssh): This should be changed
        camera_origin: tuple = (0.1, 0.0, 0.6)
        grasp_z = 0.42      
        # One-hot encoding table
        action_type_encoding: Dict = field( default_factory = lambda: { "PICK": 0,
                                                                        "PLACE": 1 } )
        action_target_encoding: Dict = field( default_factory = lambda: { "O": 0,
                                                                          "X": 1 } )


    def __init__(self, config: Config, 
                       data_path: List[str]):
        """Fetching filename class
        Args:
            config (Config): configuration class
            data_path (List): List of data file name
        """
        self.config = config
        self.data_path = data_path
        self.filenames = sorted(os.listdir(self.data_path))
        self.reprojection_fn = DepthToPointcloudOpenGL(
            cx = 32,
            cy = 32,
            near_val = 0.1,
            far_val = 0.7,
            focal_length = 64 / ( 2.0 * math.tan(math.radians(60.0) / 2.0) ),)


    def __len__(self):
        return len(self.filenames)


    def __merge_pcd_segments(self, observation_depth_image: List, 
                                   observation_seg_mask: Dict[str, List]) -> np.ndarray:
        """Convert and merge the pointcloud of all segments in the depth image.

        Args:
            observation_depth_image (Dict): depth image 
            observation_seg_mask (Dict[str, np.ndarray]): segmentation mask
        Returns:
            np.ndarray: Merged point cloud
        """
        segment_pcds = []
        for seg_mask in observation_seg_mask.values():
            pcd = self.reprojection_fn(np.array(observation_depth_image), np.array(seg_mask))
            segment_pcds.append(pcd)
        observation_pcd = np.concatenate(segment_pcds, axis=0)

        return observation_pcd


    def __getitem__(self, index) -> Dict[str, th.Tensor]:

        # Open the data
        with open(os.path.join(self.data_path, self.filenames[index]), "rb") as f:
            trajectory: List[Dict] = pickle.load(f)

        # Select a search trajectory
        trajectory_init_observation = trajectory["init_observation"]
        trajectory_exec_action      = trajectory["exec_action"]
        trajectory_exec_observation = trajectory["exec_observation"]
        trajectory_exec_reward      = trajectory["exec_reward"]
        trajectory_sim_action       = trajectory["sim_action"]
        trajectory_sim_observation  = trajectory["sim_observation"]
        trajectory_sim_reward       = trajectory["sim_reward"]

        trajectory_action      = trajectory_exec_action + trajectory_sim_action
        trajectory_observation = trajectory_exec_observation + trajectory_sim_observation
        trajectory_reward      = trajectory_exec_reward + trajectory_sim_reward

        # Process initial observation first
        init_observation_depth_image = trajectory_init_observation[0]
        init_observation_seg_mask = trajectory_init_observation[1]
        init_observation_pcd = self.__merge_pcd_segments(init_observation_depth_image, 
                                                         init_observation_seg_mask)

        # Iterating through each time step
        list_action_type = []
        list_action_target = []
        list_action_x_y_yaw = []
        list_observation_pcd = []
        list_reward = []

        for action, observation, reward \
            in zip(trajectory_action, trajectory_observation, trajectory_reward):

            # Action data
            action_type, action_target, action_pos, action_orn_e = action
            # Observation data
            observation_depth_image, observation_seg_mask = observation

            # Converting format (with one-hot encoding)
            # Action data
            action_type = np.eye(2)[self.config.action_type_encoding[action_type]]          # To one-hot
            action_target = np.eye(2)[self.config.action_target_encoding[action_target]]    # To one-hot
            action_x_y_yaw = np.array([action_pos[0], action_pos[1], action_orn_e[2]])
            # Observation data
            observation_pcd = self.__merge_pcd_segments(observation_depth_image,
                                                        observation_seg_mask)

            # Collect
            list_action_type.append(action_type)
            list_action_target.append(action_target)
            list_action_x_y_yaw.append(action_x_y_yaw)
            list_observation_pcd.append(observation_pcd)
            list_reward.append(reward)

        # Select pick action and corresponding observation
        list_prev_observation_pcd = deepcopy(list_observation_pcd)
        list_prev_observation_pcd.insert(0, init_observation_pcd)
        trajectory_length = len(trajectory_action)
        while True:
            # Randomly select the time step.
            i = np.random.randint(0, trajectory_length)
            
            # Parse action
            action_type = list_action_type[i]        
            # Break if pick
            if np.array_equal(action_type, np.eye(2)[self.config.action_type_encoding["PICK"]]):
                # Parse the rest of action
                action_target = list_action_target[i]
                action_x_y_yaw = list_action_x_y_yaw[i]            

                # Parse observation
                prev_observation: np.ndarray = list_prev_observation_pcd[i]
                # Transform pcds to world coordinate
                prev_observation += self.config.camera_origin

                break

        # |NOTE(ssh)|: Let's temporarily match to the point nearest to the object center.
        center_x_y_z = np.concatenate([action_x_y_yaw[0:2], [self.config.grasp_z]], axis=0)
        dist_center_to_point = np.linalg.norm(prev_observation - center_x_y_z, axis=1)
        target_point_idx = np.argmin(dist_center_to_point)

        # One-hot label
        num_observation_points = prev_observation.shape[0]
        target_point_label = np.zeros(num_observation_points)
        target_point_label[target_point_idx] = 1

        # Resampling
        if num_observation_points > self.config.num_points:
            resampled_indices = np.random.choice(num_observation_points, self.config.num_points, replace=False)        
        else:
            resampled_indices = np.arange(num_observation_points)
            resampled_indices = np.concatenate([resampled_indices, 
                                                np.random.choice(num_observation_points, 
                                                                 self.config.num_points - num_observation_points, 
                                                                 replace=True)])
        prev_observation = prev_observation[resampled_indices, :]
        target_point_label = target_point_label[resampled_indices]

        # To tensor
        prev_observation = th.from_numpy(prev_observation).float()
        target_point_label = th.from_numpy(target_point_label).float()

        return prev_observation, target_point_label


class PlacePolicySimulationBeliefDataset(Dataset):
    def __init__(self, config, data_path):
        self.config = config
        self.device = config.device
        self.dataset = glob.glob(f'{data_path}/*.pickle')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Open the data
        with open(self.dataset[index], 'rb') as f:
            traj = pickle.load(f)
        
        traj_b = traj['belief']
        traj_a = traj['action']
        
        place_steps = []
        for i, a in enumerate(traj_a):
            if a[0] == 'PLACE':
                place_steps.append(i)
        
        sampled_step = random.choice(place_steps)
        
        b = traj_b[sampled_step]
        num_point_in_b = len(b)
        p_weights = b[:,-1]
        p_weights /= np.sum(p_weights)
        b[:,-1] = p_weights

        if num_point_in_b >= self.config.num_point:
            p_indices = np.random.choice(num_point_in_b, self.config.num_point, replace=False, p = b[:,-1])
        else:
            p_indices = np.random.choice(num_point_in_b, self.config.num_point, replace=True, p = b[:,-1])
        
        g = th.tensor(traj['goal'], dtype=th.float32).to(self.device)
        b = th.tensor(traj_b[sampled_step][p_indices,:], dtype=th.float32).to(self.device)
        a = th.tensor([traj_a[sampled_step][2][0], traj_a[sampled_step][2][1], traj_a[sampled_step][4]], dtype=th.float32).to(self.device)
        
        data = {
            'goal': g,
            'belief': b,
            'target_action': a
        }
        
        return data


if __name__ == '__main__':

    # # Debugging scripts
    # BATCH_SIZE = 4
    # data_path = os.path.join(os.path.expanduser("~"), "vessl", "dev-pick", "sim_dataset")

    # # Dataset
    # dataset = ProcessObservationPolicyDataset(
    #     config = ProcessObservationPolicyDataset.Config(),
    #     data_path = data_path)

    # prev_observation, target_point_label = dataset[0]

    # colors = ["Red", "Blue", "Green", "tab:orange", "magenta", "tab:blue", "tab:purple", "tab:olive"]
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([0, 0.5])
    # ax.set_ylim([-0.25, 0.25])
    # ax.set_zlim([0, 0.5])
    # # Plot points
    # for i, pcd in enumerate([prev_observation]):

    #     ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2, c=colors[i % len(colors)])
        
    #     # Draw target point
    #     target_point_indices = np.where(target_point_label == 1)
    #     ax.scatter(pcd[target_point_indices,0], pcd[target_point_indices,1], pcd[target_point_indices,2], s=3.0, c=colors[i+1 % len(colors)])

    # plt.show()
    
    
    # test for PlacePolicySimulationBeliefDataset    
    class Setting():
        train_data_path = '/home/ajy8456/workspace/POMDP/dataset/sim_dataset_fixed_belief_train'
        eval_data_path = '/home/ajy8456/workspace/POMDP/dataset/sim_dataset_fixed_belief_eval'
        device = 'cuda' if th.cuda.is_available() else 'cpu'
        batch_size = 16
        num_point = 512
    
    config = Setting()
    
    dataset = PlacePolicySimulationBeliefDataset(config, config.train_data_path)
    d = dataset[0]
    print(d['goal'].shape, d['belief'].shape, d['target_action'].shape)
    
    # pcd = d['belief'][:, 0:3].to('cpu').numpy()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([0.25, 0.75])
    # ax.set_ylim([-0.25, 0.25])
    # ax.set_zlim([0.25, 0.75])

    # ax.scatter(pcd[:,0], pcd[:,1], pcd[:,2], s=0.2)

    # plt.show()
    
    loader = DataLoader(dataset, config.batch_size)
    
    for batch_ndx, d in enumerate(loader):
        print(d['goal'].shape, d['belief'].shape, d['target_action'].shape)