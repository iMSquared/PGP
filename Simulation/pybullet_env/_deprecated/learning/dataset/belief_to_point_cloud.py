import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from typing import (Union, Callable, List, Dict, Tuple, Optional, Any)
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from copy import deepcopy

# |TODO(Jiyong)|: move to process_pointcloud.py
def generate_point_cloud_from_pose(mesh_file: str, pose: tuple, num_point: int):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_point, init_factor=2)
    pos = pose[0]
    orn = pose[1]
    r = R.from_euler('zyx', orn, degrees=False)
    rot_mat = r.as_matrix()
    
    trans_mat = np.array(
        [[1, 0, 0, pos[0]],
         [0, 1, 0, pos[1]],
         [0, 0, 1, pos[2]],
         [0, 0, 0, 1]],
    )
    trans_mat[:3, :3] = rot_mat
    pcd.transform(trans_mat)
    pcd = np.asarray(pcd.points)
    
    return pcd

# |TODO(Jiyong)|: move to process_pointcloud.py & make it compatitable with different shapes
def transform_point_cloud_to_pose(point_cloud_file, pose: tuple, num_point: int):
    with open(point_cloud_file, 'rb') as f:
        pcd = pickle.load(f)
    
    pos = pose[0]
    orn = pose[1]
    r = R.from_euler('zyx', orn, degrees=False)
    rot_mat = r.as_matrix()
    
    trans_mat = np.array(
        [[1, 0, 0, pos[0]],
         [0, 1, 0, pos[1]],
         [0, 0, 1, pos[2]],
         [0, 0, 0, 1]],
    )
    trans_mat[:3, :3] = rot_mat
    
    if len(pcd['points']) >= num_point:
        sampled_indices = np.random.choice(len(pcd['points']), num_point, replace=False)
    else:
        sampled_indices = np.random.choice(len(pcd['points']), num_point, replace=True)
    
    sampled_pcd = np.tile(1, (num_point, 1))
    sampled_pcd = np.concatenate([pcd['points'][sampled_indices], sampled_pcd], axis=1)
    sampled_pcd = np.matmul(sampled_pcd, trans_mat.T)
    
    return sampled_pcd[:, 0:3]


def belief_to_point_cloud(data_path: str,
                          num_point: int=256,
                          target_obj_mesh_file: str="Simulation/pybullet_env/urdf/cuboid/cuboid.obj",
                          non_target_obj_mesh_file: str="Simulation/pybullet_env/urdf/cuboid/cuboid.obj",
                          target_obj_color: List=[0.5804, 0.0, 0.8275],
                          non_target_obj_color: List=[0.0, 0.0, 0.0],
                          goal_condition: List=[0.5804, 0.0, 0.8275, 0.0, 0.45]):
    
    target_obj_color = np.tile(target_obj_color, (num_point, 1))
    non_target_obj_color = np.tile(non_target_obj_color, (num_point, 1))
    
    data_path = Path(data_path)
    save_path = data_path.parent.joinpath("sim_dataset_fixed_belief")
    if not save_path.exists():
        os.mkdir(str(save_path))
    
    filenames = os.listdir(str(data_path))
    for file in filenames:
        if not os.path.exists(os.path.join(str(save_path), file)):
            with open(os.path.join(str(data_path), file), 'rb') as f:
                data = pickle.load(f)
            
            traj_a = []
            for a in data['exec_action']:
                traj_a.append((a[0], a[1], a[2], a[3], a[4]))
            for a in data['sim_action']:
                traj_a.append((a[0], a[1], a[2], a[3], a[4]))
                
            traj_r = data['exec_reward'] + data['sim_reward']
            
            traj_b = data['exec_belief']
            for b in data['sim_belief']:
                if isinstance(b, list):
                    traj_b.append(b)
                elif isinstance(b, dict):
                    traj_b.append([b])
            
            traj_b_pcd = []
            for b in traj_b:
                b_pcd = []
                for s in b:
                    s_pcd = []
                    if s['weight'] <= 0.001:
                        continue
                    weight = np.tile(s['weight'], (num_point, 1))
                    for v in s['object'].values():
                        if v[-1]:
                            # pcd_target = generate_point_cloud_from_pose(target_obj_mesh_file, v[0:2], num_point)
                            pcd_target = transform_point_cloud_to_pose('Simulation/pybullet_env/urdf/cuboid/cuboid_point_cloud.pickle', v[0:2], num_point)
                            pcd_target = np.hstack([pcd_target, target_obj_color, weight])
                            s_pcd.append(pcd_target)
                        else:
                            # pcd_non_target = generate_point_cloud_from_pose(non_target_obj_mesh_file, v[0:2], num_point)
                            pcd_non_target = transform_point_cloud_to_pose('Simulation/pybullet_env/urdf/cuboid/cuboid_point_cloud.pickle', v[0:2], num_point)
                            pcd_non_target = np.hstack([pcd_non_target, non_target_obj_color, weight])
                            s_pcd.append(pcd_non_target)
                    b_pcd.append(np.concatenate(s_pcd, axis=0))
                traj_b_pcd.append(np.concatenate(b_pcd, axis=0))
            
            belief_data = {
                'belief': traj_b_pcd,
                'action': traj_a,
                'reward': traj_r,
                'goal': goal_condition
            }
            
            with open(os.path.join(str(save_path), file), 'wb') as f:
                pickle.dump(belief_data, f)
    
    return


if __name__ == '__main__':

    data_path = "/home/sanghyeon/vessl/sim_dataset_fixed"
    # data_path = "dataset/sim_dataset_fixed"

    belief_to_point_cloud(data_path, 256)