import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from pathlib import Path


def generate_point_cloud_from_mesh(mesh_file: str, save_dir: str, name: str, num_point: int=2048):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_point, init_factor=2)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    point_cloud = {'points': points, 'normals': normals}
    save_file = os.path.join(save_dir, name)
    with open(f'{save_file}.pickle', 'wb') as f:
        pickle.dump(point_cloud, f)
    
    return point_cloud

if __name__ == '__main__':
    mesh_file = '/home/ajy8456/workspace/POMDP/Simulation/pybullet_env/urdf/cuboid/cuboid.obj'
    save_dir = '/home/ajy8456/workspace/POMDP/Simulation/pybullet_env/urdf/cuboid/'
    name = 'cuboid_point_cloud'
    generate_point_cloud_from_mesh(mesh_file, save_dir, name)