from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt

from scipy.spatial.distance import directed_hausdorff
import scipy.stats

from utils.process_pointcloud import visualize_point_cloud
from imm.pybullet_util.bullet_client import BulletClient
from envs.robot import Robot
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.global_object_id_table import Gid_T

from copy import deepcopy

import matplotlib.pyplot as plt



def reproject_observation_to_pointcloud(bc: BulletClient, 
                                        env: BinpickEnvPrimitive,
                                        depth_image: npt.NDArray,
                                        seg_mask: Dict[int, npt.NDArray]) -> npt.NDArray:
    """ Depth image to point cloud...

    Args:
        bc (BulletClient)
        env (BinpickEnvPrimitive)
        depth_image (npt.NDArray)
        seg_mask (Dict[int, npt.NDArray]): Key is GID!!

    Returns:
        npt.NDArray: pointcloud
    """

    # Excluding robot and background from the observation
    gid_list = list(seg_mask.keys())     # Select key

    # Merge to the foreground segmentation  (Merge pixelwise boolean mask)
    seg_mask_merged = np.zeros_like(seg_mask[gid_list[0]])
    for gid in gid_list:
        seg_mask_merged = np.logical_or(seg_mask_merged, seg_mask[gid])
    
    # Reproject
    pcd = env.reprojection_function(depth_image, seg_mask_merged)

    # plt.imshow(seg_mask_merged)
    # plt.show()
    # visualize_point_cloud([pcd])

    return pcd




def get_hausdorff_distance_norm_pdf_value(ref_pcd: npt.NDArray,
                                          eval_pcd: npt.NDArray,
                                          sigma: float = 0.0085) -> float: 
    """Hausdorff distance based observation model, 
    P( eval_pcd | ref_pcd=PCD(state) )

    Args:
        ref_pcd (npt.NDArray): PCD simulated from the state
        eval_pcd (npt.NDArray): The observation to evaluate P(O|S)
    Returns:
        pdf (float): pdf value of the Hausdorff distance from N(0, sigma**2)
    """

    # Hausdorff distance
    hausdorff_distance_ref_to_eval, _, _ = directed_hausdorff(ref_pcd, eval_pcd)
    hausdorff_distance_eval_to_ref, _, _ = directed_hausdorff(eval_pcd, ref_pcd)
    hausdorff_distance_undirected = max(hausdorff_distance_ref_to_eval, hausdorff_distance_eval_to_ref)
    
    # Get PDF value
    observation_density_fn = scipy.stats.norm(0, sigma) 
    pdf = observation_density_fn.pdf(hausdorff_distance_undirected)

    return pdf



