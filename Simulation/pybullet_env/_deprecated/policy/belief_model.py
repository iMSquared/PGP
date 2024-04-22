import os
import time
import random
import copy
import numpy as np
import torch as th
from typing import List

from dataclasses import dataclass, replace, field
from simple_parsing import Serializable
from scipy.spatial.transform import Rotation

from POMDP_framework import *
from fetching_POMDP_primitive_object import FetchingState, FetchingAction, FetchingObservation
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import Robot, UR5, UR5Suction
from envs.manipulation import Manipulation
from envs.grasp_pose_sampler_suction_gripper import PointSamplingGraspSuction

from Simulation.pybullet_env._deprecated.learning.model.place import PolicyModelPlaceBelief
from learning.model.policy.place_gpt import GPT2FetchingPlacePolicyCVAE
from learning.script.utils import load_checkpoint
from learning.dataset.belief_to_point_cloud import belief_to_point_cloud

from utils.process_pointcloud import visualize_point_cloud



class FetchingPlacePolicyModelBelief(PolicyModel):
    """A belief-model implementation of policy interface for POMCPOW"""

    # |TODO(jiyong)|: make parsing from configutation file
    @dataclass
    class Settings(Serializable):
        # dataset
        train_data_path        : str   = '/home/ajy8456/workspace/POMDP/dataset/sim_dataset_fixed_belief_train'
        eval_data_path         : str   = '/home/ajy8456/workspace/POMDP/dataset/sim_dataset_fixed_belief_eval'
        # input
        dim_action_place       : int   = 3
        dim_action_embed       : int   = 16        
        dim_point              : int   = 7
        dim_goal               : int   = 5
        # PointNet
        num_point              : int   = 512
        dim_pointnet           : int   = 128
        dim_goal_hidden        : int   = 8
        # CVAE
        dim_vae_latent         : int   = 16
        dim_vae_condition      : int   = 16
        vae_encoder_layer_sizes: Tuple = (dim_action_embed, dim_action_embed + dim_vae_condition, dim_vae_latent)
        vae_decoder_layer_sizes: Tuple = (dim_vae_latent, dim_vae_latent + dim_vae_condition, dim_action_place)
        # Training
        device                 : int   = 'cuda' if th.cuda.is_available() else 'cpu'
        resume                 : int   = 'best.pth' # checkpoint file name for resuming
        pre_trained            : bool  = None
        epochs                 : int   = 5000
        batch_size             : int   = 16
        learning_rate          : float = 1e-4
        # Logging
        exp_dir                : str   = '/home/ajy8456/workspace/POMDP/learning/exp'
        model_name             : str   = '2.26_place_policy_belief'
        print_freq             : int   = 10 # per training step
        train_eval_freq        : int   = 100 # per training step
        eval_freq              : int   = 10 # per epoch
        save_freq              : int   = 100 # per epoch


    def __init__(self, bc     : BulletClient, 
                       sim_env: BinpickEnvPrimitive, 
                       robot  : Robot, 
                       config : Settings,
                       manip  : Manipulation):  
        """ SCOPE?

        Args:
            bc (BulletClient): skipped
            sim_env (BinpickEnvPrimitive): skippied
            robot (Robot): skipped
            config (Settings): skipped
            manip (Manipulation, optional): skipped
        """
        # configuration for NN
        self.nn_config = self.Settings()

        # Model
        self.model_dir = os.path.join(self.nn_config.exp_dir, self.nn_config.model_name)
        self.device    = self.nn_config.device
        self.model     = PolicyModelPlaceBelief(self.nn_config).to(self.device)
        self.model.eval()
        # Optimizer
        # |TODO(jiyong)|: Make it compatible with the nn_config
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.nn_config.learning_rate)
        self.scheduler = th.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000, 4000, 4500])
        # Load checkpoint for resuming
        filename = os.path.join(self.model_dir, self.nn_config.resume)
        start_epoch, self.model, self.optimizer, self.scheduler = load_checkpoint(self.nn_config, 
                                                                                  filename, 
                                                                                  self.model, 
                                                                                  self.optimizer, 
                                                                                  self.scheduler)
        start_epoch += 1
        print("Loaded checkpoint '{}' (epoch {})".format(self.nn_config.resume, start_epoch))

        # Initialize policy in fetching domain
        self.bc = bc
        self.sim_env = sim_env
        self.robot = robot
        self.config = config
        self.manip = manip
        self.grasp_pose_sampler = PointSamplingGraspSuction()
        self.num_filter_trials = config["pose_sampler_params"]["num_filter_trials"]


    def sample(self, history: Tuple, state: FetchingState, goal: List):
        """
        Infer next_action using neural network

        Args:
            history (Tuple): Tuple of current history 
                ((b0, a0, o0, s'0, r0)), (b1, a1, o1, s'1, r1), ... )
                where b=WeightedParticles  
            state (FetchingState): Current state
            goal (List): Goal condition, [r,g,b,x,y]
        
        Returns:
            pred(tuple): prediction of next_action
            infer_time(float)
        """

        # Action selection
        holding_obj = state.holding_obj
        if holding_obj is None:   # not holding an object
            type = "PICK"
        else:
            type = "PLACE"

        # PICK: Random sampling
        if type == "PICK":
            if self.config['project_params']['debug']['get_data']:
                print('Start to find a grasp pose')
                check_time_g = time.time()
            
            # Find filtered grasp pose
            for i in range(self.num_filter_trials):
                
                # Select target
                obj_target = random.choice(['O', 'X']) 
                obj_uid = self.sim_env.target_to_uid[obj_target]       
                obj_pose = state.object[obj_target][0:2]
                obj_pcd = self.sim_env.point_clouds[obj_target] # Key: "O", "X"...
                obstacle_list = copy.deepcopy(self.sim_env.objects_uid)
                obstacle_list.remove(obj_uid)
                
                # sample grasp pose
                grasp_pos, grasp_orn = self.grasp_pose_sampler(obj_pcd, obj_pose)
                grasp_orn_q = self.bc.getQuaternionFromEuler(grasp_orn)
                
                modified_pos_back, modified_orn_q_back, modified_pos, modified_orn_q = self.manip.get_ee_pose_from_target_pose(grasp_pos, grasp_orn_q)
                
                joint_pos_src, joint_pos_dst = self.manip.solve_ik_numerical(modified_pos_back, modified_orn_q_back)
                if joint_pos_dst is None:
                    target, pos, orn, traj, delta_theta = None, None, None, None, None
                    continue
                
                # |TODO(Jiyong)|: If we want to append affordance check for suction, append here
                
                grasp_pose = (tuple(grasp_pos), tuple(grasp_orn))
                    
                traj = self.manip.motion_plan(joint_pos_src, joint_pos_dst, holding_obj)
                
                if traj is not None:
                    target = obj_target
                    pos = grasp_pose[0]
                    orn = grasp_pose[1]
                    delta_theta = None
                    break
                # fail to find collision free trajectory
                else:
                    target, pos, orn, traj, delta_theta = None, None, None, None, None
                    
            if (self.config['project_params']['debug']['get_data']) and (target is None):
                print('Fail to find a grasp pose')
                        
            if self.config['project_params']['debug']['get_data']:
                print(f'End to find a grasp pose - {i} trials')
                check_time_g = time.time() - check_time_g
        

        # PLACE: Use guided policy
        elif type == "PLACE":

            # Fitting form of traj to input of network
            goal = [0.5804, 0.0, 0.8275, 0.0, 0.45] # |NOTE(ssh)|: Temporarily force override...
            p, c = self._process_history(history, goal)

            # Try until find a pose without collision
            for i in range(self.num_filter_trials):
                
                # Predict (x, y, delta_theta) of place action with NN
                with th.no_grad():
                    time_start = time.time()
                    p = p.to(self.nn_config.device)
                    c = c.to(self.nn_config.device)
                    pred = self.model.inference(p, c).squeeze()
                    time_end = time.time()
                    pred = tuple(pred.tolist())
                    infer_time = time_end - time_start
                    
                x, y, delta_theta = pred
                z = history[-1][1].pos[2]   # Use same z from the last action.
                place_pos = (x, y, z)
                
                # Sample orientation
                z_axis_rot_mat = np.asarray([
                    [np.cos(delta_theta), -np.sin(delta_theta), 0],
                    [np.sin(delta_theta),  np.cos(delta_theta), 0],
                    [0                  , 0                   , 1]])
                prev_orn = history[-1][1].orn       # |TODO(Jiyong)|: Should use forward kinematics
                prev_orn_q = self.bc.getQuaternionFromEuler(prev_orn)
                prev_orn_rot_mat = np.asarray(self.bc.getMatrixFromQuaternion(prev_orn_q)).reshape(3, 3)
                place_orn_rot_mat = Rotation.from_matrix(np.matmul(z_axis_rot_mat, prev_orn_rot_mat))
                place_orn = place_orn_rot_mat.as_euler("zyx", degrees=False)
                
                target = holding_obj
                
                # filtering by motion plan
                place_orn_q = self.bc.getQuaternionFromEuler(place_orn)
                modified_pos_back, modified_orn_q_back, modified_pos, modified_orn_q = self.manip.get_ee_pose_from_target_pose(place_pos, place_orn_q)
                joint_pos_src, joint_pos_dst = self.manip.solve_ik_numerical(modified_pos, modified_orn_q)
                if joint_pos_dst is None:
                    target, pos, orn, traj = None, None, None, None
                    continue
                
                traj = self.manip.motion_plan(joint_pos_src, joint_pos_dst, holding_obj) 
                if traj is not None:
                    pos = place_pos
                    orn = tuple(place_orn.tolist())
                    break
                else:
                    target, pos, orn, traj, delta_theta = None, None, None, None, None
                
        return FetchingAction(type, target, pos, orn, traj, delta_theta)
    

    def _process_history(self, history: Tuple, 
                               goal   : List[float]):
        """Process history to the data for inference
        1. Re-normalize weight
        2. Union the point cloud of the last belief
        3. Label goal condition
        
        Args:
            history (Tuple): 
                Tuple of current history ((b0, a0, o0, s'0, r0)), (b1, a1, o1, s'1, r1), ... )
                where b=WeightedParticles  
            goal (List): Goal condition, [r,g,b,x,y]


        Returns:
            belief_pointcloud (th.Tensor): Union point cloud
            goal_condition (th.Tensor): conditioning code (Goal embedding)
        """

        # |NOTE(ssh)|: Typing
        # xyz, rgb, weight (saved before norm -> Make sure to norm) (dim=7)
        # dim 5 condition
        
        last_belief: WeightedParticles = history[-1][0]
        if last_belief is not None:
            particles = last_belief.particles
            # 1. Re-normalize weight first
            list_states = []
            list_weights = []
            key: FetchingState
            value: float
            for key, value in particles.items():
                list_states.append(key)
                list_weights.append(value)
            #   Normalizing...
            list_weights = np.asarray(list_weights)
            list_weights = list_weights / np.sum(list_weights)
        else: # For rollout history (It use state particle instead belief)
            list_states = [history[-1][-1]]
            list_weights = [1.0]

        # 2. Union the point cloud of the entire belief
        list_state_pointclouds = []
        state: FetchingState
        weight: float
        for state, weight in zip(list_states, list_weights):

            # Get pcd of objects in one state particle
            list_object_pointclouds = []    # list of N (xyzrgbw)
            for target, (pos, orn, _) in state.object.items():

                # key: "points", "normals"
                shape_pcd: Dict[np.ndarray, np.ndarray] = self.sim_env.point_clouds[target]                
                points = shape_pcd["points"]
                normals = shape_pcd["normals"]
                num_points_in_shape = points.shape[0]

                # Transform points back to where it was...
                r = Rotation.from_euler('zyx', orn, degrees=False)
                rot_mat = r.as_matrix()
                trans_mat = np.array(
                    [[1, 0, 0, pos[0]],
                    [0, 1, 0, pos[1]],
                    [0, 0, 1, pos[2]],
                    [0, 0, 0, 1]])
                trans_mat[:3, :3] = rot_mat
                points_homogen = np.concatenate([points, np.ones((num_points_in_shape, 1))], axis=1)
                points_homogen = np.matmul(trans_mat, points_homogen.transpose()).transpose()
                points = points_homogen[:, :3]
                # Select color
                if target=="O":
                    color = np.array([0.5804, 0.0, 0.8275])
                else:
                    color = np.array([0, 0, 0])

                # Embedding: xyzrgbw
                rgb = np.tile(color, (num_points_in_shape, 1))
                w = np.full((num_points_in_shape, 1), weight)
                pcd = np.concatenate([points, rgb, w], axis=1)

                # Collect
                list_object_pointclouds.append(pcd)

            # Union the point clouds of one state particle (we do not use normal for now...)
            state_pointcloud = np.concatenate(list_object_pointclouds, axis=0)
            list_state_pointclouds.append(state_pointcloud)
        # Union belief
        belief_pointcloud = np.concatenate(list_state_pointclouds, axis=0)

        # Debugging
        # visualize_point_cloud([belief_pointcloud[:, :3]])

        # 3. Goal condition
        goal_condition = np.array(goal)
        # print(goal_condition)


        belief_pointcloud = th.from_numpy(belief_pointcloud).float()
        belief_pointcloud = belief_pointcloud.unsqueeze(0)  # batchify
        goal_condition = th.from_numpy(goal_condition).float()
        goal_condition = goal_condition.unsqueeze(0)        # batchify

        # print(belief_pointcloud.shape)
        # print(goal_condition.shape)

        return belief_pointcloud, goal_condition