import copy
import random
import numpy as np
import numpy.typing as npt
from typing import List, NamedTuple
import time
import open3d as o3d
from scipy.spatial.transform import Rotation


from imm.pybullet_util.bullet_client import BulletClient
from imm.pybullet_util.typing_extra import TranslationT, EulerT, QuaternionT
from envs.global_object_id_table import Gid_T
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import Robot, UR5, UR5Suction
from envs.manipulation import Manipulation
from envs.suction_grasp_pose_samplers import PointSamplingGraspSuction, TopPointSamplingGraspSuction
from utils.process_geometry import draw_coordinate, l3_norm, matrix_visual_to_com, matrix_com_to_base, random_sample_array_from_config

from pomdp.POMDP_framework import *
from pomdp.fetching_POMDP_primitive_object import (FetchingState, FetchingAction, FetchingObservation, BulletObjectPose,
                                                   ACTION_PICK, ACTION_PLACE,
                                                   set_binpickenv_state)


line_uids = None



# class SampleRandomTopPick:

#     def __init__(self, NUM_FILTER_TRIALS: int):
#         """Random pick action sampler"""
#         self.NUM_FILTER_TRIALS = NUM_FILTER_TRIALS


#     def __call__(self, bc: BulletClient,
#                        env: BinpickEnvPrimitive,
#                        robot: Robot,
#                        manip: Manipulation,
#                        state: FetchingState) -> FetchingAction:
#         """Sample a random pick action

#         Args:
#             bc (BulletClient)
#             env (BinpickEnvPrimitive)
#             robot (Robot)
#             manip (Manipulation)
#             state (FetchingState)

#         Raises:
#             ValueError: Pick sampler being called while holding an object
#             ValueError: No GID is sampled

#         Returns:
#             FetchingAction: Sampled Pick action
#         """
#         global line_uids


#         # Validation
#         holding_obj_gid = state.holding_obj_gid
#         if holding_obj_gid is not None:
#             raise ValueError("PICK sampler called while the robot is holding an object.")

#         # This variables prevents calculating affordance multiple times.
#         gid_to_boundary      : Gid_T       = dict()
#         gid_to_filteredpoints: Gid_T       = dict()

#         # Find a filtered grasp pose
#         for i in range(self.NUM_FILTER_TRIALS):
            
#             # Recover joint state before action. 
#             # This sampler may have modified the src pose in previous loop.
#             for value_i, joint_i in enumerate(robot.joint_indices_arm):
#                 bc.resetJointState(robot.uid, joint_i, state.robot_state[value_i])

#             # Select one of the objects.
#             aimed_gid = env.gid_table.select_random_gid()

#             # Parse the state of the target object
#             aimed_obj_uid = env.gid_to_uid[aimed_gid]
#             aimed_obj_pose = state.object_states[aimed_gid]

#             # Reuse the information if possible.
#             if aimed_gid not in gid_to_boundary.keys():
#                 # Before sampling, transform pcd and normals.
#                 pcd_points = env.gid_table[aimed_gid].shape_info.pcd_points
#                 pcd = o3d.geometry.PointCloud()
#                 pcd.points = o3d.utility.Vector3dVector(pcd_points)
#                 pcd_normals = env.gid_table[aimed_gid].shape_info.pcd_normals
#                 # Transform pcd to base frame.
#                 # Mesh frame -> URDF link frame -> URDF base frame
#                 T_v2l = matrix_visual_to_com(bc, aimed_obj_uid)
#                 T_l2b = matrix_com_to_base(bc, aimed_obj_uid)
#                 T_v2b = np.matmul(T_l2b, T_v2l)
#                 pcd_points = np.asarray(pcd.points)
#                 # asdf?
#                 afford_boundary, filtered_points \
#                     = TopPointSamplingGraspSuction\
#                         .make_afford_boundary_and_filtered_points(
#                             pcd_points = pcd_points, 
#                             pcd_normals = pcd_normals,
#                             obj_pos = aimed_obj_pose.pos,
#                             obj_orn = aimed_obj_pose.orn,
#                             T_v2b = T_v2b)
#                 gid_to_boundary[aimed_gid] = afford_boundary
#                 gid_to_filteredpoints[aimed_gid] = filtered_points
#             else:
#                 # Reuse
#                 afford_boundary = gid_to_boundary[aimed_gid]
#                 filtered_points = gid_to_filteredpoints[aimed_gid]
            
#             # Pick fail if no points can be picked.
#             if filtered_points is None:
#                 continue

#             # Sample one point from afford boundary
#             grasp_pos, grasp_orn, point_idx \
#                 = TopPointSamplingGraspSuction\
#                     .sample_grasp_in_boundary(
#                         afford_boundary,
#                         filtered_points)

#             # Handling permanently ungraspable object. (Hoping not exist...)
#             if grasp_pos is None:
#                 print(f"No plausible grasp pose exists for object gid={aimed_gid}")
#                 break

#             # Convert to ee base pose (inward, ee_base)
#             grasp_orn_q = bc.getQuaternionFromEuler(grasp_orn)                  # Outward orientation (surface normal)
#             modified_pos_back, modified_orn_q_back, \
#                 modified_pos, modified_orn_q \
#                 = manip.get_ee_pose_from_target_pose(grasp_pos, grasp_orn_q)    # Inward orientation (ee base)
            
#             # Solve IK
#             joint_pos_src, joint_pos_dst \
#                 = manip.solve_ik_numerical(modified_pos_back, modified_orn_q_back)

#             # Try motion planning
#             traj = manip.motion_plan(joint_pos_src, joint_pos_dst, None)
#             if traj is not None:
#                 # Found some nice motion planning. Return here.
#                 return FetchingAction(type        = ACTION_PICK, 
#                                       aimed_gid   = aimed_gid,
#                                       pos         = tuple(grasp_pos),
#                                       orn         = tuple(grasp_orn),
#                                       traj        = traj,
#                                       delta_theta = None)
#             else:
#                 # Failed to find collision free trajectory.
#                 # Continue filtering until the iteration reaches the NUM_FILTER_TRIALS.

#                 # line_uids = draw_coordinate(bc = bc, 
#                 #                             target_pos=modified_pos_back, 
#                 #                             target_orn_e = bc.getEulerFromQuaternion(modified_orn_q_back), 
#                 #                             line_uid_xyz = line_uids)

#                 continue

#         # If reached here, it means that no feasible grasp pose is found.
#         # Returning infeasible action.
#         return FetchingAction(ACTION_PICK, None, None, None, None, None)            




class SampleRandomTopPickUncertainty:

    def __init__(self, NUM_FILTER_TRIALS: int):
        """Random pick action sampler"""
        self.NUM_FILTER_TRIALS = NUM_FILTER_TRIALS


    def __call__(self, bc: BulletClient,
                       env: BinpickEnvPrimitive,
                       robot: Robot,
                       manip: Manipulation,
                       init_observation: FetchingObservation,
                       history: Tuple[HistoryEntry],
                       state: FetchingState,
                       goal: Tuple) -> FetchingAction:
        """Sample a random pick action

        Args:
            bc (BulletClient)
            env (BinpickEnvPrimitive)
            robot (Robot)
            manip (Manipulation)
            init_observation (FetchingObservation)
            history (Tuple[HistoryEntry])
            state (FetchingState)
            goal (Tuple)
            
        Raises:
            ValueError: Pick sampler being called while holding an object
            ValueError: No GID is sampled

        Returns:
            FetchingAction: Sampled Pick action
        """

        # Validation
        holding_obj_gid = state.holding_obj_gid
        if holding_obj_gid is not None:
            raise ValueError("PICK sampler called while the robot is holding an object.")

        # This variables prevents calculating affordance multiple times.
        gid_to_boundary      : Gid_T       = dict()
        gid_to_filteredpoints: Gid_T       = dict()

        # Conditioned on the history (last observation)
        last_observation \
            = self.extract_last_observation_from_history(
                init_observation, history)
        
        # Randomization: check the target visibility
        target_gid = env.gid_table.select_target_gid()
        is_target_in_obs: bool = np.array(np.sum(last_observation.seg_mask[target_gid]), dtype=bool).item()
        if not is_target_in_obs:
            #   If occluded, select the target object in position in the occluded region.
            pick_state = self.randomize_occluded_target_state(bc, env, robot, manip, state)
        else:
            #   If not occluded, use the particle pose.
            pick_state = state


        # Find a filtered grasp pose
        for i in range(self.NUM_FILTER_TRIALS):
            
            # Restore the joint state before action. 
            #   The sampler may have modified the src pose in previous loop.
            set_binpickenv_state(bc, env, robot, state)

            # Select one of the objects.
            aimed_gid = env.gid_table.select_random_gid()

            # Parse the state of the target object
            aimed_obj_uid = env.gid_to_uid[aimed_gid]
            aimed_obj_pose = pick_state.object_states[aimed_gid]

            # Reuse the information if possible.
            if aimed_gid not in gid_to_boundary.keys():
                # Before sampling, transform pcd and normals.
                pcd_points = env.gid_table[aimed_gid].shape_info.pcd_points
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_points)
                pcd_normals = env.gid_table[aimed_gid].shape_info.pcd_normals
                # Transform pcd to base frame.
                # Mesh frame -> URDF link frame -> URDF base frame
                T_v2l = matrix_visual_to_com(bc, aimed_obj_uid)
                T_l2b = matrix_com_to_base(bc, aimed_obj_uid)
                T_v2b = np.matmul(T_l2b, T_v2l)
                pcd_points = np.asarray(pcd.points)
                # asdf?
                afford_boundary, filtered_points \
                    = TopPointSamplingGraspSuction\
                        .make_afford_boundary_and_filtered_points(
                            pcd_points = pcd_points, 
                            pcd_normals = pcd_normals,
                            obj_pos = aimed_obj_pose.pos,
                            obj_orn = aimed_obj_pose.orn,
                            T_v2b = T_v2b)
                gid_to_boundary[aimed_gid] = afford_boundary
                gid_to_filteredpoints[aimed_gid] = filtered_points
            else:
                # Reuse
                afford_boundary = gid_to_boundary[aimed_gid]
                filtered_points = gid_to_filteredpoints[aimed_gid]
            
            # Pick fail if no points can be picked.
            if filtered_points is None:
                continue

            # Sample one point from afford boundary
            grasp_pos, grasp_orn, point_idx \
                = TopPointSamplingGraspSuction\
                    .sample_grasp_in_boundary(
                        afford_boundary,
                        filtered_points)

            # Handling permanently ungraspable object. (Hoping not exist...)
            if grasp_pos is None:
                print(f"No plausible grasp pose exists for object gid={aimed_gid}")
                break

            # Convert to ee base pose (inward, ee_base)
            grasp_orn_q = bc.getQuaternionFromEuler(grasp_orn)                  # Outward orientation (surface normal)
            modified_pos_back, modified_orn_q_back, \
                modified_pos, modified_orn_q \
                = manip.get_ee_pose_from_target_pose(grasp_pos, grasp_orn_q)    # Inward orientation (ee base)
            
            # Solve IK
            joint_pos_src, joint_pos_dst \
                = manip.solve_ik_numerical(modified_pos_back, modified_orn_q_back)

            # Try motion planning
            traj = manip.motion_plan(joint_pos_src, joint_pos_dst, None)
            if traj is not None:
                # Found some nice motion planning. Return here.
                return FetchingAction(type        = ACTION_PICK, 
                                      aimed_gid   = aimed_gid,
                                      pos         = tuple(grasp_pos),
                                      orn         = tuple(grasp_orn),
                                      traj        = traj,
                                      delta_theta = None)
            else:
                # Failed to find collision free trajectory.
                # Continue filtering until the iteration reaches the NUM_FILTER_TRIALS.
                continue

        # If reached here, it means that no feasible grasp pose is found.
        # Returning infeasible action.
        return FetchingAction(ACTION_PICK, None, None, None, None, None)            


    def extract_last_observation_from_history(self, init_observation: FetchingObservation,
                                                    history: Tuple[HistoryEntry]) -> FetchingObservation:
        """This function selects the last observation from the history.
        This function exists to handle when history is empty."""
        if len(history)!=0:
            return history[-1].observation
        else:
            return init_observation


    def randomize_occluded_target_state(self, bc: BulletClient,
                                              env: BinpickEnvPrimitive,
                                              robot: UR5Suction,
                                              manip: Manipulation,
                                              state: FetchingState,
                                              trials: int = 100) -> FetchingState:
        """ If the target object is not visible, 
        we can just randomly sample a pick pose in the occluded region.
        This is inspired from the initial belief generator
        
        Args:
            bc (BulletClient)
            env (BinpickEnvPrimitive)
            robot (UR5Suction)
            manip (Manipulation)
            state (FetchingState)
            trials (int)
        
        Returns:
            randomized_state (FetchingState)
        """
        # Original state
        robot_state   = state.robot_state
        object_states = state.object_states
        robot.release()

        # Randomize
        for i in range(trials):
            new_object_states: dict[Gid_T, BulletObjectPose] = dict()
            for obj_gid, obj_state in object_states.items():
                obj_uid = env.gid_to_uid[obj_gid]
                base_pos, base_orn_q = bc.getBasePositionAndOrientation(obj_uid)
                # Noising only the target object
                if env.gid_table[obj_gid].is_target:
                    random_pos = random_sample_array_from_config(center      = env.RANDOM_INIT_TARGET_POS_CENTER,
                                                                 half_ranges = env.RANDOM_INIT_TARGET_POS_HALF_RANGES)
                    random_orn = random_sample_array_from_config(center      = env.RANDOM_INIT_TARGET_ORN_CENTER,
                                                                 half_ranges = env.RANDOM_INIT_TARGET_ORN_HALF_RANGES)
                    random_pos[2] = base_pos[2]
                    new_obj_state = BulletObjectPose(tuple(random_pos), tuple(random_orn))
                else:
                    new_obj_state = copy.deepcopy(obj_state)
                # Save the state
                new_object_states[obj_gid] = new_obj_state
            # Force reset
            for obj_gid, obj_state in new_object_states.items():
                uid = env.gid_to_uid[obj_gid]
                bc.resetBasePositionAndOrientation(
                    uid, obj_state.pos, bc.getQuaternionFromEuler(obj_state.orn))
            # Check collision
            has_contact = env.check_objects_close(closest_threshold=0.003)
            # Check target object occlusion
            _, _, seg_mask = env.capture_rgbd_image(sigma = 0)
            target_gid     = env.gid_table.select_target_gid()
            is_target_in_obs: bool = np.array(np.sum(seg_mask[target_gid]), dtype=bool).item()
            # End randomization when no collision exist
            if not has_contact and not is_target_in_obs:
                break  
    
        # Create state from randomization
        randomized_state = FetchingState(robot_state, new_object_states, None)
        # Restore the argument state
        set_binpickenv_state(bc, env, robot, state)

        return randomized_state



# class SampleRandomPick:

#     def __init__(self, grasp_pose_sampler: PointSamplingGraspSuction,
#                        NUM_FILTER_TRIALS: int):
#         """Random pick action sampler"""
#         self.NUM_FILTER_TRIALS = NUM_FILTER_TRIALS
#         self.grasp_pose_sampler = grasp_pose_sampler


#     def __call__(self, bc: BulletClient,
#                        env: BinpickEnvPrimitive,
#                        robot: Robot,
#                        manip: Manipulation,
#                        state: FetchingState) -> FetchingAction:
#         """Sample a random pick action

#         Args:
#             bc (BulletClient)
#             env (BinpickEnvPrimitive)
#             robot (Robot)
#             manip (Manipulation)
#             state (FetchingState)

#         Raises:
#             ValueError: Pick sampler being called while holding an object
#             ValueError: No GID is sampled

#         Returns:
#             FetchingAction: Sampled Pick action
#         """
#         # Validation: Check holding object.
#         holding_obj_gid = state.holding_obj_gid
#         if holding_obj_gid is not None:
#             raise ValueError("PICK sampler called while the robot is holding an object.")

#         # Find a filtered grasp pose
#         for i in range(self.NUM_FILTER_TRIALS):
            
#             # Recover joint state before action. 
#             # This sampler may have modified the src pose in previous loop.
#             for value_i, joint_i in enumerate(robot.joint_indices_arm):
#                 bc.resetJointState(robot.uid, joint_i, state.robot_state[value_i])

#             # Select one of the objects.
#             aimed_gid = env.gid_table.select_random_gid()

#             # Parse the state of the target object
#             aimed_obj_uid = env.gid_to_uid[aimed_gid]
#             aimed_obj_pose = state.object_states[aimed_gid]
            
#             # Sample the grasp pose
#             # NOTE(ssh): This is an inappropriate bullet transform.
#             grasp_pos, grasp_orn = self.grasp_pose_sampler(pcd_points = env.gid_table[aimed_gid].shape_info.pcd_points, 
#                                                     pcd_normals = env.gid_table[aimed_gid].shape_info.pcd_normals,
#                                                     obj_pos = aimed_obj_pose.pos,
#                                                     obj_orn = aimed_obj_pose.orn)
#             # Convert to ee base pose (inward, ee_base)
#             grasp_orn_q = bc.getQuaternionFromEuler(grasp_orn)                  # Outward orientation (surface normal)
#             modified_pos_back, modified_orn_q_back, \
#                 modified_pos, modified_orn_q \
#                 = manip.get_ee_pose_from_target_pose(grasp_pos, grasp_orn_q)    # Inward orientation (ee base)

#             # Solve IK
#             joint_pos_src, joint_pos_dst \
#                 = manip.solve_ik_numerical(modified_pos_back, modified_orn_q_back)

#             # Try motion planning
#             traj = manip.motion_plan(joint_pos_src, joint_pos_dst, None)
#             if traj is not None:
#                 # Found some nice motion planning. Return here.
#                 return FetchingAction(type        = ACTION_PICK, 
#                                       aimed_gid   = aimed_gid,
#                                       pos         = tuple(grasp_pos),
#                                       orn         = tuple(grasp_orn),
#                                       traj        = traj,
#                                       delta_theta = None)
#             else:
#                 # Failed to find collision free trajectory.
#                 # Continue filtering until the iteration reaches the NUM_FILTER_TRIALS.
#                 continue

#         # If reached here, it means that no feasible grasp pose is found.
#         # Returning infeasible action.
#         return FetchingAction(ACTION_PICK, None, None, None, None, None)



class SampleRandomPlace:

    def __init__(self, NUM_FILTER_TRIALS: int):
        """Random pick action sampler"""
        self.NUM_FILTER_TRIALS      = NUM_FILTER_TRIALS


    def __call__(self, bc: BulletClient,
                       env: BinpickEnvPrimitive,
                       robot: Robot,
                       manip: Manipulation,
                       state: FetchingState,
                       prev_action: FetchingAction,
                       goal: Tuple[float, float, float, float, float]) -> FetchingAction:
        """Place action sampler
 
        Args:
            bc (BulletClient)
            env (BinpickEnvPrimitive)
            robot (Robot)
            manip (Manipulation)
            state (FetchingState)
            prev_action (FetchingAction)
            goal (Tuple[float, float, float, float, float]): [rgbxy]

        Returns:
            FetchingAction: Generated PLACE action
        """
        GOAL_POS_X = goal[3]
        GOAL_POS_Y = goal[4]

        # Find holding object.
        holding_obj_uid = env.gid_to_uid[state.holding_obj_gid]

        # Find a filtered grasp pose
        for i in range(self.NUM_FILTER_TRIALS):

            # Recover joint state before action. 
            # This sampler may have modified the src pose in previous loop.
            for value_i, joint_i in enumerate(robot.joint_indices_arm):
                bc.resetJointState(robot.uid, joint_i, state.robot_state[value_i])

            # Sample a place position
            place_region = np.random.choice(['cabinet', 'goal'], p=[0.8, 0.2])
            # Temp policy for debugging
            # if env.gid_table[state.holding_obj_gid].is_target:
            #     place_region = "goal"
            # else:
            #     place_region = "cabinet"
            # Compose position
            if place_region == 'cabinet':
                place_pos = random_sample_array_from_config(center      = env.TASKSPACE_CENTER, 
                                                            half_ranges = env.TASKSPACE_HALF_RANGES)
                place_pos[2] = prev_action.pos[2]
                place_pos = tuple(place_pos)
            elif place_region == 'goal':
                place_pos = tuple([GOAL_POS_X, GOAL_POS_Y, prev_action.pos[2]+0.02])
            # Compose orientation (relative yaw-rotation)
            prev_orn = prev_action.orn
            prev_orn_q = bc.getQuaternionFromEuler(prev_orn)
            delta_theta = random.uniform(-np.pi, np.pi)
            yaw_rot_in_world = np.array([0, 0, delta_theta])
            yaw_rot_in_world_q = bc.getQuaternionFromEuler(yaw_rot_in_world) 
            _, place_orn_q = bc.multiplyTransforms([0, 0, 0], yaw_rot_in_world_q,
                                                [0, 0, 0], prev_orn_q)
            place_orn = bc.getEulerFromQuaternion(place_orn_q)


            # Filtering by motion plan
            place_orn_q = bc.getQuaternionFromEuler(place_orn)                 # Outward orientation (surface normal)
            modified_pos_back, modified_orn_q_back, \
                modified_pos, modified_orn_q \
                = manip.get_ee_pose_from_target_pose(place_pos, place_orn_q)   # Inward orientation (ee base)
            joint_pos_src, joint_pos_dst \
                = manip.solve_ik_numerical(modified_pos, modified_orn_q)       # src is captured from current pose.


            # Try motion planning
            traj = manip.motion_plan(joint_pos_src, joint_pos_dst, holding_obj_uid) 
            if traj is not None:
                # Found some nice motion planning. Break here.
                # Double check: type, target, pos, orn, traj, delta_theta
                return FetchingAction(type        = ACTION_PLACE,
                                      aimed_gid   = prev_action.aimed_gid,
                                      pos         = tuple(place_pos),
                                      orn         = tuple(place_orn),
                                      traj        = traj,
                                      delta_theta = delta_theta)
            else:
                # Continue filtering until the iteration reaches the num_filter_trials.
                continue

        # If reached here, it means that no feasible place pose is found.
        # Returning infeasible action.
        return FetchingAction(ACTION_PLACE, None, None, None, None, None)