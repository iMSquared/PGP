import argparse
import os
import yaml
import numpy as np
import math

from copy import deepcopy

# PyBullet
import pybullet as pb
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env import BinpickEnv
from envs.robot import UR5, UR5Suction
from envs.manipulation import Manipulation
import time

# Typing
from typing import Optional, Tuple, List, Dict
from imm.pybullet_util.typing_extra import TranslationT, Tuple3, QuaternionT




# Global variables for visualizer...
line_uid_xyz_ur5_link_in_world: Tuple = None
line_uid_xyz_tip_end_target_in_world: Tuple = None
line_uid_xyz_ee_base_target_in_world: Tuple = None




def main(config):    
    
    # Connect bullet client
    if config["project_params"]["debug"]["show_gui"]:
        sim_id = pb.connect(pb.GUI)
    else:
        sim_id = pb.connect(pb.DIRECT)
    if sim_id < 0:
        raise ValueError("Failed to connect to pybullet!")
    bc = BulletClient(sim_id)

    # Use external GPU for rendering when available
    if config["project_params"]["use_nvidia"] == True:
        import pkgutil
        egl = pkgutil.get_loader('eglRenderer')
        if (egl):
            eglPluginId = bc.loadPlugin(egl.get_filename(), "_eglRendererPlugin")

    # Sim params
    CONTROL_DT = 1. / config["sim_params"]["control_hz"]
    bc.setTimeStep(CONTROL_DT)
    bc.setGravity(0, 0, config["sim_params"]["gravity"])
    bc.resetDebugVisualizerCamera(
        cameraDistance       = config["sim_params"]["debug_camera"]["distance"], 
        cameraYaw            = config["sim_params"]["debug_camera"]["yaw"], 
        cameraPitch          = config["sim_params"]["debug_camera"]["pitch"], 
        cameraTargetPosition = config["sim_params"]["debug_camera"]["target_position"])
    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING)
    

    # Simulation initialization
    binpick_env = BinpickEnv(bc, config)
    robot = UR5Suction(bc, config)


    # Stabilize
    for i in range(1000):
        bc.stepSimulation()


    # Let's start with some arbitrary object pose
    obj_uid = binpick_env.objects_uid[1]
    obj_pos, obj_orn_q = bc.getBasePositionAndOrientation(obj_uid)
    obj_pos = list(obj_pos)
    # Assume the grasp point is sampled somehow... This is a simulated normal.
    relative_contact_pos = [-0.05, -0.00, 0.02]
    relative_contact_normal = np.array([-0.2, -0.2, 1])/np.linalg.norm([-0.2, -0.2, 1]) # Some vector in object base frame


    # Make a grasp pose
    ee_base_link_target_pos, ee_base_link_target_orn_q \
        = simulated_grasp_pose_sampler(bc, robot, binpick_env, obj_pos, obj_orn_q, relative_contact_pos, relative_contact_normal)

    # Realign the end effector wrist with robot... 
    ee_base_link_target_pos, ee_base_link_target_orn_q \
        = robot.get_overriden_pose_wrist_joint_zero_position(ee_base_link_target_pos, ee_base_link_target_orn_q)


    # Pick!
    pick_success = pick(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)
    
    # Move! (Deprecate?)
    ee_base_link_target_pos += np.array([-0.1, 0.1, 0.3])
    move(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)


    # Place!
    # Rotating end effector pose (yaw in world frame)
    place_rotation_e_in_world_frame = [0.0, 0.0, -1.57]
    _, ee_base_link_target_orn_q = bc.multiplyTransforms([0, 0, 0], bc.getQuaternionFromEuler(place_rotation_e_in_world_frame),
                                                                [0, 0, 0], ee_base_link_target_orn_q)
    ee_base_link_target_pos += np.array([0.0, 0.1, -0.3])
    place(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)

    # Move! (Deprecate?)
    ee_base_link_target_pos += np.array([0.0, -0.3, 0.3])
    move(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)



    # Let's start with some arbitrary object pose
    obj_uid = binpick_env.objects_uid[1]
    obj_pos, obj_orn_q = bc.getBasePositionAndOrientation(obj_uid)
    obj_pos = list(obj_pos)
    # Assume the grasp point is sampled somehow... This is a simulated normal
    relative_contact_pos = [-0.05, -0.00, 0.02]
    relative_contact_normal = np.array([-0.2, -0.2, 1])/np.linalg.norm([-0.2, -0.2, 1]) # Some vector in object base frame


    # Make a grasp pose
    ee_base_link_target_pos, ee_base_link_target_orn_q \
        = simulated_grasp_pose_sampler(bc, robot, binpick_env, obj_pos, obj_orn_q, relative_contact_pos, relative_contact_normal)

    # Realign the end effector wrist with robot... 
    ee_base_link_target_pos, ee_base_link_target_orn_q \
        = robot.get_overriden_pose_wrist_joint_zero_position(ee_base_link_target_pos, ee_base_link_target_orn_q)

    # Pick!
    pick_success = pick(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)

    # Move! (Deprecate?)
    ee_base_link_target_pos += np.array([0.0, -0.1, 0.3])
    move(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)

    # Place!
    ee_base_link_target_pos += np.array([0.0, -0.1, -0.3])
    place(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)

    # Move! (Deprecate?)
    ee_base_link_target_pos += np.array([0.0, 0.2, 0.3])
    move(bc, robot, binpick_env, ee_base_link_target_pos, ee_base_link_target_orn_q)



    while True:
        bc.stepSimulation()
        time.sleep(1/240.)



def target_surface_normal_to_ee_link_pose(bc: BulletClient, 
                                          robot: UR5Suction,
                                          target_surface_normal_pos_in_world, 
                                          target_surface_normal_orn_e_in_world) -> Tuple[TranslationT, QuaternionT]:
    """Get the pos of the end effector base from the surface normal.

    Args:
        bc (_type_): _description_
        target_surface_normal_pos_in_world (_type_): _description_
        target_surface_normal_orn_e_in_world (_type_): _description_

    Returns:
        _type_: _description_
    """

    global line_uid_xyz_tip_end_target_in_world
    global line_uid_xyz_ee_base_target_in_world


    # Get the tip-end coordinate in world frame. This flips the contact point normal inward.
    tip_end_target_pos_in_world, tip_end_target_orn_q_in_world \
        = bc.multiplyTransforms(target_surface_normal_pos_in_world, bc.getQuaternionFromEuler(target_surface_normal_orn_e_in_world),
                                [0.0, 0.0, 0.0], bc.getQuaternionFromEuler([3.1416, 0.0, 0.0]))
    # Draw debug line
    line_uid_xyz_tip_end_target_in_world \
        = draw_coordinate(bc,
                          tip_end_target_pos_in_world, 
                          bc.getEulerFromQuaternion(tip_end_target_orn_q_in_world),
                          line_uid_xyz = line_uid_xyz_tip_end_target_in_world,
                          brightness = 0.5)

    # Target in world frame -> tip-end frame -> ur5 ee base link frame
    ee_base_link_pos_in_tip_end_frame = [0.0, 0.0, -robot.gripper_base_to_tip_stroke]   # NOTE(ssh): Okay... 12cm is appropriate
    ee_base_link_orn_q_in_tip_end_frame = bc.getQuaternionFromEuler([0.0, 0.0, 0.0])
    ee_base_link_target_pos_in_world, ee_base_link_target_orn_q_in_world \
        = bc.multiplyTransforms(tip_end_target_pos_in_world, tip_end_target_orn_q_in_world,
                                ee_base_link_pos_in_tip_end_frame, ee_base_link_orn_q_in_tip_end_frame)
    
    
    line_uid_xyz_ee_base_target_in_world = draw_coordinate(bc,
                                                            ee_base_link_target_pos_in_world,
                                                            bc.getEulerFromQuaternion(ee_base_link_target_orn_q_in_world),
                                                            line_uid_xyz = line_uid_xyz_ee_base_target_in_world, 
                                                            brightness = 0.5)


    return ee_base_link_target_pos_in_world, ee_base_link_target_orn_q_in_world





def simulated_grasp_pose_sampler(bc: BulletClient,
                      robot: UR5Suction,
                      binpick_env: BinpickEnv,
                      obj_pos, 
                      obj_orn_q, 
                      relative_contact_pos, 
                      relative_contact_normal: np.ndarray):

    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
        
        Args:
            vec1: A 3d "source" vector
            vec2: A 3d "destination" vector
        
        Returns:
            mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

        return rotation_matrix

    # 1. Represent the orientation of contact point in world frame
    # NOTE(ssh): I think this formulation can have a null space in z-direction... But I'm going to override z anyway.
    from scipy.spatial.transform import Rotation
    relative_contact_orn_m = rotation_matrix_from_vectors(np.array([0, 0, 1]), relative_contact_normal)    
    R_relative_contact_orn = Rotation.from_matrix(relative_contact_orn_m)
    relative_contact_orn_q = R_relative_contact_orn.as_quat()
    tip_end_target_in_world_pos, tip_end_target_in_world_orn_q = bc.multiplyTransforms(obj_pos, obj_orn_q,
                                                                                       relative_contact_pos, relative_contact_orn_q)
    

    # 2. Converting the contact point to ur5_ee_link pose.
    tip_end_target_in_world_orn_e = bc.getEulerFromQuaternion(tip_end_target_in_world_orn_q)
    #    Target contact point -> target tip-end pose -> target UR5 EE link pose
    ee_base_link_target_pos_in_world, ee_base_link_target_orn_q_in_world \
         = target_surface_normal_to_ee_link_pose(bc, robot,
                                                 tip_end_target_in_world_pos, 
                                                 tip_end_target_in_world_orn_e)

    return ee_base_link_target_pos_in_world, ee_base_link_target_orn_q_in_world







def pick(bc: BulletClient,
         robot: UR5Suction,
         binpick_env: BinpickEnv,
         ee_base_link_target_pos,
         ee_base_link_target_orn_q) -> bool:

    global line_uid_xyz_ur5_link_in_world

    # 1. Suction gripper is z-invariant. Solve IK and override gripper rotation
    joint_position_list = list( bc.calculateInverseKinematics(robot.uid, 
                                                              robot.link_index_endeffector_base, 
                                                              ee_base_link_target_pos, 
                                                              ee_base_link_target_orn_q, 
                                                              maxNumIterations = 10000, 
                                                              residualThreshold = 1e-6) )
    joint_position_list[-1] = 0.0

    # 2. Control
    current_joint_state_list = bc.getJointStates(robot.uid, robot.joint_indices_arm)
    current_joint_position_list = [state[0] for state in current_joint_state_list]
    trajectory = interpolate_trajectory(current_joint_position_list, joint_position_list, 2.5, 1/240.)
    for traj in trajectory:
        # Control
        bc.setJointMotorControlArray(robot.uid, 
                                     robot.joint_indices_arm, 
                                     bc.POSITION_CONTROL, 
                                     traj)
        bc.stepSimulation()
        # Debug
        link_info = bc.getLinkState(robot.uid, robot.link_index_endeffector_base)
        link_pos = link_info[4]
        link_orn_e = bc.getEulerFromQuaternion(link_info[5])
        line_uid_xyz_ur5_link_in_world = draw_coordinate(bc, 
                                                        link_pos,
                                                        link_orn_e,
                                                        line_uid_xyz = line_uid_xyz_ur5_link_in_world) 
        time.sleep(1/240.)
    # Stabilize
    for i in range(50):
        bc.stepSimulation()
        # Debug
        link_info = bc.getLinkState(robot.uid, robot.link_index_endeffector_base)
        link_pos = link_info[4]
        link_orn_e = bc.getEulerFromQuaternion(link_info[5])
        line_uid_xyz_ur5_link_in_world = draw_coordinate(bc, 
                                                        link_pos,
                                                        link_orn_e,
                                                        line_uid_xyz = line_uid_xyz_ur5_link_in_world) 
        time.sleep(1/240.)
        
    # 6. Grasp
    if robot.detect_contact():
        robot.activate(binpick_env.objects_uid)


    # 7. Return relative transform
    return robot.check_grasp()



def move(bc: BulletClient,
         robot: UR5Suction,
         binpick_env: BinpickEnv,
         ee_base_link_target_pos, 
         ee_base_link_target_orn_q):
    

    global line_uid_xyz_ur5_link_in_world

    # NOTE(ssh): Okay... much simpler than pick because all the transformation was done already at there.
    # 1. This time, we do not override the joint.
    joint_position_list = bc.calculateInverseKinematics(robot.uid, 
                                                              robot.link_index_endeffector_base, 
                                                              ee_base_link_target_pos, 
                                                              ee_base_link_target_orn_q, 
                                                              maxNumIterations = 1000, 
                                                              residualThreshold = 1e-6)

    # 2. Control
    current_joint_state_list = bc.getJointStates(robot.uid, robot.joint_indices_arm)
    current_joint_position_list = [state[0] for state in current_joint_state_list]
    trajectory = interpolate_trajectory(current_joint_position_list, joint_position_list, 2.5, 1/240.)
    for traj in trajectory:
        # Control
        bc.setJointMotorControlArray(robot.uid, 
                                     robot.joint_indices_arm, 
                                     bc.POSITION_CONTROL, 
                                     traj)
        bc.stepSimulation()
        # Debug
        link_info = bc.getLinkState(robot.uid, robot.joint_index_last)
        link_pos = link_info[4]
        link_orn_e = bc.getEulerFromQuaternion(link_info[5])
        line_uid_xyz_ur5_link_in_world = draw_coordinate(bc, 
                                                        link_pos,
                                                        link_orn_e,
                                                        line_uid_xyz = line_uid_xyz_ur5_link_in_world) 
        time.sleep(1/240.)
    # Stabilize
    for i in range(50):
        bc.stepSimulation()
        time.sleep(1/240.)



def place(bc: BulletClient,
         robot: UR5Suction,
         binpick_env: BinpickEnv,
         ee_base_link_target_pos, 
         ee_base_link_target_orn_q):
    
    global line_uid_xyz_ur5_link_in_world

    # 1. This time, we do not override the joint.
    joint_position_list = bc.calculateInverseKinematics(robot.uid, 
                                                              robot.link_index_endeffector_base, 
                                                              ee_base_link_target_pos, 
                                                              ee_base_link_target_orn_q, 
                                                              maxNumIterations = 1000, 
                                                              residualThreshold = 1e-6)


    # 2. Control
    current_joint_state_list = bc.getJointStates(robot.uid, robot.joint_indices_arm)
    current_joint_position_list = [state[0] for state in current_joint_state_list]
    trajectory = interpolate_trajectory(current_joint_position_list, joint_position_list, 2.5, 1/240.)
    for traj in trajectory:
        # Control
        bc.setJointMotorControlArray(robot.uid, 
                                     robot.joint_indices_arm, 
                                     bc.POSITION_CONTROL, 
                                     traj)
        bc.stepSimulation()
        # Debug
        link_info = bc.getLinkState(robot.uid, robot.joint_index_last)
        link_pos = link_info[4]
        link_orn_e = bc.getEulerFromQuaternion(link_info[5])
        line_uid_xyz_ur5_link_in_world = draw_coordinate(bc, 
                                                        link_pos,
                                                        link_orn_e,
                                                        line_uid_xyz = line_uid_xyz_ur5_link_in_world) 
        time.sleep(1/240.)
    # Stabilize
    for i in range(50):
        bc.stepSimulation()
        time.sleep(1/240.)



    # 4. Deactivate
    robot.release()



def interpolate_trajectory(cur: List, goal: List, action_duration: float, control_dt: float) -> np.ndarray:
    '''
    This function returns linear-interpolated (dividing straight line)
    trajectory between current and goal pose.
    Acc, Jerk is not considered.
    '''
    # Interpolation steps
    steps = math.ceil(action_duration/control_dt)
    
    # Calculate difference
    delta = [ goal[i] - cur[i] for i in range(len(cur)) ]
    
    # Linear interpolation
    trajectory:Tuple[np.ndarray, ...] = ([
        np.array([
            cur[j] + ( delta[j] * float(i)/float(steps) ) 
            for j in range(len(cur))
        ])
        for i in range(1, steps+1)
    ])

    return trajectory



def draw_coordinate(bc: BulletClient, 
                    target_pos: Optional[TranslationT]=None, 
                    target_orn_e: Optional[Tuple3]=None, 
                    parent_object_unique_id: Optional[int] = None, 
                    parent_link_index: Optional[int] = None, 
                    line_uid_xyz: Optional[Tuple[int, int, int]] = None,
                    brightness: float = 1.0) -> Tuple[int, int, int]:
    """Draw coordinate frame

    Args:
        bc (BulletClient): PyBullet client
        target_pos (Optional[TranslationT], optional): Position of local frame in global frame
        target_orn_e (Optional[Tuple3], optional): Orientation of local frame in global frame
        parent_object_unique_id (Optional[int], optional): Local frame? Defaults to None.
        parent_link_index (Optional[int], optional): Local frame? Defaults to None.
        line_uid (Tuple[int, int, int], optional): Replace uid. Defaults to None.
        brightness (float): Color brightness

    Returns:
        line_uid_xyz (Tuple[int, int, int]): Line uid
    """

    origin_pos = [0.0, 0.0, 0.0]
    x_pos = [0.1, 0.0, 0.0]
    y_pos = [0.0, 0.1, 0.0]
    z_pos = [0.0, 0.0, 0.1]
    origin_orn_e = [0.0, 0.0, 0.0]


    if parent_object_unique_id is not None:
        if line_uid_xyz is not None:
            line_uid_x = bc.addUserDebugLine(origin_pos, x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[0])
            line_uid_y = bc.addUserDebugLine(origin_pos, y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[1])
            line_uid_z = bc.addUserDebugLine(origin_pos, z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index,
                                            replaceItemUniqueId = line_uid_xyz[2])
        else:
            line_uid_x = bc.addUserDebugLine(origin_pos, x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
            line_uid_y = bc.addUserDebugLine(origin_pos, y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
            line_uid_z = bc.addUserDebugLine(origin_pos, z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01, 
                                            parentObjectUniqueId = parent_object_unique_id,
                                            parentLinkIndex = parent_link_index)
    else:
        target_origin_pos, target_origin_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                                       origin_pos, bc.getQuaternionFromEuler(origin_orn_e)) 
        target_x_pos, target_x_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                             x_pos, bc.getQuaternionFromEuler(origin_orn_e))
        target_y_pos, target_y_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                             y_pos, bc.getQuaternionFromEuler(origin_orn_e))
        target_z_pos, target_z_orn_q = bc.multiplyTransforms(target_pos, bc.getQuaternionFromEuler(target_orn_e),
                                                             z_pos, bc.getQuaternionFromEuler(origin_orn_e))
        
        if line_uid_xyz is not None:
            line_uid_x = bc.addUserDebugLine(target_origin_pos, target_x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01,
                                            replaceItemUniqueId = line_uid_xyz[0])
            line_uid_y = bc.addUserDebugLine(target_origin_pos, target_y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01,
                                            replaceItemUniqueId = line_uid_xyz[1])
            line_uid_z = bc.addUserDebugLine(target_origin_pos, target_z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01,
                                            replaceItemUniqueId = line_uid_xyz[2])
        else:
            line_uid_x = bc.addUserDebugLine(target_origin_pos, target_x_pos, [1*brightness, 0, 0], 
                                            lineWidth = 0.01)
            line_uid_y = bc.addUserDebugLine(target_origin_pos, target_y_pos, [0, 1*brightness, 0], 
                                            lineWidth = 0.01)
            line_uid_z = bc.addUserDebugLine(target_origin_pos, target_z_pos, [0, 0, 1*brightness], 
                                            lineWidth = 0.01)

    return (line_uid_x, line_uid_y, line_uid_z)



if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_test_suction_gripper.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)