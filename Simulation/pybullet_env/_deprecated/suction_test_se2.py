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


    # Make a grasp pose
    target_pos = deepcopy(obj_pos)
    target_pos[2] = 0.42
    target_yaw = 0.0

    # Pick!
    pick_success = pick_se2(bc, robot, binpick_env, target_pos, target_yaw)

    # Move! (Deprecate?)
    target_pos += np.array([0.0, -0.1, 0.2])
    move_se2(bc, robot, binpick_env, target_pos, target_yaw)

    # Place!
    target_pos += np.array([-0.1, -0.15, -0.2])
    target_yaw -= 0.5
    place_se2(bc, robot, binpick_env, target_pos, target_yaw)

    # Move! (Deprecate?)
    target_pos += np.array([-0.1, 0.2, 0.2])
    move_se2(bc, robot, binpick_env, target_pos, target_yaw)


    # Let's start with some arbitrary object pose
    obj_uid = binpick_env.objects_uid[1]
    obj_pos, obj_orn_q = bc.getBasePositionAndOrientation(obj_uid)
    obj_pos = list(obj_pos)


    # Make a grasp pose
    target_pos = deepcopy(obj_pos)
    target_pos[2] = 0.42
    target_yaw = -0.5

    # Pick!
    pick_success = pick_se2(bc, robot, binpick_env, target_pos, target_yaw)

    # Move! (Deprecate?)
    target_pos += np.array([-0.1, 0.0, 0.2])
    move_se2(bc, robot, binpick_env, target_pos, target_yaw)

    # Place!
    target_pos += np.array([0.1, 0.0, -0.2])
    target_yaw -= 1.2
    place_se2(bc, robot, binpick_env, target_pos, target_yaw)

    # Move! (Deprecate?)
    target_pos += np.array([0.0, 0.45, 0.2])
    move_se2(bc, robot, binpick_env, target_pos, target_yaw)


    while True:
        bc.stepSimulation()
        time.sleep(1/240.)





def pick_se2(bc: BulletClient,
             robot: UR5Suction,
             binpick_env: BinpickEnv,
             target_pos,
             target_yaw) -> bool:

    global line_uid_xyz_ur5_link_in_world

    # Make SE3 gripper pose from pos and yaw
    ee_base_link_target_pos, ee_base_link_target_orn_e = robot.get_target_ee_pose_from_se2(target_pos, target_yaw)
    ee_base_link_target_orn_q = bc.getQuaternionFromEuler(ee_base_link_target_orn_e)


    # 1. Backward pose first
    poke_backward = robot.grasp_poke_backward
    ee_base_link_backward_pos, ee_base_link_backward_orn_q \
        = bc.multiplyTransforms(ee_base_link_target_pos, ee_base_link_target_orn_q,
                                [0, 0, -poke_backward], [0, 0, 0, 1])
    #   Solving IK
    joint_position_list = list( bc.calculateInverseKinematics(robot.uid, 
                                                              robot.link_index_endeffector_base, 
                                                              ee_base_link_backward_pos, 
                                                              ee_base_link_backward_orn_q, 
                                                              maxNumIterations = 10000, 
                                                              residualThreshold = 1e-6) )

    #   Control
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

    
    # 2. Approach for poke
    joint_position_list = list( bc.calculateInverseKinematics(robot.uid, 
                                                              robot.link_index_endeffector_base, 
                                                              ee_base_link_target_pos, 
                                                              ee_base_link_target_orn_q, 
                                                              maxNumIterations = 10000, 
                                                              residualThreshold = 1e-6) )

    #   Control
    current_joint_state_list = bc.getJointStates(robot.uid, robot.joint_indices_arm)
    current_joint_position_list = [state[0] for state in current_joint_state_list]
    trajectory = interpolate_trajectory(current_joint_position_list, joint_position_list, 2.5, 1/240.)
    for traj in trajectory:
        # Control
        bc.setJointMotorControlArray(robot.uid, 
                                     robot.joint_indices_arm, 
                                     bc.POSITION_CONTROL,
                                     traj,
                                     positionGains=[1.2]*len(robot.joint_indices_arm))
        bc.stepSimulation()
        # Debug
        link_info = bc.getLinkState(robot.uid, robot.link_index_endeffector_base)
        link_pos = link_info[4]
        link_orn_e = bc.getEulerFromQuaternion(link_info[5])
        line_uid_xyz_ur5_link_in_world = draw_coordinate(bc, 
                                                        link_pos,
                                                        link_orn_e,
                                                        line_uid_xyz = line_uid_xyz_ur5_link_in_world) 

        # Check grasp and break when success
        if robot.detect_contact():
            robot.activate(binpick_env.objects_uid)
        if robot.check_grasp():
            break

        time.sleep(1/240.)


    # Stabilize
    for i in range(20):
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
        

    # 7. Return grasp success flag
    return robot.check_grasp()



def move_se2(bc: BulletClient,
             robot: UR5Suction,
             binpick_env: BinpickEnv,
             target_pos, 
             target_yaw):
    

    global line_uid_xyz_ur5_link_in_world

    # 1. Grasp pose
    # Make SE3 gripper pose from pos and yaw
    ee_base_link_target_pos, ee_base_link_target_orn_e = robot.get_target_ee_pose_from_se2(target_pos, target_yaw)
    ee_base_link_target_orn_q = bc.getQuaternionFromEuler(ee_base_link_target_orn_e)
    #    Solve IK
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
    for i in range(20):
        bc.stepSimulation()
        time.sleep(1/240.)



def place_se2(bc: BulletClient,
         robot: UR5Suction,
         binpick_env: BinpickEnv,
         target_pos, 
         target_yaw):
    
    global line_uid_xyz_ur5_link_in_world

    # 1. Grasp pose
    #    Make SE3 gripper pose from pos and yaw
    ee_base_link_target_pos, ee_base_link_target_orn_e = robot.get_target_ee_pose_from_se2(target_pos, target_yaw)
    ee_base_link_target_orn_q = bc.getQuaternionFromEuler(ee_base_link_target_orn_e)
    #    Solve IK
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



    # 5. Backward pose
    poke_backward = robot.grasp_poke_backward
    ee_base_link_backward_pos, ee_base_link_backward_orn_q \
        = bc.multiplyTransforms(ee_base_link_target_pos, ee_base_link_target_orn_q,
                                [0, 0, -poke_backward], [0, 0, 0, 1])
    #   Solving IK
    joint_position_list = list( bc.calculateInverseKinematics(robot.uid, 
                                                              robot.link_index_endeffector_base, 
                                                              ee_base_link_backward_pos, 
                                                              ee_base_link_backward_orn_q, 
                                                              maxNumIterations = 10000, 
                                                              residualThreshold = 1e-6) )

    # 6. Retreat backward
    #   Control
    current_joint_state_list = bc.getJointStates(robot.uid, robot.joint_indices_arm)
    current_joint_position_list = [state[0] for state in current_joint_state_list]
    trajectory = interpolate_trajectory(current_joint_position_list, joint_position_list, 2.5, 1/240.)
    for traj in trajectory:
        # Control
        bc.setJointMotorControlArray(robot.uid, 
                                     robot.joint_indices_arm, 
                                     bc.POSITION_CONTROL,
                                     traj,
                                     positionGains=[1.2]*len(robot.joint_indices_arm))
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