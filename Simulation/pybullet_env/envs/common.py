import pybullet as pb

from typing import Dict, Tuple

from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import UR5Suction
from envs.manipulation import Manipulation




def init_new_bulletclient(config      : Dict, 
                          force_direct: bool = False,
                          stabilize   : bool = True) \
        -> Tuple[ BulletClient, BinpickEnvPrimitive, UR5Suction, Manipulation ]:
    """Some common routines of bullet client initialization.
    This function does not destroy previous client.

    Args:
        config (Dict): Configuration file
        force_direct (bool): Force the pybullet to be in direct mode. Used in the belief update.
        stabilize (bool): When true, steps several times to stabilize the environment.
        
    Returns:
        bc (BulletClient): New client
        sim_env (BinpickEnvPrimitive): New env
        robot (UR5Suction): New robot
    """

    # Configuration
    DEBUG_SHOW_GUI         = config["project_params"]["debug"]["show_gui"]
    CONTROL_HZ             = config["sim_params"]["control_hz"]
    GRAVITY                = config["sim_params"]["gravity"]
    CAMERA_DISTANCE        = config["sim_params"]["debug_camera"]["distance"]
    CAMERA_YAW             = config["sim_params"]["debug_camera"]["yaw"]
    CAMERA_PITCH           = config["sim_params"]["debug_camera"]["pitch"]
    CAMERA_TARGET_POSITION = config["sim_params"]["debug_camera"]["target_position"]

    # Connect bullet client
    if DEBUG_SHOW_GUI and not force_direct:
        sim_id = pb.connect(pb.GUI)
    else:
        sim_id = pb.connect(pb.DIRECT)
    bc = BulletClient(sim_id)

    # Sim params
    CONTROL_DT = 1. / CONTROL_HZ
    bc.setTimeStep(CONTROL_DT)
    bc.setGravity(0, 0, GRAVITY)
    bc.resetDebugVisualizerCamera(
        cameraDistance       = CAMERA_DISTANCE, 
        cameraYaw            = CAMERA_YAW, 
        cameraPitch          = CAMERA_PITCH, 
        cameraTargetPosition = CAMERA_TARGET_POSITION )
    bc.configureDebugVisualizer(bc.COV_ENABLE_RENDERING)

    # Simulation initialization
    sim_env      = BinpickEnvPrimitive(bc, config)
    robot        = UR5Suction(bc, config)
    manip        = Manipulation(bc, sim_env, robot, config)
    if stabilize:
        manip.wait(240)

    return bc, sim_env, robot, manip