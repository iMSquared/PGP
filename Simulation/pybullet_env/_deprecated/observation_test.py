import argparse
import os
import yaml

# PyBullet
import pybullet as pb
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env import BinpickEnv
from envs.robot import UR5, FrankaPanda
from envs.manipulation import Manipulation

# POMDP
from POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from online_planner_framework import Planner
from fetching_POMDP import *
from POMCPOW import POMCPOW

# Metric
from metric.mesh_iou import calculate_mesh_iou_between_pair

def main(config):
    """ Simply run this script, and it will generate `observation.csv` in root directory."""

    
    # Connect bullet client
    sim_id = pb.connect(pb.GUI)
    if sim_id < 0:
        raise ValueError("Failed to connect to pybullet!")
    bc = BulletClient(sim_id)
    #bc.setPhysicsEngineParameter(enableFileCaching=0)   # Turn off file caching


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
    # env
    binpick_env = BinpickEnv(bc, config)
    # robot
    if config["project_params"]["robot"] == "ur5":
        robot = UR5(bc, config)
    elif config["project_params"]["robot"] == "franka_panda":
        robot = FrankaPanda(bc, config)
    else:
        raise Exception("invalid robot")
    # manipulation
    manipulation = Manipulation(bc, binpick_env, robot, config)

    # Stabilize the environment
    manipulation.wait(100)
    
    # POMDP
    observation_model = FetchingObservationModel(bc, binpick_env, robot, config)
    # Set initial state    
    robot_state, object_state = get_state_info(bc, binpick_env, robot, config)  # Array values
    init_state = FetchingState(bc, binpick_env, robot_state, object_state)      # Keeps uid
    init_obs = observation_model.get_sensor_observation(init_state, None)       # Wrapping class for depth image and segmentation mask
    # Set initial belief
    shape_folder = os.path.join(os.path.expanduser('~'), "workspace/POMDP/Initial_Belief/gaussian/created")
    shape_table, cls_indices = make_shape_table(shape_folder)                   # shape_table stores all initial belief urdfs.            
    binpick_env.shape_table = shape_table                                       # cls_indices stores the start and last indices of the class
    # WeightedParticels.particles is the dictionary, which consists of...
    # - key: FetchingState instance
    # - value: particle weight
    init_belief:WeightedParticles = make_belief(bc, binpick_env, robot, config, shape_table, cls_indices, config["plan_params"]["num_sims"])
    # Make agent... for convenience.
    agent = FetchingAgent(bc, binpick_env, robot, config, None, None, None)


    # For 3D IoU check, create a copy of ground truth object
    gt_objects_uid, gt_pos_list, gt_orn_q_list = create_ground_truth_objects_uid(bc, binpick_env, config)
    send_ground_truth_faraway(bc, gt_objects_uid)

    # Partly cited from `agent.update()`
    urdf_paths = []          # 2D list
    object_positions = []    # 2D list
    object_orientations_q = [] # 2D list
    object_scales = []       # 2D list
    pdfs = []
    ious = []
    cur_particle = init_belief.particles
    for i, (p, v) in enumerate(cur_particle.items()):

        # Force set simulation
        agent._set_simulation(p, None, None)
        
        # Observation update
        pdf = observation_model.probability(init_obs, None, None, log=True)

        # Now, let's measure the 3D IoU
        restore_ground_truth(bc, gt_objects_uid, gt_pos_list, gt_orn_q_list)
        iou_mean = calculate_mesh_iou_between_pair(bc, binpick_env.objects_uid, gt_objects_uid, reduction="mean")

        # Log
        print(f"{i}) iou: {iou_mean}, pdf: {pdf}")
        ious.append(iou_mean)
        pdfs.append(pdf)

        # Send them away again
        send_ground_truth_faraway(bc, gt_objects_uid)

        # Log current particle information
        urdf_paths_per_instance = []
        object_positions_per_instance = []
        object_orientations_per_instance = []
        object_scales_per_instance = []
        for shape, (pos, orn, scale, target) in p.object.items():
            urdf_paths_per_instance.append(shape)
            object_positions_per_instance.append(pos)
            object_orientations_per_instance.append(orn)
            object_scales_per_instance.append(scale)    
        urdf_paths.append(urdf_paths_per_instance)
        object_positions.append(object_positions_per_instance)
        object_orientations_q.append(object_orientations_per_instance)
        object_scales.append(object_scales_per_instance)

    # Normalize pdf weights
    pdfs = np.asarray(pdfs)
    pdfs = pdfs / np.sum(pdfs)

    # Below here is for writing the csv log.
    data = []
    # First row contains the ground truth information
    gt_objects_list = config["env_params"]["binpick_env"]["objects"]["path"]
    data.append([gt_objects_list[0], gt_pos_list[0], bc.getEulerFromQuaternion(gt_orn_q_list[0]),
                 gt_objects_list[1], gt_pos_list[1], bc.getEulerFromQuaternion(gt_orn_q_list[1]), 0, 0])
    # Add rows
    for i in range(len(urdf_paths)):
        from pathlib import Path
        data_row = [os.path.join(*Path(urdf_paths[i][0]).parts[5:]), object_positions[i][0], object_orientations_q[i][0],
                    os.path.join(*Path(urdf_paths[i][1]).parts[5:]), object_positions[i][1], object_orientations_q[i][1],
                    ious[i], pdfs[i]]
        data.append(data_row)
    # Write csv file
    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv("./observation.csv")

    # Hold the simulation
    while True:
        bc.stepSimulation()


def create_ground_truth_objects_uid(bc, binpick_env:BinpickEnv, config)->List[int]:
    """Please call this function after stabilization."""

    # Back up pos and orn
    gt_pos_list = []
    gt_orn_q_list = []
    for uid in binpick_env.objects_uid:
        pos, orn = bc.getBasePositionAndOrientation(uid)
        gt_pos_list.append(pos)
        gt_orn_q_list.append(orn)

    # Spawn object somewhere far away...
    env_params = config["env_params"]["binpick_env"]
    gt_objects_uid = ([    
        bc.loadURDF(
            fileName        = os.path.join(ycb_objects.getDataPath(), env_params["objects"]["path"][i]),
            basePosition    = [10.0, 0.0, 0.0],
            baseOrientation = bc.getQuaternionFromEuler([0.0, 0.0 ,0.0]),
            useFixedBase    = False,
            globalScaling   = env_params["objects"]["scale"][i])
        for i in range(env_params["objects"]["num_objects"])
    ])

    # Adjust dynamics
    for uid in gt_objects_uid:
        bc.changeDynamics(
            uid, 
            -1, 
            lateralFriction=0.8,
            rollingFriction=0.0004,
            spinningFriction=0.0004,
            restitution=0.2)


    return gt_objects_uid, gt_pos_list, gt_orn_q_list


def send_ground_truth_faraway(bc, gt_objects_uid):
    """ Some ridiculus function that throws the objects away somewhere """    
    for uid in gt_objects_uid:
        bc.resetBasePositionAndOrientation(uid, [10.0, 0.0, 0.0], bc.getQuaternionFromEuler([0.0, 0.0 ,0.0]))


def restore_ground_truth(bc, gt_objects_uid, gt_pos_list, gt_orn_q_list):
    """ This function brings the objects back to its original place... """

    for i in range(len(gt_objects_uid)):
        uid = gt_objects_uid[i]
        pos = gt_pos_list[i]
        orn = gt_orn_q_list[i]
        bc.resetBasePositionAndOrientation(uid, pos, orn)



    

if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)

    