import argparse
import os
import yaml
import time
import copy

import numpy as np

# PyBullet
import pybullet as pb
from pybullet_object_models import ycb_objects
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env import BinpickEnv
from envs.robot import UR5, FrankaPanda
from envs.manipulation import Manipulation

# POMDP
from POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from online_planner_framework import Planner
from fetching_POMDP import *

# Debugging
from grasp_test_result import grasp_test_result, reset
import pickle

    
def _make_shake_action(env:BinpickEnv, policy_model:FetchingRolloutPolicyModel, cur_state:FetchingState, pick_action:FetchingAction):
    obj_uid = env.objects_uid[0]
    obj_urdf = env.objects_urdf[obj_uid]
    orn = list(pick_action.orn)
    pos_1= list(cur_state.object[obj_urdf][0])
    pos_1[2] += 0.03
    pos_2 = copy.deepcopy(pos_1)
    pos_2[1] -= 0.1
    pos_3 = copy.deepcopy(pos_2)
    pos_3[1] += 0.2
    
    pos_shake = [pos_1, pos_2, pos_3]
    # pos_shake = [pos_1]
        
    shake_actions = [FetchingAction('MOVE', obj_uid, tuple(pos), tuple(orn), None) for pos in pos_shake]
    
    return shake_actions
    
    
def test(config, obj_info, time_limit=100):    
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
    transition_model = FetchingTransitionModel(bc, binpick_env, robot, config)
    observation_model = FetchingObservationModel(bc, binpick_env, robot, config)
    reward_model = FetchingRewardModel(bc, binpick_env, robot, config)
    blackbox_model = BlackboxModel(transition_model, observation_model, reward_model)

    policy_model = FetchingRolloutPolicyModel(bc, binpick_env, robot, config)
    rollout_policy_model = FetchingRolloutPolicyModel(bc, binpick_env, robot, config)

    agent = FetchingAgent(bc, binpick_env, robot, config, blackbox_model, policy_model, rollout_policy_model)
    env = Environment(transition_model, observation_model, reward_model)

    pomdp = POMDP(agent, env, "FetchingProblem")

    robot_init_state, _ = get_state_info(bc, binpick_env, robot, config)

    time_taken = 0.0
    while time_taken < time_limit:
        grasp_test_result['num_trial'] += 1
        print(f"Start {grasp_test_result['num_trial']} trial")
        
        # Reset robot joints
        robot.last_pose[robot.joint_indices_arm] = robot_init_state[0]
        robot.last_pose[robot.joint_index_finger] = robot_init_state[1]
        for i, idx in enumerate(robot.joint_indices_arm):
            bc.resetJointState(robot.uid, idx, robot_init_state[0][i])
        bc.resetJointState(robot.uid, robot.joint_index_finger, robot_init_state[1])
        target_position = -1.0 * robot.joint_gear_ratio_mimic * np.asarray(robot_init_state[1])
        for i, idx in enumerate(robot.joint_indices_finger_mimic):
            bc.resetJointState(robot.uid, idx, target_position[i])
        robot.last_pose[robot.joint_indices_finger_mimic] = np.asarray(robot_init_state[1])

        
        # Remove objects
        for obj_uid in binpick_env.objects_uid:
            bc.removeBody(obj_uid)

        # Reload objects
        objects_uids = []
        uid_to_urdf = {}
        uid = bc.loadURDF(
            fileName        = obj_info['urdf'],
            basePosition    = obj_info['pos'],
            baseOrientation = bc.getQuaternionFromEuler(obj_info['orn']),
            useFixedBase    = False,
            globalScaling   = obj_info['scale'])
        objects_uids.append(uid)
        uid_to_urdf[uid] = obj_info['urdf']
        
        # Store the uids to state and env
        binpick_env.objects_uid = objects_uids
        binpick_env.objects_urdf = uid_to_urdf
        
        # Adjust dynamics
        for uid in objects_uids:
            bc.changeDynamics(
                uid, 
                -1, 
                lateralFriction=0.8,
                rollingFriction=0.0004,
                spinningFriction=0.0004,
                restitution=0.2)

        for _ in range(100): 
            robot.update_arm_control()
            robot.update_finger_control()
            bc.stepSimulation()

        manipulation.wait(1000)

        # Set initial state    
        robot_state, object_state = get_state_info(bc, binpick_env, robot, config)

        init_state = FetchingState(bc, binpick_env, robot_state, object_state)

        pomdp.env.set_state(init_state, True)
        
        # Set actions
        t_taken_start = time.time()
        pick_action = policy_model.sample(None, init_state, prior=['PICK', None, None])
        t_taken_end = time.time()
        time_taken = time_taken + (t_taken_end - t_taken_start)
        
        if pick_action.pos is None:
            print("Pick fail: can't find a grasp pose")
            grasp_test_result["num_fail"] += 1
            grasp_test_result["reason"][0] += 1
        else:
            # t_taken_start = time.time()
            next_state, result = transition_model.sample(init_state, pick_action)
            # t_taken_end = time.time()
            # time_taken = time_taken + (t_taken_end - t_taken_start)
            
            if result:
                print("PICK success")
                grasp_test_result["num_grasp_success"] += 1
                
                shake_actions = _make_shake_action(binpick_env, policy_model, next_state, pick_action)
                for a in shake_actions:
                    next_state, result = transition_model.sample(next_state, a)
                    if result:
                        continue
                    else:
                        print("Shake fail")
                        grasp_test_result["num_fail"] += 1
                        grasp_test_result["reason"][2] += 1
                        break
            
            else:
                print("Pick fail: fail to close")
                grasp_test_result["num_fail"] += 1
                grasp_test_result["reason"][1] += 1
    
    bc.disconnect()

if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obj_class = [
        # "YcbBanana",
        # "YcbChipsCan",
        # "YcbCrackerBox",
        # "YcbFoamBrick",
        # "YcbGelatinBox",
        # "YcbHammer",
        # "YcbMasterChefCan",
        # "YcbMediumClamp",
        # "YcbMustardBottle",
        # "YcbPear",
        # "YcbPottedMeatCan",
        # "YcbPowerDrill",
        # "YcbScissors",
        "YcbStrawberry",
        "YcbTennisBall",
        "YcbTomatoSoupCan",
    ]
    obj_urdfs = [os.path.join(
            ycb_objects.getDataPath(), f"{cls}/model.urdf"
            ) for cls in obj_class]

    obj_poses = [
        # [0.55, 0.0, 0.54],
        # [0.65, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.65, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.65, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.55, 0.0, 0.54],
        # [0.65, -0.03, 0.525],
        # [0.55, 0.0, 0.54],
        [0.55, 0.0, 0.54],
        [0.55, 0.0, 0.54],
        [0.65, 0.0, 0.54],
        ]
    obj_orns = [
        # [0.0, 0.0, -1.5708],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [0.0, 0.0, 0.0],
        # [-1.5708, -3.6, 0.0],
        # [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        ]
    obj_scales = [
        # 1.0,
        # 0.6,
        # 0.6,
        # 1.0,
        # 1.0,
        # 1.0,
        # 0.8,
        # 1.0,
        # 0.6,
        # 1.0,
        # 1.0,
        # 0.6,
        # 1.0,
        1.0,
        1.0,
        0.8,
    ]

    for i in range(len(obj_class)):
        reset()
        
        obj_info = {
            'urdf': obj_urdfs[i],
            'pos': obj_poses[i],
            'orn': obj_orns[i],
            'scale': obj_scales[i]
        }        
        
        name = obj_class[i]
        print(f"=========== Start: {name} ===========")
        
        test(config, obj_info, 100)
        
        # with open(f"exp_grasp_test_12.26_fix_closing/{name}.pickle", 'wb') as f:
        #     pickle.dump(grasp_test_result, f)
        
        print(f"=========== End: {name} ===========")
    
    # for j in range(10):
    #     for i, cls in enumerate(obj_class):
    #         object_urdf = os.path.join(
    #             os.path.expanduser('~'),
    #             "workspace/POMDP/Initial_Belief/gaussian/created",
    #             f"{cls}/{j}/model.urdf")
    #         reset()
    #         name = f"{cls}_{j}"
    #         obj_info = {
    #             'urdf': object_urdf,
    #             'pos': obj_poses[i],
    #             'orn': obj_orns[i],
    #             'scale': obj_scales[i]
    #         }  
    #         print(f"=========== Start: {name} ===========")
    #         test(config, obj_info, 10)
            
    #         with open(f"exp_grasp_test_12.28_particle/{name}.pickle", 'wb') as f:
    #             pickle.dump(grasp_test_result, f)
                
    #         print(f"=========== End: {name} ===========")