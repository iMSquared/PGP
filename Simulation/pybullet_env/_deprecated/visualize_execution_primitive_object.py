import argparse
import os
import yaml

# PyBullet
import pybullet as pb
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import UR5Suction, UR5
from envs.manipulation import Manipulation

# POMDP
from POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from online_planner_framework import Planner
from fetching_POMDP_primitive_object import *
from POMCPOW import POMCPOW

# Debugging
import pickle


def main(config, data):
    
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
    binpick_env = BinpickEnvPrimitive(bc, config)
    binpick_env.target_to_uid = {"X": 3, "O": 4}
    binpick_env.uid_to_target = {3: "X", 4: "O"}
    # robot
    robot = UR5Suction(bc, config)
    # manipulation
    manipulation = Manipulation(bc, binpick_env, robot, config)

    # Stabilize the environment
    manipulation.wait(240)

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

    # Set initial state    
    robot_state, object_state = get_state_info(bc, binpick_env, robot, config)

    init_state = FetchingState(bc, binpick_env, robot_state, object_state, None)

    pomdp.env.set_state(init_state, True)

    actions = []
    for a in data['action']:
        print(a)
        actions.append(FetchingAction(a[0], a[1], a[2], a[3], np.asarray(a[4])))
        # actions.append(FetchingAction((a.type, a.uid, a.pos, a.orn)))

    # actions.append(policy_model.sample(None, None, ['PICK', 3, None, None]))
    # actions.append(policy_model.sample(None, None, ['MOVE', 3, ((0.25, 0.20, 0.60), (0, 0.75*np.pi, 0)), None]))
    
    for action in actions:
        # observation, reward, termination = pomdp.env.execute(action)
        next_state, *rest_T = transition_model.sample(env._cur_state, action, True)
    
    bc.disconnect()
    
    

if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    with open("data/1.24_fix/data_1.24_4_20.pickle", 'rb') as f:
        data = pickle.load(f)

    main(config, data=data)
