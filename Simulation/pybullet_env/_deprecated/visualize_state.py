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
from debug import debug_data, debug_data_reset
import pickle


def vis_belief(bc: BulletClient, p_ids: List, belief: Dict, scale: float=1.0):
    while p_ids:
        p_id = p_ids.pop()
        bc.removeBody(p_id)
            
    for k, v in belief.items():
        if v >= 0.001:
            vis_box_shape_id = bc.createVisualShape(
                shapeType = bc.GEOM_BOX,
                halfExtents = [0.015, 0.015, 0.04],
                rgbaColor = [1.0, 0.5, 0.3, v*scale]
            )
            p_ids.append(bc.createMultiBody(
                baseVisualShapeIndex = vis_box_shape_id,
                basePosition = k,
            ))
        
    return p_ids
        

def main(config, data):
    # Connect bullet client
    sim_id = pb.connect(pb.GUI)
    if sim_id < 0:
        raise ValueError("Failed to connect to pybullet!")
    bc = BulletClient(sim_id)
    #bc.setPhysicsEngineParameter(enableFileCaching=0)   # Turn off file caching

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
    # Stabilize the environment
    manipulation = Manipulation(bc, binpick_env, robot, config)
    manipulation.wait(240)

    # POMDP
    transition_model = FetchingTransitionModel(bc, binpick_env, robot, config)
    observation_model = FetchingObservationModel(bc, binpick_env, robot, config)
    reward_model = FetchingRewardModel(bc, binpick_env, robot, config)
    blackbox_model = BlackboxModel(transition_model, observation_model, reward_model)

    policy_model = FetchingRolloutPolicyModel(bc, binpick_env, robot, config)
    rollout_policy_model = FetchingRolloutPolicyModel(bc, binpick_env, robot, config, manipulation)

    agent = FetchingAgent(bc, binpick_env, robot, config, blackbox_model, policy_model, rollout_policy_model)
    env = Environment(transition_model, observation_model, reward_model)

    pomdp = POMDP(agent, env, "FetchingProblem")

    # Set initial state    
    robot_state, object_state = get_state_info(bc, binpick_env, robot, config)
    init_state = FetchingState(bc, binpick_env, robot_state, object_state, None)
    pomdp.env.set_state(init_state, True)
    # init_obs = observation_model.get_sensor_observation(init_state, None)

    # Set initial belief
    init_belief = make_belief(bc, binpick_env, robot, config, config["plan_params"]["num_sims"])
    pomdp.agent.set_belief(init_belief, True)

    # Planner
    planner = POMCPOW(pomdp, config)

    # Visualize the initial belief
    belief = {}
    for k, v in agent.belief.particles.items():
        obj_pos = k.object['O'][0]
        belief[obj_pos] = v

    p_ids = []
    p_ids = vis_belief(bc, p_ids, belief, 10.0)
    
    # logged actions
    actions = data['action']

    for a in actions:
        action = FetchingAction(a[0], a[1], a[2], a[3], np.asarray(a[4]))
        observation, reward, termination = pomdp.env.execute(action)
        bc.disconnect()
        agent.update(action, observation, reward)
        
        # Simulation re-initialization
        sim_id = pb.connect(pb.GUI)
        bc = BulletClient(sim_id)
        
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
        
        # env
        binpick_env = BinpickEnvPrimitive(bc, config)
        binpick_env.target_to_uid = {"X": 3, "O": 4}
        binpick_env.uid_to_target = {3: "X", 4: "O"}
        # robot
        robot = UR5Suction(bc, config)
        
        # Restore true state in simulation
        agent.imagine_state(env.state, reset=True, simulator=config["plan_params"]["simulator"])
        
        # Visualize the updated belief
        belief = {}
        for k, v in agent.belief.particles.items():
            obj_pos = k.object['O'][0]
            belief[obj_pos] = v
            
        p_ids = []
        p_ids = vis_belief(bc, p_ids, belief)
        
    bc.disconnect()



if __name__ == '__main__':
    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    with open("data/1.24_fix/data_1.24_4_20.pickle", 'rb') as f:
        data = pickle.load(f)
    
    main(config, data)