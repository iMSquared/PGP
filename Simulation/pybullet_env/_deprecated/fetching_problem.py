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

# Debugging
from debug import debug_data, debug_data_reset
import pickle


def main(config, n):    
        
    debug_data_reset()
    
    # Connect bullet client
    if config["project_params"]["debug"]["show_gui"]:
        sim_id = pb.connect(pb.GUI)
    else:
        sim_id = pb.connect(pb.DIRECT)
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

    # # observation
    # observation_model = DistanceModel(bc, config)
    # # grasp pose sampler
    # sample_grasp_pose = PointSamplingGrasp()

    # Stabilize the environment
    manipulation.wait(100)
    # start_pos, start_orn = robot.get_endeffector_pose()
    
    # true_state_id = bc.saveState()

    # # Main loop
    # while True:

    #     # Stabilize the environment
    #     manipulation.wait(100)

    #     # Query state
    #     state_pcd = binpick_env.get_pcd()

    #     # Select one pick
    #     grasp_pos, grasp_orn_q = manipulation.filter_valid_grasp_pose(
    #         sample_grasp_pose, 
    #         state_pcd[binpick_env.objects_uid[1]])

    #     # Pick demo
    #     manipulation.pick(grasp_pos, grasp_orn_q)

    #     # Observation demo
    #     measurement = binpick_env.get_measurement()
    #     state_pcd = binpick_env.get_pcd()
    #     likelihood, maplines = observation_model(measurement, state_pcd, average_by_segment=False)
    #     #visualize_point_cloud([state_pcd, measurement], maplines)
    #     print(f"likelihood: {likelihood}")

    #     # Stabilize the environment
    #     manipulation.place(start_pos, start_orn)    # This should be automatically identified.

    #     print("Done")
    #     manipulation.wait()

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

    init_state = FetchingState(bc, binpick_env, robot_state, object_state)

    pomdp.env.set_state(init_state, True)
    
    init_obs = observation_model.get_sensor_observation(init_state, None)

    # Set initial belief
    shape_folder = os.path.join(os.path.expanduser('~'), "workspace/POMDP/Initial_Belief/gaussian/created")
    shape_table, cls_indices = make_shape_table(shape_folder)
    binpick_env.shape_table = shape_table

    init_belief = make_belief(bc, binpick_env, robot, config, shape_table, cls_indices, config["plan_params"]["num_sims"])
    pomdp.agent.set_belief(init_belief, True)

    # Planner
    planner = POMCPOW(pomdp, config)
    
    
    # # ===================== Testing =============================
    # # POMDP
    # transition_model = FetchingTransitionModel(bc, binpick_env, robot, config)
    # observation_model = FetchingObservationModel(bc, binpick_env, config)
    # reward_model = FetchingRewardModel(bc, binpick_env, robot, config)
    # blackbox_model = BlackboxModel(transition_model, observation_model, reward_model)

    # policy_model = FetchingRolloutPolicyModel(bc, binpick_env, robot, config)
    # rollout_policy_model = FetchingRolloutPolicyModel(bc, binpick_env, robot, config)

    # agent = FetchingAgent(bc, binpick_env, robot, config, blackbox_model, policy_model, rollout_policy_model)
    # env = Environment(transition_model, reward_model)

    # pomdp = POMDP(agent, env, "FetchingProblem")

    # # Set initial state    
    # robot_state, object_state = get_state_info(bc, binpick_env, robot, config)

    # init_state = FetchingState(bc, binpick_env, robot_state, object_state)

    # pomdp.env.set_state(init_state, True)

    # # Set initial belief

    # # |FIXME(Jiyong)|: temporaliy use
    # def make_belief(bc, env, num_particle):
    #     """
    #     This is temporal initial belief which has the uncertainty only for position.
    #     """
    #     robot_state, object_state = get_state_info(bc, binpick_env, robot, config)
        
    #     init_belief = {}
    #     for _ in range(num_particle):
    #         objs_state = {}
    #         for k, (pos, orn, target) in object_state.items():
    #             pos = list(pos)
    #             pos[0] += random.gauss(0, 0.05)
    #             pos[1] += random.gauss(0, 0.05)
    #             pos = tuple(pos)
    #             objs_state[k] = (pos, orn, target)
    #         particle = FetchingState(bc, env, robot_state, objs_state)
    #         init_belief[particle]    
    debug_data_reset()
    
    # init_belief = make_belief(bc, binpick_env, 100)
    # pomdp.agent.set_belief(init_belief, True)

    # # Planner
    # planner = POMCPOW(pomdp, config)
    # # ===========================================================

    # for i in range(5):
        
    #     planning_result = planner.plan()
    
    #     if config["project_params"]["debug"]["get_data"]:
    #         with open(f'exp/debug_11.18_GelatinBox_PottedMeatCan_0.9/debug_data_{i+1}.pickle', 'wb') as f:
    #             pickle.dump(debug_data, f)
        
    #     debug_data_reset()

    total_reward = 0.0
    while len(agent.history) < config["plan_params"]["max_depth"]:        
        # |TODO(Jiyong)|: make env_reset()
        bc.disconnect()
        # Connect bullet client
        if config["project_params"]["debug"]["show_gui"]:
            sim_id = pb.connect(pb.GUI)
        else:
            sim_id = pb.connect(pb.DIRECT)
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
    
        next_action, *rest = planner.plan()
        
        # Store debug data
        # if config["project_params"]["debug"]["get_data"]:
        #     with open(f'exp_12.30/debug_data_{n}_{len(agent.history)}.pickle', 'wb') as f:
        #         pickle.dump(debug_data, f)

        # |FIXME(Jiyong)|: need to handle changing uid or use communication with multiprocessing
        # Connect bullet client with GUI    
        bc.disconnect()
        sim_id = pb.connect(pb.DIRECT)
        bc = BulletClient(sim_id)
        # bc.resetSimulation()
        
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
        
        # Restore true state in simulation
        # bc.restoreState(true_state_id)
        agent.imagine_state(env.state, reset=True, simulator=config["plan_params"]["simulator"])
        
        # Execution
        observation, reward, termination = pomdp.env.execute(next_action)
        
        # # Save again true state simulation
        # bc.removeState(true_state_id)
        # true_state_id = bc.saveState()
        
        total_reward = reward + config["plan_params"]["discount_factor"] * total_reward
        
        # Update history and belief state
        bc.disconnect()
        agent.update(next_action, observation, reward)
        # Update search tree
        planner.update(agent, next_action, observation)
        
        debug_data_reset()
        
        if (termination == "success") or (termination == "fail"):
            break
    
    data_action = []
    data_observation = []
    data_reward = []
    for a, o, r in agent.history:
        traj = np.asarray(a.traj)
        data_action.append((a.type, a.uid, a.pos, a.orn, traj.tolist()))
        data_observation.append((o.observation, o.seg_mask))
        data_reward.append(r)
    data = {"action": data_action, "observation": data_observation, "reward": data_reward}
    
    return termination, total_reward, data
    

if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for n in range(10):
        termination, total_reward, data = main(config, n)
        print(f"Execution Result of {n}:", termination, total_reward)
    
        # with open(f"data/data_12.30_{n}.pickle", "wb") as f:
        #     pickle.dump(data, f)
    