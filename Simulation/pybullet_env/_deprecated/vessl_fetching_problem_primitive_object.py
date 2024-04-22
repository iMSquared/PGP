import argparse
import os
import yaml
import paramiko

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

# from policy import FetchingPlacePolicyModelBelief

# Debugging
from debug import debug_data, debug_data_reset
import pickle

# Collecting data
from Simulation.pybullet_env.data_collection.collect_data import save_sim_data, process_exec_data


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

    sim_env = BinpickEnvPrimitive(bc, config)
    
    # Simulation initialization
    if config["project_params"]["robot"] == "ur5_suction":
        robot = UR5Suction(bc, config)
    elif config["project_params"]["robot"] == "ur5":
        robot = UR5(bc, config)
    else:
        raise Exception("invalid robot")
    # manipulation
    manipulation = Manipulation(bc, sim_env, robot, config)

    # Stabilize the environment
    manipulation.wait(240)


    # POMDP
    transition_model = FetchingTransitionModel(bc, sim_env, robot, config)
    observation_model = FetchingObservationModel(bc, sim_env, robot, config)
    reward_model = FetchingRewardModel(bc, sim_env, robot, config)
    blackbox_model = BlackboxModel(transition_model, observation_model, reward_model)

    # policy_model = FetchingPlacePolicyModelBelief(bc, sim_env, robot, config, manipulation)
    policy_model = FetchingRolloutPolicyModel(bc, sim_env, robot, config)
    rollout_policy_model = FetchingRolloutPolicyModel(bc, sim_env, robot, config, manipulation)

    env = Environment(transition_model, observation_model, reward_model)
    agent = FetchingAgent(bc, sim_env, robot, config, blackbox_model, policy_model, rollout_policy_model)
    
    pomdp = POMDP(agent, env, "FetchingProblem")


    # Set initial state    
    robot_state, object_state = get_state_info(bc, sim_env, robot, config)

    init_state = FetchingState(bc, sim_env, robot_state, object_state, None)

    pomdp.env.set_state(init_state, True)

    goal_condi = config["env_params"]["binpick_env"]["goal"]["color"] + config["env_params"]["binpick_env"]["goal"]["pos"][0:2]
    agent.add_attr("goal_condition", goal_condi)
    
    init_obs = observation_model.get_sensor_observation(init_state, None)
    seg_mask = {}
    for uid, obj in agent.env.uid_to_target.items():
        seg_mask[obj] = init_obs.seg_mask[uid].tolist()
    init_obs = (init_obs.observation.tolist(), seg_mask)
    agent.add_attr("init_observation", init_obs)

    # Set initial belief
    init_belief = make_belief(bc, sim_env, agent, robot, config, config["plan_params"]["num_sims"])
    pomdp.agent.set_belief(init_belief, True)

    # Planner
    planner = POMCPOW(pomdp, config)
    
    bc.disconnect()
    
    # Planning
    total_reward = 0.0
    total_time_taken = 0.0
    sim_data = []
    while len(agent.history) < config["plan_params"]["max_depth"]:        
        # |TODO(Jiyong)|: make env_reset()
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
        sim_env = BinpickEnvPrimitive(bc, config)
        # robot
        if config["project_params"]["robot"] == "ur5_suction":
            robot = UR5Suction(bc, config)
        elif config["project_params"]["robot"] == "ur5":
            robot = UR5(bc, config)
        else:
            raise Exception("invalid robot")
     
        next_action, time_taken, sims_count, sim_success_trajs = planner.plan()
        total_time_taken += time_taken
        
        # # Store debug data
        # if config["project_params"]["debug"]["get_data"]:
        #     with open(f'exp/exp_1.25_noise/debug_data_4_{n}_{len(agent.history)}.pickle', 'wb') as f:
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
        sim_env = BinpickEnvPrimitive(bc, config)
        # robot
        if config["project_params"]["robot"] == "ur5_suction":
            robot = UR5Suction(bc, config)
        elif config["project_params"]["robot"] == "ur5":
            robot = UR5(bc, config)
        else:
            raise Exception("invalid robot")
        
        # Restore true state in simulation
        agent.imagine_state(env.state, reset=True, simulator=config["plan_params"]["simulator"])
        
        # Execution
        observation, reward, termination = pomdp.env.execute(next_action)
        total_reward = reward + config["plan_params"]["discount_factor"] * total_reward
        
        # Update history and belief state
        bc.disconnect()
        agent.update(next_action, observation, reward)
        # Update search tree
        planner.update(agent, next_action, observation)
        
        debug_data_reset()
        
        if (termination == "success") or (termination == "fail"):
            break
    
    # # For visualizing execution
    # data_action = []
    # data_observation = []
    # data_reward = []
    # for a, o, r in agent.history:
    #     traj = np.asarray(a.traj)
    #     data_action.append((a.type, a.target, a.pos, a.orn, traj.tolist()))
    #     data_observation.append((o.observation, o.seg_mask))
    #     data_reward.append(r)
    # data = {"action": data_action, "observation": data_observation, "reward": data_reward}
    
    # For collecting data
    # Simulation data
    sim_data = sim_success_trajs
    # Execution data
    exec_data = process_exec_data(agent.history, agent.env.uid_to_target, agent.goal_condition, agent.init_observation)
        
    return termination, total_reward, total_time_taken, exec_data, sim_data
    


if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    parser.add_argument("--save_path", type=str, default="./output", help="Execution data save path to override default config.")
    parser.add_argument("--data_name_prefix_override", type=str, default=None, help="Name of the dataset when overriding. Automatically use timestamp if not given.")
    parser.add_argument("--num_executions", type=int, default=50, help="Number of executions")
    params = parser.parse_args()


    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    # NOTE(ssh): This part automates the vessl script, i.e.
    # output/       (at the root of Filesystem)
    # └ dataset/
    #   ├ exec_dataset
    #   │ ├ data_1.30_19:18:31_0.pickle
    #   │ └ ...
    #   └ sim_dataset
    #     ├ data_1.30_19:18:31_0_0.pickle
    #     └ ...
    # Override save path if given
    if params.save_path is not None:

        # Naming the dataset. Automatically name to current datetime defaultly.
        if params.data_name_prefix_override is None:
            from datetime import datetime
            now = datetime.now()

            current_date = now.date()
            month = current_date.month
            day = current_date.day
            current_time = now.time()
            hour = current_time.hour
            minute = current_time.minute
            second = current_time.second
            millis = current_time.microsecond

            data_name_prefix = f"data_{month}.{day}_{hour}:{minute}:{second}.{int(millis/1000)}"
        else:
            data_name_prefix = params.dataset_name_prefix_override

        # Re-configure the save path
        output_path = params.save_path
        if not os.path.exists(os.path.join(output_path, "exec_dataset")):
            os.mkdir(os.path.join(output_path, "exec_dataset"))
        if not os.path.exists(os.path.join(output_path, "sim_dataset")):
            os.mkdir(os.path.join(output_path, "sim_dataset"))
        path_exec_data = os.path.join(output_path, "exec_dataset", data_name_prefix)
        path_sim_data = os.path.join(output_path, "sim_dataset", data_name_prefix)


    # SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("", username="", password="")
    sftp = ssh.open_sftp()
    print("ssh connected.")


    num_executions = params.num_executions
    for n in range(num_executions):
        termination, total_reward, total_time_taken, exec_data, sim_data = main(config, n)
        print(f"Execution Result of {n}:", termination, "total reward -", total_reward, "time taken -", total_time_taken)


        # For collecting data
        if config["project_params"]["debug"]["get_data"]:

            # Saving simulation data
            path_sim_data = path_sim_data + f"_{n}"
            num_sim_data = save_sim_data(path_sim_data, sim_data)   
            print("# saved simulation data:", num_sim_data)
            
            # Send SFTP
            for i, d in enumerate(sim_data):
                src = f"{path_sim_data}_{i}.pickle"
                filename = src.split("/")[-1]
                dst = os.path.join("/home/shared_directory/vessl/sim_dataset_2.27", filename)
                sftp.put(src, dst)
                
            # Saving execution data
            if termination == 'success':
                path_data = path_exec_data
                name_data = f"{path_data}_{n}"
                with open(f"{name_data}.pickle", "wb") as f:
                    pickle.dump(exec_data,f)

                # Send SFTP
                src = f"{name_data}.pickle"
                filename = src.split("/")[-1]
                dst = os.path.join("/home/shared_directory/vessl/exec_dataset_2.27", filename)
                sftp.put(src, dst)

