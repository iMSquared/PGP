import argparse
import os
import yaml
import pathlib

# PyBullet
import pybullet as pb
from envs.common import init_new_bulletclient
from envs.global_object_id_table import GlobalObjectIDTable

import json

# POMDP
from pomdp.POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from pomdp.online_planner_framework import Planner
from pomdp.fetching_POMDP_primitive_object import *
from pomdp.POMCPOW import POMCPOW
from pomdp.policy.guided_policy import FetchingGuidedPolicyPlace
from pomdp.policy.rollout_policy import FetchingRolloutPolicyModel
from pomdp.value.guided_value_mse import FetchingGuidedValueRegression


# Debugging
from debug.debug import debug_data, debug_data_reset

# Collecting data
from data_generation.collect_data import ( time_stamp,
                                          log_single_episode,
                                          collect_trajectory_data, 
                                          save_trajectory_data_json_numpy,
                                          mkdir_data_save_path,
                                          open_sftp,
                                          mkdir_data_save_path_sftp,
                                          save_routine )



def episode(config: Dict, 
            episode_n: int, 
            use_guided_policy: bool,
            use_guided_value: bool):
    """Lifetime of this main function is one episode.

    Args:
        config (Dict): Configuration file
        episode_n (int): The number of current episode.
        use_guided_policy (bool): Use guided policy when true
        use_guided_value (bool):Used guided value when true

    Returns:
        termination (str)
        total_reward (float)
        episode_num_sim_total (int)
        episode_num_sim_success (int)
        episode_time_taken_per_step (List[float])
        episode_sim_trajs_total (List[Dict])
        exec_traj (List[Dict])
    """

    # Configuration
    PROJECT_INIT_RANDOM  = config["project_params"]["init_random"]
    COLLECT_DATA         = config["project_params"]["collect_data"]
    PLAN_MAX_DEPTH       = config["plan_params"]["max_depth"]
    PLAN_NUM_SIMS        = config["plan_params"]["num_sims"]
    PLAN_DISCOUNT_FACTOR = config["plan_params"]["discount_factor"]
    debug_data_reset()

    # Connect to a new bullet client
    bc, sim_env, robot, manip = init_new_bulletclient(config, stabilize=True)
    # Randomize the environment
    if PROJECT_INIT_RANDOM:
            sim_env.reset_object_poses_to_random()


    # POMDP initialization
    #   setup 1: initialize models
    transition_model  = FetchingTransitionModel(bc, sim_env, robot, manip, config)
    observation_model = FetchingObservationModel(bc, sim_env, robot, config)
    reward_model      = FetchingRewardModel(bc, sim_env, robot, config)
    blackbox_model    = BlackboxModel(transition_model, observation_model, reward_model)
    if use_guided_policy:
        print("Selected policy: History")
        policy_model         = FetchingGuidedPolicyPlace(bc, sim_env, robot, manip, config)
        rollout_policy_model = FetchingGuidedPolicyPlace(bc, sim_env, robot, manip, config)
    else:
        print("Selected policy: Random")
        policy_model         = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)
        rollout_policy_model = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)
    if use_guided_value:
        print("Selected value: Guided")
        value_model = FetchingGuidedValueRegression(bc, sim_env, robot, manip, config)
    else:
        print("Selected value: Rollout")
        value_model = None
    #   setup 2: gt_init_state, goal_condition, init_observation, and inital_belief
    gt_init_state    = make_gt_init_state(bc, sim_env, robot, config)                               # Initial ground truth state 
    goal_condition   = make_goal_condition(config)                                                  # Goal condition
    init_observation = get_initial_observation(sim_env, observation_model, gt_init_state)           # Initial observation instance
    if PROJECT_INIT_RANDOM:
        init_belief = make_belief_random_problem(bc, sim_env, robot, config, PLAN_NUM_SIMS)
    else:
        init_belief = make_belief_fixed_problem(bc, sim_env, robot, config, PLAN_NUM_SIMS)          # Initial belief
    #   setup 3: initialize POMDP
    env     = Environment(transition_model, observation_model, reward_model, gt_init_state)
    agent   = FetchingAgent(bc, sim_env, robot, manip, config, 
                            blackbox_model, policy_model, rollout_policy_model, value_model, 
                            init_belief, init_observation, goal_condition)    
    pomdp   = POMDP(agent, env, "FetchingProblem")
    agent.imagine_state(gt_init_state, reset=True)
    #   setup 4: Planner initialization
    planner = POMCPOW(pomdp, config)

    # =====
    # Simulation (planning)
    # =====
    bc.disconnect()
    bc, sim_env, robot, manip = init_new_bulletclient(config, stabilize=False)
    transition_model.set_new_bulletclient(bc, sim_env, robot, manip)
    observation_model.set_new_bulletclient(bc, sim_env, robot)
    reward_model.set_new_bulletclient(bc, sim_env, robot)
    policy_model.set_new_bulletclient(bc, sim_env, robot, manip)
    rollout_policy_model.set_new_bulletclient(bc, sim_env, robot, manip)
    agent.set_new_bulletclient(bc, sim_env, robot)

    # Plan to the agent's goal
    next_action, \
        time_taken, num_sim_total, num_sim_success, \
        sim_trajs = planner.plan()

    # Logging
    next_action: FetchingAction
    print(f"[next_action at depth {len(agent.history)}] {next_action}")
    action_children = agent.tree.children

    bc.disconnect()
    return next_action, action_children, time_taken, num_sim_success, num_sim_total



def log_tree_debug(TREE_DEBUG_LOG_DIR: str,
                   episode_n: int,
                   episode_time_taken_per_planning: float,
                   episode_sim_success_count: int,
                   episode_total_sim_count: int,
                   next_action: FetchingAction,
                   action_children: Dict):
    """What I need?
    
    - Time taken for planning
    - Selected next action
    - Branching factor
    - Avg and std of q values
    - For each action childs
        - Their Q values
        - Their visit counts
        - Action type and target, placement pose of each.
        - Num obs childs..?
    """
    tree_debug_log_dir_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), TREE_DEBUG_LOG_DIR)
    if not os.path.exists(tree_debug_log_dir_path):
        os.mkdir(tree_debug_log_dir_path)

    tree_debug_fname = time_stamp()
    tree_debug_fname += f"_exp{episode_n}"
    tree_debug_fname += f"_particles{episode_n}.json"
    
    list_action_child_data = []
    for action_key in action_children.keys():
        child = action_children[action_key]
        q_value = child.value
        num_visits = child.num_visits
        num_children = len(child.children)

        data = {
            "action": {
                "type": action_key.type,
                "target": action_key.aimed_gid,
                "pos": action_key.pos,
                "orn": action_key.orn },
            "q_value": q_value,
            "num_visits": num_visits,
            "num_obs_children": num_children }
        list_action_child_data.append(data)


    with open(os.path.join(tree_debug_log_dir_path, tree_debug_fname), "w") as f:
        tree_debug_data = {
            "time_taken_per_planning": episode_time_taken_per_planning,
            "sim_success_count": episode_sim_success_count,
            "total_sim_count": episode_total_sim_count,
            "next_action": {
                "type": next_action.type,
                "target": next_action.aimed_gid,
                "pos": next_action.pos,
                "orn": next_action.orn },
            "list_action_child_data": list_action_child_data}

        json.dump(tree_debug_data, f, indent=4)







def main(config           : Dict, 
         num_episodes     : int, 
         dataset_save_path: str, 
         use_guided_policy: bool,
         use_guided_value : bool,
         save_sftp        : bool):  
    """Project main"""


    COLLECT_DATA          : bool = config["project_params"]["collect_data"]
    DATASET_SAVE_PATH     : str  = config["project_params"]["default_dataset_save_path"] \
                                    if dataset_save_path is None \
                                    else dataset_save_path
    DATASET_SAVE_PATH_SFTP: str  = config["project_params"]["default_dataset_save_path_sftp"] 
    SAVE_SFTP             : bool = save_sftp
    TREE_DEBUG_LOG_DIR    : str  = "./tree_debug_log"


    # Make subdirectories
    if COLLECT_DATA:
        mkdir_data_save_path(DATASET_SAVE_PATH)
        # ssh
        if SAVE_SFTP:
            ssh, sftp = open_sftp()
            mkdir_data_save_path_sftp(sftp, DATASET_SAVE_PATH_SFTP)
        else:
            ssh, sftp = None, None


    # Repeat executions
    for episode_n in range(num_episodes):
        
        # One episode
        next_action, action_children, time_taken, num_sim_success, num_sim_total \
            = episode(config, episode_n, use_guided_policy, use_guided_value)

        log_tree_debug(TREE_DEBUG_LOG_DIR              = TREE_DEBUG_LOG_DIR,
                       episode_n                       = episode_n,
                       episode_time_taken_per_planning = time_taken,
                       episode_sim_success_count       = num_sim_success,
                       episode_total_sim_count         = num_sim_total,
                       next_action                     = next_action,
                       action_children                 = action_children)





if __name__=="__main__":

    
    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config",             type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    parser.add_argument("--num_episodes",       type=int, default=1, help="Number of episodes")
    parser.add_argument("--dataset_save_path",  type=str, default=None, help="Execution data save path to override default config.")
    parser.add_argument("--use_guided_policy",  type=bool, default=False, help="Use guided policy when True")
    parser.add_argument("--use_guided_value",   type=bool, default=False, help="Use guided value when True")
    parser.add_argument("--sftp",               type=bool, default=False, help="Send data via SFTP")
    params = parser.parse_args()


    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    # main...
    main(config            = config, 
         num_episodes      = params.num_episodes, 
         dataset_save_path = params.dataset_save_path, # Overrides default when given
         use_guided_policy = params.use_guided_policy, 
         use_guided_value  = params.use_guided_value,
         save_sftp         = params.sftp)

    