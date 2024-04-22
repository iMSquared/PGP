import argparse
import os
import yaml
import pathlib

# PyBullet
import pybullet as pb
from envs.common import init_new_bulletclient
from envs.global_object_id_table import GlobalObjectIDTable


# POMDP
from pomdp.POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from pomdp.online_planner_framework import Planner
from pomdp.fetching_POMDP_primitive_object import *
from pomdp.POMCPOW import POMCPOW
from pomdp.policy.guided_policy import FetchingGuidedPolicyPlace
from pomdp.policy.rollout_policy import FetchingRolloutPolicyModel
from pomdp.value.guided_v_value_mse import FetchingGuidedValueMSE
from pomdp.value.guided_v_value_preference import FetchingGuidedValuePreference
from pomdp.value.guided_q_value_mse import FetchingGuidedQValueMSE
from pomdp.value.guided_q_value_preference import FetchingGuidedQValuePreference


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
            episode_n: int) -> Tuple:
    """Lifetime of this main function is one episode.

    Args:
        config (Dict): Configuration file
        episode_n (int): The number of current episode.

    Returns:
        Too much data...
    """

    # Configuration
    USE_GUIDED_POLICY    = config["project_params"]["overridable"]["use_guided_policy"]
    USE_GUIDED_VALUE     = config["project_params"]["overridable"]["use_guided_value"]
    GUIDE_Q_VALUE        = config["project_params"]["overridable"]["guide_q_value"]
    GUIDE_PREFERENCE     = config["project_params"]["overridable"]["guide_preference"]
    COLLECT_DATA         = config["project_params"]["overridable"]["collect_data"]
    PLAN_MAX_DEPTH       = config["plan_params"]["max_depth"]
    PLAN_NUM_SIMS        = config["plan_params"]["num_sims"]
    PLAN_DISCOUNT_FACTOR = config["plan_params"]["discount_factor"]
    debug_data_reset()

    # Connect to a new bullet client
    bc, sim_env, robot, manip = init_new_bulletclient(config, stabilize=True)
    # Randomize the environment
    sim_env.reset_object_poses_to_random()


    # POMDP initialization
    #   setup 1: initialize models
    transition_model  = FetchingTransitionModel(bc, sim_env, robot, manip, config)
    observation_model = FetchingObservationModel(bc, sim_env, robot, config)
    reward_model      = FetchingRewardModel(bc, sim_env, robot, config)
    blackbox_model    = BlackboxModel(transition_model, observation_model, reward_model)
    if USE_GUIDED_POLICY:
        print("Selected policy: Guided")
        policy_model         = FetchingGuidedPolicyPlace(bc, sim_env, robot, manip, config)
        rollout_policy_model = policy_model
        print(f"\tmodel={policy_model.nn_config.exp_name}/{policy_model.nn_config.weight}")
    else:
        print("Selected policy: Random")
        policy_model         = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)
        rollout_policy_model = policy_model
    if USE_GUIDED_VALUE:
        if not GUIDE_Q_VALUE and not GUIDE_PREFERENCE:
            print("Selected value: V-MSE")
            value_model = FetchingGuidedValueMSE(bc, sim_env, robot, manip, config)
        elif GUIDE_Q_VALUE and not GUIDE_PREFERENCE:
            print("Selected value: Q-MSE")
            value_model = FetchingGuidedQValueMSE(bc, sim_env, robot, manip, config)
        elif not GUIDE_Q_VALUE and GUIDE_PREFERENCE:
            print("Selected value: V-Preference")
            value_model = FetchingGuidedValuePreference(bc, sim_env, robot, manip, config)
        elif GUIDE_Q_VALUE and GUIDE_PREFERENCE:
            print("Selected value: Q-Preference")
            value_model = FetchingGuidedQValuePreference(bc, sim_env, robot, manip, config)
        print(f"\tmodel={value_model.nn_config.exp_name}/{value_model.nn_config.weight}")
    else:
        print("Selected value: Rollout")
        value_model = None
        
    
    #   setup 2: gt_init_state, goal_condition, init_observation, and inital_belief
    gt_init_state    = make_gt_init_state(bc, sim_env, robot, config)                               # Initial ground truth state 
    goal_condition   = make_goal_condition(config)                                                  # Goal condition
    init_observation = get_initial_observation(sim_env, observation_model, gt_init_state)           # Initial observation instance
    init_belief      = make_belief_random_problem(bc, sim_env, robot, config, PLAN_NUM_SIMS)
    #   setup 3: initialize POMDP
    env     = Environment(transition_model, observation_model, reward_model, gt_init_state)
    agent   = FetchingAgent(bc, sim_env, robot, manip, config, 
                            blackbox_model, policy_model, rollout_policy_model, value_model, 
                            init_belief, init_observation, goal_condition)    
    pomdp   = POMDP(agent, env, "FetchingProblem")
    #   setup 4: Planner initialization
    planner = POMCPOW(pomdp, config)


    # # Visualize the belief!
    # agent.imagine_state(gt_init_state, reset=True)
    # list_belief_uids = draw_belief(bc, sim_env, agent.belief)
    # print("Continue planning...")
    # for uid in list_belief_uids:
    #     bc.removeBody(uid)


    # Plan+Execution loop
    #   Debugging data
    total_reward = 0.0
    episode_tree_sequence        = []
    episode_groundtruth_sequence = []
    #   Profiling data
    episode_num_sim_total       = 0
    episode_num_sim_success     = 0
    episode_time_taken_per_step = []
    episode_sim_trajs_total     = []
    while len(agent.history) < PLAN_MAX_DEPTH:
        # =====
        # Simulation (planning)
        # =====
        bc.disconnect()
        bc, sim_env, robot, manip = init_new_bulletclient(config, stabilize=False)
        transition_model.set_new_bulletclient(bc, sim_env, robot, manip)
        observation_model.set_new_bulletclient(bc, sim_env, robot)
        reward_model.set_new_bulletclient(bc, sim_env, robot)
        policy_model.set_new_bulletclient(bc, sim_env, robot, manip)
        if value_model is not None:
            value_model.set_new_bulletclient(bc, sim_env, robot, manip)
        agent.set_new_bulletclient(bc, sim_env, robot)

        # Plan to the agent's goal
        next_action, \
            time_taken, num_sim_total, num_sim_success, \
            sim_trajs = planner.plan()

        # Collect debug data
        episode_num_sim_total   += num_sim_total
        episode_num_sim_success += num_sim_success
        episode_time_taken_per_step.append(time_taken)

        #   Planning data
        episode_groundtruth_sequence.append(pomdp.env.state)    # State before the execution. This is very useful even for logging.
        episode_tree_sequence.append(agent.tree)
        if COLLECT_DATA:
            episode_sim_trajs_total += sim_trajs
            
        # =====
        # Execution
        # =====
        bc.disconnect()
        bc, sim_env, robot, manip = init_new_bulletclient(config, stabilize=False)
        transition_model.set_new_bulletclient(bc, sim_env, robot, manip)
        observation_model.set_new_bulletclient(bc, sim_env, robot)
        reward_model.set_new_bulletclient(bc, sim_env, robot)
        policy_model.set_new_bulletclient(bc, sim_env, robot, manip)
        if value_model is not None:
            value_model.set_new_bulletclient(bc, sim_env, robot, manip)
        agent.set_new_bulletclient(bc, sim_env, robot)

        # Restore the ground truth state in simulation
        print("In execution...")
        agent.imagine_state(pomdp.env.state, reset=True)
        # Execution in real world
        observation, reward, termination = pomdp.env.execute(next_action)
        total_reward = reward + PLAN_DISCOUNT_FACTOR * total_reward

        # Logging & Data collection
        next_action: FetchingAction
        observation: FetchingObservation
        print(f"[next_action at depth {len(agent.history)}] {next_action}")
        print(f"[observation at depth {len(agent.history)}] contact={observation.grasp_contact}")

        # Update history and belief state
        # Update search tree (clean or reuse whatever...)
        # NOTE(jshan): We only get environment just to get the gt pose + noise. This will be deleted later.
        agent.update(next_action, observation, reward, pomdp.env) # pomdp.env)
        planner.update(agent, next_action, observation)
        # Check termination condition!
        if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
            break

        # Show the gt again (for visualization)
        agent.imagine_state(pomdp.env.state, reset=True)
        debug_data_reset()
        # # Visualize the belief!
        # list_belief_uids = draw_belief(bc, sim_env, agent.belief)
        # print("Inital belief...")
    

    # Finalizing the planning!
    bc.disconnect()
    # Always collect history data. It is useful for debugging.
    #   GT seq, tree seq
    episode_groundtruth_sequence.append(pomdp.env.state)
    episode_agent_history = agent.history
    if COLLECT_DATA:
        episode_exec_traj = collect_trajectory_data(
            agent       = agent,
            history     = agent.history,
            termination = termination)
    else:
        episode_sim_trajs_total = None
        episode_exec_traj       = None

    return termination, total_reward, \
            episode_num_sim_total, episode_num_sim_success, \
            episode_time_taken_per_step, \
            episode_agent_history, \
            episode_sim_trajs_total, episode_exec_traj, \
            episode_tree_sequence, episode_groundtruth_sequence




def main(config      : Dict, 
         num_episodes: int): 
    """Project main"""

    # Set configs.
    NUM_SIMS_PER_PLAN     : int  = config["plan_params"]["num_sims"]
    PROCESS_TIME_STAMP    : str  = time_stamp()
    USE_GUIDED_POLICY     : bool = config["project_params"]["overridable"]["use_guided_policy"]
    USE_GUIDED_VALUE      : bool = config["project_params"]["overridable"]["use_guided_value"]
    GUIDE_Q_VALUE         : bool = config["project_params"]["overridable"]["guide_q_value"]
    GUIDE_PREFERENCE      : bool = config["project_params"]["overridable"]["guide_preference"]
    COLLECT_DATA          : bool = config["project_params"]["overridable"]["collect_data"]
    SAVE_SFTP             : bool = config["project_params"]["overridable"]["sftp"]
    DATASET_SAVE_PATH     : str  = config["project_params"]["overridable"]["default_dataset_save_path"]
    DATASET_SAVE_PATH_SFTP: str  = config["project_params"]["overridable"]["default_dataset_save_path_sftp"] 
    EXP_LOG_DIR           : str  = config["project_params"]["overridable"]["default_exp_log_dir_path"]
    print(f"NUM_SIMS_PER_PLAN     : {NUM_SIMS_PER_PLAN}")
    print(f"PROCESS_TIME_STAMP    : {PROCESS_TIME_STAMP}")
    print(f"USE_GUIDED_POLICY     : {USE_GUIDED_POLICY}")
    print(f"USE_GUIDED_VALUE      : {USE_GUIDED_VALUE}")
    print(f"GUIDE_Q_VALUE         : {GUIDE_Q_VALUE}")
    print(f"GUIDE_PREFERENCE      : {GUIDE_PREFERENCE}")
    print(f"COLLECT_DATA          : {COLLECT_DATA}")
    print(f"SAVE_SFTP             : {SAVE_SFTP}")
    print(f"DATASET_SAVE_PATH     : {DATASET_SAVE_PATH}")
    print(f"DATASET_SAVE_PATH_SFTP: {DATASET_SAVE_PATH_SFTP}")
    print(f"EXP_LOG_DIR           : {EXP_LOG_DIR}")


    # Make subdirectories
    if COLLECT_DATA:
        mkdir_data_save_path(DATASET_SAVE_PATH)
        # ssh
        if SAVE_SFTP:
            print("Saving SFTP...")
            ssh, sftp = open_sftp()
            mkdir_data_save_path_sftp(sftp, DATASET_SAVE_PATH_SFTP)
        else:
            ssh, sftp = None, None


    # Repeat executions
    num_exec_success = 0
    num_sim          = 0
    num_sim_success  = 0
    for episode_n in range(num_episodes):
        
        # One episode
        episode_termination, episode_total_reward, \
            episode_num_sim_total, episode_num_sim_success, \
            episode_time_taken_per_step, \
            episode_agent_history, \
            episode_sim_trajs_total, episode_exec_traj, \
            episode_tree_sequence, episode_groundtruth_sequence \
                = episode(config, episode_n)

        # Log one execution
        total_time_taken = sum(episode_time_taken_per_step)
        print(f"Execution Result of {episode_n}: {episode_termination}, total reward - {episode_total_reward}, time taken - {total_time_taken}s")
        for i, t in enumerate(episode_time_taken_per_step):
            print(f"Planning time at depth {i}: {t}")
        # Log success count
        if episode_termination == TERMINATION_SUCCESS:
            num_exec_success += 1
        num_sim += episode_num_sim_total
        num_sim_success += episode_num_sim_success
        print("Total execution success:", num_exec_success, '/', episode_n+1)
        print("Total simulation success:", num_sim_success, '/', num_sim)


        # Collecting data
        if COLLECT_DATA:
            # Let this function do everything...
            save_routine(episode_sim_trajs_total      = episode_sim_trajs_total,
                         episode_exec_traj            = episode_exec_traj,
                         episode_tree_sequence        = episode_tree_sequence,
                         episode_groundtruth_sequence = episode_groundtruth_sequence,
                         episode_n                    = episode_n,
                         PROCESS_TIME_STAMP           = PROCESS_TIME_STAMP,
                         DATASET_SAVE_PATH            = DATASET_SAVE_PATH,
                         DATASET_SAVE_PATH_SFTP       = DATASET_SAVE_PATH_SFTP,
                         SAVE_SFTP                    = SAVE_SFTP,
                         sftp                         = sftp)


        # Save experiment result
        # TODO(ssh): Pass timestamp here. Some processes are dying during the evaluation for unknown reason.
        log_single_episode(episode_termination                  = episode_termination,
                           episode_agent_history                = episode_agent_history,
                           episode_tree_sequence                = episode_tree_sequence,
                           episode_list_time_taken_per_planning = episode_time_taken_per_step,
                           episode_total_sim_success_count      = episode_num_sim_success,
                           episode_total_sim_count              = episode_num_sim_total,
                           episode_n                            = episode_n,
                           PROCESS_TIME_STAMP                   = PROCESS_TIME_STAMP,
                           USE_GUIDED_POLICY                    = USE_GUIDED_POLICY,
                           USE_GUIDED_VALUE                     = USE_GUIDED_VALUE,
                           NUM_SIMS_PER_PLAN                    = NUM_SIMS_PER_PLAN,
                           EXP_LOG_DIR                          = EXP_LOG_DIR)




if __name__=="__main__":

    
    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config",                          type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    parser.add_argument("--num_episodes",                    type=int, default=1, help="Number of episodes")
    parser.add_argument("--override_num_sims",               type=int, default=None, help="overrides number of simulations when called")
    parser.add_argument("--override_collect_data",           action='store_true', help="Overrides to collect data when called")
    parser.add_argument("--override_sftp",                   action='store_true', help="Overrides to send data via SFTP when called")
    parser.add_argument("--override_exp_log_dir_path",       type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_exp_learning_dir_path",  type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_dataset_save_path",      type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_dataset_save_path_sftp", type=str, default=None, help="Overrides default config when passed.")
    parser.add_argument("--override_use_guided_policy",      action='store_true', help="Overrides to use guided policy when called")
    parser.add_argument("--override_use_guided_value",       action='store_true', help="Overrides to use guided value when called")
    parser.add_argument("--override_guide_q_value",          action='store_true', help="Overrides to guide Q value when called")
    parser.add_argument("--override_guide_preference",       action='store_true', help="Overrides to guide preference when called")
    parser.add_argument("--override_inference_device",       type=str, default=None, help="Overrides the default inference device in config.")
    params = parser.parse_args()


    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    # Override configuration file when given the arguments.
    if params.override_num_sims is not None :
        config["plan_params"]["num_sims"] = params.override_num_sims
    if params.override_collect_data == True:
        config["project_params"]["overridable"]["collect_data"] \
            = params.override_collect_data
    if params.override_sftp == True:
        config["project_params"]["overridable"]["sftp"] \
            = params.override_sftp
    if params.override_exp_log_dir_path is not None:
        config["project_params"]["overridable"]["default_exp_log_dir_path"] \
            = params.override_exp_log_dir_path
    if params.override_exp_learning_dir_path is not None:
        config["project_params"]["overridable"]["default_exp_learning_dir_path"] \
            = params.override_exp_learning_dir_path
    if params.override_dataset_save_path is not None:
        config["project_params"]["overridable"]["default_dataset_save_path"] \
            = params.override_dataset_save_path
    if params.override_dataset_save_path_sftp is not None:
        config["project_params"]["overridable"]["default_dataset_save_path_sftp"] \
            = params.override_dataset_save_path_sftp

    if params.override_use_guided_policy == True:
        config["project_params"]["overridable"]["use_guided_policy"] \
            = params.override_use_guided_policy
    if params.override_use_guided_value == True:
        config["project_params"]["overridable"]["use_guided_value"] \
            = params.override_use_guided_value
    if params.override_guide_q_value == True:
        config["project_params"]["overridable"]["guide_q_value"] \
            = params.override_guide_q_value
    if params.override_guide_preference == True:
        config["project_params"]["overridable"]["guide_preference"] \
            = params.override_guide_preference

    if params.override_inference_device is not None:
        config["project_params"]["overridable"]["inference_device"] \
            = params.override_inference_device


    # main...
    main(config       = config, 
         num_episodes = params.num_episodes)
