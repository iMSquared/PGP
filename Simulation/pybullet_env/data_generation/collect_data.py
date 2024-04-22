import json
import numpy as np
import numpy.typing as npt
import os
import paramiko
from typing import Dict, Tuple, List, Union, Optional
import pathlib
import pickle

from envs.global_object_id_table import GlobalObjectIDTable, Gid_T
from pomdp.POMDP_framework import HistoryEntry, TerminationT, PhaseT, PHASE_EXECUTION, PHASE_ROLLOUT, PHASE_SIMULATION
from pomdp.fetching_POMDP_primitive_object import FetchingAction, FetchingObservation, FetchingState, FetchingAgent


def time_stamp() -> str:
    """Get the unqiue time stamp for a process or file"""
    from datetime import datetime
    now = datetime.now()
    current_date = now.date()
    month = current_date.month
    day = current_date.day
    current_time = now.time()
    hour = current_time.hour
    minute = current_time.minute
    second = current_time.second
    millis = current_time.microsecond/1000
    TIME_STAMP = f"data_{month}m{day}d{hour}h{minute}m{second}s{f'{millis}'.zfill(3)}ms"

    return TIME_STAMP



def log_single_episode(episode_termination: TerminationT,
                       episode_agent_history: Tuple[HistoryEntry],
                       episode_tree_sequence: Tuple,
                       episode_list_time_taken_per_planning: List[float],
                       episode_total_sim_success_count: int,
                       episode_total_sim_count: int,
                       episode_n: int,
                       PROCESS_TIME_STAMP: str,
                       USE_GUIDED_POLICY: bool,
                       USE_GUIDED_VALUE: bool,
                       NUM_SIMS_PER_PLAN: int,
                       EXP_LOG_DIR: str):
    """Log a single execution information"""

    # Make log directory if not exist.
    exp_log_root_dir_path = os.path.join(pathlib.Path(__file__).parent.parent.resolve(), EXP_LOG_DIR)
    if not os.path.exists(exp_log_root_dir_path):
        os.mkdir(exp_log_root_dir_path)
    exp_log_history_json_dir_path = os.path.join(exp_log_root_dir_path, "history_json")
    if not os.path.exists(exp_log_history_json_dir_path):
        os.mkdir(exp_log_history_json_dir_path)
    exp_log_history_pickle_dir_path = os.path.join(exp_log_root_dir_path, "history_pickle")
    if not os.path.exists(exp_log_history_pickle_dir_path):
        os.mkdir(exp_log_history_pickle_dir_path)
    exp_log_tree_pickle_dir_path = os.path.join(exp_log_root_dir_path, "tree_pickle")
    if not os.path.exists(exp_log_tree_pickle_dir_path):
        os.mkdir(exp_log_tree_pickle_dir_path)

    # Compose filename
    exp_log_fname = PROCESS_TIME_STAMP
    exp_log_fname += f"_exp{episode_n}"
    if USE_GUIDED_POLICY:
        exp_log_fname += "_policy"
    if USE_GUIDED_VALUE:
        exp_log_fname += "_value"
    
    exp_log_history_json_fname   = exp_log_fname + "_history.json"
    exp_log_history_pickle_fname = exp_log_fname + "_history.pickle"
    exp_log_tree_pickle_fname    = exp_log_fname + "_tree.pickle"

    # Reuse some information from trajectory dict..
    episode_termination = episode_termination
    episode_action_history = [str(entry.action) for entry in episode_agent_history]
    episode_observation_history = [str(entry.observation) for entry in episode_agent_history]
    episode_total_reward = sum([entry.reward for entry in episode_agent_history])
    # Dump history log!
    with open(os.path.join(exp_log_history_json_dir_path, exp_log_history_json_fname), "w") as f:
        exp_data = {
            "termination": episode_termination,
            "total_reward": episode_total_reward,
            "list_time_taken_per_planning": episode_list_time_taken_per_planning,
            "episode_action_history": episode_action_history,
            "episode_observation_history": episode_observation_history,
            "total_sim_success_count": episode_total_sim_success_count,
            "total_sim_count": episode_total_sim_count,
            "use_guided_policy": USE_GUIDED_POLICY,
            "use_guided_value": USE_GUIDED_VALUE,
            "num_particles": NUM_SIMS_PER_PLAN,}
        json.dump(exp_data, f, indent=4)

    # Dump history log in pickle!
    dump_pickle(episode_agent_history, os.path.join(exp_log_history_pickle_dir_path, exp_log_history_pickle_fname))

    # Dump tree log!
    dump_pickle(episode_tree_sequence, os.path.join(exp_log_tree_pickle_dir_path, exp_log_tree_pickle_fname))



def convert_key_and_join_seg_mask(gid_table: GlobalObjectIDTable,
                                  seg_mask: Dict[Gid_T, npt.NDArray]) \
                                        -> Dict[str, npt.NDArray]:
    """
    1. Convert the key to 'O', 'X' format. 
    2. Join all mask for non-target objects
    
    NOTE(ssh): I think rather passing the obs.seg_mask, the entire observation is more strict for typing...
    """
    seg_mask_converted = {}
    for obj_gid, segment in seg_mask.items():
        # Get target mask (assume single target)
        if gid_table[obj_gid].is_target:
            seg_mask_converted['O'] = segment
        # Unioning multiple non-target mask
        else:
            if 'X' in seg_mask_converted.keys():
                seg_mask_converted['X'] \
                    = np.logical_or(seg_mask_converted['X'], segment)
            else:
                seg_mask_converted['X'] = segment

    return seg_mask_converted
    


def remove_observation_background(observation_depth_image: np.ndarray, 
                                  observation_rgb_image: np.ndarray,
                                  observation_seg_mask: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Remove background in depth image using segmentation mask.

    Args:
        observation_depth_image (Dict): depth image 
        observation_seg_mask (Dict[str, np.ndarray]): segmentation mask with keys of 'O' and 'X'
    Returns:
        np.ndarray: Foreground segmented depth image
        np.ndarray: Foreground segmented rgb image
    """
    seg_target = observation_seg_mask['O']
    seg_other = observation_seg_mask['X']
    seg_all = np.logical_or(seg_target, seg_other)            
    observation_masked_depth_image = np.copy(observation_depth_image)
    observation_masked_depth_image[~seg_all] = 0.0     # Set background to 0.0
    observation_masked_rgb_image = np.copy(observation_rgb_image)
    observation_masked_rgb_image[~seg_all] = 0.0

    return observation_masked_depth_image, observation_masked_rgb_image



def collect_trajectory_data(agent: FetchingAgent,
                            history: Tuple[ HistoryEntry, ...],
                            termination: TerminationT) -> Dict[str, object]:
    """ Collect the trajectory data into a dictionary. (Called inside POMCPOW)
    See `FetchingAgent._update_history` and `POMCPOW._simulate` and `POMCPOW._rollout` for the type of trajectory.

    Args:
        agent (FetchingAgent): Fetching agent instance
        history (Tuple[HistoryEntry]): A trajectory tuple
        termination (TerminationT): Termination condition of the trajectory.
    Returns:
        Dict: Some formatted dictionary.
    """
    gid_table = agent.env.gid_table
    goal_condition = agent.goal_condition
    init_observation = agent.init_observation

    # Saving execution and simulation histories in the trajectory separately.
    data_exec_action         = []
    data_exec_observation    = []
    data_exec_reward         = []
    data_sim_action          = []
    data_sim_observation     = []
    data_sim_reward          = []
    data_rollout_action      = []
    data_rollout_observation = []
    data_rollout_reward      = []

    # Formatting initial observation
    init_seg_mask_converted = convert_key_and_join_seg_mask(gid_table, init_observation.seg_mask)
    init_masked_depth_image, init_masked_rgb_image \
        = remove_observation_background(init_observation.depth_image,
                                        init_observation.rgb_image,
                                        init_seg_mask_converted)
    init_grasp_contact = init_observation.grasp_contact
    init_observation_retyped = (init_masked_depth_image, init_masked_rgb_image, init_grasp_contact)


    # Formatting history
    for i, history_entry in enumerate(history):
        action     : FetchingAction      = history_entry.action
        observation: FetchingObservation = history_entry.observation
        reward     : float               = history_entry.reward
        phase      : PhaseT              = history_entry.phase

        if action.is_feasible():
            action_type      = action.type
            action_is_target = gid_table[action.aimed_gid].is_target
            action_pos       = action.pos
            action_orn       = action.orn
            action_dyaw      = action.delta_theta
        else:
            action_type      = action.type
            action_is_target = None
            action_pos       = None
            action_orn       = None
            action_dyaw      = None

        # Change uid to gid in segmentation mask
        seg_mask_converted = convert_key_and_join_seg_mask(gid_table, observation.seg_mask)
        masked_depth_image, masked_rgb_image \
            = remove_observation_background(observation.depth_image,
                                            observation.rgb_image, 
                                            seg_mask_converted)
        grasp_contact = observation.grasp_contact

        # Saving the execution part
        if phase==PHASE_EXECUTION:
            data_exec_action.append((action_type, action_is_target, action_pos, action_orn, action_dyaw))
            data_exec_observation.append((masked_depth_image, masked_rgb_image, grasp_contact))
            data_exec_reward.append([reward])
        # Saving the simulation part
        elif phase==PHASE_SIMULATION:
            data_sim_action.append((action_type, action_is_target, action_pos, action_orn, action_dyaw))
            data_sim_observation.append((masked_depth_image, masked_rgb_image, grasp_contact))
            data_sim_reward.append([reward])
        elif phase==PHASE_ROLLOUT:
            data_rollout_action.append((action_type, action_is_target, action_pos, action_orn, action_dyaw))
            data_rollout_observation.append((masked_depth_image, masked_rgb_image, grasp_contact))
            data_rollout_reward.append([reward])
        else:
            raise ValueError("Not a valid phase type.")    
    
    data = {
        "goal_condition": goal_condition,
        "init_observation": init_observation_retyped, 
        "exec_action": data_exec_action, 
        "exec_observation": data_exec_observation, 
        "exec_reward": data_exec_reward,
        "sim_action": data_sim_action,
        "sim_observation": data_sim_observation,
        "sim_reward": data_sim_reward, 
        "rollout_action": data_rollout_action,
        "rollout_observation": data_rollout_observation,
        "rollout_reward": data_rollout_reward,
        "termination": termination }

    return data



def save_trajectory_data_json_numpy(json_file_path: str, 
                                    numpy_file_path: str, 
                                    data: Dict[str, object]):
    """Save trajectory data into json and numpy format.
    It detects the number of execution and simulation history from the `data` automatically.
    """

    # To json
    goal_condition      = data["goal_condition"]
    exec_action         = data["exec_action"]
    exec_reward         = data["exec_reward"]
    sim_action          = data["sim_action"]
    sim_reward          = data["sim_reward"]
    rollout_action      = data["rollout_action"]
    rollout_reward      = data["rollout_reward"]
    termination         = data["termination"]
    # To numpy
    init_observation    = data["init_observation"]
    exec_observation    = data["exec_observation"]
    sim_observation     = data["sim_observation"]
    rollout_observation = data["rollout_observation"]
    num_exec    = len(exec_action)
    num_sim     = len(sim_action)
    num_rollout = len(rollout_action)

    # Dump json
    json_data = {}
    json_data["goal_condition"] = goal_condition
    json_data["exec_action"] = exec_action
    json_data["exec_reward"] = exec_reward
    json_data["sim_action"] = sim_action
    json_data["sim_reward"] = sim_reward
    json_data["rollout_action"] = rollout_action
    json_data["rollout_reward"] = rollout_reward
    json_data["termination"] = termination
    with open(json_file_path, "w") as f:
        json.dump(json_data, f, indent=4)

    # Dump numpy
    numpy_data = {}
    numpy_data["init_observation_depth"] = init_observation[0]
    numpy_data["init_observation_rgb"]   = init_observation[1]
    numpy_data["init_observation_grasp"] = init_observation[2]
    for i in range(num_exec):
        numpy_data[f"exec_observation_{i}_depth"] = exec_observation[i][0]
        numpy_data[f"exec_observation_{i}_rgb"]   = exec_observation[i][1]
        numpy_data[f"exec_observation_{i}_grasp"] = exec_observation[i][2]
    for i in range(num_sim):
        numpy_data[f"sim_observation_{i}_depth"] = sim_observation[i][0]
        numpy_data[f"sim_observation_{i}_rgb"]   = sim_observation[i][1]
        numpy_data[f"sim_observation_{i}_grasp"] = sim_observation[i][2]
    for i in range(num_rollout):
        numpy_data[f"rollout_observation_{i}_depth"] = rollout_observation[i][0]
        numpy_data[f"rollout_observation_{i}_rgb"]   = rollout_observation[i][1]
        numpy_data[f"rollout_observation_{i}_grasp"] = rollout_observation[i][2]

    np.savez(numpy_file_path, **numpy_data)
    


def mkdir_data_save_path(dataset_save_path: str):
    """Double checking the directory exists"""

    if not os.path.exists(os.path.join(dataset_save_path, "exec_dataset_json")):
        os.mkdir(os.path.join(dataset_save_path, "exec_dataset_json"))
    if not os.path.exists(os.path.join(dataset_save_path, "exec_dataset_numpy")):
        os.mkdir(os.path.join(dataset_save_path, "exec_dataset_numpy"))
    if not os.path.exists(os.path.join(dataset_save_path, "sim_dataset_json")):
        os.mkdir(os.path.join(dataset_save_path, "sim_dataset_json"))
    if not os.path.exists(os.path.join(dataset_save_path, "sim_dataset_numpy")):
        os.mkdir(os.path.join(dataset_save_path, "sim_dataset_numpy"))
    if not os.path.exists(os.path.join(dataset_save_path, "tree_sequence")):
        os.mkdir(os.path.join(dataset_save_path, "tree_sequence"))
    if not os.path.exists(os.path.join(dataset_save_path, "groundtruth_sequence")):
        os.mkdir(os.path.join(dataset_save_path, "groundtruth_sequence"))



def open_sftp():
    """Open sftp"""
    # SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("", username="", password="")
    sftp = ssh.open_sftp()
    print("ssh connected.")

    return ssh, sftp



def mkdir_data_save_path_sftp(sftp: paramiko.SFTPClient, 
                              dataset_save_path_sftp: str):
    """Triple checking the directory exists"""
    # exec positive json
    try:
        sftp.chdir(os.path.join(dataset_save_path_sftp, "exec_dataset_json"))
    except:
        sftp.mkdir(os.path.join(dataset_save_path_sftp, "exec_dataset_json"))
        sftp.chdir(os.path.join(dataset_save_path_sftp, "exec_dataset_json"))
    # exec positive numpy
    try:
        sftp.chdir(os.path.join(dataset_save_path_sftp, "exec_dataset_numpy"))
    except:
        sftp.mkdir(os.path.join(dataset_save_path_sftp, "exec_dataset_numpy"))
        sftp.chdir(os.path.join(dataset_save_path_sftp, "exec_dataset_numpy"))
    # sim positive json
    try:
        sftp.chdir(os.path.join(dataset_save_path_sftp, "sim_dataset_json"))
    except:
        sftp.mkdir(os.path.join(dataset_save_path_sftp, "sim_dataset_json"))
        sftp.chdir(os.path.join(dataset_save_path_sftp, "sim_dataset_json"))
    # sim positive numpy
    try:
        sftp.chdir(os.path.join(dataset_save_path_sftp, "sim_dataset_numpy"))
    except:
        sftp.mkdir(os.path.join(dataset_save_path_sftp, "sim_dataset_numpy"))
        sftp.chdir(os.path.join(dataset_save_path_sftp, "sim_dataset_numpy"))
    # tree sequence pickle
    try:
        sftp.chdir(os.path.join(dataset_save_path_sftp, "tree_sequence"))
    except:
        sftp.mkdir(os.path.join(dataset_save_path_sftp, "tree_sequence"))
        sftp.chdir(os.path.join(dataset_save_path_sftp, "tree_sequence"))
    # groundtruth sequence pickle
    try:
        sftp.chdir(os.path.join(dataset_save_path_sftp, "groundtruth_sequence"))
    except:
        sftp.mkdir(os.path.join(dataset_save_path_sftp, "groundtruth_sequence"))
        sftp.chdir(os.path.join(dataset_save_path_sftp, "groundtruth_sequence"))



def dump_pickle(data: object, path: str):
    """Dump pickle... just a wrapper for better looking."""
    with open(path, "wb") as f:
        pickle.dump(data, f)




def save_routine(episode_sim_trajs_total: List[Dict[str, object]],
                 episode_exec_traj: Dict[str, object],
                 episode_tree_sequence: List[Dict],
                 episode_groundtruth_sequence: List[FetchingState],
                 episode_n: int, 
                 PROCESS_TIME_STAMP: str,
                 DATASET_SAVE_PATH: str,
                 DATASET_SAVE_PATH_SFTP: str,
                 SAVE_SFTP: bool,
                 sftp: Optional[paramiko.SFTPClient] = None):
    """Some giant function for saving all things."""

    # Data name is shared for all kinds of data.
    EP_DATA_NAME = f"{PROCESS_TIME_STAMP}_ep{episode_n}"


    # Saving simulation data
    sim_save_dir_path_json = os.path.join(DATASET_SAVE_PATH, "sim_dataset_json")
    sim_save_dir_path_numpy = os.path.join(DATASET_SAVE_PATH, "sim_dataset_numpy")
    #   Make list of newly saved files for SFTP transfer
    list_saved_file_names_json = []
    list_saved_file_names_numpy = []
    #   Saving one by one...
    for i, sim_data in enumerate(episode_sim_trajs_total):
        sim_data_file_name_json = f"{EP_DATA_NAME}_sim{i}.json"
        sim_data_file_name_numpy = f"{EP_DATA_NAME}_sim{i}.npz"
        sim_data_file_path_json = os.path.join(sim_save_dir_path_json, sim_data_file_name_json)
        sim_data_file_path_numpy = os.path.join(sim_save_dir_path_numpy, sim_data_file_name_numpy)
        save_trajectory_data_json_numpy(sim_data_file_path_json,
                                        sim_data_file_path_numpy,
                                        sim_data)
        # Keep the list of newly saved file for SFTP...
        list_saved_file_names_json.append(sim_data_file_name_json)
        list_saved_file_names_numpy.append(sim_data_file_name_numpy)
    print("# newly saved simulation data: ", len(episode_sim_trajs_total))
    if SAVE_SFTP:
        # Sending simulation data via SFTP
        sim_save_dir_path_sftp_json = os.path.join(DATASET_SAVE_PATH_SFTP, "sim_dataset_json")
        sim_save_dir_path_sftp_numpy = os.path.join(DATASET_SAVE_PATH_SFTP, "sim_dataset_numpy")
        for sim_fname_json, sim_fname_numpy in zip(list_saved_file_names_json, list_saved_file_names_numpy):
            src = os.path.join(sim_save_dir_path_json, sim_fname_json)
            dst = os.path.join(sim_save_dir_path_sftp_json, sim_fname_json)
            sftp.put(src, dst)
            src = os.path.join(sim_save_dir_path_numpy, sim_fname_numpy)
            dst = os.path.join(sim_save_dir_path_sftp_numpy, sim_fname_numpy)
            sftp.put(src, dst)
        print("# sent simulation data via SFTP: ", len(episode_sim_trajs_total))


    # Saving execution data
    exec_save_dir_path_json = os.path.join(DATASET_SAVE_PATH, "exec_dataset_json")
    exec_save_dir_path_numpy = os.path.join(DATASET_SAVE_PATH, "exec_dataset_numpy")
    exec_data_file_name_json = f"{EP_DATA_NAME}_exec.json"
    exec_data_file_name_numpy = f"{EP_DATA_NAME}_exec.npz"
    exec_data_file_path_json = os.path.join(exec_save_dir_path_json, exec_data_file_name_json)
    exec_data_file_path_numpy = os.path.join(exec_save_dir_path_numpy, exec_data_file_name_numpy)
    save_trajectory_data_json_numpy(exec_data_file_path_json,
                                    exec_data_file_path_numpy,
                                    episode_exec_traj)
    print("@ newly saved a success execution data")
    if SAVE_SFTP:
        # Sending execution data via SFTP
        exec_save_dir_path_sftp_json = os.path.join(DATASET_SAVE_PATH_SFTP, "exec_dataset_json")
        exec_save_dir_path_sftp_numpy = os.path.join(DATASET_SAVE_PATH_SFTP, "exec_dataset_numpy")
        src = os.path.join(exec_save_dir_path_json, exec_data_file_name_json)
        dst = os.path.join(exec_save_dir_path_sftp_json, exec_data_file_name_json)
        sftp.put(src, dst)
        src = os.path.join(exec_save_dir_path_numpy, exec_data_file_name_numpy)
        dst = os.path.join(exec_save_dir_path_sftp_numpy, exec_data_file_name_numpy)
        sftp.put(src, dst)
        print("@ sent a success execution data via SFTP")


    # Saving tree data
    tree_save_dir_path = os.path.join(DATASET_SAVE_PATH, "tree_sequence")
    tree_data_file_name = f"{EP_DATA_NAME}_tree_sequence.pickle"
    tree_data_file_path = os.path.join(tree_save_dir_path, tree_data_file_name)
    dump_pickle(episode_tree_sequence, tree_data_file_path)
    print("@ newly saved a tree sequence data")
    if SAVE_SFTP:
        # Sending tree data via SFTP
        tree_save_dir_path_sftp = os.path.join(DATASET_SAVE_PATH_SFTP, "tree_sequence")
        src = os.path.join(tree_save_dir_path, tree_data_file_name)
        dst = os.path.join(tree_save_dir_path_sftp, tree_data_file_name)
        sftp.put(src, dst)
        print("@ sent a tree sequence data via SFTP")
    

    # Saving groundtruth data
    groundtruth_save_dir_path = os.path.join(DATASET_SAVE_PATH, "groundtruth_sequence")
    groundtruth_data_file_name = f"{EP_DATA_NAME}_groundtruth.pickle"
    groundtruth_data_file_path = os.path.join(groundtruth_save_dir_path, groundtruth_data_file_name)
    dump_pickle(episode_groundtruth_sequence, groundtruth_data_file_path)
    print("@ newly saved a groundtruth sequence data")
    if SAVE_SFTP:
        # Sending tree data via SFTP
        groundtruth_save_dir_path_sftp = os.path.join(DATASET_SAVE_PATH_SFTP, "groundtruth_sequence")
        src = os.path.join(groundtruth_save_dir_path, groundtruth_data_file_name)
        dst = os.path.join(groundtruth_save_dir_path_sftp, groundtruth_data_file_name)
        sftp.put(src, dst)
        print("@ sent a groundtruth sequence data via SFTP")