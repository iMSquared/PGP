import argparse
import os
import yaml
from typing import Tuple

import torch

# PyBullet
import pybullet as pb
import pybullet_data
from imm.pybullet_util.bullet_client import BulletClient
from envs.common import init_new_bulletclient
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import UR5Suction, UR5
from envs.manipulation import Manipulation

# POMDP
from pomdp.POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from pomdp.online_planner_framework import Planner
from pomdp.fetching_POMDP_primitive_object import *
from pomdp.policy import FetchingPlacePolicyModelHistory, FetchingRolloutPolicyModel
from pomdp.POMCPOW import POMCPOW

# Tensor processing
from learning.dataset.placepolicy_dataset import format_data, add_zero_pre_padding

# Debugging
import pickle




def main(config):
    """This script tests the execution!"""

    FILE = "/home/sanghyeon/workspace/POMDP/Simulation/pybullet_env/bug log/crash_log_next_state6.pickle"      # PICK-PLACE
    BELIEF_FILE = "/home/sanghyeon/workspace/POMDP/Simulation/pybullet_env/bug log/crash_log_next_state6.pickle"
    with open(FILE, "rb") as f:
        data = pickle.load(f)

    episode(config, data)





def episode(config, data):
    """Lifetime of this main function is one episode.

    Args:
        config (Dict): Configuration file
        n (int): The number of current episode.

    Returns:
        Tuple: logs
    """


    # Configuration
    PROJECT_INIT_RANDOM  = config["project_params"]["init_random"]
    PLAN_MAX_DEPTH       = config["plan_params"]["max_depth"]
    PLAN_NUM_SIMS        = config["plan_params"]["num_sims"]
    PLAN_SIMULATOR       = config["plan_params"]["simulator"] # NOTE(ssh): What is this for????
    PLAN_DISCOUNT_FACTOR = config["plan_params"]["discount_factor"]
    config["project_params"]["debug"]["show_gui"]=True


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
    policy_model         = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)
    rollout_policy_model = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)
    #   setup 2: gt_init_state, goal_condition, init_observation, and inital_belief
    gt_init_state        = make_gt_init_state(bc, sim_env, robot, config)                               # Initial ground truth state 
    goal_condition       = make_goal_condition(config)                                                  # Goal condition
    init_obs_reformatted = make_reformatted_init_observation(sim_env, observation_model, gt_init_state) # Reformatted initial observation
    init_belief = make_belief_random_problem(bc, sim_env, robot, config, PLAN_NUM_SIMS)
    #   setup 3: initialize POMDP
    env     = Environment(transition_model, observation_model, reward_model, gt_init_state)
    agent   = FetchingAgent(bc, sim_env, robot, config, 
                                        goal_condition, init_obs_reformatted,
                                        blackbox_model, policy_model, rollout_policy_model,
                                        None, init_belief)    
    pomdp   = POMDP(agent, env, "FetchingProblem")

    # Planner initialization
    planner = POMCPOW(pomdp, config)

    
    bc.disconnect()
    bc, sim_env, robot, manip = init_new_bulletclient(config, stabilize=False)
    transition_model.set_new_bulletclient(bc, sim_env, robot, manip)
    observation_model.set_new_bulletclient(bc, sim_env, robot)
    reward_model.set_new_bulletclient(bc, sim_env, robot)
    policy_model.set_new_bulletclient(bc, sim_env, robot, manip)
    rollout_policy_model.set_new_bulletclient(bc, sim_env, robot, manip)
    agent.set_new_bulletclient(bc, sim_env, robot)


    state = FetchingState(robot_state=data["robot"],
                            object_state=data["object"],
                            holding_obj=data["holding_obj"])    
    agent.imagine_state(state, None, simulator=PLAN_SIMULATOR)

    term = reward_model._check_termination(state)
    print(term)
    

    # Hold
    while True:
        time.sleep(5000)


    






def predict_next_place_action(bc: BulletClient, 
                              model: torch.nn.Module, 
                              pickled_data: Dict, 
                              last_action: FetchingAction,
                              action_type_encoding: Dict,
                              time_step_to_predict: int, 
                              num_predictions: int) -> List[ Tuple[float, float, float] ]:
    """ Infer given number of next actions from the sequence.

    Args:
        bc (BulletClient): Bullet client
        model (torch.nn.Module): Policy torch model
        pickled_data (Dict): Some raw data
        last_action (FetchingAction): Last action
        action_type_encoding (Dict): Table for one-hot encoding "PICK" and "PLACE"
        time_step_to_predict (int): Time step to PREDICT
        num_predictions (int): Number of predictions to show

    Returns:
        list_pred_poses (List[ Tuple[float, float, float] ]): List of predicted x, y, dyaw
    """


    init_obs, goal, \
        seq_action, seq_obs, seq_reward, \
        mask, time_step_to_predict, next_action_label \
            = format_data(pickled_data, time_step_to_predict, 
                          action_type_encoding,
                          to_tensor=True)

    # Padding
    num_paddings = 3 - (time_step_to_predict.item())    # Tensor index starts from 0. Actual starts from 1.
    seq_action = add_zero_pre_padding(seq_action, num_paddings)
    seq_obs    = add_zero_pre_padding(seq_obs, num_paddings)
    seq_reward = add_zero_pre_padding(seq_reward, num_paddings)
    mask       = add_zero_pre_padding(mask, num_paddings)
    # Batchify
    init_obs   = init_obs.unsqueeze(0)    
    goal       = goal.unsqueeze(0)
    seq_action = seq_action.unsqueeze(0)
    seq_obs    = seq_obs.unsqueeze(0)
    seq_reward = seq_reward.unsqueeze(0)
    mask       = mask.unsqueeze(0)
    time_step_to_predict = time_step_to_predict.unsqueeze(0)   


    # Prediction!
    list_pred_poses = []

    #   Try until a pose without collision is found
    for i in range(num_predictions):

        # Predit (x, y, delta_theta) of place action with NN
        with torch.no_grad():
            time_start = time.time()
            init_obs   = init_obs.to("cuda:0")
            goal       = goal.to("cuda:0")
            seq_action = seq_action.to("cuda:0")
            seq_obs    = seq_obs.to("cuda:0")
            seq_reward = seq_reward.to("cuda:0")
            mask       = mask.to("cuda:0")
            time_step_to_predict = time_step_to_predict.to("cuda:0")
            pred = model.inference(init_obs, goal,
                                        seq_action, seq_obs, seq_reward,
                                        mask).squeeze(0)   # Shape=(1, 3)->(3)
            time_end = time.time()
            pred = tuple(pred.tolist())
            infer_time = time_end - time_start

        x, y, delta_theta = pred
        z = last_action.pos[2]   # Use same z from the last action.
        place_pos = (x, y, z)
        
        # Sample orientation
        z_axis_rot_mat = np.asarray([
            [np.cos(delta_theta), -np.sin(delta_theta), 0],
            [np.sin(delta_theta),  np.cos(delta_theta), 0],
            [0                  , 0                   , 1]])
        prev_orn = last_action.orn       # |TODO(Jiyong)|: Should use forward kinematics
        prev_orn_q = bc.getQuaternionFromEuler(prev_orn)
        prev_orn_rot_mat = np.asarray(bc.getMatrixFromQuaternion(prev_orn_q)).reshape(3, 3)
        place_orn_rot_mat = Rotation.from_matrix(np.matmul(z_axis_rot_mat, prev_orn_rot_mat))
        place_orn = place_orn_rot_mat.as_euler("zyx", degrees=False)

        pose = np.concatenate((place_pos, place_orn), axis=0)
        list_pred_poses.append(pose)

    return list_pred_poses






if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)









































