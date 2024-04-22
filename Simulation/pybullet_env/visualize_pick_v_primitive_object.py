import argparse
import os
import yaml
import pathlib

# PyBullet
import pybullet as pb
import pybullet_data
from envs.common import init_new_bulletclient
from envs.global_object_id_table import GlobalObjectIDTable

import matplotlib.pyplot as plt

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



def episode(config):

    # DEBUGGER
    TIMESTEP_TO_VISUALIZE = 2
    NUM_INFERENCE = 200
    config["project_params"]["overridable"]["use_guided_policy"] = False
    config["project_params"]["overridable"]["use_guided_value"] = True
    config["project_params"]["overridable"]["guide_q_value"]    = False
    config["project_params"]["overridable"]["guide_preference"] = True



    # Configuration
    config["plan_params"]["num_sims"] = 1
    USE_GUIDED_POLICY    = config["project_params"]["overridable"]["use_guided_policy"]
    USE_GUIDED_VALUE     = config["project_params"]["overridable"]["use_guided_value"]
    GUIDE_Q_VALUE        = config["project_params"]["overridable"]["guide_q_value"]
    GUIDE_PREFERENCE     = config["project_params"]["overridable"]["guide_preference"]
    PLAN_MAX_DEPTH       = config["plan_params"]["max_depth"]
    PLAN_NUM_SIMS        = config["plan_params"]["num_sims"]

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
    else:
        print("Selected value: Rollout")
        value_model = None


    list_predv_visible_pick_target_success     = []
    list_predv_visible_pick_target_fail        = []
    list_predv_visible_pick_nontarget_success  = []
    list_predv_visible_pick_nontarget_fail     = []
    list_predv_occluded_pick_target_success    = []
    list_predv_occluded_pick_target_fail       = []
    list_predv_occluded_pick_nontarget_success = []
    list_predv_occluded_pick_nontarget_fail    = []
    count = 0
    while count < NUM_INFERENCE:

        # Reset the environment
        robot.release()
        env_params = config["env_params"]["binpick_env"]
        OBJECT_CONFIGS = env_params["objects"]
        for uid, obj_config in zip(sim_env.object_uids, OBJECT_CONFIGS):
            # Read configs
            pos = obj_config["pos"]
            orn = obj_config["orn"]
            bc.resetBasePositionAndOrientation(uid, pos, bc.getQuaternionFromEuler(orn))
        for i in range(200):
            bc.stepSimulation()
        # Randomize the environment
        sim_env.reset_object_poses_to_random()
        for value_i, joint_i in enumerate(robot.joint_indices_arm):
            bc.resetJointState(robot.uid, joint_i, robot.rest_pose[joint_i])
        gt_init_state    = make_gt_init_state(bc, sim_env, robot, config)                               # Initial ground truth state 
        goal_condition   = make_goal_condition(config)                                                  # Goal condition
        init_observation = get_initial_observation(sim_env, observation_model, gt_init_state)           # Initial observation instance
        robot_state, object_states = capture_binpickenv_state(bc, sim_env, robot, config)
        particles = []
        particles.append(FetchingState(robot_state, object_states, None))
        init_belief = UnweightedParticles(tuple(particles))
        #   setup 3: initialize POMDP
        env     = Environment(transition_model, observation_model, reward_model, gt_init_state)
        agent   = FetchingAgent(bc, sim_env, robot, manip, config, 
                                blackbox_model, policy_model, rollout_policy_model, value_model, 
                                init_belief, init_observation, goal_condition)    
        pomdp   = POMDP(agent, env, "FetchingProblem")
        agent.imagine_state(gt_init_state, reset=True)

        # Plan+Execution loop
        while len(agent.history) < TIMESTEP_TO_VISUALIZE:

            # Randomly select action...
            next_action = policy_model.sample(
                agent.init_observation, 
                agent.history, 
                pomdp.env.state, 
                agent.goal_condition)
            # Do not pick target when benchmark.            
            if next_action.aimed_gid == 0:
                continue

            # Restore the ground truth state in simulation
            agent.imagine_state(pomdp.env.state, reset=True)
            # Execution in real world
            observation, reward, termination = pomdp.env.execute(next_action)
            if termination != TERMINATION_CONTINUE:
                break
            agent._update_history(next_action, observation, reward)
            agent.imagine_state(pomdp.env.state, reset=True)

        # Only check pick at time=2
        if not (termination == TERMINATION_CONTINUE and next_action.type=="PLACE"):
            print("termination skip")
            continue
        
        # Count the number of picks.
        count += 1
        print(count)

        # Next action
        next_action = policy_model.sample(
            agent.init_observation, 
            agent.history, 
            pomdp.env.state, 
            agent.goal_condition)
        # Adding some noise to the pick pose to simulate the pose uncertainty
        if sim_env.gid_table[next_action.aimed_gid].is_target:
            x = random.gauss(0, 0.01)
            y = random.gauss(0, 0.01)
            sampled_pick_pos = list(next_action.pos)
            sampled_pick_pos[0] += x
            sampled_pick_pos[1] += y
            next_action.pos = tuple(sampled_pick_pos)
        # Execute
        observation: FetchingObservation
        observation, reward, termination = pomdp.env.execute(next_action)

        # Infer V value
        entry = HistoryEntry(next_action, observation, reward, phase=PHASE_EXECUTION)
        inference_history = agent.history + (entry,)
        pred_v_value = value_model.sample(agent.init_observation, inference_history, pomdp.env.state, agent.goal_condition)

        # Previous observation check
        if check_visibility(agent.history[-1].observation.rgb_image, agent.history[-1].observation.seg_mask):
            if sim_env.gid_table[next_action.aimed_gid].is_target:
                if observation.grasp_contact:
                    list_predv_visible_pick_target_success.append(pred_v_value) 
                else:
                    list_predv_visible_pick_target_fail.append(pred_v_value)
            else:
                if observation.grasp_contact:
                    list_predv_visible_pick_nontarget_success.append(pred_v_value)
                else:
                    list_predv_visible_pick_nontarget_fail.append(pred_v_value)
        else:
            if sim_env.gid_table[next_action.aimed_gid].is_target:
                if observation.grasp_contact:
                    list_predv_occluded_pick_target_success.append(pred_v_value) 
                else:
                    list_predv_occluded_pick_target_fail.append(pred_v_value)
            else:
                if observation.grasp_contact:
                    list_predv_occluded_pick_nontarget_success.append(pred_v_value)
                else:
                    list_predv_occluded_pick_nontarget_fail.append(pred_v_value)


    t2_occlude = len(list_predv_occluded_pick_target_success) \
               + len(list_predv_occluded_pick_target_fail) \
               + len(list_predv_occluded_pick_nontarget_success) \
               + len(list_predv_occluded_pick_nontarget_fail)
    t2_visible = len(list_predv_visible_pick_target_success) \
               + len(list_predv_visible_pick_target_fail) \
               + len(list_predv_visible_pick_nontarget_success) \
               + len(list_predv_visible_pick_nontarget_fail)

    t2_occlude_pick_target = len(list_predv_occluded_pick_target_success) \
                           + len(list_predv_occluded_pick_target_fail)
    t2_occlude_pick_nontarget = len(list_predv_occluded_pick_nontarget_success) \
                              + len(list_predv_occluded_pick_nontarget_fail)
    t2_visible_pick_target = len(list_predv_visible_pick_target_success) \
                           + len(list_predv_visible_pick_target_fail) 
    t2_visible_pick_nontarget = len(list_predv_visible_pick_nontarget_success) \
                              + len(list_predv_visible_pick_nontarget_fail)
    

    print(f"total: {NUM_INFERENCE}, t2_occlude: {t2_occlude}, t2_visible: {t2_visible}")
    print(f"\tt2_occlude - pick_target: {t2_occlude_pick_target}, pick_non_target: {t2_occlude_pick_nontarget}")
    print(f"\t\tt2_occlude pick_target success: {len(list_predv_occluded_pick_target_success)}, fail {len(list_predv_occluded_pick_target_fail)}")
    print(f"\t\tt2_occlude pick_nontarget success: {len(list_predv_occluded_pick_nontarget_success)}, fail {len(list_predv_occluded_pick_nontarget_fail)}")

    print(f"\tt2_visible - pick_target: {t2_visible_pick_target}, pick_non_target: {t2_visible_pick_nontarget}")
    print(f"\t\tt2_visible pick_target success: {len(list_predv_visible_pick_target_success)}, fail {len(list_predv_visible_pick_target_fail)}")
    print(f"\t\tt2_visible pick_nontarget success: {len(list_predv_visible_pick_nontarget_success)}, fail {len(list_predv_visible_pick_nontarget_fail)}")

    # print
    ylim = 10 if GUIDE_PREFERENCE else 100

    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.boxplot([list_predv_occluded_pick_target_success, list_predv_occluded_pick_target_fail, 
                 list_predv_occluded_pick_nontarget_success, list_predv_occluded_pick_nontarget_fail])
    plt.title("Occluded (t-1)")
    plt.ylabel("V Value")
    plt.xticks([1, 2, 3, 4], 
               ["PICK(target)-success", "PICK(target)-fail", "PICK(non-target)-success", "PICK(non-target)-fail"], 
               rotation=15)
    plt.ylim(-ylim, ylim)
    plt.subplot(1, 2, 2)
    plt.boxplot([list_predv_visible_pick_target_success, list_predv_visible_pick_target_fail,
                 list_predv_visible_pick_nontarget_success, list_predv_visible_pick_nontarget_fail])
    plt.title("Visible (t-1)")
    plt.ylabel("V Value")
    plt.xticks([1, 2, 3, 4], 
               ["PICK(target)-success", "PICK(target)-fail", "PICK(non-target)-success", "PICK(non-target)-fail"],
               rotation=15)
    plt.ylim(-ylim, ylim)
    plt.show()

    



def check_visibility(observation_rgb_image, instance_seg_mask):
    # Merge to the foreground segmentation  (Merge pixelwise boolean mask)
    seg_mask_merged = np.zeros_like(list(instance_seg_mask.values())[0])
    for id in instance_seg_mask.keys():
        seg_mask_merged = np.logical_or(seg_mask_merged, instance_seg_mask[id])
           
    observation_masked_rgb_image = np.copy(observation_rgb_image)
    observation_masked_rgb_image[~seg_mask_merged] = 0.0
    is_target_visible = bool(np.sum((observation_masked_rgb_image[:,:,0]>1)))

    return is_target_visible




def main(config):
    """This script tests the execution!"""

    episode(config)




if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)









































