import argparse
import os
import yaml
import pathlib

# PyBullet
import pybullet as pb
import pybullet_data
from envs.common import init_new_bulletclient
from envs.global_object_id_table import GlobalObjectIDTable


# POMDP
from pomdp.POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from pomdp.online_planner_framework import Planner
from pomdp.fetching_POMDP_primitive_object import *
from pomdp.POMCPOW import POMCPOW
from pomdp.policy.guided_policy import FetchingGuidedPolicyPlace
from pomdp.policy.rollout_policy import FetchingRolloutPolicyModel



def episode(config):

    # DEBUGGER
    TIMESTEP_TO_VISUALIZE = 1
    NUM_VISUALIZATION = 100



    # Configuration
    config["plan_params"]["num_sims"] = 1
    USE_GUIDED_POLICY    = config["project_params"]["overridable"]["use_guided_policy"]
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
    #   asdf
    policy_model_1 = FetchingGuidedPolicyPlace(bc, sim_env, robot, manip, config)
    policy_model_2 = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)

    #   setup 2: gt_init_state, goal_condition, init_observation, and inital_belief
    gt_init_state    = make_gt_init_state(bc, sim_env, robot, config)                               # Initial ground truth state 
    goal_condition   = make_goal_condition(config)                                                  # Goal condition
    init_observation = get_initial_observation(sim_env, observation_model, gt_init_state)           # Initial observation instance
    init_belief      = make_belief_random_problem(bc, sim_env, robot, config, PLAN_NUM_SIMS)
    #   setup 3: initialize POMDP
    env     = Environment(transition_model, observation_model, reward_model, gt_init_state)
    agent   = FetchingAgent(bc, sim_env, robot, manip, config, 
                            blackbox_model, policy_model_1, policy_model_2, None, 
                            init_belief, init_observation, goal_condition)    
    pomdp   = POMDP(agent, env, "FetchingProblem")
    agent.imagine_state(gt_init_state, reset=True)
    

    # Plan+Execution loop
    while len(agent.history) < TIMESTEP_TO_VISUALIZE:
        # =====
        # Simulation (planning)
        # =====
        # Randomly select action...
        if len(agent.history) < 3:
            next_action = policy_model_2.sample(
                agent.init_observation, 
                agent.history, 
                pomdp.env.state, 
                agent.goal_condition)
        else:
            next_action = policy_model_1.sample(
                agent.init_observation, 
                agent.history, 
                pomdp.env.state, 
                agent.goal_condition)

        # =====
        # Execution
        # =====
        # Restore the ground truth state in simulation
        print("In execution...")
        agent.imagine_state(pomdp.env.state, reset=True)
        # Execution in real world
        observation, reward, termination = pomdp.env.execute(next_action)

        # Logging
        next_action: FetchingAction
        observation: FetchingObservation
        print(f"[next_action at depth {len(agent.history)}] {next_action}")
        print(f"[observation at depth {len(agent.history)}] contact={observation.grasp_contact}")
        agent.update(next_action, observation, reward) 
        agent.imagine_state(pomdp.env.state, reset=True)

        # Check termination condition!
        if (termination == TERMINATION_SUCCESS) or (termination == TERMINATION_FAIL):
            break


    # Draw taskspace plot
    draw_taskpace(bc, sim_env)
    # Sample new actions!
    count = 0
    list_sampled_actions = []
    while len(list_sampled_actions) < NUM_VISUALIZATION:
        sampled_action: FetchingAction \
            = agent._policy_model.sample(
                agent.init_observation, 
                agent.history, 
                pomdp.env.state, 
                agent.goal_condition)
        if sampled_action.pos is not None:
            list_sampled_actions.append(sampled_action)
        else:
            print("fail")
            count += 1
    # Draw actions
    action: FetchingAction
    for action in list_sampled_actions:
        pos = action.pos
        if pos is not None:
            _ = bc.loadURDF(os.path.join(pybullet_data.getDataPath(), "sphere_small.urdf"), pos, globalScaling=0.15)
            pass


    # # Imshow
    # import matplotlib.pyplot as plt
    # state_before_action = pomdp.env.state
    # for sam_action in list_sampled_actions:
    #     print("In execution...")
    #     agent.imagine_state(state_before_action, reset=True)
    #     observation, reward, termination = pomdp.env.execute(sam_action)
    #     plt.imshow(observation.rgb_image)






    # Hold
    while True:
        time.sleep(5000)


    
def draw_taskpace(bc: BulletClient, env: BinpickEnvPrimitive):
    
    center = env.TASKSPACE_CENTER
    half_ranges = env.TASKSPACE_HALF_RANGES

    point1 = np.array(center)
    point1[0] += half_ranges[0]
    point1[1] += half_ranges[1]

    point2 = np.array(center)
    point2[0] += half_ranges[0]
    point2[1] -= half_ranges[1]

    point3 = np.array(center)
    point3[0] -= half_ranges[0]
    point3[1] -= half_ranges[1]

    point4 = np.array(center)
    point4[0] -= half_ranges[0]
    point4[1] += half_ranges[1]

    bc.addUserDebugLine(point1, point2, [0, 0, 1])
    bc.addUserDebugLine(point2, point3, [0, 0, 1])
    bc.addUserDebugLine(point3, point4, [0, 0, 1])
    bc.addUserDebugLine(point4, point1, [0, 0, 1])    



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









































