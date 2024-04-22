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
from pomdp.value.guided_v_value_mse import FetchingGuidedValueMSE
from pomdp.value.guided_v_value_preference import FetchingGuidedValuePreference
from pomdp.value.guided_q_value_mse import FetchingGuidedQValueMSE
from pomdp.value.guided_q_value_preference import FetchingGuidedQValuePreference



def episode(config):

    # DEBUGGER
    TIMESTEP_TO_VISUALIZE = 1
    MESHGRID_RESOLUTION = 20
    config["project_params"]["overridable"]["use_guided_value"] = True
    config["project_params"]["overridable"]["guide_q_value"]    = False
    config["project_params"]["overridable"]["guide_preference"] = True



    # Configuration
    config["plan_params"]["num_sims"] = 1
    USE_GUIDED_POLICY    = config["project_params"]["overridable"]["use_guided_policy"]
    USE_GUIDED_VALUE     = config["project_params"]["overridable"]["use_guided_value"]
    GUIDE_Q_VALUE        = config["project_params"]["overridable"]["guide_q_value"]
    GUIDE_PREFERENCE     = config["project_params"]["overridable"]["guide_preference"]
    COLLECT_DATA         = config["project_params"]["overridable"]["collect_data"]
    PLAN_MAX_DEPTH       = config["plan_params"]["max_depth"]
    PLAN_NUM_SIMS        = config["plan_params"]["num_sims"]
    PLAN_DISCOUNT_FACTOR = config["plan_params"]["discount_factor"]

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
    #   asdf
    policy_model_1 = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)
    policy_model_2 = FetchingRolloutPolicyModel(bc, sim_env, robot, manip, config)

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
    agent.imagine_state(gt_init_state, reset=True)
    

    # Plan+Execution loop
    while len(agent.history) < TIMESTEP_TO_VISUALIZE:
        # =====
        # Simulation (planning)
        # =====
        # Randomly select action...
        if len(agent.history) < 2:
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

    # 1. Generate query point.
    query_state = pomdp.env.state
    queries = meshgrid_querypoint(bc, sim_env, agent.goal_condition, MESHGRID_RESOLUTION)
    position = []
    values = []
    for q in queries:
        mse_V = -1000

        # 2. Generate V history sequence. (10 direction and take max)
        # dyaw_bin = np.linspace(-3.14, 3.14, 6)
        dyaw_bin = [0]
        for dyaw in dyaw_bin:
            # Generate place action
            last_action: FetchingAction = agent.history[-1].action
            last_gid = last_action.aimed_gid
            place_pos  = tuple([q[0], q[1], last_action.pos[2]])
            dyaw     = dyaw
            prev_orn = last_action.orn
            prev_orn_q = bc.getQuaternionFromEuler(prev_orn)
            # delta_theta = random.uniform(-np.pi, np.pi)
            yaw_rot_in_world = np.array([0, 0, dyaw])
            yaw_rot_in_world_q = bc.getQuaternionFromEuler(yaw_rot_in_world) 
            _, place_orn_q = bc.multiplyTransforms([0, 0, 0], yaw_rot_in_world_q,
                                                [0, 0, 0], prev_orn_q)
            place_orn = bc.getEulerFromQuaternion(place_orn_q)

            # Filtering by motion plan
            place_orn_q = bc.getQuaternionFromEuler(place_orn)                 # Outward orientation (surface normal)
            modified_pos_back, modified_orn_q_back, \
                modified_pos, modified_orn_q \
                = manip.get_ee_pose_from_target_pose(place_pos, place_orn_q)   # Inward orientation (ee base)
            joint_pos_src, joint_pos_dst \
                = manip.solve_ik_numerical(modified_pos, modified_orn_q)       # src is captured from current pose.

            holding_obj_uid = sim_env.gid_to_uid[query_state.holding_obj_gid]
            traj = manip.motion_plan(joint_pos_src, joint_pos_dst, holding_obj_uid) 
            # Restore
            agent.imagine_state(query_state, reset=True)

            if traj is not None:
                query_action = FetchingAction(
                    type        = ACTION_PLACE, 
                    aimed_gid   = last_gid,
                    pos         = place_pos,
                    orn         = place_orn,
                    traj        = traj,
                    delta_theta = dyaw)
            else:
                query_action = FetchingAction(ACTION_PLACE, None, None, None, None, None)
            # Execution in real world
            observation, reward, termination = pomdp.env.execute(query_action)
            query_history = agent.history + (HistoryEntry(query_action, observation, reward, phase=PHASE_EXECUTION),)   

            if termination == "continue":
                # Infer the mse V.
                dir_mse_V = agent._value_model.sample(agent.init_observation, query_history, pomdp.env.state, agent.goal_condition)
            else:
                dir_mse_V = 0

            if dir_mse_V > mse_V:
                mse_V = dir_mse_V
            
            # Restore
            agent.imagine_state(query_state, reset=True)
            
        print(mse_V)
        values.append(mse_V)
    

    print("visualizing values.")
    # 4. Draw the color
    values = minmax_to_01(values, ignore_zero=True)
    for q, v in zip(queries, values):
        
        if v == 0:
            continue
        pos = [q[0], q[1], 0.681]
        vizshape_id = bc.createVisualShape(bc.GEOM_SPHERE, radius=0.005)
        colshape_id = bc.createVisualShape(bc.GEOM_SPHERE, radius=0.005)
        uid = bc.createMultiBody(0, vizshape_id, colshape_id, pos)
        bc.changeVisualShape(uid, -1, rgbaColor=[v, v, v, 1.0])


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


def meshgrid_querypoint(bc: BulletClient, 
                        env: BinpickEnvPrimitive, 
                        goal_condition: Tuple[float], 
                        MESHGRID_RESOLUTION: int) -> npt.NDArray: 

    center = env.TASKSPACE_CENTER
    half_ranges = env.TASKSPACE_HALF_RANGES
    goal_pose = goal_condition[3:]
    x_max = center[0] + half_ranges[0]
    x_min = center[0] - half_ranges[0]
    y_max = center[1] + half_ranges[1]
    y_min = center[1] - half_ranges[1]

    x = np.linspace(x_min, x_max, num=MESHGRID_RESOLUTION)
    y = np.linspace(y_min, y_max, num=MESHGRID_RESOLUTION)

    grid_x, grid_y = np.meshgrid(x, y)
    grid_x = np.expand_dims(grid_x, axis=-1)
    grid_y = np.expand_dims(grid_y, axis=-1)
    queries = np.reshape(np.concatenate((grid_x, grid_y), axis=-1), newshape=(-1, 2))
    # queries = np.concatenate((queries, [goal_pose]), axis=0)

    return queries


def np_softmax(x, temperature):
    x = np.asarray(x)
    x /= temperature

    list_s = []
    for v in x:
        _x = x-v
        s = np.exp(0) / np.exp(_x).sum()
        list_s.append(s)

    list_s = np.asarray(list_s)

    return list_s


def minmax_to_01(weights, ignore_zero):
    weights = np.asarray(weights)
    zeros = weights == 0
    if ignore_zero:
        minval = np.min(weights[np.nonzero(weights)])
        maxval = np.max(weights[np.nonzero(weights)])
    else:
        minval = min(weights)
        maxval = max(weights)

    weights = np.asarray(weights)
    weights = (weights - minval) / (maxval - minval)
    weights[zeros] = 0

    return weights


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









































