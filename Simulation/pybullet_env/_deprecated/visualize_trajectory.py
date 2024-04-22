import argparse
import os
import yaml

# PyBullet
import pybullet as pb
from imm.pybullet_util.bullet_client import BulletClient
from envs.binpick_env_primitive_object import BinpickEnvPrimitive
from envs.robot import UR5, FrankaPanda
from envs.manipulation import Manipulation

# POMDP
from POMDP_framework import Agent, Environment, POMDP, BlackboxModel
from fetching_POMDP import *

# Debugging
import pickle


def vis_traj(config, state, action, margin=0.01):
    
    # Connect bullet client
    sim_id = pb.connect(pb.GUI)
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
    bc.setPhysicsEngineParameter(enableFileCaching=0)
    
    # Simulation initialization
    # env
    env = BinpickEnvPrimitive(bc, config)

    # Remove objects
    for obj_uid in env.objects_uid:
        bc.removeBody(obj_uid)
        
    # Reload objects
    env.objects_uid = [bc.createMultiBody(
        baseMass = 3.0,
        baseCollisionShapeIndex = env.col_cylinder_shape_id,
        baseVisualShapeIndex = env.vis_cylinder_shape_id,
        basePosition = pos
    )
    for pos in state["object"].values()]
    
    # Path to URDF
    project_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(project_path, config["project_params"]["custom_urdf_path"])
    # visualize robots for each trajectory step    
    robot_params = config["robot_params"]["ur5"]
    joint_indices_arm = np.array(robot_params["joint_indices_arm"])
    link_index_endeffector_base         = robot_params["link_index_endeffector_base"]
    joint_index_finger                  = robot_params["joint_index_finger"]
    joint_value_finger_open             = robot_params["joint_value_finger_open"]
    distance_finger_to_endeffector_base = robot_params["distance_finger_to_endeffector_base"]
    joint_indices_finger_mimic = np.array(robot_params["joint_indices_finger_mimic"])
    joint_gear_ratio_mimic = np.array(robot_params["joint_gear_ratio_mimic"])

    for step, traj in enumerate(action):
        uid = bc.loadURDF(
                fileName        = os.path.join(urdf_path, robot_params["path"]),
                basePosition    = robot_params["pos"],
                baseOrientation = bc.getQuaternionFromEuler(robot_params["orn"]),
                useFixedBase    = True)
        
        for idx in joint_indices_arm:
            bc.resetJointState(uid, idx, traj[idx])
            bc.resetJointState(uid, joint_index_finger, traj[8])
            target_position = -1.0 * joint_gear_ratio_mimic * np.asarray(traj[8])
        for i, idx in enumerate(joint_indices_finger_mimic):
            bc.resetJointState(uid, idx, target_position[i])
        
        bc.performCollisionDetection()
        
        contacts = []
        contact_with_cabinet = bc.getContactPoints(bodyA=uid, bodyB=env.cabinet_uid)
        if contact_with_cabinet:
            contacts.append(contact_with_cabinet)
        for obj_uid in env.objects_uid:
            contact_with_obj = bc.getContactPoints(bodyA=uid, bodyB=obj_uid)
            if contact_with_obj:
                contacts.append(contact_with_obj)
        
        if contacts:
            for contact in contacts:
                for c in contact:
                    if c[8] < 0.0:
                        print(c[5], c[6], c[8])
                        start_pos = np.asarray(c[5])
                        end_pos = start_pos + 0.0025 * np.asarray(c[7])
                        bc.addUserDebugLine(
                            start_pos,
                            end_pos,
                            lineColorRGB = [1.0, 0.0, 0.0],
                            lineWidth = 5,
                            lifeTime = 0.0
                        )

    bc.disconnect()
    
    

if __name__=="__main__":

    # Specify the config file
    parser = argparse.ArgumentParser(description="Config")
    parser.add_argument("--config", type=str, default="config_primitive_object.yaml", help="Specify the config file to use.")
    params = parser.parse_args()

    # Open yaml config file
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg", params.config), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # with open('vis_traj_data_4_1_PICK_fail_unstable.pickle', 'rb') as f:
    #     data = pickle.load(f)
    datas = []
    with open('vis_traj_strange_path_2.pickle', 'rb') as f:
        while True:
            try:
                datas.append(pickle.load((f)))
            except EOFError:
                break
    
    data = datas[-1]
    
    state = data['state'][0]
    action = data['action']

    vis_traj(config, state, action)

    # state = {
    #     "object": {
    #         1: (0.699991303179041, -3.1892360956485677e-06, 0.5599883433037834),
    #         2: (0.8499930433173661, 0.029994561675481382, 0.5599894406151616)
    #     },
    #     "robot": ((1.7892911709415383e-08, -2.093999919836774, 2.0700001577977276, -2.617999997227532, -1.5700000349231766, 1.973266609425047e-12), 0.7999790958903468)
    # }
    # action = [
    #     np.asarray([ 0.00000000e+00,  3.57942022e-08, -2.09399984e+00,  2.07000032e+00,
    #    -2.61799999e+00, -1.57000007e+00,  1.97326662e-12,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  3.57942022e-08, -2.09399984e+00,  2.07000032e+00,
    #    -2.61799999e+00, -1.57000007e+00,  1.97326662e-12,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  3.24822975e-03, -2.06675573e+00,  2.06363471e+00,
    #    -2.65171726e+00, -1.58998952e+00, -2.01472856e-05,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  6.49642371e-03, -2.03951163e+00,  2.05726910e+00,
    #    -2.68543452e+00, -1.60997896e+00, -4.02945732e-05,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  9.74461766e-03, -2.01226752e+00,  2.05090349e+00,
    #    -2.71915179e+00, -1.62996841e+00, -6.04418608e-05,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.29928116e-02, -1.98502341e+00,  2.04453788e+00,
    #    -2.75286905e+00, -1.64995785e+00, -8.05891483e-05,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.62410056e-02, -1.95777930e+00,  2.03817227e+00,
    #    -2.78658631e+00, -1.66994730e+00, -1.00736436e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.94891995e-02, -1.93053520e+00,  2.03180666e+00,
    #    -2.82030358e+00, -1.68993674e+00, -1.20883724e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  2.27373935e-02, -1.90329109e+00,  2.02544105e+00,
    #    -2.85402084e+00, -1.70992619e+00, -1.41031011e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  2.59855874e-02, -1.87604698e+00,  2.01907544e+00,
    #    -2.88773810e+00, -1.72991563e+00, -1.61178299e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  2.92337814e-02, -1.84880288e+00,  2.01270983e+00,
    #    -2.92145537e+00, -1.74990508e+00, -1.81325586e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  3.24819754e-02, -1.82155877e+00,  2.00634423e+00,
    #    -2.95517263e+00, -1.76989453e+00, -2.01472874e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  3.57301693e-02, -1.79431466e+00,  1.99997862e+00,
    #    -2.98888989e+00, -1.78988397e+00, -2.21620161e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  3.89783633e-02, -1.76707056e+00,  1.99361301e+00,
    #    -3.02260716e+00, -1.80987342e+00, -2.41767449e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  4.22265572e-02, -1.73982645e+00,  1.98724740e+00,
    #    -3.05632442e+00, -1.82986286e+00, -2.61914737e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  4.54747512e-02, -1.71258234e+00,  1.98088179e+00,
    #    -3.09004168e+00, -1.84985231e+00, -2.82062024e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  4.87229451e-02, -1.68533824e+00,  1.97451618e+00,
    #    -3.12375895e+00, -1.86984175e+00, -3.02209312e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  5.19711391e-02, -1.65809413e+00,  1.96815057e+00,
    #    -3.15747621e+00, -1.88983120e+00, -3.22356599e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  5.52193331e-02, -1.63085002e+00,  1.96178496e+00,
    #    -3.19119348e+00, -1.90982064e+00, -3.42503887e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  5.84675270e-02, -1.60360591e+00,  1.95541935e+00,
    #    -3.22491074e+00, -1.92981009e+00, -3.62651174e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  6.17157210e-02, -1.57636181e+00,  1.94905374e+00,
    #    -3.25862800e+00, -1.94979954e+00, -3.82798462e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  6.49639149e-02, -1.54911770e+00,  1.94268814e+00,
    #    -3.29234527e+00, -1.96978898e+00, -4.02945750e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  6.82121089e-02, -1.52187359e+00,  1.93632253e+00,
    #    -3.32606253e+00, -1.98977843e+00, -4.23093037e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  7.14603028e-02, -1.49462949e+00,  1.92995692e+00,
    #    -3.35977979e+00, -2.00976787e+00, -4.43240325e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  7.47084968e-02, -1.46738538e+00,  1.92359131e+00,
    #    -3.39349706e+00, -2.02975732e+00, -4.63387612e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  7.79566908e-02, -1.44014127e+00,  1.91722570e+00,
    #    -3.42721432e+00, -2.04974676e+00, -4.83534900e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  8.12048847e-02, -1.41289717e+00,  1.91086009e+00,
    #    -3.46093158e+00, -2.06973621e+00, -5.03682188e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  8.44530787e-02, -1.38565306e+00,  1.90449448e+00,
    #    -3.49464885e+00, -2.08972565e+00, -5.23829475e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  8.77012726e-02, -1.35840895e+00,  1.89812887e+00,
    #    -3.52836611e+00, -2.10971510e+00, -5.43976763e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  9.09494666e-02, -1.33116484e+00,  1.89176326e+00,
    #    -3.56208337e+00, -2.12970454e+00, -5.64124050e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  9.41976605e-02, -1.30392074e+00,  1.88539765e+00,
    #    -3.59580064e+00, -2.14969399e+00, -5.84271338e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  9.74458545e-02, -1.27667663e+00,  1.87903205e+00,
    #    -3.62951790e+00, -2.16968344e+00, -6.04418625e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.00694048e-01, -1.24943252e+00,  1.87266644e+00,
    #    -3.66323517e+00, -2.18967288e+00, -6.24565913e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.03942242e-01, -1.22218842e+00,  1.86630083e+00,
    #    -3.69695243e+00, -2.20966233e+00, -6.44713201e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.07190436e-01, -1.19494431e+00,  1.85993522e+00,
    #    -3.73066969e+00, -2.22965177e+00, -6.64860488e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.10438630e-01, -1.16770020e+00,  1.85356961e+00,
    #    -3.76438696e+00, -2.24964122e+00, -6.85007776e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.13686824e-01, -1.14045610e+00,  1.84720400e+00,
    #    -3.79810422e+00, -2.26963066e+00, -7.05155063e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.16935018e-01, -1.11321199e+00,  1.84083839e+00,
    #    -3.83182148e+00, -2.28962011e+00, -7.25302351e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.20183212e-01, -1.08596788e+00,  1.83447278e+00,
    #    -3.86553875e+00, -2.30960955e+00, -7.45449638e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.23431406e-01, -1.05872378e+00,  1.82810717e+00,
    #    -3.89925601e+00, -2.32959900e+00, -7.65596926e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    #     np.asarray([ 0.00000000e+00,  1.26679600e-01, -1.03147967e+00,  1.82174156e+00,
    #    -3.93297327e+00, -2.34958845e+00, -7.85744214e-04,  0.00000000e+00,
    #    -7.99979438e-01, -7.99979448e-01,  7.99980456e-01,  7.99979438e-01,
    #     7.99979327e-01, -7.99979739e-01]),
    # ]
