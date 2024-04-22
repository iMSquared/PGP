import os
import pickle
import numpy as np


global grasp_test_result
grasp_test_result = {}

def reset():
    grasp_test_result["num_trial"] = 0
    grasp_test_result["num_grasp_success"] = 0
    grasp_test_result["num_fail"] = 0
    grasp_test_result["reason"] = [0, 0, 0] # Fail to find grasp pose / Fail to close / Drop the object
    
    # Time
    grasp_test_result["time_grasp_pose_sampling"] = []
    grasp_test_result["time_ik"] = []
    grasp_test_result["time_collision_check"] = []
    grasp_test_result["time_affordance_check"] = []
    grasp_test_result["time_rrt"] = []
    
def result():
    result_dir = "exp_grasp_test_12.28_particle"
    
    obj_class = [
        "YcbBanana",
        "YcbChipsCan",
        "YcbCrackerBox",
        "YcbFoamBrick",
        "YcbGelatinBox",
        "YcbHammer",
        "YcbMasterChefCan",
        "YcbMediumClamp",
        "YcbMustardBottle",
        "YcbPear",
        "YcbPottedMeatCan",
        "YcbPowerDrill",
        "YcbScissors",
        "YcbStrawberry",
        "YcbTennisBall",
        "YcbTomatoSoupCan",
    ]
    
    with open(f"{result_dir}/result.txt", 'a') as f:
        f.write(f"YCB dataset\n")

    # for cls in obj_class:
    #     name = cls
    #     print(f"=========== Start: {name} ===========")
    #     test_result = {}
    #     with open(f"{result_dir}/{name}.pickle", 'rb') as f:
    #         data = pickle.load(f)
    #         test_result["num_trial"] = data["num_trial"]
    #         test_result["num_success"] = data["num_trial"] - data["num_fail"]
    #         test_result["num_fail"] = data["num_fail"]
    #         test_result["reason"] = data["reason"]

    #     with open(f"{result_dir}/result.txt", 'a') as f:
    #         f.write(f"{name}: {test_result}\n")
    #     print(f"=========== End: {name} ===========")
    
    # time_grasp_pose_sampling = []
    # time_ik = []
    # time_collision_check = []
    # time_affordance_check = []
    # time_rrt = []
    # for cls in obj_class:
    #     name = cls
    #     print(f"=========== Start: {name} ===========")
    #     with open(f"{result_dir}/{name}.pickle", 'rb') as f:
    #         data = pickle.load(f)
    #         time_grasp_pose_sampling = time_grasp_pose_sampling + data["time_grasp_pose_sampling"]
    #         time_ik = time_ik + data["time_ik"]
    #         time_collision_check = time_collision_check + data["time_collision_check"]
    #         time_affordance_check = time_affordance_check + data["time_affordance_check"]
    #         time_rrt = time_rrt + data["time_rrt"]
    #     print(f"=========== End: {name} ===========")
    
    # print("# grasp pose sampling:", len(time_grasp_pose_sampling))
    # print("Time grasp pose sampling:", np.mean(np.asarray(time_grasp_pose_sampling)), np.std(np.asarray(time_grasp_pose_sampling)))
    # print("# ik:", len(time_ik))
    # print("Time ik:", np.mean(np.asarray(time_ik)), np.std(np.asarray(time_ik)))
    # print("# collision check:", len(time_collision_check))
    # print("Time collision check:", np.mean(np.asarray(time_collision_check)), np.std(np.asarray(time_collision_check)))
    # print("# affordance check:", len(time_affordance_check))
    # print("Time affordance check:", np.mean(np.asarray(time_affordance_check)), np.std(np.asarray(time_affordance_check)))
    # print("# rrt:", len(time_rrt))
    # print("Time rrt:", np.mean(np.asarray(time_rrt)), np.std(np.asarray(time_rrt)))
    
  
    with open(f"{result_dir}/result.txt", 'a') as f:
        f.write(f"Gaussian belief\n")
        
    for cls in obj_class:
        test_result = {
            "num_trial": 0,
            "num_success": 0,
            "num_fail": 0,
            "reason": [0, 0, 0]
        }
        for i in range(10):
            name = f"{cls}_{i}"
            print(f"=========== Start: {name} ===========")
            with open(f"{result_dir}/{name}.pickle", 'rb') as f:
                data = pickle.load(f)
                # test_result["num_success"] += data["num_success"]
                # test_result["num_fail"] += data["num_fail"]
                # test_result["reason"][0] += data["reason"][0]
                # test_result["reason"][1] += data["reason"][1]
                # test_result["reason"][2] += data["reason"][2]
                test_result["num_trial"] += data["num_trial"]
                test_result["num_success"] += (data["num_trial"] - data["num_fail"])
                test_result["num_fail"] += data["num_fail"]
                test_result["reason"][0] += data["reason"][0]
                test_result["reason"][1] += data["reason"][1]
                test_result["reason"][2] += data["reason"][2]
            print(f"=========== End: {name} ===========")

        with open(f"{result_dir}/result.txt", 'a') as f:
            f.write(f"{cls}: {test_result}\n")
            
    return


if __name__ == '__main__':
    result()