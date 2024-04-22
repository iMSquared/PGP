import os
import pickle

global debug_data

debug_data = {}

def debug_data_reset():
    debug_data['simulation_time'] = []
    debug_data['planning_time'] = []
    debug_data['num_sim_success'] = []
    debug_data['rollout_time'] = []
    debug_data['TransitionModel.sample()'] = []
    debug_data['ObservationModel.sample()'] = []
    debug_data['ObservationModel.probability()'] = []
    debug_data['PolicyModel.sample()'] = []
    debug_data['Agent._set_simulation()'] = []
    debug_data['fetch_state_info()'] = []
    debug_data['filter_grasp_pose_time'] = []
    debug_data['rrt_time'] = []
    debug_data['arm_control'] = []
    debug_data['finger_control'] = []
    debug_data['wait()'] = []
    debug_data['check_holding_object()'] = []
    debug_data['fail_case'] = {
        'PICK': [0, 0, 0],  # sampling / MP / close
        'PLACE': [0, 0, 0], # -        / MP / -
        'termination': 0,
        'max_depth': 0
    }
    debug_data['num_action'] = [0, 0, 0]    # PICK / PLACE
    

# def result():
#     root_dir = "exp_12.28"
#     data_file_list = os.listdir(root_dir)
#     print(data_file_list)

#     fail_case = {
#         'PICK': [0, 0, 0],  # sampling / MP / close
#         'MOVE': [0, 0, 0],  # -        / MP / drop
#         'PLACE': [0, 0, 0], # -        / MP / -
#         'termination': 0,
#         'max_depth': 0
#     }
#     num_success = 0
#     num_action = [0, 0, 0]

#     for i, data_file in enumerate(data_file_list):
#         with open(f"{root_dir}/{data_file}", "rb") as f:
#             print(f"{root_dir}/{data_file}")
            
#             data = pickle.load(f)
#             fail_case['PICK'][0] += data['fail_case']['PICK'][0]
#             fail_case['PICK'][1] += data['fail_case']['PICK'][1]
#             fail_case['PICK'][2] += data['fail_case']['PICK'][2]
            
#             fail_case['MOVE'][0] += data['fail_case']['MOVE'][0]
#             fail_case['MOVE'][1] += data['fail_case']['MOVE'][1]
#             fail_case['MOVE'][2] += data['fail_case']['MOVE'][2]
            
#             fail_case['PLACE'][0] += data['fail_case']['PLACE'][0]
#             fail_case['PLACE'][1] += data['fail_case']['PLACE'][1]
#             fail_case['PLACE'][2] += data['fail_case']['PLACE'][2]
            
#             fail_case['termination'] += data['fail_case']['termination']
#             fail_case['max_depth'] += data['fail_case']['max_depth']
            
#             print(data['num_sim_success'])
#             num_success += data['num_sim_success'][0]
            
#             num_action[0] += data['num_action'][0]
#             num_action[1] += data['num_action'][1]
#             num_action[2] += data['num_action'][2]
            
#     print(num_action)
#     print(fail_case)
#     print(num_success)

# if __name__ == '__main__':
#     result()