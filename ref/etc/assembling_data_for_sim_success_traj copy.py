import os
import pickle
import numpy as np

data_dir = os.path.join(os.getcwd(), 'result/dataset')

# For train dataset
version = ['recur_1_sim100_1_big', 'recur_1_sim100_2_big', 'recur_1_sim100_3_big', 'recur_1_sim100_4_big', 'recur_1_sim100_5_big', 'recur_1_sim100_6_big', 'recur_1_sim100_7_big', 'recur_1_sim100_8_big']
# version = ['long_1K']
# version = ['sim10K_test']

print('='*20, 'train dataset', '='*20)
print('='*20, 'loading', '='*20)

data = []
for ver in version:
    with open(os.path.join(data_dir, f'{ver}/simulation_history_data.pickle'), 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    print(f'Loading #{ver} is finished!')

print('#total planning:', len(data))

# save all path information into separate lists
print('='*20, 'saving', '='*20)
action = []
observation = []
next_state = []
reward = []
traj_len = []

for k in range(len(data)): # about all pickle
    for i in range(len(data[k])): # about all traj.
        tmp_action = []
        tmp_observation = []
        tmp_next_state = []
        tmp_reward = []
        # append trajectory length to dataset
        tmp_traj_len = []
        for j in range(len(data[k][i])): # about one traj.
            tmp_action.append(data[k][i][j][0])
            tmp_observation.append(data[k][i][j][1])
            tmp_next_state.append(data[k][i][j][2])
            tmp_reward.append(data[k][i][j][3])

        action.append(np.asarray(tmp_action))
        observation.append(np.asarray(tmp_observation))
        next_state.append(np.asarray(tmp_next_state))
        reward.append(np.asarray(tmp_reward))
        traj_len.append(len(data[k][i]))


print('#total traj.:', len(action))
print(len(traj_len))

dataset = {}
dataset['action'] = action
dataset['observation'] = observation
dataset['next_state'] = next_state
dataset['reward'] = reward
dataset['traj_len'] = traj_len

with open(os.path.join(data_dir, 'light_dark_recur_train.pickle'), 'wb') as f:
    pickle.dump(dataset, f)

print('Saving is finished!!')



# For test dataset

version = ['recur_1_sim100_9_big', 'recur_1_sim100_10_big']

print('='*20, 'test dataset', '='*20)
print('='*20, 'loading', '='*20)

data = []
for ver in version:
    with open(os.path.join(data_dir, f'{ver}/simulation_history_data.pickle'), 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    print(f'Loading #{ver} is finished!')

print('#total planning:', len(data))

# save all path information into separate lists
print('='*20, 'saving', '='*20)
action = []
observation = []
next_state = []
reward = []
traj_len = []

for k in range(len(data)): # about all pickle
    for i in range(len(data[k])): # about all traj.
        tmp_action = []
        tmp_observation = []
        tmp_next_state = []
        tmp_reward = []
        # append trajectory length to dataset
        tmp_traj_len = []
        for j in range(len(data[k][i])): # about one traj.
            tmp_action.append(data[k][i][j][0])
            tmp_observation.append(data[k][i][j][1])
            tmp_next_state.append(data[k][i][j][2])
            tmp_reward.append(data[k][i][j][3])
            tmp_traj_len.append(len(data[k][i][j][1]))

        action.append(np.asarray(tmp_action))
        observation.append(np.asarray(tmp_observation))
        next_state.append(np.asarray(tmp_next_state))
        reward.append(np.asarray(tmp_reward))
        traj_len.append(tmp_traj_len)


print('#total traj.:', len(action))
print(len(traj_len))


dataset = {}
dataset['action'] = action
dataset['observation'] = observation
dataset['next_state'] = next_state
dataset['reward'] = reward
dataset['traj_len'] = traj_len

with open(os.path.join(data_dir, 'light_dark_recur_test.pickle'), 'wb') as f:
    pickle.dump(dataset, f)

print('Saving is finished!!')