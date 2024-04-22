from run import *
from load import get_loader, LightDarkDataset
import matplotlib.pyplot as plt

config = Settings()
comparison_traj_num = 2
compare_threshold = 0.0001
sample_traj_num=100

split_ratio=0.8
_, pos_test = load_dataset('/home/share_folder/dataset/', 'success_mini', split_ratio)
_, neg_test = load_dataset('/home/share_folder/dataset/', 'fail_mini', split_ratio)
save_traj = '/home/pomdp/workspace/POMDP/toy_domain/Learning/same_traj_dist'

test = pos_test[sample_traj_num:] + neg_test[sample_traj_num:]
sample_traj = pos_test[:sample_traj_num] + neg_test[:sample_traj_num]
test_dataset = LightDarkDataset(config, test, None)
sample_dataset = LightDarkDataset(config, sample_traj, None)

test_np_obs = np.inf * np.ones((len(test_dataset), 2, comparison_traj_num))
test_np_action = np.inf * np.ones((len(test_dataset), 2, comparison_traj_num))
for i in range(len(test_dataset)):
    traj_len = np.array(test_dataset[i]['observation']).shape[0]
    if traj_len < comparison_traj_num:
        test_np_obs[i][:, :traj_len] = np.transpose(np.array(test_dataset[i]['observation']))
        test_np_action[i][:, :traj_len] = np.transpose(np.array(test_dataset[i]['action']))
    else:
        test_np_obs[i] = np.transpose(np.array(test_dataset[i]['observation']), (1,0))[:,:comparison_traj_num]
        test_np_action[i] = np.transpose(np.array(test_dataset[i]['action']), (1,0))[:,:comparison_traj_num]

for i in range(len(sample_dataset)):
    if np.array(sample_dataset[i]['observation']).shape[0] > comparison_traj_num:
        sample_traj_obs = np.expand_dims(np.transpose(np.array(sample_dataset[i]['observation']), (1,0))[:,:comparison_traj_num], axis=0)
        sample_traj_action = np.expand_dims(np.transpose(np.array(sample_dataset[i]['action']), (1,0))[:,:comparison_traj_num], axis=0)
        tmp_comparison_obs = (np.abs(sample_traj_obs - test_np_obs) > compare_threshold).sum(axis=2).sum(axis=1)
        tmp_comparison_action = (np.abs(sample_traj_action - test_np_action) > compare_threshold).sum(axis=2).sum(axis=1)
        same_obs = np.argwhere(tmp_comparison_obs==0)
        same_action = np.argwhere(tmp_comparison_action==0)
        final = []
        for j in same_obs:
            if j in same_action:
                final.append(int((np.sum(test_dataset[j.item()]['reward'])-(-30))/130*config.num_bin))
        if len(final) > 3:
            print(i, final)
            plt.hist(final, config.num_bin, density=True)
            plt.xlim(0, config.num_bin)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(save_traj, f'{str(i)}.png'))
        plt.clf()
    else:
        print(i,"pass")
        

if __name__ == '__main__':
    