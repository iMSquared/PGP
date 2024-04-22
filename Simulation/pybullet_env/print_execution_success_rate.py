import os
import json
import matplotlib.pyplot as plt


def main():
    EXP_DIR_PATH = "/home/sanghyeon/workspace/vessl/experiments/3obj3000_prefV+prefpolicy_tree_reinv/history_json"
    MAX_DEPTH = 8

    files = sorted(os.listdir(EXP_DIR_PATH))
    jsons = []
    for fn in files:
        with open(os.path.join(EXP_DIR_PATH, fn), "r") as f:
            data = json.load(f)
            jsons.append(data)

    num_lucky = 0
    num_success = 0
    num_continue = 0
    num_fail = 0
    num_trial = 0
    count_pick_target = [0 for i in range(MAX_DEPTH)]
    count_pick_nontarget = [0 for i in range(MAX_DEPTH)]
    list_planning_time = []

    for d in jsons:
        num_trial += 1
        if d["termination"] == "continue":
            num_continue += 1
        if d["termination"] == "success":
            num_success += 1
        if d["termination"] == "fail":
            num_fail += 1
        if d["termination"] == "success" and len(d["episode_action_history"]) == 2:
            num_lucky += 1

        for time_step, action in enumerate(d["episode_action_history"]):
            try:
                action = action.split(",")
                action_type   = action[0]
                action_target = int(action[1].strip())

                if action_type=="PICK" and action_target==0:
                    count_pick_target[time_step] += 1
                elif action_type=="PICK" and action_target!=0:
                    count_pick_nontarget[time_step] += 1
            except:
                continue

        list_planning_time.append(d["list_time_taken_per_planning"][0])


    success_rate = float(num_success)/float(num_trial)
    net_success_rate = float(num_success-num_lucky)/float(num_trial)
    continue_rate = float(num_continue)/float(num_trial)
    fail_rate = float(num_fail)/float(num_trial)
    lucky_rate = float(num_lucky)/float(num_trial)
    print(f"success rate: {success_rate:.4f}, continue rate: {continue_rate:.4f}, fail_rate: {fail_rate:.4f}, episodes: {num_trial}")
    print(f"net success rate: {net_success_rate:4f}, lucky?: {lucky_rate:.4f}")
    print(f"avg planning time: {sum(list_planning_time)/len(list_planning_time)}")


    plt.figure(figsize=(8,8))
    plt.subplot(2, 1, 1)
    plt.bar(range(MAX_DEPTH), count_pick_target)
    plt.title("PICK(target) - Does not guarantee grasp success")
    plt.ylabel("# picks")
    plt.ylim(0, len(jsons))
    plt.subplot(2, 1, 2)
    plt.bar(range(MAX_DEPTH), count_pick_nontarget, color="#e35f62")
    plt.title("PICK(non-target)")
    plt.ylabel("# picks")
    plt.ylim(0, len(jsons))
    plt.show()





if __name__=="__main__":
    main()