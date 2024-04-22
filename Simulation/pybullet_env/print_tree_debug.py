import os
import json
import numpy as np


files = sorted(os.listdir("./tree_debug_log"))
jsons = []
for fn in files:
    with open(os.path.join("./tree_debug_log", fn), "r") as f:
        data = json.load(f)
        jsons.append(data)


next_action_target = 0
null_count = 0

num_children = []
num_visits = []

q_to_target = []
q_to_nontarget = []
for d in jsons:


    if d["next_action"]["target"] is None:
        null_count += 1
        continue

    # Count action dist.
    if d["next_action"]["target"] == 0:    
        next_action_target += 1
    # Count num children
    num_children = len(d["list_action_child_data"])
    # Num visit dist.
    for child in d["list_action_child_data"]:
        num_visits.append(child["num_visits"])

    # Avg and std of Q?
    for child in d["list_action_child_data"]:
        if child["action"]["target"] == 0:
            q_to_target.append(child["q_value"])
        else:
            q_to_nontarget.append(child["q_value"])        



print(f"num null exp... {null_count}")
print(f"num valid experiments: {len(jsons)-null_count}")
print(f"num target pick: {next_action_target} ({float(next_action_target)/(len(jsons)-null_count)*100:.2f}%)")
print(f"average branching factor: {np.average(num_children)} ({np.std(num_children)})")
print(f"average num visits: {np.average(num_visits)} ({np.std(num_visits)})")


print(f"q_to_target: {np.average(q_to_target)} ({np.std(q_to_target)})")
print(f"q_to_nontarget: {np.average(q_to_nontarget)} ({np.std(q_to_nontarget)})")
