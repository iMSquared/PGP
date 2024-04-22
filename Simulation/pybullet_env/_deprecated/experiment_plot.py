import os
import matplotlib.pyplot as plt
import pickle

def main():

    file_list = sorted(os.listdir("Simulation/pybullet_env/exp_fetching"))

    data_list = []
    for file_name in file_list:

        with open(os.path.join("Simulation/pybullet_env//exp_fetching", file_name), "rb") as f:
            data = pickle.load(f)
            data_list.append(data)


    num_particle_lists = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    policy_lists = ["random", "history"]

    record = {}
    for v1 in num_particle_lists:
        record[v1] = {}
        for v2 in policy_lists:
            record[v1][v2] = {
                "num_exec": 0,
                "num_exec_success": 0, }
    

    for data in data_list:

        num_particles = data["num_particles"]
        policy        = data["policy"]
        record[num_particles][policy]["num_exec"] += 1
        if data["termination"] == "success":
            record[num_particles][policy]["num_exec_success"] += 1


    
    
    for policy in policy_lists:
        x = []
        y = []
        for np in num_particle_lists:

            if record[np][policy]["num_exec"] != 0:
                rate = float(record[np][policy]["num_exec_success"]) / float(record[np][policy]["num_exec"])
                x.append(np)
                y.append(rate)

        plt.plot(x, y, label=policy)
    
    plt.xlabel("# Simulations")
    plt.ylabel("Execution success rate")
    plt.legend()

    plt.show()
    

if __name__=="__main__":
    main()