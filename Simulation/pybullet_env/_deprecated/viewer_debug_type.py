import pickle
import os


with open("/home/sanghyeon/vessl/sim_dataset/data_2.6_23:36:36.382_0_74.pickle", "rb") as f:
    data = pickle.load(f)


print(data)