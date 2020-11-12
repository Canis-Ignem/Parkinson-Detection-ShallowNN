import pandas as pd
import numpy as np
import tensorflow as tf
import glob

path = "Data/control"
all_files = glob.glob(path + "/*.txt")

control = []

for file in all_files:
    df = pd.read_csv(file,sep=";")
    control.append(df)
    print(df.shape)


print(np.shape(control))