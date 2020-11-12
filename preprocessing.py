import pandas as pd
import numpy as np
import os
import csv

#data = pd.read_csv('./Data/control/C_0002.txt')

#print(data.shape)


data = os.listdir("Data/control")

from time import time
t1 = time()
cont = 0
for file in data:
        f = open(os.path.join("Data/control",file),'r')
        lines = f.readlines()
        cont += len(lines)
        f.close()
        f = open(os.path.join("Data/control",file),'w+')
        for line in lines:
            line = str( line[:len(line)-1] + ";0\n")
            #print(line)
            f.write(line)
        f.close()
print(cont)
'''
data = os.listdir("Data/parkinson")


for file in data:
        f = open(os.path.join("Data/parkinson",file),'r')
        lines = f.readlines()
        f.close()
        f = open(os.path.join("Data/parkinson",file),'w+')
        for line in lines:
            line = str( line[:len(line)-1] + ";1\n")
            #print(line)
            f.write(line)
        f.close()
'''
