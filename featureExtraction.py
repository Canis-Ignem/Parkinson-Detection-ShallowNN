import pandas as pd
import numpy as np
import glob

control_path = "Data/control"
parkinson_path = "Data/parkinson"


def getFiles(path):
    all_files = glob.glob(path + "/*.txt")

    data = []

    for file in all_files:
        df = pd.read_csv(file,sep=";")
        data.append(df)
    return data
         

def extractFeatures(df):
    features = []
    
    for i in range(df.shape[1]):
        mean = 0
        for j in range(df.shape[0]):
            mean += df.values[j][i]
        features.append(mean/len(df))
    return features

def getControlFeatures():
    control_data = []
    control = getFiles(control_path)
    for i in control:
        control_data.append(extractFeatures(i))
    return control_data

def getParkinsonFeatures():
    parkinson_data = []
    parkinson = getFiles(parkinson_path)
    for i in parkinson:
        parkinson_data.append(extractFeatures(i))
    return parkinson_data