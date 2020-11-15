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
    
    
    features.append(np.var( df.values[:,0]))
    features.append(np.var( df.values[:,1]))
    features.append(np.var( df.values[:,2]))
    features.append(np.var( df.values[:,3]))
    features.append(np.var( df.values[:,4]))
    #print("AAAAAAAAA")
    #print(len(df))
    #print(len(features))
    return features
    

def getControlFeatures():
    control_data = []
    control = getFiles(control_path)
    for i in control:
        new_df = np.array_split(i,40)
        for j in new_df:
            control_data.append(extractFeatures(j))
    return control_data

def getParkinsonFeatures():
    parkinson_data = []
    parkinson = getFiles(parkinson_path)
    for i in parkinson:
        new_df = np.array_split(i,7)
        for j in new_df:
            parkinson_data.append(extractFeatures(j))
    return parkinson_data

def getClasses():
    classes = []
    for i in range(np.shape(getControlFeatures())[0]):
        classes.append(0)
    for i in range(np.shape(getParkinsonFeatures())[0]):
        classes.append(1)
    #res = np.array(classes)
    return np.array(classes)
'''
a = getClasses()
print(a)

a = getControlFeatures()

print(np.shape(a))

df = getFiles(parkinson_path)
a = extractFeatures(df[0])
print(np.shape(a))

new_df = np.array_split(df[0],100)

print(new_df[0].values[:,1])
'''
