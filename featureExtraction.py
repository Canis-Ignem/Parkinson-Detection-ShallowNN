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
    features.append(np.argmax( df.values[:,0]) - np.argmin( df.values[:,0]))
    features.append(np.argmax( df.values[:,1]) - np.argmin( df.values[:,1]))
    features.append(np.argmax( df.values[:,2]) - np.argmin( df.values[:,2]))
    features.append(np.argmax( df.values[:,3]) - np.argmin( df.values[:,3]))
    features.append(np.argmax( df.values[:,4]) - np.argmin( df.values[:,4]))
    features.append(df.values[0][6])
    return features
    

def getControlFeatures():
    control_data = []
    control = getFiles(control_path)
    for i in control:
        new_df = splitByTest(i)
        for k in range(new_df.shape[0]):
            last_df = np.array_split(i,700)
            for j in last_df:
                control_data.append(extractFeatures(j))
    return pd.DataFrame(control_data)

def getParkinsonFeatures():
    parkinson_data = []
    parkinson = getFiles(parkinson_path)
    for i in parkinson:
        new_df = splitByTest(i)
        for k in range(new_df.shape[0]):
            last_df = np.array_split(i,400)
            for j in last_df:
                parkinson_data.append(extractFeatures(j))
    return pd.DataFrame(parkinson_data)

def getClasses(control, parkinson):
    classes = []
    for i in range(control.shape[0]):
        classes.append(0)
    for i in range(parkinson.shape[0]):
        classes.append(1)
    res = np.array(classes)

    return res

def splitByTest(df):
    data = []
    test = df.values[0][6]
    split = []
    cont = 0
    for i in range(df.shape[0]):
        if df.values[i][6] == test:
            split.append(df.values[i])
        else:
            aux = pd.DataFrame(split)
            data.append(aux)
            split = []
            test = df.values[i][6]
            split.append(df.values[i])
    data.append(split)
    return pd.DataFrame(data)

