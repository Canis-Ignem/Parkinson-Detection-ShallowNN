import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import featureExtraction as fe

'''
control_path = "Data/control"
parkinson_path = "Data/parkinson"

parkinson = fe.getFiles(parkinson_path)
control = fe.getFiles(control_path)

cont0 =0
for i in range(len(control)):
 cont0 += control[i].shape[0]
#print(cont0)

classes = np.zeros(cont0)

cont1 =0
for i in range(len(parkinson)):
 cont1 += parkinson[i].shape[0]
#print(cont1)

classes1 = np.ones(cont1)

#print(classes.shape)
#print(classes1.shape)
target = np.concatenate((classes,classes1),axis=0)
#print(target.shape)

all_data = np.array(control[0])
#print(all_data.shape)
for i in range(1,len(control)):
    all_data = np.concatenate((all_data,control[i]),axis=0)
for i in parkinson:
    all_data = np.concatenate((all_data,i),axis=0)
#print(all_data.shape)
#all_data = np.concatenate((control,parkinson),axis=1)
#print(all_data.shape)
'''

data = fe.getControlFeatures()
park = fe.getParkinsonFeatures()


all_data = np.concatenate((data,park),axis= 0)
target = fe.getClasses(data,park)

train_data, test_data, train_target, test_target = train_test_split(all_data,target, test_size= 0.3, random_state=30 )



mlp = MLPClassifier(alpha=1e-08, hidden_layer_sizes=(70,10),solver='adam',
                    activation='relu',  learning_rate_init = .0005,
                    max_iter=5000, momentum = 0.7)

rbm = BernoulliRBM( verbose=True)

rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('MLP', mlp)])

rbm.learning_rate = 0.0002
rbm.n_iter = 300

rbm.n_components = 200


rbm_features_classifier.fit(train_data, train_target)

Y_pred = rbm_features_classifier.predict(test_data)

print(confusion_matrix(Y_pred, test_target))
print(accuracy_score(Y_pred, test_target))
