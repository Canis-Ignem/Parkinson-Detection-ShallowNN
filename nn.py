import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from IPython.display import display
from IPython.display import Image
import featureExtraction as fe
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold



def custom_MLP(data, classes, test_data, test_classes,input_size, layer_sizes):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, 2])
    weights = []
    biases = []
    layers = [x]
    
    last_layer = input_size
    for layer, neurons in enumerate(layer_sizes):  
        weights = weights + [tf.Variable(tf.zeros([last_layer, neurons]),name="layer" + str(len(layers)) + "_weights")]
                       
        biases = biases + [tf.Variable(tf.zeros([neurons]),name="layer" + str(len(layers)) + "_biases")]
                       
        layers += [tf.sigmoid(tf.add(tf.matmul(layers[len(layers)-1],weights[len(weights)-1]),biases[len(biases)-1]))]
        last_layer = neurons
        
    weights = weights + [tf.Variable(tf.zeros([last_layer, 2]),name="last_layer_weights")]
    biases = biases +[tf.Variable(tf.zeros([2]),name="last_layer_biases")]                                                 
    prediction = tf.add(tf.matmul(layers[len(layers)-1],weights[len(weights)-1]),biases[len(biases)-1])

    #loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=prediction)
    class_pred = tf.argmax(prediction, axis=1)

    optimizer = tf.train.AdamOptimizer(learning_rate=.0005).minimize(loss)

    init = tf.global_variables_initializer()

    training_epochs = 1000 *16
    train_n_samples = data.shape[0]
    display_step = 2000

    mini_batch_size = 250
    n_batch = train_n_samples // mini_batch_size + (train_n_samples % mini_batch_size != 0)
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            i_batch = (epoch % n_batch)*mini_batch_size
            batch = data[i_batch:i_batch+mini_batch_size, :], classes[i_batch:i_batch+mini_batch_size, :]
            sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
            if (epoch+1) % display_step == 0:
                cost = sess.run(loss, feed_dict={x: batch[0], y: batch[1]})
                acc = np.sum(np.argmax(test_classes, axis=1) == sess.run(class_pred, feed_dict={x: test_data}))/test_classes.shape[0]
                print("Epoch:", str(epoch+1), "Error:", np.mean(cost), "Accuracy:", acc)
        parameters = sess.run(weights+biases)
        test_predictions = sess.run(class_pred, feed_dict={x: test_data})
    return parameters, test_predictions, acc



data = fe.getControlFeatures()
park = fe.getParkinsonFeatures()


all_data = np.concatenate((data,park),axis= 0)
target = fe.getClasses(data,park)

'''
skf = StratifiedKFold(n_splits=50,shuffle=True)

for train_index, test_index in skf.split(all_data, target):
    train_data, test_data = all_data[train_index], all_data[test_index]
    train_target, test_target = target[train_index], target[test_index]
'''
train_data, test_data, train_target, test_target = train_test_split(all_data,target, test_size= 0.3, random_state=30 )

n_classes = np.unique(train_target).shape[0]
train_target = np.eye(n_classes)[train_target]
test_target = np.eye(n_classes)[test_target]


input_size = np.shape(train_data)[1]
print(input_size)
#params, test_preds = MLP(train_data, train_target, test_data, test_target)
layers = np.array([7])
params, test_preds, _  = custom_MLP(train_data, train_target, test_data, test_target,input_size, layers)
#print(test_preds)

cont1 =np.count_nonzero(test_preds)
cont0 = len(test_preds) - cont1
print(confusion_matrix(test_preds, np.argmax(test_target, axis=1)))
print("0: " ,cont0)
print("1: " , cont1)

#16, 7 , .0005, 0.849
#18, 6 , .0005, 0.843

