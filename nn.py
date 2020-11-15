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


'''
def MLP(data, classes, test_data, test_classes):
    tf.reset_default_graph()
    
    # One placeholder for all features (as a matrix) and another for the prediction
    x = tf.placeholder(tf.float32, [None, 7])
    y = tf.placeholder(tf.float32, [None, 2])
    
    # First layer: one matrix of weights (on top of the known features) and a vector of biases
    w1 = tf.Variable(tf.zeros([7, 4]), name="layer1_weights")
    b1 = tf.Variable(tf.zeros([4]), name="layer1_biases")
    
    # Second layer: one matrix of weights (on top of the abstract features) and a vector of biases
    w2 = tf.Variable(tf.zeros([4, 2]), name="layer2_weights")
    b2 = tf.Variable(tf.zeros([2]), name="layer2_biases")
    
    linear_model = tf.matmul(x, w1) + b1
    # First layer and probability prediction
    layer1 = tf.sigmoid(linear_model)
    
    linear_model2 = tf.matmul(layer1,w2) + b2
    prediction = tf.add(tf.matmul(layer1,w2),b2)

    # Loss and class prediction
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
    class_pred = tf.argmax(prediction, axis =1)

    # Optimizer & initialization
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)
    init = tf.global_variables_initializer()

    training_epochs = 10000
    train_n_samples = data.shape[0]
    display_step = 200

    mini_batch_size = 100
    n_batch = train_n_samples // mini_batch_size + (train_n_samples % mini_batch_size != 0)
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            i_batch = (epoch % n_batch)*mini_batch_size
            batch = data[i_batch:i_batch+mini_batch_size,:], classes[i_batch:i_batch+mini_batch_size, :]
            sess.run(optimizer, feed_dict={x: batch[0], y: batch[1]})
            if (epoch+1) % display_step == 0:
                cost = sess.run(loss, feed_dict={x: batch[0], y: batch[1]})
                acc = np.sum(np.argmax(test_classes, axis=1) == sess.run(class_pred, feed_dict={x: test_data}))/test_classes.shape[0]
                print("Epoch:", str(epoch+1), "Error:", np.mean(cost), "Accuracy:", acc)
        parameters = sess.run([w1, b1, w2, b2])
        test_predictions = sess.run(class_pred, feed_dict={x: test_data})
    return parameters, test_predictions
'''
def custom_MLP(data, classes, test_data, test_classes, layer_sizes):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 5])
    y = tf.placeholder(tf.float32, [None, 2])
    # At the end of the loop, this variable should contain all the weight variables
    weights = []
    # At the end of the loop, this variable should contain all the biases variables
    biases = []
    # At the end of the loop, this variable should contain all the layers variables (the first one is the data itself)
    layers = [x]
    
    # We advise the usage of an auxiliary variable that contains the number of neurons in the last layer
    # It should initialized as the number of features in the data
    last_layer = 5
    #tf.matmul(x, w1) + b1
    # For giving names to the variables, you can use something like name="layer" + str(len(layers)) + "_{biases|weights}"
    for layer, neurons in enumerate(layer_sizes):  # For each layer specified in the list
        # "+": Concatenation between two lists
        weights = weights + [tf.Variable(tf.zeros([last_layer, neurons]),name="layer" + str(len(layers)) + "_weights")]
                       
        biases = biases + [tf.Variable(tf.zeros([neurons]),name="layer" + str(len(layers)) + "_biases")]
                       
        layers += [tf.sigmoid(tf.add(tf.matmul(layers[len(layers)-1],weights[len(weights)-1]),biases[len(biases)-1]))]
        # Update the number of neurons in the last layer
        last_layer = neurons
        
    # Once we have built the DNN structure, we create the last layer, where the results will be collected
    weights = weights + [tf.Variable(tf.zeros([last_layer, 2]),name="last_layer_weights")]
    biases = biases +[tf.Variable(tf.zeros([2]),name="last_layer_biases")]                                                 
    prediction = tf.add(tf.matmul(layers[len(layers)-1],weights[len(weights)-1]),biases[len(biases)-1])

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
    bce = tf.keras.losses.BinaryCrossentropy()
    #loss = bce(y,prediction)
    class_pred = tf.argmax(prediction, axis=1)

    #optimizer = tf.train.AdamOptimizer(learning_rate=.00001).minimize(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(loss)

    init = tf.global_variables_initializer()

    training_epochs = 24000
    train_n_samples = data.shape[0]
    display_step = 2000

    mini_batch_size = 5
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
target = fe.getClasses()

'''
train_data = all_data[0::2]
train_target = target[0::2]

test_data = all_data[1::2]
test_target = target[1::2]
'''

train_data, test_data, train_target, test_target = train_test_split(all_data,target, test_size= 0.2, random_state=30 )

ohe = OneHotEncoder()

n_classes = np.unique(train_target).shape[0]
train_target = np.eye(n_classes)[train_target]
test_target = np.eye(n_classes)[test_target]

'''
print(train_data.shape)
print(train_target.shape)
print(test_data.shape)
print(test_target.shape)
'''
#print(test_preds)
#print(test_target)
#print(confusion_matrix(test_preds,test_target[:][1]))

#print(train_data[:,3])

#params, test_preds = MLP(train_data, train_target, test_data, test_target)
layers = np.array([3])
params, test_preds, _  = custom_MLP(train_data, train_target, test_data, test_target, layers)
#print(test_preds)

cont0 = 0
cont1 = 0
for i in test_preds:
    if i != 1:
        cont0 += 1
    else:
        cont1 += 1
        
print(confusion_matrix(test_preds, np.argmax(test_target, axis=1)))
print("0: " ,cont0)
print("1: " , cont1)