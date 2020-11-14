import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from IPython.display import display
from IPython.display import Image
import featureExtraction as fe
from sklearn.preprocessing import OneHotEncoder

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

data = fe.getControlFeatures()


park = fe.getParkinsonFeatures()


all_data = np.concatenate((data,park),axis= 0)
'''
ohe = OneHotEncoder()

n_classes = np.unique(y_mnist_tr).shape[0]
y_mnist_tr_oh = np.eye(n_classes)[y_mnist_tr]
y_mnist_ts_oh = np.eye(n_classes)[y_mnist_ts]
'''
print(all_data.shape)