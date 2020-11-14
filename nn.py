import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()    
from IPython.display import display
from IPython.display import Image


def double_linear_model(data):
    tf.reset_default_graph()

    x1 = tf.placeholder("float", name="x1")
    x2 = tf.placeholder("float", name="x2")
    y = tf.placeholder("float", name="y")

    w11 = tf.Variable(0.0, name="weight1_model1")
    w21 = tf.Variable(0.0, name="weight2_model1")
    b1 = tf.Variable(0.0, name="bias_model1")
    
    w12 = tf.Variable(0.0, name="weight1_model2")
    w22 = tf.Variable(0.0, name="weight2_model2")
    b2 = tf.Variable(0.0, name="bias_model2")

    model1 = tf.sigmoid(tf.add(tf.add(tf.multiply(x1, w11), tf.multiply(x2, w21)), b1))
    model2 = tf.sigmoid(tf.add(tf.add(tf.multiply(x1, w12), tf.multiply(x2, w22)), b2))
    
    combined = tf.multiply(model1, 2.) + model2

    loss = tf.losses.mean_squared_error(y, combined)

    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(loss)

    init = tf.global_variables_initializer()

    training_epochs = 40000
    train_n_samples = data.shape[0]
    display_step = 10000

    mini_batch_size = 100
    n_batch = train_n_samples // mini_batch_size + (train_n_samples % mini_batch_size != 0)
    
    for i in range(data.shape[0]):
        plt.plot(data[i,0], data[i,1],'o', c=cmap[np.int(data[i,2])])
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            i_batch = (epoch % n_batch)*mini_batch_size
            batch = data[i_batch:i_batch+mini_batch_size, 0], data[i_batch:i_batch+mini_batch_size, 1], data[i_batch:i_batch+mini_batch_size, 2]
            sess.run(optimizer, feed_dict={x1: batch[0], x2: batch[1], y: batch[2]})
            if (epoch+1) % display_step == 0:
                malda1, c1 = -sess.run(w11)/sess.run(w21), -sess.run(b1)/sess.run(w21)
                malda2, c2 = -sess.run(w12)/sess.run(w22), -sess.run(b2)/sess.run(w22)
                plt.plot([0, 4], [c1,c1+malda1*4], label="Model1, Epoch " + str(epoch+1))
                plt.plot([0, 4], [c2,c2+malda2*4], label="Model2, Epoch " + str(epoch+1))
                cost = sess.run(loss, feed_dict={x1: data[:, 0], x2: data[:, 1], y: data[:, 2]})
                print("Epoch:", str(epoch+1), "Error:", np.mean(cost))
        params = [sess.run(w11), sess.run(w21), sess.run(b1), sess.run(w12), sess.run(w22), sess.run(b2)]
    plt.legend()
    plt.ylim((-3.2,3.2))
    plt.xlim((-0.2,4.2))
    return params