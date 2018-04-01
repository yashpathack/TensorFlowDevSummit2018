import tensorflow as tf
import numpy as np

xTrain = np.array([[1,0,1],
                   [0,0,0],
                   [1,1,0],
                   [1,0,0],
                   [0,0,1]])
                   
yTrain = np.array([
                  [0,1],
                  [1,0],
                  [1,0],
                  [1,0],
                  [1,0]])

xTest = np.array([[0,1,1],
                  [1,1,1],
                  [0,1,0]])

yTest = np.array([[1,0],
                  [0,1],
                  [1,0]])

x = tf.placeholder(dtype=tf.float32,shape=[None,3])
y = tf.placeholder(dtype=tf.float32,shape=[None,2])

# Placeholder is just like a promise for the declaration of a datatype

w0 = tf.Variable(initial_value=tf.truncated_normal([3,2],stddev=0.5))
b0 = tf.Variable(tf.zeroes([2]))
