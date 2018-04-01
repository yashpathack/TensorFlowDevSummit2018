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
