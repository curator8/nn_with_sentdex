# batch layers and objects.
# batch refers the amount of data that's going into an neural network

# the bigger the batch the more 
# parallel operations we can run
# A GPU is gonna have 100s or 1000s 
# cores to run calculations on 

# inputs are values/features we're trying to analyze 

# we can show algorithm multiple at 
# time then we can generalize it 

import numpy as np 
import nnfs 
from nnfs.datasets import spiral_data 
nnfs.init()

# 100 feature sets of 3 classes 
X, y = spiral_data(100, 3) 

#this is a new comment

inputs = [[1,2,3,2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# this would be tedious after a while 
# weights = [[0.2, 0.8, -0.5, 1.0 ],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]



# biases = [2,3, 0.5]



# weights2 = [[0.1, 0.8, -0.5, 1.0 ],
#            [0.5, -0.21, 0.26, -0.5],
#            [-0.26, -0.27, 0.47, 0.87]]



# biases2 = [2,-3, 0.5]

# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases 
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# print(layer2_outputs) 

# matrix multiplication 

# X = [[1,2,3,2.5],
#       [2.0, 5.0, -1.0, 2.0],
#       [-1.5, 2.7, 3.3, -0.8]]

# X is convention for input data to Neural Network 

class Layer_Dense: 
  def __init__(self, n_inputs, n_neurons): 
    # size of input coming in. 
    # parameters are the shape
    # we don't have to do transpose here 
    self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons)) # first parameter is shape 
    
  def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases 



class Activation_ReLU:
  def forward(self, inputs):
    self.output = np.maximum(0, inputs)



# n_inputs how many features in each sample 
# neurons...anything we want 


# 4 is number of inputs and 5 number of neurons 
layer1 = Layer_Dense(2,5)
activation1 = Activation_ReLU()
layer1.forward(X)
# will print negatives 
print(layer1.output)
# will make negative to 0 
activation1.forward(layer1.output)
print(activation1.output)


# notes 

# output of layer 1 be the input of layer2
# print(layer1.output)
# print(0.10 * np.random.randn(4,3))
# how do we intialize a layer 
# trained model and load the model is load weights and biases 
# with new model we intialize weights and biases 
# initialize weight as random values 
# we want small values in the range of -1 and 1 
# normalize and scale dataset so value are -1 and 1 
# weights good starting point -.01 to .01 
# initialize biases other than 0 otherwise might lead to dead network 

# three examples of 2 neurons we have. 