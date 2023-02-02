# batch layers and objects.

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
import math 
import numpy as np

# 100 feature sets of 3 classes 
X, y = spiral_data(100, 3) 


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

# X = [[1,2,3, 2.5],
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

# output of layer 1 be the input of layer2
   


# print(layer1.output)

# print(0.10 * np.random.randn(4,3))

# how do we intialize a layer 
# trained model and load the model is load weights and biases 
# with new model we intialize weights and biases 
# initialize weight as random values 
# we want small values in the range   of -1 and 1 
# normalize and scale dataset so value are -1 and 1 
# weights good starting point -.01 to .01 
# initialize biases other than 0 otherwise might lead to dead network 

# three examples of 2 neurons we have. 



layer_outputs = [4.8, 1.21, 2.385]

# layer_outputs = [4.8, 4.79, 4.25]

# there are cases when we want -1 numbers so reLu activation function
# won't work here...

# so we rely another function called exponential function
# where we maintain the meaning of the negative number 

# we need to exponentiate these values 

E = math.e 
# Euler = 2.7....

#exp_values = []
exp_values = np.exp(layer_outputs)

# for output in layer_outputs: 
#   exp_values.append(E**output) 

# print(exp_values) 

# we then can get the probabilty distribution by dividing 
# by the sum of of each value
# we exponentiate to not loose meaning of negative values 
# we then normalize the values 

# norm_base = sum(exp_values) 
# norm_values = []
 
# for value in exp_values: 
#    norm_values.append(value / norm_base) 

norm_values = exp_values / np.sum(exp_values)

print(norm_values) 
print(sum(norm_values))  




# to summarize 
# input -> softmax (exponentiate then normalize) -> output 

# now we're trying to convert code to work with batch inputs/outputs



# softmax is used during the last layer to output a distribution 
# meanswhile reLu is used in hidden layers to avoid 
# vanishing gradient problem 









