# a basic neural network with 4 input and 3 output with python list code 

input = [1, 2, 3, 2.5]

weights1 = [.2, .8, -.5, 1.0]
weights2 = [0.5, -.91, 0.26, -0.5]
weights3 = [-.26, -.27, .17, .87]

bias1 = 2
bias2 = 3
bias3 = 0.5

output = [input[0] * weights1[0]  + input[1] * weights1[1] + input[2] * weights1[2] + input[3] * weights1[3] + bias1,
          input[0] * weights2[0]  + input[1] * weights2[1] + input[2] * weights2[2] + input[3] * weights2[3] + bias2,
          input[0] * weights3[0]  + input[1] * weights3[1] + input[2] * weights3[2] + input[3] * weights3[3] + bias3]  
print(output)


# simplification of this code using data structures and loops 


inputs = [1, 2, 3, 2.5]

weights = [[.2, .8, -.5, 1.0],
            [0.5, -.91, 0.26, -0.5],
            [-.26, -.27, .17, .87]]

biases = [2, 3, .5]

# gonna use zip 
layer_output = []
for neuron_weights, neuron_bias in zip(weights, biases): 
    neuron_output = 0 
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)


print(layer_output)

# simplication using numpy 

import numpy as np 

# we use dot product to multiple vectors with matrix 
# the dot product of two vectors results to a scalar...

input = [1, 2, 3, 2.5]

weights1 = [.2, .8, -.5, 1.0]

bias1 = 2

output = np.dot(input, weights1) + bias1

print(output) 


# dot product with a layer of neurons

inputs = [1, 2, 3, 2.5]

# three neuron with weights
weights = [[.2, .8, -.5, 1.0],
            [0.5, -.91, 0.26, -0.5],
            [-.26, -.27, .17, .87]]
 
biases = [2, 3, .5]


output = np.dot(weights, inputs) + biases 

print(output)

# our input will be batch eventually 