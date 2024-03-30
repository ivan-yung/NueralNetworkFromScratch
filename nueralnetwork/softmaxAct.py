import numpy as np
import nnfs
nnfs.innit()

#given a vector of layer_outputs from prev neurons, softmax
#activation returns a vector of probability scores as a method
#to classify objects

#when given input, we exponentiate the input values, then normalize(value divided by total sum), which becomes output

layer_outputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]


exp_values = np.exp(layer_outputs)

print(np.sum(layer_outputs, axis=1, keepdims=True))

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

print(norm_values)