
import math
#categorical cross entropy calculates the loss/ inacuracy against a one hot vector
#<1,0,0> where 1 represents the classification the object should be
#if we get a softmax output, and compare those probabilies to a target output

#exvalues
softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0,0 ]

#calculate loss via categorical cross entropy (summation of: log(soft_output * onehot_vector[target_output]))
loss = -(math.log(softmax_output[0] * target_output[0] + softmax_output[1] * target_output[1]
        + softmax_output[2] * target_output[2]))

'''
#implimenting loss
import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2], [0.1, 0.5, 0.4], [0.02, 0.9, 0.08]])

class_targets = [0,1,1]

print(softmax_outputs[[0, 1, 2], class_targets])
'''