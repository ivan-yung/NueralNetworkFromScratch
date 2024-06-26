import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

#initialize weights and biases
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #weights initilized randomly as nonzero values
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #biases initialized as vector of 0's
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        #forward is the dot product
        self.output = np.dot(inputs, self.weights) + self.biases
        
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dvalues = np.dot(dvalues, self.weights.T)
        
    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases


class Activation_ReLU:
    def forward(self, inputs):
        #returns input if its greater than 0
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()  # Copy dvalues to avoid overwriting original array
        self.dvalues[self.inputs <= 0] = 0  # Derivative of ReLU

#softmax is used on ouput layer to determine
#the probability of a guess/ outcome
class Activation_Softmax:
    def forward(self, inputs):
        #exponentiate list, then
        #subtract max from inputs to prevent overflow from exp values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        #normalize values with formula, value/sum of all values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        
    def backward(self, dvalues):
        self.dvalues = dvalues.copy()  # Copy dvalues to avoid overwriting original array

    # No need to implement update method for Activation layers
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
        
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dvalues = self.calculate_dvalues(dvalues, y_true) / samples


    # Placeholder method for calculating dvalues
    def calculate_dvalues(self, dvalues, y_true):
        raise NotImplementedError("Subclasses must implement calculate_dvalues method")
    

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        #check if inputted are one hot encoded or scalars
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis = 1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    
    def calculate_dvalues(self, dvalues, y_true):
        # Convert class labels to one-hot encoded vectors if they're not already
        if len(y_true.shape) == 1:
            y_true = np.eye(dvalues.shape[1])[y_true]

        # Calculate gradient
        self.dvalues = -(y_true - dvalues)
        return self.dvalues




X,y = spiral_data(samples=100, classes = 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

print("Loss:", loss)
