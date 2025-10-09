import numpy as np

# activation function(s) class: has step function, sigmoid, ReLu, Softmax
class ActivationFunction:
    # step activation function, so any values (weighted inputs) greater than 0 return 1, else 0
    def step_function(self, x):
        if x > 0:
            return 1
        else:
            return 0
    
    # sigmoid activation function
    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x*(x-1)
        return 1/(1+np.exp(-x))

    # ReLu activation function
    def relu(self, x):
        return max(0, x)
    
    # softmax activation function
    def softmax(self, x):
        exponentiation_values = np.exp(x)
        normalization_sum = exponentiation_values.sum()

        return exponentiation_values / normalization_sum

class OptimizationAlgorithms:
    # will have stochastic gradient descent, adam, etc.
    def stochastic_gradient_descent(self):
        print('placeholder')

# Loss function(s) class: for regression has MAE, MSE and for categoricalization has Cross-entropy and its types, Hinge
class LossFunction:
    def mean_absolute_error(self, y_true, y_pred):
        return (np.abs(y_true - y_pred)).mean()

    def mean_squared_error(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def cross_entropy(self, cross_entropy_type):
        if cross_entropy_type == 'categorical':
            self.categorical_ce()
        elif cross_entropy_type == 'binary':
            self.binary_ce()

    def categorical_ce(self):
        print('placeholder')

    def binary_ce(self):
        print('placeholder')
    
    def hinge(self):
        print('placeholder')

# Neuron class capable of forward propagation, takes weights, bias, and its inputs
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward_propagation(self, inputs, activation_function):
        weighted_inputs = np.dot(self.weights, inputs) + self.bias
        return getattr(ActivationFunction(), activation_function)(weighted_inputs)

# NeuronLayer class: capable of making a layer in a neural network by taking a node_nums int that specifies how much nodes are in the layer.
# Also takes the weights for that layer (so the weights coming from past nodes), and the biases at the layer.
# And is able to do forward propagation or feedforward by taking the inputs (from the previous layer, except for the input layer).
class NeuralLayer:
    nodes = []

    def __init__(self, node_nums, weights, biases_at_layer):
        for iter in range(node_nums):
            node_name = "node_" + str(iter)
            self.node_name = Neuron(weights, biases_at_layer[iter])
            self.nodes.append(self.node_name)

    def  feedforward(self, inputs, activation_function):
        outputs = []

        for node in self.nodes:
            outputs.append(node.forward_propagation(inputs, activation_function))
        
        return outputs

# NeuralNetwork class: is initiated with an array called node_array whose integer values represent how much nodes per layer (including input layer), 
# to check how much layers there are in the network we get the length of node_array
# In initiation the __init__ method calls two functions that generate randomized weights and biases (generate_weights and generate_biases)
# train_compile method: takes the data and true outputs,
# also takes an activation function, can be called with a learning rate (default to 0.1), and epochs (default to 100)
class NeuralNetwork:
    def __init__(self, node_array):
        self.num_layers = len(node_array)
        self.num_nodes = node_array

        self.generate_weights()
        self.generate_biases()

    def generate_weights(self):
        # seed random for determinism
        np.random.seed(1)

        # will be an array of matrices (so [[[]]]), and will hold all the matrices of all randomized weights for each layer in an array
        randomized_weights = []

        for iter in range(self.num_layers - 1):
            # a matrix or array of the array of weights for each node in a layer
            weights_matrix = 2*np.random.random((self.num_nodes[iter], self.num_nodes[iter + 1])) - 1

            randomized_weights.append(weights_matrix)

        self.weights = randomized_weights 
        
    def generate_biases(self):
        # seed random for determinism
        np.random.seed(1)

        # will be matrix, which will hold the biases that are generated for each layer (for the input layer it will be 0)
        randomized_biases = []

        input_layer_flag = True
        for iter in self.num_nodes:
            if input_layer_flag == True:
                input_biases = np.zeros(iter)
                randomized_biases.append(input_biases)

                input_layer_flag = False
            else:
                layer_biases = np.random.random(iter)
                randomized_biases.append(layer_biases)

        self.biases = randomized_biases

    def train_compile(self, inputs, true_outputs, activation_function, loss_function, optimizer_algo, learning_rate = 0.1, epochs=100):
        # do n amount of forward and back propagation based off the value of epochs
        for epoch in range(epochs):
            # forward propagation
            # we start the iteration from the first hidden layer instead of the input layer so that 
            for iter in range(1, self.num_layers):
                layer_name = "layer_" + str(iter)
                weights_at_layer = self.weights[iter - 1] # takes the weights that go from the previous layer to the current layer

                self.layer_name = NeuralLayer(self.nodes_num[iter], weights_at_layer, self.biases[iter])
                outputs = self.layer_name.feedforward(inputs, self.activation_function)

                # for the inner/hidden layers the outputs become the inputs for the next layer
                if iter < self.num_layers:
                    # the outputs become inputs for the next nodes
                    inputs = outputs
                # else we have the final outputs for this pass of forward propagation

            # then do back propagation (using Stochastic Gradient Descent)
            # calculate the total error 
            total_error = getattr(LossFunction(), loss_function)(true_outputs, outputs)
            # then compute the gradients of the parameters
            # ...


#network = NeuralNetwork([3, 6, 1])
#network.train_compile([0, 1, 0], [0], "sigmoid", mean_squared_error, stochastic_gradient_descent, 0.3, 1000)