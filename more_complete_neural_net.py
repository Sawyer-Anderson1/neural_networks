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
    def gradient_descent(self, layer_idx, gradient_mat, bias_gradient_mat):
        # calculate changes in weights and biases
        delta_weight = (self.momentum * self.last_weight_change[layer_idx]) - (self.learning_rate * gradient_mat)

        delta_bias = (self.momentum * self.last_bias_change[layer_idx]) - (self.learning_rate * bias_gradient_mat)

        # update weights
        self.weights[layer_idx] += delta_weight
        self.biases[layer_idx] += delta_bias

        # keep track of the latest changes to weights and biases
        self.last_weight_change[layer_idx] = 1 * delta_weight
        self.last_bias_change[layer_idx] = 1 * delta_bias

    # will have stochastic gradient descent, adam, etc.
    def stochastic_gradient_descent(self, 
                                    learning_rate, 
                                    momentum, 
                                    node_net_inputs, 
                                    true_outputs, 
                                    node_outputs, 
                                    activation_function, 
                                    loss_function, 
                                    weights, 
                                    biases, 
                                    last_weight_change, 
                                    last_bias_change):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weights = weights
        self.biases = biases

        self.last_weight_change = last_weight_change
        self.last_bias_change = last_bias_change

        for iter in range(len(node_outputs) + 1):
            # node_outputs only has the values for the hidden nodes and the output nodes (so do back_index - 1)
            back_index = len(node_outputs) - iter - 1
            
            print(node_outputs[back_index])
            if iter == 0: # the final/output layer
                d_activation = getattr(ActivationFunction(), activation_function)(node_net_inputs[back_index], deriv=True)
                d_error = getattr(LossFunction(), loss_function)(true_outputs, node_outputs[back_index], deriv=True)
                delta = d_error * d_activation

                gradient_mat = np.dot(node_outputs[back_index].T, delta)
                bias_gradient_mat = 1 * delta

                # apply gradient descent
                self.gradient_descent(back_index, gradient_mat, bias_gradient_mat)

            else: # the hidden layers
                W_trans = weights[back_index + 1].T
                d_activation = getattr(ActivationFunction(), activation_function)(node_net_inputs[back_index], deriv=True)
                d_error = getattr(LossFunction(), loss_function)(delta, W_trans)
                delta = d_error * d_activation

                gradient_mat = np.dot(node_outputs[back_index].T, delta)
                bias_gradient_mat = 1 * delta

                # apply gradient descent
                self.gradient_descent(back_index, gradient_mat, bias_gradient_mat)

# Loss function(s) class: for regression has MAE, MSE and for categoricalization has Cross-entropy and its types, Hinge
class LossFunction:
    def mean_absolute_error(self, y_true, y_pred, deriv=False):
        return (np.abs(y_true - y_pred)).mean()

    def mean_squared_error(self, y_true, y_pred, deriv=False):
        if deriv == True:
            N = y_true.shape[0]
            return -2*(y_true - y_pred) / N
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
        weighted_inputs = np.dot(inputs, self.weights) + self.bias
        return weighted_inputs, getattr(ActivationFunction(), activation_function)(weighted_inputs)

# NeuronLayer class: capable of making a layer in a neural network by taking a node_nums int that specifies how much nodes are in the layer.
# Also takes the weights for that layer (so the weights coming from past nodes), and the biases at the layer.
# And is able to do forward propagation or feedforward by taking the inputs (from the previous layer, except for the input layer).
class NeuralLayer:
    def __init__(self, node_nums, weights_at_layer, biases_at_layer):
        self.nodes = []

        for iter in range(node_nums):
            node_name = "node_" + str(iter)
            weights_to_node = weights_at_layer[:, iter].T

            self.node_name = Neuron(weights_to_node, biases_at_layer[iter])
            self.nodes.append(self.node_name)

    def  feedforward(self, inputs, activation_function):
        outputs = []
        net_inputs = []

        for node in self.nodes:
            net_input, output = node.forward_propagation(inputs, activation_function)
            net_inputs.append(net_input)
            outputs.append(output)

        return net_inputs, outputs

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

        self.last_weight_change = [np.zeros(mat.shape) for mat in self.weights]
        self.last_bias_change = [np.zeros(mat.shape) for mat in self.biases]

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
    
    def forward_propagation(self, layer_inputs):
        layers = []
        node_outputs = []
        node_net_inputs = []
        
        # we start the iteration from the first hidden layer instead of the input layer 
        for iter in range(1, self.num_layers):
            layer_name = "layer_" + str(iter)
            weights_at_layer = self.weights[iter - 1] # takes the weights that go from the previous layer to the current layer
        
            self.layer_name = NeuralLayer(self.num_nodes[iter], weights_at_layer, self.biases[iter])
            net_inputs, outputs = self.layer_name.feedforward(layer_inputs, self.activation_function)

            layers.append(self.layer_name)
            node_outputs.append(np.array(outputs))
                
            # for the inner/hidden layers the outputs become the inputs for the next layer
            if iter < self.num_layers - 1:
                # the outputs become inputs for the next nodes
                layer_inputs = outputs
            # else we have the final outputs for this pass of forward propagation

            node_net_inputs.append(np.array(net_inputs))
        return layers, node_net_inputs, node_outputs
    
    def back_propagation(self, true_outputs, node_outputs):
        # call the specific back propagation algorithm
        getattr(OptimizationAlgorithms(), self.optimizer_algo)(self.learning_rate, 
                                                               self.momentum, 
                                                               self.node_net_inputs, 
                                                               true_outputs, 
                                                               node_outputs, 
                                                               self.activation_function, 
                                                               self.loss_function, 
                                                               self.weights, 
                                                               self.biases,
                                                               self.last_weight_change,
                                                               self.last_bias_change)

    def train_compile(self, 
                      inputs, 
                      true_outputs, 
                      activation_function, 
                      loss_function, 
                      optimizer_algo, 
                      learning_rate = 0.01, 
                      momentum = 0.1, 
                      epochs=100):
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.optimizer_algo = optimizer_algo
        self.learning_rate = learning_rate
        self.momentum = momentum

        # convert the inputs to an numpy array
        inputs = np.array(inputs)
        true_outputs = np.array(true_outputs)

        # do n amount of forward and back propagation based off the value of epochs
        for epoch in range(epochs):
            # create an array of the layers and all the outputs that were calculated at each node
            layer_inputs = inputs

            # call forward propagation
            layers, node_net_inputs, node_outputs = self.forward_propagation(layer_inputs)
            self.node_net_inputs = node_net_inputs
            self.layers = layers

            # then do back propagation
            self.back_propagation(true_outputs, node_outputs)

            # calculate the total error 
            total_error = getattr(LossFunction(), loss_function)(true_outputs, node_outputs[-1])

network = NeuralNetwork([3, 6, 2, 2])
network.train_compile([0, 1, 0], [0, 1], "sigmoid", "mean_squared_error", "stochastic_gradient_descent", 0.3, 0.1, 1)