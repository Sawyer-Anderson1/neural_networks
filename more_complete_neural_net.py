import numpy as np

# activation function(s) class: has step function, sigmoid, ReLu, Softmax
class ActivationFunction:
    # step activation function, so any values (weighted inputs) greater than 0 return 1, else 0
    def step_function(self, x):
        # if x is array-like, behave elementwise and return numpy array
        if np.isscalar(x) or getattr(x, "shape", ()) == ():
            return 1 if x > 0 else 0
        x = np.asarray(x)
        return (x > 0).astype(int)
    
    # sigmoid activation function
    def sigmoid(self, x, deriv=False):
        # have to implement a stable version of this function for very large negative and positive values
        if np.isscalar(x) or getattr(x, "shape", ()) == ():
            x = float(x)
            if x >= 0:
                out = 1.0 / (1.0 + np.exp(-x))
            else:
                ex = np.exp(x)
                out = ex / (1.0 + ex)
            return out * (1.0 - out) if deriv else out

        # array path (elementwise, avoids overflow)
        x = np.asarray(x, dtype=float)
        out = np.empty_like(x)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))   # safe for positive entries
        neg = ~pos
        ex = np.exp(x[neg])
        out[neg] = ex / (1.0 + ex)                 # safe for negative entries

        return out * (1.0 - out) if deriv else out
    
    # ReLu activation function
    def relu(self, x, deriv=False):
        if deriv:
            # derivative of ReLU: 1 for x>0 else 0
            if np.isscalar(x) or getattr(x, "shape", ()) == ():
                return 1 if x > 0 else 0
            x = np.asarray(x)
            return (x > 0).astype(float)
        
        # forward
        if np.isscalar(x) or getattr(x, "shape", ()) == ():
            return max(0.0, x)
        x = np.asarray(x)
        return np.maximum(0.0, x)
    
    # softmax activation function
    def softmax(self, x, deriv=False):
        # expects 1D array of logits; subtract max for stability
        x = np.asarray(x, dtype=float)
        x_max = np.max(x)
        exps = np.exp(x - x_max)
        softmax = exps / np.sum(exps)

        if deriv == True:
            softmax.reshape(1, -1)
            return softmax * np.identity(softmax.size) - softmax.transpose() @ softmax
        return softmax

    # tanh activation funciton
    def tanh(self, x, deriv=False):
        y = np.tanh(x)
        
        if deriv == True:
            return 1 - np.sqrt(y)
        return y
    
    # retrieve the thresholds
    def retrieve_threshold(self, activation_function):
        if activation_function == 'sigmoid':
            return 0.5
        if activation_function == 'tanh':
            return 0.0

# Loss function(s) class: for regression has MAE, MSE and for categoricalization has Cross-entropy and its types, Hinge
class LossFunction:
    def mean_absolute_error(self, y_true, y_pred, deriv=False):
        if deriv == True:
            if y_pred > y_true:
                return 1
            elif y_pred < y_true:
                return -1
            else:
                return 0
            
        return (np.abs(y_true - y_pred)).mean()

    def mean_squared_error(self, y_true, y_pred, deriv=False):
        if deriv == True:
            N = y_true.shape[0]
            return -2*(y_true - y_pred) / N
        return ((y_true - y_pred) ** 2).mean()

    def categorical_ce(self, y_true, y_pred, deriv=False):
        if deriv == True:
            return y_pred - y_true

        cross_entropy_loss = 0

        for iter in range(len(y_true)):
            cross_entropy_loss += y_true[iter] * np.log(y_pred[iter])

        return -(cross_entropy_loss)

    def binary_ce(self, y_true, y_pred, deriv=False):
        if deriv == True:
            return (y_pred - y_true) / (y_pred * (1 - y_pred))
        
        if y_true == 1:
            return -y_true * np.log(y_pred)
        else:
            return -(1 - y_true) * np.log(1 - y_pred)
    
    def hinge(self, y_true, y_pred, deriv=False):
        if deriv == True:
            return -1 * y_true * (y_pred / (y_pred * (1 - y_pred)))
        
        return np.max(0, 1 - y_true * y_pred)

class OptimizationAlgorithms:
    def __init__(self,
                learning_rate, 
                momentum,
                last_weight_change, 
                last_bias_change):
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.last_weight_change = last_weight_change
        self.last_bias_change = last_bias_change

    def gradient_descent(self, layer_idx, gradient_mat, bias_gradient_mat, weights, biases):
        # calculate changes in weights and biases
        delta_weight = (self.momentum * self.last_weight_change[layer_idx]) - (self.learning_rate * gradient_mat)

        delta_bias = (self.momentum * self.last_bias_change[layer_idx + 1]) - (self.learning_rate * bias_gradient_mat.reshape(-1))

        # update weights
        weights[layer_idx] += delta_weight
        biases[layer_idx + 1] += delta_bias

        # keep track of the latest changes to weights and biases
        self.last_weight_change[layer_idx] = 1 * delta_weight
        self.last_bias_change[layer_idx + 1] = 1 * delta_bias
        
        # update the value of momentum
        #new_momentum = 
    
    def stochastic_gradient_descent(self,
                                    node_net_inputs, 
                                    true_outputs, 
                                    node_outputs, 
                                    activation_function, 
                                    loss_function,
                                    weights,
                                    biases,
                                    input_activation = None):
        L = len(node_outputs)

        last_idx = L - 1

        # computing delta for output layer
        d_activation = getattr(ActivationFunction(), activation_function)(node_net_inputs[last_idx], deriv=True)
        d_error = getattr(LossFunction(), loss_function)(true_outputs, node_outputs[last_idx], deriv=True)

        delta = d_error.reshape(-1, 1) * d_activation.reshape(-1, 1)

        for l in range(last_idx, -1, -1):
            if l == 0: # the previous layer is the network's input
                a_prev = input_activation.reshape(-1, 1)
            else: # the hidden layers
                a_prev = node_outputs[l - 1].reshape(-1, 1)

            gradient_mat = a_prev @ delta.T
            bias_gradient_mat = 1 * delta.reshape(-1)

            # apply gradient descent
            self.gradient_descent(l, gradient_mat, bias_gradient_mat, weights, biases)

            if l > 0:
                # prepare delta for next layer
                delta = (weights[l] @ delta).reshape(-1, 1)

                d_act_prev = getattr(ActivationFunction(), activation_function)(node_net_inputs[l - 1], deriv=True)
        
                delta = delta * d_act_prev.reshape(-1, 1)
    
    def adagrad(self):
        print('placeholder')
    
    def rmsprop(self):
        print('placeholder')
    
    def adadelta(self):
        print('placeholder')
    
    def adam(self):
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
    def __init__(self, node_array, activation_function):
        self.num_layers = len(node_array)
        self.num_nodes = node_array
        self.activation_function = activation_function  

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
            node_net_inputs.append(np.array(net_inputs))
            node_outputs.append(np.array(outputs))

            # for the inner/hidden layers the outputs become the inputs for the next layer
            if iter < self.num_layers - 1:
                # the outputs become inputs for the next nodes
                layer_inputs = outputs
            # else we have the final outputs for this pass of forward propagation

        return layers, node_net_inputs, node_outputs
    
    def back_propagation(self, input_id, true_outputs, node_outputs):
        # call the specific back propagation algorithm
        getattr(OptimizationAlgorithms(self.learning_rate, self.momentum, self.last_weight_change, self.last_bias_change), self.optimizer_algo)(self.node_net_inputs, 
                                                                                                                                                true_outputs, 
                                                                                                                                                node_outputs, 
                                                                                                                                                self.activation_function, 
                                                                                                                                                self.loss_function, 
                                                                                                                                                self.weights, 
                                                                                                                                                self.biases,
                                                                                                                                                self.input_activation[input_id])

    def apply_threshold(self, outputs):
        # get threshold for activation function
        threshold = getattr(ActivationFunction(), 'retrieve_threshold')(self.activation_function)

        # then apply threshold
        output_id = 0
        for output_values in outputs:
            for index in range(len(output_values)):
                if output_values[index] >= threshold:
                    output_values[index] = 1
                else:
                    output_values[index] = 0

            outputs[output_id] = output_values
            output_id += 1

        # then return the classified output
        return outputs
    
    def calculate_accuracy(self, true_outputs, outputs):
        # apply threshold to the outputs calculated by the activation function
        outputs = self.apply_threshold(outputs)

        correct_count = 0
        output_id = 0
        for true_output in true_outputs:
            #print(true_output, outputs[output_id])
            if np.array_equal(true_output, outputs[output_id]):
                correct_count += 1
            output_id += 1

        return float(correct_count) / len(true_outputs)

    def compile(self, 
                optimizer_algo, 
                loss_function,
                learning_rate = 0.01, 
                momentum = 0.5):
        self.loss_function = loss_function
        self.optimizer_algo = optimizer_algo
        self.learning_rate = learning_rate
        self.momentum = momentum

    def fit(self,
            train_inputs, 
            train_outputs,
            epochs=100):
        # dictionary to save the history of each epochs
        history = {}

        # convert the inputs to an numpy array
        train_inputs = np.array(train_inputs)
        self.input_activation = train_inputs
        true_outputs = np.array(train_outputs)

        # do n amount of forward and back propagation based off the value of epochs
        epoch_errors = []
        epoch_accuracies = []
        for epoch in range(epochs):
            input_id = 0
            epoch_errors = []
            epoch_final_outputs = []
            for inputs in train_inputs:
                # call forward propagation
                layers, node_net_inputs, node_outputs = self.forward_propagation(inputs)
                self.node_net_inputs = node_net_inputs
                self.layers = layers
                
                # then do back propagation
                self.back_propagation(input_id, true_outputs[input_id], node_outputs)

                # calculating the error for this input in the epoch, which will later be averaged for the error for the whole epoch
                epoch_errors.append(getattr(LossFunction(), self.loss_function)(true_outputs[input_id], node_outputs[-1]))

                # add the final output for an input to epoch_final_outputs
                epoch_final_outputs.append(node_outputs[-1])

                input_id += 1
            # averaging the error/loss of all the errors of the epoch and displaying it
            error_for_epoch = np.array(epoch_errors).mean()
            epoch_errors.append(error_for_epoch)
            print('Training Loss for epoch', epoch, ":", error_for_epoch)

            # calculate accuracy
            epoch_accuracy = self.calculate_accuracy(true_outputs, epoch_final_outputs)
            epoch_accuracies.append(epoch_accuracy)
            print('Training Accuracy for epoch', epoch, ':', epoch_accuracy)

            # have not yet added validation and batching
            log = {'accuracy': epoch_accuracy, 'loss': error_for_epoch, 'val_accuracy': None, 'val_loss': None} 

            # add the log to the history
            history['epoch_' + str(epoch)] = log

        # displaying the final error/loss, which was the error of the final epoch
        print('Final Loss for training:', epoch_errors[-1])

        # displaying the final accuracy of the model
        print('Final accuracy for training:', epoch_accuracies[-1])

        # display final weights
        print("Final weights:", self.weights)

        return history
    
    def evaluate(self, test_inputs, test_outputs):
        model_outputs = []
        test_loss = []
        for input_id in range(len(test_inputs)):
            # do forward propagation
            _, _, node_outputs = self.forward_propagation(test_inputs[input_id])

            # then calculate the error/loss
            test_loss.append(getattr(LossFunction(), self.loss_function)(test_outputs[input_id], node_outputs[-1]))

            model_outputs.append(node_outputs[-1])
        
        # average the error/loss for all the outputs of the test and display as the testing loss
        avg_test_loss = np.array(test_loss).mean()
        print('Testing Loss:', avg_test_loss)

        # calculate the accuracy for the testing
        testing_accuracy = self.calculate_accuracy(test_outputs, model_outputs)
        print('Testing accuracy:', testing_accuracy)

        return avg_test_loss, testing_accuracy
    
    def predict(self, inputs):
        outputs = []
        for input in inputs:
            _, _, node_outputs = self.forward_propagation(input)

            outputs.append(node_outputs[-1])
        
        return outputs

# akin to making the sequential models in tensor
network = NeuralNetwork([3, 6, 3], "sigmoid")

# compiling
network.compile(optimizer_algo = "stochastic_gradient_descent", 
                loss_function = "categorical_ce", 
                learning_rate = 0.3, 
                momentum = 0.5)

# training/fitting
model_history = network.fit(train_inputs = [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 1]],
            train_outputs = [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1]],
            epochs = 200)
print(model_history['epoch_0'])

# testing/evaluating
network.evaluate([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]], [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 0, 1]])

# predicting outputs (so just doing the forward propagation)
predictions = network.predict([[0, 1, 0]])
print(network.apply_threshold(predictions))