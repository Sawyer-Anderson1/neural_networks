import numpy as np

# sigmoid activation function
def nonlin(x, deriv=False):
    if deriv is True:
        return x*(x-1)
    return 1/(1+np.exp(-x))

'''
# input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# output dataset
y = np.array([[0,0,1,1]]).T

# seed random nums for deterministism 
np.random.seed(1)

# initialize weights randomly around 0
weights = 2*np.random.random((3,1)) - 1 # creating a numpy array of 3 rows and one column that have random weights
# can also name this syn0 for synapse 0

for iter in range(10000):
    # forward propagatin
    input_layer = X
    output_layer = nonlin(np.dot(input_layer, weights))

    # backpropagation
    # calculate error
    output_error = y - output_layer

    # multiplying error with the slope/derivative of the sigmoid at the values in the output layer
    output_delta = output_error * nonlin(output_layer, True)

    # update weights
    weights += np.dot(input_layer.T, output_delta)

print('Output after training, with only input and output layers: ', output_layer)
'''

# doing multiple layer nueral network
# input
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# output
y = np.array([[0,1,1,0]]).T

# initialize weights for the layers, the hidden/inner layer has 6 nodes
syn0 = 2*np.random.random((3,6)) - 1
syn1 = 2*np.random.random((6,1)) - 1
print(syn0)

for iter in range(60000):
    # forward propagation
    input_layer = X
    hidden_layer = nonlin(np.dot(input_layer, syn0))
    output_layer = nonlin(np.dot(hidden_layer, syn1))

    # backpropagation
    # calculate error for output layer
    output_error = y - output_layer
    
    # calculate the error weighted derivative
    output_delta = output_error * nonlin(output_layer, True)

    # then calculate the error for the hidden layer based off output_delta, 
    # to find the contributions of each hidden nodes to the output's error
    hidden_layer_error = np.dot(output_delta, syn1.T)

    # calculate the error weighted derivative for the hidden layer
    hidden_layer_delta = hidden_layer_error * nonlin(hidden_layer, True)

    # then update both weights, syn0 and syn1
    syn1 += np.dot(hidden_layer.T, output_delta)
    syn0 += np.dot(input_layer.T, hidden_layer_delta)

print('Output of multi-layer net: ', output_layer)