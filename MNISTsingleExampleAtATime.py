def softmax_activation_function(Z):
    A = np.exp(Z) / np.sum(np.exp(Z))
    return A

def softmax_activation_function_Prime(Z):
    A = softmax_activation_function(Z)
    APrime = np.multiply(A, 1 - A)
    return APrime

def sigmoid_activation_function(Z):
    A = 1 / (1 + np.exp(- 1* Z))
    return A

def sigmoid_activation_function_Prime(Z):
    A = sigmoid_activation_function(Z)
    APrime = np.multiply(A, 1 - A)
    return APrime

class Network(object): #eg. layer_sizes = [3,2,4], initial_... = [2x3 matrix,4x2 matrix]
    def __init__(self,layer_sizes=None,initial_weights=None,initial_biases=None,inputX=None,labelY=None,lastWeights=None,lastBiases=None,lastA=None,lastZ=None):
        self.layer_sizes = layer_sizes
        self.initial_weights = initial_weights
        self.initial_biases = initial_biases
        self.inputX = inputX
        self.labelY = labelY
        self.lastWeights = lastWeights
        self.lastBiases = lastBiases
        self.lastA = lastA
        self.lastZ = lastZ

    def feedforward(self):
        if self.lastA is None:
            self.lastA = [np.zeros([x,1]) for x in self.layer_sizes]
            self.lastA[0] = self.inputX
        if self.lastZ is None:
            self.lastZ = [np.zeros([x,1]) for x in self.layer_sizes]
            self.lastZ[0] = self.inputX
        if self.lastWeights is None:
            self.lastWeights = self.initial_weights
        if self.lastBiases is None:
            self.lastBiases = self.initial_biases
        lastWeights = self.lastWeights
        lastBiases = self.lastBiases
        lastA = self.lastA
        lastZ = self.lastZ
        for x in range(1,len(layer_sizes)):
            lastZ[x] = np.dot(lastWeights[x-1],lastA[x-1]) + lastBiases[x-1]
            lastA[x] = sigmoid_activation_function(lastZ[x])
        l = len(layer_sizes) - 1
        lastZ[l] = np.dot(lastWeights[l-1],lastA[l-1]) + lastBiases[l-1]
        lastA[l] = softmax_activation_function(lastZ[l])
        self.lastA = lastA
        self.lastZ = lastZ


    def backprop(self):
        lastA = self.lastA
        lastZ = self.lastZ
        labelY = self.labelY
        lastWeights = self.lastWeights
        lastBiases = self.lastBiases
        gradCost = -1*(labelY - lastA[-1])
        deltaL = np.multiply(gradCost,softmax_activation_function_Prime(lastZ[-1]))
        lastDelta = deltaL
        lastBiases[-1] -= eta * lastDelta
        lastWeights[-1] -= eta * np.outer(lastDelta, lastA[-2])
        for l in range(len(lastA)-2,1,-1):
            delta_l = np.multiply(np.dot(np.transpose(lastWeights[l]),lastDelta),sigmoid_activation_function_Prime(lastZ[l]))
            lastDelta = delta_l
            lastBiases[l-1] -= eta * delta_l
            lastWeights[l-1] -= eta * np.outer(delta_l,lastA[l-1])
        self.lastWeights = lastWeights
        self.lastBiases = lastBiases


    def cost(self):
        lastA = self.lastA
        labelY = self.labelY
        diff = labelY - lastA[-1]
        diff = np.reshape(diff,(-1))
        cost = np.dot(diff,diff)
        return cost


import numpy as np
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

layer_sizes = np.array([784,120,10])
initial_weights = [np.random.randn(x,y) for x,y in zip(layer_sizes[1:],layer_sizes[:-1])]
for x in range(len(initial_weights)):
    for y in range(len(initial_weights[x])):
        initial_weights[x][y] /= np.sum(initial_weights[x][y]**2)
initial_biases = [np.random.randn(x,1) for x in layer_sizes[1:]]

inputX = train_X[0].reshape(784,1)
zerosForLabelY = np.zeros([layer_sizes[-1],1])
labelY = zerosForLabelY
labelY[train_y[0]-1][0] = 1

network1 = Network()
network1.layer_sizes = layer_sizes
network1.initial_weights = initial_weights
network1.initial_biases = initial_biases
network1.inputX = inputX
network1.labelY = labelY

network1.feedforward()
print("cost0:",network1.cost())
for y in range(1):
    for x in range(60000):
        if train_y[x] in [0,1,2,3,4,5,6,7,8,9]:
            eta = 0.001
            network1.inputX = train_X[x].reshape(784,1)
            zerosForLabelY = np.zeros([layer_sizes[-1], 1])
            labelY = zerosForLabelY
            labelY[train_y[x] - 1][0] = 1
            network1.labelY = labelY
            network1.backprop()
            network1.feedforward()
            print("cost"+str(y)+":",network1.cost())
print(network1.lastA[-1])