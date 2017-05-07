import numpy as np

def logistic(x):
    return 1/(1+np.exp(-x))


def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))


# ################# CLASS - NEURON  ########################
class Neuron:
    # ****************** constructor *******************
    def __init__(self, num_inputs):
        self.weights = []
        self.num_inputs = num_inputs
        self.output = 0
        # initialise weights for each input
        self.__initialise_weights__()

    # **************** methods **********************
    def activation(self, vector_input, bias):
        sum = 0
        for x in range(0, len(vector_input)):
            sum += self.weights[x] * vector_input[x]
        self.output = logistic(sum + bias)

        return self.output

    def adjust_weights(self, target_output, connected_layer, learn_rate):
        for w in range(0, len(self.weights)):
            errorloss = logistic_deriv(self.output) * (self.output - target_output)
            self.weights[w] -= (errorloss * connected_layer.neurons[w].output * learn_rate)

    def __initialise_weights__(self):
        for x in range(self.num_inputs):
            self.weights.append(np.random.uniform(-1, 1))


# #################### CLASS - LAYER ######################
class Layer:
    def __init__(self, num_neuron, num_inputs):
        self.neurons = []
        for n in range(num_neuron):
            self.__init_neurons__(num_inputs)

    def __init_neurons__(self, num_inputs):
        neuron = Neuron(num_inputs)
        self.neurons.append(neuron)


# ################# CLASS - NEURAL NETWORK ##################
class NeuralNetwork:
    # ***************** constructor ***************
    def __init__(self, num_hidden_layers, num_hidden_neurons, num_outputs, epoch=1000):
        # network properties
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.num_outputs = num_outputs

        # network objects (layers)
        self.epochs = epoch
        self.hidden_layers = []
        self.output_layer = []

    # ****************** methods *********************
    def initialise(self, bias, learn_rate):
        self.bias = bias
        self.learn_rate = learn_rate
        # initialise hidden layers and neurons
        for i in range(0, self.num_hidden_layers):
            layer = Layer(self.num_hidden_neurons, self.num_hidden_neurons)
            self.hidden_layers.append(layer)
        # initialise output layer
            self.output_layer = Layer(self.num_outputs, self.num_hidden_neurons)

    def train(self, list_input, list_target):
        # ----------- start forward propagation -----------
        for e in range(self.epochs):
            for i in range(0, len(list_input)):
                self.__forwardpropagation__(list_input[i])
                self.__backpropagation__(list_target[i])

    def recognize(self, list_input):
        output = self.__forwardpropagation__(list_input)
        print('recognized: ', output)

    def __forwardpropagation__(self, list_input):
        # foreach hidden layer activate it's neurons
        propagated_input = self.__propagation__(list_input)
        # finished hidden layer computing, calculate output
        list_output = []
        for n in range(0, len(self.output_layer.neurons)):
            temp_output = self.output_layer.neurons[n].activation(propagated_input, self.bias)
            list_output.append(temp_output)

        return list_output

    def __propagation__(self, list_input):
        activating_intput = list_input
        # foreach hidden layer activate it's neurons
        for l in range(0, len(self.hidden_layers)):
            for n in range(0, len(self.hidden_layers[l].neurons)):
                self.hidden_layers[l].neurons[n].activation(activating_intput, self.bias)

            # clear activating input list, then append just activated neurons' output
            activating_intput.clear()
            for n in range(0, len(self.hidden_layers[l].neurons)):
                activating_intput.append(self.hidden_layers[l].neurons[n].output)

        return activating_intput

    def __backpropagation__(self, list_target):
        # reverse the hidden layer
        self.hidden_layers.reverse()
        list_input = []
        # calculate the errorloss at output layer

        # adjust weights
        for n in range(0, len(self.output_layer.neurons)):
            self.output_layer.neurons[n].adjust_weights(list_target[n], self.hidden_layers[0], self.learn_rate)
            list_input.append(self.output_layer.neurons[n].output)

        # backward propagation (because hidden layer is reversed)
        self.__propagation__(list_input)
        self.hidden_layers.reverse()