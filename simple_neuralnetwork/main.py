import NeuralNetwork as nn


def main():
    input = [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    neuralnetwork = nn.NeuralNetwork(6, 40, 3, 10000)
    neuralnetwork.initialise(0.5, 0.5)
    neuralnetwork.train(input, [[0, 0, 1], [0, 1, 0], [0, 1, 1]])
    neuralnetwork.recognize([0, 1, 0])

main()