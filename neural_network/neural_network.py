import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg
import random
import math

pg.init()

SIZE = 28

class Utils:

    @staticmethod
    def dot_product(a : list, b : list) -> float:
        dot = 0
        for i in range(len(a)):
            dot += a[i] * b[i]
        return dot

    @staticmethod
    def sigmoid(x): #TODO ReLU
        return 1 / (1 + (math.e**-x))

class Neuron:
    
    def __init__(self, activation_state=0, bias=0):
        self.activation_state = activation_state
        self.synaptic_connections = []
        self.weights = []
        self.bias = bias
    
    def add_synaptic_connection(self, neuron):
        self.synaptic_connections.append(neuron)

    def update_activation(self, adjacent_activations):
        activation = Utils.sigmoid(Utils.dot_product(self.weights, adjacent_activations) + self.bias)
        self.activation_state = activation

    def __str__(self):
        return f"Neuron Activation: {round(self.activation_state, 2)} with {len(self.synaptic_connections)} connections"

class NeuralNetwork:

    def __init__(self, first_layer_neuron_count, second_layer_neuron_count):
        digit_image = pg.image.load("test_digit.png")
        self.colour_list = []
        self.create_input_layer(digit_image)
        self.first_hidden_layer = self.create_hidden_layer(first_layer_neuron_count, self.input_layer)
        self.second_hidden_layer = self.create_hidden_layer(second_layer_neuron_count, self.first_hidden_layer)
        self.output_layer = self.create_hidden_layer(10, self.second_hidden_layer)

    def create_hidden_layer(self, neuron_count, adjacent_layer):
        neurons = []
        for i in range(neuron_count):
            neuron = Neuron(bias=random.randint(-10,10))
            for other_neuron in adjacent_layer:
                neuron.weights.append(random.uniform(-4,4))
                neuron.add_synaptic_connection(other_neuron)
            neuron.update_activation(self.get_activation_list(adjacent_layer))
            neurons.append(neuron)
        return neurons

    def get_activation_list(self, layer):
        return [neuron.activation_state for neuron in layer]

    def create_input_layer(self, digit_image : pg.Surface):
        pixel_array = pg.PixelArray(digit_image)
        self.input_layer = []
        for row in range(SIZE):
            for col in range(SIZE):
                pixel_value = pixel_array[col, row]
                colour = digit_image.unmap_rgb(pixel_value)
                self.colour_list.append(colour) # DEBUG
                adjusted_greyscale_value = colour[0]/255
                self.input_layer.append(Neuron(activation_state=adjusted_greyscale_value))
        self.input_count = len(self.input_layer)

if __name__ == "__main__":

    network = NeuralNetwork(16, 16)
    layer = network.input_layer
    for index in range(70, 74):
        print(f"{index}: {layer[index]} {network.colour_list[index]}")

    # for neuron in network.input_layer:
    #     print(neuron.activation_state)
