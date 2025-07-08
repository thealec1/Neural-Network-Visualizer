import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg
import random
import math
import pickle
import numpy as np
import time
import paint
import output_visualization
from mnist import MNIST
mndata = MNIST("mnist_data")

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
    def relu(x):
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x):
        return 1.0 if x > 0 else 0.0

    # @staticmethod
    # def sigmoid(x): #TODO ReLU
    #     return 1 / (1 + (math.e**-x))

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1 / (1 + z)
        else:
            z = math.exp(x)
            return z / (1 + z)

    @staticmethod
    def sigmoid_derivative(x):
        s = Utils.sigmoid(x)
        return (1-s) * s

    @staticmethod
    def mean(list_to_average):
        averaged_list = []
        for i in range(len(list_to_average[0])):
            item = 0
            for j in range(len(list_to_average)):
                item += list_to_average[j][i]
            averaged_list.append(item/len(list_to_average))
        return averaged_list

class Neuron:
    
    def __init__(self, activation_state=0, bias=0):
        self.activation_state = activation_state
        self.synaptic_connections = []
        self.weights = []
        self.bias = bias
    
    def add_synaptic_connection(self, neuron):
        self.synaptic_connections.append(neuron)

    def update_activation(self, adjacent_activations):
        self.z = np.dot(self.weights, adjacent_activations) + self.bias
        self.activation_state = Utils.relu(self.z)

    def __str__(self):
        return f"Neuron Activation: {round(self.activation_state, 2)} with {len(self.synaptic_connections)} connections"

class NeuralNetwork:

    SERIALIZE_PATH = "serialized_network_parameters.pickle"

    def __init__(self, first_layer_neuron_count, second_layer_neuron_count, activation_function="relu"):
        self.training_images, self.training_labels = mndata.load_training()
        self.testing_images, self.testing_labels = mndata.load_testing()
        self.first_layer_neuron_count = first_layer_neuron_count
        self.second_layer_neuron_count = second_layer_neuron_count

        if activation_function == "relu":
            self.activation_function = Utils.relu
            self.activation_derivative = Utils.relu_derivative
        elif activation_function == "sigmoid":
            self.activation_function = Utils.sigmoid
            self.activation_derivative = Utils.sigmoid_derivative
        else:
            raise TypeError(f"Incorrect activation function argument {activation_function}")

        self.minibatch_size = 128
        self.learning_rate = 0.28
        self.initialize_network(random.choice(self.training_images))

    def initialize_network(self, training_example):
        self.create_input_layer_from_list(training_example)
        self.first_hidden_layer = self.create_layer(self.first_layer_neuron_count, self.input_layer)
        self.second_hidden_layer = self.create_layer(self.second_layer_neuron_count, self.first_hidden_layer)
        self.output_layer = self.create_layer(10, self.second_hidden_layer)

    def select_random_testing_example(self):
        random_index = random.randint(0, 10_000-1)
        self.image, self.label = self.testing_images[random_index], self.testing_labels[random_index]
        self.feedforward(self.image)

    def update_layer(self, bias_index, weight_index, learning_rate, averaged_weights, averaged_biases, layer):
        for neuron in layer:
            neuron.bias -= learning_rate * averaged_biases[bias_index]
            bias_index += 1
            for index in range(len(neuron.weights)):
                neuron.weights[index] -= learning_rate * averaged_weights[weight_index]
                weight_index += 1
        return bias_index, weight_index

    def feedforward(self, training_example):
        for index in range(len(self.input_layer)):
            self.input_layer[index].activation_state = training_example[index]/255
        for neuron in self.first_hidden_layer:
            neuron.update_activation(self.get_activation_list(self.input_layer))
        for neuron in self.second_hidden_layer:
            neuron.update_activation(self.get_activation_list(self.first_hidden_layer))
        for neuron in self.output_layer:
            z = np.dot(neuron.weights, self.get_activation_list(self.second_hidden_layer)) + neuron.bias
            neuron.z = z
            neuron.activation_state = Utils.sigmoid(z)
    
    def train_on_epoch(self):
        paired_data = list(zip(self.training_images, self.training_labels))
        random.shuffle(paired_data)
        minibatch_quantity = math.ceil(len(paired_data)/self.minibatch_size)
        for i in range(0, len(paired_data), self.minibatch_size):
            minibatch = paired_data[i:i+self.minibatch_size]

            # gradient_weights = [(w0, w1, w2, ..., w13001), (w0, w1, w2, ..., w13001), ..., (w0, w1, w2, ..., w13001)]
            # gradient_biases = [(b0, b1, b2, ..., b825), (b0, b1, b2, ..., b825), (b0, b1, b2, ..., b825)]

            gradient_weights = []
            gradient_biases = []
            for example, label in minibatch:
                desired_output = [0.0]*10
                desired_output[label] = 1.0
                self.feedforward(example)
                weights, biases = self.backpropagate(desired_output)
                gradient_weights.append(weights)
                gradient_biases.append(biases)

            bias_index = 0
            weight_index = 0
            averaged_weights = Utils.mean(gradient_weights)
            averaged_biases = Utils.mean(gradient_biases)
            # print("Mean weight gradient (first 5):", averaged_weights[20:60])
            # print("Mean bias gradient (first 5):", averaged_biases[20:60])
            bias_index, weight_index = self.update_layer(bias_index, weight_index, self.learning_rate, averaged_weights, averaged_biases, self.first_hidden_layer)
            bias_index, weight_index = self.update_layer(bias_index, weight_index, self.learning_rate, averaged_weights, averaged_biases, self.second_hidden_layer)
            bias_index, weight_index = self.update_layer(bias_index, weight_index, self.learning_rate, averaged_weights, averaged_biases, self.output_layer)
            # print(f"{int(i/self.minibatch_size)+1}/{minibatch_quantity}", end="\r")

    def train(self, epochs : int):
        then = time.time()
        end_char = "\r"
        for i in range(epochs):
            self.train_on_epoch()
            if i+1 == epochs:
                end_char = "\n"
            print(f"{(i+1)}/{epochs}", end=end_char)
        print(f"\033[032mTraining completed in {round((time.time()-then)/60,2)}m\033[0m")

    def backpropagate(self, desired_output) -> tuple:
        output_weights = [] # ∂C0/∂w^(L)_wjk
        output_biases = [] # ∂C0/∂b^(L)_bj
        for j, neuron in enumerate(self.output_layer): # L

            a = neuron.activation_state
            sigmoid_derivative = Utils.sigmoid_derivative(neuron.z) * 2 * (a - desired_output[j])
            for k in range(len(neuron.synaptic_connections)):
                pre_a = neuron.synaptic_connections[k].activation_state    
                output_weights.append(pre_a * sigmoid_derivative)
            output_biases.append(sigmoid_derivative)
        
        second_layer_weights = [] # ∂C0/∂w^(L-1)_wki
        second_layer_biases = [] # ∂C0/∂b^(L-1)_wbk
        derivative_cost_l1 = [] # ∂C0/∂a^(L-1)
        for k, neuron in enumerate(self.second_hidden_layer): # (L-1)
            cost_derivative = 0 # ∂C0/∂a^(L-1)_k
            for j, post_neuron in enumerate(self.output_layer): # L
                weight = post_neuron.weights[k]
                cost_derivative += weight * Utils.sigmoid_derivative(post_neuron.z) * 2 * (post_neuron.activation_state - desired_output[j])
            derivative_cost_l1.append(cost_derivative)
            relu_derivative = self.activation_derivative(neuron.z) * cost_derivative
            for i in range(len(neuron.synaptic_connections)): # i in (L-2)
                pre_a = neuron.synaptic_connections[i].activation_state
                second_layer_weights.append(pre_a * relu_derivative)
            second_layer_biases.append(relu_derivative)

        first_layer_weights = [] # ∂C0/∂w^(L-1)_wki
        first_layer_biases = [] # ∂C0/∂b^(L-1)_wbh
        for i, neuron in enumerate(self.first_hidden_layer): # (L-2)
            cost_derivative = 0
            for k, post_neuron in enumerate(self.second_hidden_layer): # (L-1)
                weight = post_neuron.weights[i]
                a_l1_k = derivative_cost_l1[k]
                z = post_neuron.z
                cost_derivative += weight * self.activation_derivative(z) * a_l1_k
            relu_derivative = self.activation_derivative(neuron.z) * cost_derivative
            for h in range(len(neuron.synaptic_connections)):
                pre_a = neuron.synaptic_connections[h].activation_state
                first_layer_weights.append(pre_a * relu_derivative)
            first_layer_biases.append(relu_derivative)
        
        return first_layer_weights+second_layer_weights+output_weights, first_layer_biases+second_layer_biases+output_biases

    def create_layer(self, neuron_count, adjacent_layer):
        neurons = []
        n = len(adjacent_layer)
        standard_deviation = math.sqrt(2/n)
        for i in range(neuron_count):
            neuron = Neuron(bias=0)
            for other_neuron in adjacent_layer:
                he_initialization = random.gauss(0, standard_deviation)
                neuron.weights.append(he_initialization) 
                neuron.add_synaptic_connection(other_neuron)
            neuron.update_activation(self.get_activation_list(adjacent_layer))
            neurons.append(neuron)
        return neurons

    def get_activation_list(self, layer):
        return [neuron.activation_state for neuron in layer]

    def create_input_layer_from_list(self, image : list):
        self.input_layer = []
        for colour_value in image:
            adjusted_greyscale_value = colour_value/255
            self.input_layer.append(Neuron(activation_state=adjusted_greyscale_value))
        self.input_count = len(self.input_layer)

    def feed_image(self, digit_image : pg.Surface):
        pixel_array = pg.PixelArray(digit_image)
        training_colour_data = []
        for row in range(SIZE):
            for col in range(SIZE):
                pixel_value = pixel_array[col, row]
                colour = digit_image.unmap_rgb(pixel_value)
                training_colour_data.append(colour[0])
        self.feedforward(training_colour_data)

    def create_input_layer_from_surface(self, digit_image : pg.Surface):
        pixel_array = pg.PixelArray(digit_image)
        self.input_layer = []
        for row in range(SIZE):
            for col in range(SIZE):
                pixel_value = pixel_array[col, row]
                colour = digit_image.unmap_rgb(pixel_value)
                adjusted_greyscale_value = colour[0]/255
                self.input_layer.append(Neuron(activation_state=adjusted_greyscale_value))
        self.input_count = len(self.input_layer)

    def get_testing_accuracy(self):
        return self.get_accuracy("testing", self.testing_images, self.testing_labels)

    def get_training_accuracy(self):
        return self.get_accuracy("training", self.training_images, self.training_labels)

    def get_accuracy(self, acc_type : str, images : list, labels : list) -> str:
        accurate_passes = 0
        total_passes = len(images)
        end_char = "\r"
        for i, image in enumerate(images):
            self.feedforward(image)
            output_activations = self.get_activation_list(self.output_layer)
            output_number = output_activations.index(max(output_activations))
            colour = "031m" # red
            if output_number == labels[i]:
                colour = "032m" # green
                accurate_passes += 1
            if i+1 == total_passes:
                end_char = "\n"
            print(f"\033[{colour}{i+1}/{total_passes}\033[0m", end=end_char)
        return f"{accurate_passes}/{total_passes} correct for a {round(100*(accurate_passes/total_passes), 2)}% {acc_type} accuracy"

def image_test():

    with open("serialized_network_parameters.pickle", "rb") as infile:
        network = pickle.load(infile)

    while True:
        image_directory = input("Name the image you would like to identify: ")
        if len(image_directory) == 0:
            image_directory = "test_digit.png"
        if image_directory.lower() == "q":
            break
        network.feed_image(pg.image.load(image_directory))
        print(network.get_activation_list(network.output_layer))

if __name__ == "__main__":

    # TRAIN NETWORK WITH 30 EPOCHS
    # network = NeuralNetwork(first_layer_neuron_count=16, second_layer_neuron_count=16, activation_function='relu')
    # network.train(epochs=30)

    # DUMP RESULTING PARAMETERS TO EXTERNAL FILE
    # with open(NeuralNetwork.SERIALIZE_PATH, "wb") as outfile:
    #     pickle.dump(network, outfile)

    # PRINTS TESTING AND TRAINING ACCURACY
    # print(network.get_testing_accuracy())
    # print(network.get_training_accuracy())

    # TEST SPECIFIC PNG/JPG FILE
    # image_test()

    # VISUALIZER
    output_visualization.loop()
