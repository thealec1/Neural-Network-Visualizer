import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pickle
import pygame as pg
from pygame import gfxdraw
from neural_network import NeuralNetwork
import math
import random

pg.font.init()
LARGE_FONT = pg.font.Font(None, 40)
SMALL_FONT = pg.font.Font(None, 18)
WHITE = (255, 255, 255)

class RenderUtils:

    @staticmethod
    def draw_alpha_line(target_surface : pg.Surface, colour : tuple, start_pos : tuple, end_pos : tuple, alpha : int, thickness : int):
        x1,y1 = start_pos
        x2,y2 = end_pos
        line_surface = pg.Surface( (x2-x1, max(abs(y2-y1), thickness) ), pg.SRCALPHA)
        line_surface.set_alpha(alpha)
        
        w,h = line_surface.get_size()
        start_y = 0 if y1 <= y2 else h
        end_y = 0 if start_y == h or y1 == y2 else h

        pg.draw.line(line_surface, colour, (0, start_y), (w, end_y), thickness)
        target_surface.blit(line_surface, (x1, min(y2,y1)))

    def get_neuron_colour(weight, max_weight, min_weight) -> tuple:
        # print(f"max{weight/max_weight} min{abs(weight/min_weight)}")
        orange_blue = True
        if weight > 0:
            return (255, int(pg.math.lerp(106, 157, weight/max_weight)), 0) if orange_blue else (0, int(pg.math.lerp(50, 255, weight / max_weight)), 0)
        else:
            return (0, int(pg.math.lerp(51, 213, abs(weight/min_weight))), 255) if orange_blue else (int(pg.math.lerp(50, 255, (weight / min_weight))), 0, 0)

class RenderNeuron:

    def __init__(self, render_network, max_activation, neuron, index, x, y, activation, radius, digit=None, in_input_layer=False):
        self.x = x
        self.y = y
        self.render_network = render_network
        self.max_activation = max_activation
        self.neuron = neuron
        self.index = index
        self.digit = digit
        self.in_input_layer = in_input_layer
        self.activation = activation
        self.radius = radius
        self.weight_colours = []

    def pattern_render(self, display : pg.Surface):
        connections = self.neuron.weights
        dim = int(math.sqrt(len(connections)))
        weight_pixel_array = pg.PixelArray(pg.Surface((dim, dim)))
        index = 0
        max_w = self.render_network.max_weight
        min_w = self.render_network.min_weight
        for row in range(dim):
            for col in range(dim):
                weight_pixel_array[col, row] = RenderUtils.get_neuron_colour(connections[index], max_w, min_w)
                index += 1
        size = (self.radius*2)-5
        weight_pixel_surface = pg.transform.scale(weight_pixel_array.surface, (size, size))
        w,h = weight_pixel_surface.get_size()
        display.blit(weight_pixel_surface, (self.x-(w//2), self.y-(h//2)))

    def basic_render(self, display : pg.Surface):
        rgb_value = int(self.activation/self.max_activation * 255)
        greyscale_shade = (rgb_value, rgb_value, rgb_value)

        gfxdraw.filled_circle(display, self.x, self.y, self.radius, greyscale_shade)
        pg.draw.circle(display, WHITE, (self.x, self.y), self.radius, 2)

        activation_ratio = self.activation/self.max_activation
        neuron_activation_label = SMALL_FONT.render(str(round(self.activation, 2)), True, (255 - (255*activation_ratio), 128*activation_ratio, 0))
        nwidth, nheight = neuron_activation_label.get_size()
        display.blit(neuron_activation_label, (self.x-nwidth//2, self.y-nheight//2))

        if self.digit != None:
            neuron_digit_label = LARGE_FONT.render(str(self.digit), True, WHITE)
            nwidth, nheight = neuron_digit_label.get_size()
            display.blit(neuron_digit_label, (self.x+self.radius*1.5, self.y-nheight//2))

    def render(self, do_pattern_render : bool, display : pg.Surface):
        if do_pattern_render and not self.in_input_layer:
            self.pattern_render(display)
        else:
            self.basic_render(display)

class NetworkRenderer:

    NEURON_START = 28*6 + 6
    NEURON_STOP = 28*21 + 21
    
    def __init__(self, display : pg.Surface):
        self.neuron_radius = 30
        self.display = display
        self.spacing = 0
        self.display_width = display.get_width()

        self.network = self.load_network()
        self.randomize(False)
    
    def randomize(self, noise_mode):
        
        if noise_mode:
            self.network.image = [random.randint(0, 255) for _ in range(784)]
            self.network.feedforward(self.network.image)
            self.network.label = "Noise"
        else:
            self.network.select_random_testing_example()

        hidden_activations = self.network.get_activation_list(self.network.first_hidden_layer + self.network.second_hidden_layer)

        self.max_hidden_activation = max(hidden_activations)
        self.number_surface = self.colour_array_to_image(self.network.image)
        
        self.generate()
        weights = []
        weights = self.append_weights(self.input_neurons, self.first_hidden_neurons, weights)
        weights = self.append_weights(self.first_hidden_neurons, self.second_hidden_neurons, weights)
        weights = self.append_weights(self.second_hidden_neurons, self.output_neurons, weights)
        self.max_weight = max(weights)
        self.min_weight = min(weights)

    def append_weights(self, selected_layer, connected_layer, weights):
        for render_neuron in selected_layer:
            for i in range(len(connected_layer)):
                connected_neuron = connected_layer[i]
                weight = connected_neuron.neuron.weights[render_neuron.index]
                weights.append(weight)
        return weights

    def load_network(self):
        with open("serialized_network_parameters.pickle", "rb") as infile:
            return pickle.load(infile)
        # try:
        #     with open("serialized_network_parameters.pickle", "rb") as infile:
        #         return pickle.load(infile)
        # except:
        #     return NeuralNetwork(16, 16)

    def get_layer_offset(self, radius, layer_size):
        sh = self.display.get_size()[1]
        spacing = self.spacing
        return (sh-((radius*2+spacing)*layer_size))//2

    def generate(self):
        self.input_neurons = []
        self.first_hidden_neurons = []
        self.second_hidden_neurons = []
        self.output_neurons = []
        w = self.display_width
        self.generate_input_neurons(int(w*1/8))
        self.generate_hidden_layer(int(w*3/8), self.network.first_hidden_layer, self.first_hidden_neurons)
        self.generate_hidden_layer(int(w*5/8), self.network.second_hidden_layer, self.second_hidden_neurons)
        self.generate_output_neurons(int(w*7/8))
        self.neurons = self.input_neurons+self.first_hidden_neurons+self.second_hidden_neurons+self.output_neurons
    
    def generate_input_neurons(self, x_position):
        neuron_index = NetworkRenderer.NEURON_START
        row_position = 0
        spacing = self.spacing
        radius = 20
        offset = self.get_layer_offset(radius, 16)
        network = self.network
        neuron_values = network.get_activation_list(network.input_layer)
        while neuron_index < NetworkRenderer.NEURON_STOP:
        
            if neuron_index == NetworkRenderer.NEURON_START+8:
                neuron_index = NetworkRenderer.NEURON_STOP-8
                row_position += 1
            
            y_position = (row_position*(radius*2+spacing))+offset

            activation = neuron_values[neuron_index]
            self.input_neurons.append(RenderNeuron(self, 1, network.input_layer[neuron_index], neuron_index, x_position, y_position, activation, radius, in_input_layer=True))
            neuron_index += 1
            row_position += 1

    def generate_hidden_layer(self, x_position, hidden_layer, render_list : list):
        network = self.network
        neuron_values = network.get_activation_list(hidden_layer)
        radius = 20
        spacing = self.spacing
        offset = self.get_layer_offset(radius, 15)
        for i in range(len(neuron_values)):
            activation = neuron_values[i]

            y_position = (i*(radius*2+spacing))+offset

            render_list.append(RenderNeuron(self, self.max_hidden_activation, hidden_layer[i], i, x_position, y_position, activation, radius))

    def generate_output_neurons(self, x_position):
        network = self.network
        neuron_values = network.get_activation_list(network.output_layer)
        radius = self.neuron_radius
        spacing = self.spacing
        offset = self.get_layer_offset(radius, 9)
        for i in range(10):

            y_position = (i*(radius*2+spacing))+offset

            activation = neuron_values[i]

            self.output_neurons.append(RenderNeuron(self, 1, network.output_layer[i], i, x_position, y_position, activation, radius, digit=i))
        
        activation_list = self.network.get_activation_list(self.network.output_layer)
        self.output_number = activation_list.index(max(activation_list))

    def draw_lines_for_layer(self, display : pg.Surface, selected_layer, connected_layer):
        for render_neuron in selected_layer:
            render_neuron.weight_colours = []
            for i in range(len(connected_layer)):
                connected_neuron = connected_layer[i]
                weight = connected_neuron.neuron.weights[render_neuron.index]
                
                line_colour = RenderUtils.get_neuron_colour(weight, self.max_weight, self.min_weight)

                render_neuron.weight_colours.append(line_colour)

                # TODO Come back to this visualization when network is trained
                line_thickness = int(max(4*abs(weight)/self.max_weight, 1))
                line_alpha = int(max(255*(abs(weight)/self.max_weight), 0))
                
                RenderUtils.draw_alpha_line(display, line_colour, (render_neuron.x, render_neuron.y), (connected_neuron.x, connected_neuron.y), line_alpha, line_thickness)

    def render_synaptic_connections(self, display):
        self.draw_lines_for_layer(display, self.input_neurons, self.first_hidden_neurons)
        self.draw_lines_for_layer(display, self.first_hidden_neurons, self.second_hidden_neurons)
        self.draw_lines_for_layer(display, self.second_hidden_neurons, self.output_neurons)

    def colour_array_to_image(self, image):
        pixel_array = pg.PixelArray(pg.Surface((28, 28)))
        index = 0
        for row in range(28):
            for col in range(28):
                colour = image[index]
                pixel_array[col,row] = (colour, colour, colour)
                index += 1
        surface = pg.transform.scale(pixel_array.surface, (100, 100))
        return surface

    def render_image_with_label(self, display):
        input_text = LARGE_FONT.render(str(self.network.label), True, (255, 255, 255))
        output_text = LARGE_FONT.render(str(self.output_number), True, (255, 255, 255))
        
        sw,sh = self.display.get_size()
        nw, nh = self.number_surface.get_size()
        input_textw, input_texth = input_text.get_size()
        display.blit(input_text, ((sw*1/8-20)//2-(input_textw//2), (sh//2)-(input_texth//2)+100))
        display.blit(output_text, (display.get_width()-output_text.get_width()-10, display.get_height()//2-output_text.get_height()//2))
        display.blit(self.number_surface, ((sw*1/8-20)//2-(nw//2), (sh//2)-(nh//2)))

    def render(self, do_pattern_render, display : pg.Surface):
        
        self.render_synaptic_connections(display)
        
        for neuron in self.neurons:
            neuron.render(do_pattern_render, display)
        
        self.render_image_with_label(display)

def loop():
    display = pg.display.set_mode((1280, 720))
    pg.display.set_caption("Neural Network Visualizer")
    space_frame_counter = 0
    continuous_input = False
    clock = pg.time.Clock()
    running = True
    network_renderer = NetworkRenderer(display)
    do_pattern_render = False
    noise_mode = False
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYUP:
                if event.key == pg.K_SPACE:
                    if not continuous_input:
                        network_renderer.randomize(noise_mode)
                    continuous_input = False
                    space_frame_counter = 0
                if event.key == pg.K_p:
                    do_pattern_render = not do_pattern_render
                if event.key == pg.K_n:
                    noise_mode = not noise_mode
                    network_renderer.randomize(noise_mode)

        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]: space_frame_counter += 1
        if space_frame_counter == 30:
            continuous_input = True
            network_renderer.randomize(noise_mode)
            space_frame_counter = 0

        pg.display.update()
        clock.tick(60)
        display.fill((10, 10, 10))
        network_renderer.render(do_pattern_render, display)

    pg.quit()

