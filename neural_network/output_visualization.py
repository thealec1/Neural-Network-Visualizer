import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame as pg
from pygame import gfxdraw
from neural_network import NeuralNetwork

display = pg.display.set_mode((1280, 720))

pg.display.set_caption("Neural Network Visualizer")

pg.font.init()
LARGE_FONT = pg.font.Font(None, 40)
SMALL_FONT = pg.font.Font(None, 18)
WHITE = (255, 255, 255)

class Neuron:

    def __init__(self, neuron, index, x, y, activation, radius, digit=None):
        self.x = x
        self.y = y
        self.neuron = neuron
        self.index = index
        self.digit = digit
        self.activation = activation
        self.radius = radius
    
    def render(self, display : pg.Surface):
        rgb_value = int(self.activation * 255)
        greyscale_shade = (rgb_value, rgb_value, rgb_value)

        gfxdraw.filled_circle(display, self.x, self.y, self.radius, greyscale_shade)
        pg.draw.circle(display, WHITE, (self.x, self.y), self.radius, 2)

        neuron_activation_label = SMALL_FONT.render(str(round(self.activation, 2)), True, (255 - (255*self.activation), 128*self.activation, 0))
        nwidth, nheight = neuron_activation_label.get_size()
        display.blit(neuron_activation_label, (self.x-nwidth//2, self.y-nheight//2))

        if self.digit != None:
            neuron_digit_label = LARGE_FONT.render(str(self.digit), True, WHITE)
            nwidth, nheight = neuron_digit_label.get_size()
            display.blit(neuron_digit_label, (self.x+self.radius*1.5, self.y-nheight//2))

class NetworkRenderer:

    def __init__(self, display : pg.Surface):
        self.neuron_radius = 30
        self.display = display
        self.spacing = 0
        self.display_width = display.get_width()
        self.network = self.new_network()
    
    def new_network(self):
        self.network = NeuralNetwork(16, 16)
        self.generate()

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
        NEURON_START = 65
        neuron_index = NEURON_START
        NEURON_STOP = 383
        row_position = 0
        spacing = self.spacing
        radius = 20
        offset = self.get_layer_offset(radius, 16)
        network = self.network
        neuron_values = network.get_activation_list(network.input_layer)
        while neuron_index < NEURON_STOP:
        
            if neuron_index == NEURON_START+8:
                neuron_index = NEURON_STOP-8
                row_position += 1
            
            y_position = (row_position*(radius*2+spacing))+offset

            activation = neuron_values[neuron_index]
            self.input_neurons.append(Neuron(network.input_layer[neuron_index], neuron_index, x_position, y_position, activation, radius))
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

            render_list.append(Neuron(hidden_layer[i], i, x_position, y_position, activation, radius))

    def generate_output_neurons(self, x_position):
        network = self.network
        neuron_values = network.get_activation_list(network.output_layer)
        swidth = self.display.get_size()[0]
        radius = self.neuron_radius
        spacing = self.spacing
        offset = self.get_layer_offset(radius, 9)
        for i in range(10):

            y_position = (i*(radius*2+spacing))+offset

            activation = neuron_values[i]

            self.output_neurons.append(Neuron(network.output_layer[i], i, x_position, y_position, activation, radius, digit=i))

    def draw_lines_for_layer(self, display : pg.Surface, selected_layer, connected_layer):
        for render_neuron in selected_layer:
            for i in range(len(connected_layer)):
                connected_neuron = connected_layer[i]
                weight = connected_neuron.neuron.weights[render_neuron.index]

                if weight > 0:
                    line_colour = (255, int(pg.math.lerp(106, 157, weight / 4)), 0) # (0, int(pg.math.lerp(50, 255, weight / 4)), 0)
                else:
                    line_colour = (0, int(pg.math.lerp(51, 213, abs(weight) / 4)), 255) # (int(pg.math.lerp(50, 255, (abs(weight) / 4))), 0, 0)
                
                # TODO Come back to this visualization when network is trained
                line_thickness = int(max(3*abs(weight)/4, 1))
                line_alpha = int(max(255*(abs(weight)/4), 0))
                surface = pg.Surface(display.get_size(), pg.SRCALPHA)
                surface.set_alpha(line_alpha)
                
                pg.draw.line(surface, line_colour+(line_alpha,), (render_neuron.x, render_neuron.y), (connected_neuron.x, connected_neuron.y), line_thickness)
                display.blit(surface, (0, 0))

    def render_synaptic_connections(self, display):
        self.draw_lines_for_layer(display, self.input_neurons, self.first_hidden_neurons)
        self.draw_lines_for_layer(display, self.first_hidden_neurons, self.second_hidden_neurons)
        self.draw_lines_for_layer(display, self.second_hidden_neurons, self.output_neurons)

    def render(self, display):
        self.render_synaptic_connections(display)
        for neuron in self.neurons:
            neuron.render(display)

def loop():
    space_frame_counter = 0
    continuous_input = False
    clock = pg.time.Clock()
    running = True
    network_renderer = NetworkRenderer(display)
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            if event.type == pg.KEYUP:
                if event.key == pg.K_SPACE:
                    if not continuous_input:
                        network_renderer.new_network()
                    continuous_input = False
                    space_frame_counter = 0
        
        keys = pg.key.get_pressed()
        if keys[pg.K_SPACE]: space_frame_counter += 1
        if space_frame_counter == 30:
            continuous_input = True
            network_renderer.new_network()
            space_frame_counter = 0

        pg.display.update()
        clock.tick(60)
        display.fill((0, 0, 0))
        network_renderer.render(display)

    pg.quit()

loop()
