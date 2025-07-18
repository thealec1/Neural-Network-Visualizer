Neural Network Visualizer
==============

<img width="1592" height="892" alt="Neural Network Visualized" src="https://github.com/user-attachments/assets/8580112f-38f6-4f5a-97bf-e69b15780aa2" />

---
# What is it?

A visualizer for a multilayer perceptron that recognizes handwritten digits.  The images are extracted from the MNIST database.

It recognizes digits at 95.79% accuracy for the testing dataset in MNIST.

The visualizer tool above shows the randomly selected image '3' being accurately identified, as seen by the network's output number.

The colour coding is as follows:

- `Purple`: Positive weights relative to the 1st layer maximum
- `Dark Blue`: Negative weights relative to the 1st layer minimum
- `Orange`: Positive weights relative to the global maximum in the 2nd and 3rd layers
- `Light Blue`: Negative weights relative to the global minimum in the 2nd and 3rd layers

# How does it work?

The network has 4 layers:
- An input layer
- Two hidden layers in the middle
- An output layer

Each layer is comprised of neurons with a corresponding number, or activation.  The input layer consists of all 784 pixels within the 28x28 grid of pixels.  However, for visualization purposes there are only 2 groups of 8 being displayed at different regions within the image.

The input layer neuron activation values are dependent on the greyscale value of the corresponding pixels of the input image.  The lines are called weights, which can be thought of as synapses in a biological neural network, and connect all of the neurons in the adjacent layers together.

These connections are just numbers, and are multiplied by the activations in the previous layer to determine the activations in the next layer.  They can be thought of as determining how 'important' a particular neuron in the network is for picking up on patterns within the input images.  Lower numbers suggest low importance, higher numbers suggest high importance, negative numbers select against that particular neuron.

By default, these weights are randomly initialized.  Through an algorithm called backpropagation, all of these weights are tweaked and adjusted based on how well the network performs on recognizing the digits in the training dataset.  The more training examples the network is given, the greater the opportunity the network has to adjust the weights, and the more accurate it will pick up on these patterns.

# How do I use it?

1. Download contents of `neural_network`
2. Unzip the `serialized_network_parameters` zip file
3. Run the `neural_network.py` script, ensuring that it's calling the visualizer's loop method.

Controls:

- `Space` Selects a new random image
- `W` Toggles weight map visualizer
- `N` Input random noise

