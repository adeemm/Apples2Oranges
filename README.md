# Apples2Oranges
A simple neural network (built without any deep learning libraries) to classify images of apples and oranges

I built this very simple neural network in Python, without the help of any machine learning libraries, in order to better understand how neural networks actually work and the concepts behind them. The network has 7500 input layer neurons (because each image is 50 x 50 pixels x 3 color channels), 2 hidden layers with 50 and 30 neurons respectively, and an output layer with 2 neurons (for apple or orange). The network starts out with random weights and biases, but as it is trained repeatedly on pictures of apples and oranges, it will update its weights and biases with stochastic gradient descent using backpropagation. Also included is an already pretrained model file with saved weights and biases.

# Requirements
* Qt & pyqt5 for the UI
* numpy for matrix math
* opencv for image processing
* matplotlib for visualization
