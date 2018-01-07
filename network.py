import numpy as np


class Network(object):

    # initialize a network with a list of layer sizes (number of neurons), then generate random weights and biases
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        # for plotting data
        self.losses = []
        self.acc1s = []
        self.acc2s = []


    # feeds a feature matrix (image) into the network and returns the activations / outputs at each layer of the network
    def feedforward(self, x):
        activation = x
        activations = [x]
        outputs = []

        for bias, weight in zip(self.biases, self.weights):
            output = np.dot(weight, activation) + bias
            outputs.append(output)
            activation = self.sigmoid(output)
            activations.append(activation)

        return outputs, activations


    # calculate gradients of weights and biases (partial derivatives) and update weights / biases
    def backpropagate(self, x, y):
        # initialize partial derivatives (gradients) of biases and weights
        delta_bias = [np.zeros(bias.shape) for bias in self.biases]
        delta_weight = [np.zeros(weight.shape) for weight in self.weights]

        # get outputs of neural net and calculate loss / cost
        outputs, activations = self.feedforward(x)
        loss = self.loss(activations[-1], y)

        # calculate derivative of loss
        delta_loss = activations[-1] - y
        delta = delta_loss
        delta_bias[-1] = delta
        delta_weight[-1] = np.dot(delta, activations[-2].T)

        # update gradients of each layer in the network using reverse / negative indexing
        for l in range(2, self.num_layers):
            output = outputs[-l]
            delta_activation = self.sigmoid_prime(output)
            delta = np.dot(self.weights[-l + 1].T, delta) * delta_activation
            delta_bias[-l] = delta
            delta_weight[-l] = np.dot(delta, activations[-l - 1].T)

        return loss, delta_bias, delta_weight


    # train the network (update weights / biases) with stochastic gradient descent using backpropagation to compute gradients
    def train(self, x, y, x_validation, y_validation, learning_rate, epochs, batch_size, app, ui):
        batches = x.shape[0] / batch_size
        self.losses = []
        self.acc1s = []
        self.acc2s = []

        for e in range(epochs):
            batch_gen = self.batch(x, y, batch_size)

            for b in range(int(batches)):
                batch = batch_gen.__next__()
                delta_bias = [np.zeros(bias.shape) for bias in self.biases]
                delta_weight = [np.zeros(weight.shape) for weight in self.weights]

                # calculate the change (delta) for weights and biases
                for batch_x, batch_y in batch:
                    loss, delta2_bias, delta2_weight = self.backpropagate(batch_x, batch_y)
                    delta_bias = [db + d2b for db, d2b in zip(delta_bias, delta2_bias)]
                    delta_weight = [dw + d2w for dw, d2w in zip(delta_weight, delta2_weight)]

            # update weights / biases by multiplying the gradients with the ratio of learning rate to batch size
            self.weights = [weight - (learning_rate / batch_size) * dw for weight, dw in zip(self.weights, delta_weight)]
            self.biases = [bias - (learning_rate / batch_size) * db for bias, db in zip(self.biases, delta_bias)]

            # output info to gui
            output = "<strong>Epoch {}</strong> - Loss: {:f}".format(e, loss)
            ui.netOutput.append(output)
            app.processEvents()
            print("Epoch {} - Loss: {:f}".format(e, loss))

            acc1 = self.validate(x, y)
            output = "- Train Accuracy: {:f}".format(acc1)
            ui.netOutput.append(output)

            acc2 = self.validate(x_validation, y_validation)
            output = "- Valid Accuracy: {:f}\n".format(acc2)
            ui.netOutput.append(output)

            # update gui (training occurs on main thread and hangs gui updates)
            app.processEvents()

            # epoch results used for visualization
            self.losses.append(loss)
            self.acc1s.append(acc1)
            self.acc2s.append(acc2)


    # using validation data, predict the label for input, then compare it with the actual label and calculate accuracy
    def validate(self, x, y):
        count = 0

        for x2, y2 in zip(x, y):
            outputs, activations = self.feedforward(x2)

            # check if predicted output (output layer neuron with stronger activation) matches the actual label
            if np.argmax(activations[-1]) == np.argmax(y2):
                count += 1

        accuracy = self.accuracy(count, x.shape[0])
        print("   - Accuracy: {:f}".format(accuracy))
        return accuracy


    # predict the label for an unlabeled input image
    def predict(self, x):
        outputs, activations = self.feedforward(x)
        prediction = np.argmax(activations[-1])

        # convert activation to string representation of the category
        if prediction == 0:
            return "apple"

        elif prediction == 1:
            return "orange"


    # generator that yields label and feature lists in batches
    @staticmethod
    def batch(x, y, batch_size):
        for i in range(0, x.shape[0], batch_size):
            batch = zip(x[i:i + batch_size], y[i:i + batch_size])
            yield batch

    # activation function (output between 0 and 1) for neurons
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # derivative of sigmoid function
    @staticmethod
    def sigmoid_prime(x):
        s = Network.sigmoid(x)
        return s * (1 - s)

    # calculate loss by comparing the output with the one hot vector target (using sigmoid cross entropy)
    @staticmethod
    def loss(prediction, target):
        return np.sum(np.nan_to_num(-target * np.log(prediction) - (1 - target) * np.log(1 - prediction)))

    # calculate total accuracy
    @staticmethod
    def accuracy(correct, total):
        return (float(correct) / total) * 100
