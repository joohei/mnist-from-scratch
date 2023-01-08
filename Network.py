import pickle
import datetime
from torch.utils.tensorboard import SummaryWriter
from Layer import Activation, Dropout
import numpy as np


def create_batches(x, y, batch_size):
    num_batches = np.int(np.ceil(x.shape[0] / batch_size))
    x_train = np.array_split(x, num_batches)
    y_train = np.array_split(y, num_batches)
    return zip(x_train, y_train)


class DataSet:
    def __init__(self, x_train, y_train, batch_size=32):
        self.x = np.asarray(x_train)
        self.y = np.asarray(y_train)
        self.batch_size = batch_size
        self.batches = [Batch(x, y) for x, y in create_batches(self.x, self.y, self.batch_size)]

    def update_batches(self, batch_size):
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.batches = [Batch(x, y) for x, y in create_batches(self.x, self.y, self.batch_size)]

    @staticmethod
    def load(filename):
        return pickle.load(open(filename, "rb"))

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))


class Batch:
    def __init__(self, x_train, y_train):
        self.x = np.asarray(x_train)
        self.y = np.asarray(y_train)
        self.points = [Point(x, y) for x, y in zip(self.x, self.y)]


class Point:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs


class TrainingContext:
    def __init__(self, net):
        self.net = net

    def __enter__(self):
        self.net.training = True

    def __exit__(self, *exc):
        self.net.training = False


class Network:
    def __init__(self, loss, *layers):
        self.layers = layers
        self.loss = loss
        self.training = False

    def forward(self, inputs):

        for layer in self.layers:
            if isinstance(layer, Dropout):
                if self.training:
                    inputs = layer.forward(inputs)
            else:
                inputs = layer.forward(inputs)

        return inputs

    def backward(self, gradient, learn_rate, momentum, clip_value):

        for layer in reversed(self.layers):
            if isinstance(layer, Activation):
                gradient = layer.backward(gradient)
            else:
                gradient = layer.backward(gradient, learn_rate, momentum, clip_value)

    def learn(self, data_set, learn_rate, momentum, clip_value):

        with TrainingContext(self):

            for batch in np.random.permutation(data_set.batches):
                outputs = self.forward(batch.x)
                gradient = self.loss.derivative(outputs, batch.y)
                self.backward(gradient, learn_rate, momentum, clip_value)

    def train(self, data_set, test_set, learn_rate=0.01, momentum=0.9, clip_value=1.0, epochs=1000, verbose=True):

        with SummaryWriter(f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}") as logger:

            for epoch in range(epochs):
                self.learn(data_set, learn_rate, momentum, clip_value)

                if verbose:
                    logger.add_scalar("loss/training", self.cost(data_set), global_step=epoch)
                    logger.add_scalar("loss/validation", self.cost(test_set), global_step=epoch)
                    logger.add_scalar("accuracy/training", self.accuracy(data_set), global_step=epoch)
                    logger.add_scalar("accuracy/validation", self.accuracy(test_set), global_step=epoch)

    def accuracy(self, data_set):
        return np.mean(self.classify(data_set.x) == self.labelify(data_set.y))

    def cost(self, data_set):
        return self.loss.calculate(self.forward(data_set.x), data_set.y)

    def classify(self, inputs):
        return self.labelify(self.forward(inputs))

    @staticmethod
    def labelify(outputs):
        return np.argmax(outputs, axis=1)

    @staticmethod
    def load(filename):
        return pickle.load(open(filename, "rb"))

    def save(self, filename):
        pickle.dump(self, open(filename, "wb"))
