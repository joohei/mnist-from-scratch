# mnist-from-scratch
Creating a neural network from scratch to solve the MNIST dataset. The MNIST dataset consists of 70000 handwritten digits. My goal is to achieve an efficient solution that can recognize digits with at least a 98% accuracy.

## Creating a Neural Network to Classify the MNIST Dataset

### Prerequisites

To use this code, you will need to install the following libraries:

- numpy
- scipy
- matplotlib (optional, you need it to display augmented images)
- cupy
- torch
- keras

You can install these libraries using pip:

`pip install numpy scipy matplotlib cupy torch keras`

### Using the Code

The code consists of four Python files:

- `Network.py`: This file contains the `Network` class, which represents a neural network. It has functions for training and evaluating the network, as well as saving and loading the network to and from disk.

- `Loss.py`: This file contains the `Loss` class, which represents a loss function. It has functions for calculating the loss and the gradient of the loss with respect to the output of the network.

- `Layer.py`: This file contains the `Layer` class, which represents a layer in a neural network. It has functions for forward and backward propagation through the layer.

- `utils.py`: This file contains utility functions for augmenting images and plotting images.

To use the code to create and train a neural network to classify the MNIST dataset, you can follow these steps:

1. Load the MNIST data from keras:

```python
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

2. Augment the training data by rotating and shifting the images:

```python
from utils import augment

num_repeats = 5  # How many times to repeat the 60000 images from the MNIST dataset?
x_train = np.repeat(x_train, num_repeats, axis=0)
y_train = np.repeat(y_train, num_repeats, axis=0)

x_train = np.stack([augment(image) for image in x_train])  # This creates a single array of uniquely augmented images
```

3. Reshape the data to the correct format:

```python
from keras.utils import to_categorical

# Reshape the data to the correct format
x_train = np.reshape(x_train, (-1, 784, 1)) / 255.0  # Making the images a flattened 1D array and normalizing the rgb values to range [0, 1]
y_train = np.reshape(to_categorical(y_train, 10), (-1, 10, 1))  # One-Hot encoding the labels for the images
x_test = np.reshape(x_test, (-1, 784, 1)) / 255.0  # Making the images a flattened 1D array and normalizing the rgb values
```

4. Create the `DataSet` objects for training and testing:

```python
import Network as nn

# Create the training and the validation datasets with a specified batch size
data_set = nn.DataSet(x_train, y_train, batch_size=32)
test_set = nn.DataSet(x_test, y_test, batch_size=32)
```

5. Create the neural network:

```python
import Network as nn
from Layer import Dense, ReLU, Softmax
from Loss import CrossEntropyLoss

# Create a neural network with 2 layers
network = nn.Network(
    CrossEntropyLoss(),
    Dense(784, 256),
    ReLU(),
    Dense(256, 10),
    Softmax()
)
```

6. Train the neural network:

```python
# Train the network for 10 epochs with a learning rate of 0.001 and a momentum of 0.9
network.train(data_set, test_set, epochs=10, learn_rate=0.001, momentum=0.9)
```

7. Save the trained neural network to disk:

```python
network.save("mnist_network.pkl")
```

8. Load the trained neural network from disk and evaluate it on the test set:

```python
# Load the network from disk
network = nn.Network.load("mnist_network.pkl")

# Evaluate the network on the test set
accuracy = network.evaluate(test_set)
print(f"Accuracy: {accuracy:.2%}")
```

### Results

With the above steps, you should be able to achieve an accuracy of at least 98% on the MNIST dataset.

### Additional notes

- If you don't have cuda or rocm available please check the other branch in this repository which uses plain numpy instead of cupy. Please note that this makes the training process much slower.
- You can specify a `verbose` argument on the `network.train()` function. If set to `True`, the training process will be logged using Tensorboard. You can initialize Tensorboard using the command line. `tensorboard --logdir logs` You can then see the logs at http://localhost:6006
