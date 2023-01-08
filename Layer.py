import cupy as cp


class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.weights = None
        self.biases = None
        self.previous_weight_momentum = None
        self.previous_bias_momentum = None

    def forward(self, inputs):
        pass

    def backward(self, output_gradient, learn_rate, momentum, clip_value):
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):

        """
        Initialize a fully connected layer.

        Args:
            input_size: Number of input features
            output_size: Number of output features
        """

        super().__init__()
        self.weights = cp.random.normal(0, 0.01, size=(output_size, input_size))
        self.biases = cp.zeros((output_size, 1))
        self.previous_weight_momentum = cp.zeros_like(self.weights)
        self.previous_bias_momentum = cp.zeros_like(self.biases)

    def forward(self, inputs):

        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, output_size)
        """

        self.inputs = inputs
        self.outputs = self.weights @ self.inputs + self.biases
        return self.outputs

    def backward(self, output_gradient, learn_rate, momentum, clip_value):

        """
        Perform a backward pass through the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer, of shape (batch_size, output_size)
            learn_rate: Learning rate to use during the gradient descent update
            momentum: Momentum to use during the gradient descent update
            clip_value: Value to clip gradients to prevent exploding gradients
        Returns:
            Gradient of the loss with respect to the input of the layer, of shape (batch_size, input_size)
        """

        weight_gradient = cp.mean(output_gradient @ cp.swapaxes(self.inputs, 1, 2), axis=0)
        bias_gradient = cp.mean(output_gradient, axis=0)

        weight_gradient = cp.clip(weight_gradient, -clip_value, clip_value)
        bias_gradient = cp.clip(bias_gradient, -clip_value, clip_value)

        weight_momentum = momentum * self.previous_weight_momentum - learn_rate * weight_gradient
        bias_momentum = momentum * self.previous_bias_momentum - learn_rate * bias_gradient

        self.weights = self.weights + weight_momentum
        self.biases = self.biases + bias_momentum

        self.previous_weight_momentum = weight_momentum
        self.previous_bias_momentum = bias_momentum

        return self.weights.T @ output_gradient


class Activation:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        pass

    def backward(self, output_gradient):
        pass


class Dropout(Activation):
    def __init__(self, rate):

        """
        Initialize a Dropout layer.

        Args:
            rate: Probability of setting a neuron to zero during training
        """

        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, inputs):

        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        self.mask = cp.random.binomial(1, 1 - self.rate, inputs.shape)
        return self.mask * inputs

    def backward(self, output_gradient):

        """
        Perform a backward pass through the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer, of shape (batch_size, input_size)
        Returns:
            Gradient of the loss with respect to the input of the layer, of shape (batch_size, input_size)
        """

        return output_gradient * self.mask


class ReLU(Activation):
    def __init__(self):

        """
        Initialize a ReLU activation layer.
        """

        super().__init__()

    @staticmethod
    def calculate(inputs):

        """
        Perform ReLU activation on inputs.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        return cp.where(inputs > 0, inputs, cp.zeros_like(inputs))

    @staticmethod
    def derivative(inputs):

        """
        Calculate derivative of ReLU activation.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Derivative of shape (batch_size, input_size)
        """

        return cp.where(inputs > 0, cp.ones_like(inputs), cp.zeros_like(inputs))

    def forward(self, inputs):

        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        self.inputs = inputs
        return self.calculate(self.inputs)

    def backward(self, output_gradient):

        """
        Perform a backward pass through the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer, of shape (batch_size, input_size)
        Returns:
            Gradient of the loss with respect to the input of the layer, of shape (batch_size, input_size)
        """

        return output_gradient * self.derivative(self.inputs)


class Sigmoid(Activation):
    def __init__(self):

        """
        Initialize a Sigmoid activation layer.
        """

        super().__init__()

    @staticmethod
    def calculate(inputs):

        """
        Perform Sigmoid activation on inputs.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        return 1 / (1 + cp.exp(-inputs))

    @staticmethod
    def derivative(outputs):

        """
        Calculate derivative of Sigmoid activation.

        Args:
            outputs: Output data of shape (batch_size, input_size)
        Returns:
            Derivative of shape (batch_size, input_size)
        """

        return outputs * (1 - outputs)

    def forward(self, inputs):

        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        self.outputs = self.calculate(inputs)
        return self.outputs

    def backward(self, output_gradient):

        """
        Perform a backward pass through the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer, of shape (batch_size, input_size)
        Returns:
            Gradient of the loss with respect to the input of the layer, of shape (batch_size, input_size)
        """

        return output_gradient * self.derivative(self.outputs)


class Tanh(Activation):
    def __init__(self):

        """
        Initialize a Tanh activation layer.
        """

        super().__init__()

    @staticmethod
    def calculate(inputs):

        """
        Perform Tanh activation on inputs.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        return cp.tanh(inputs)

    @staticmethod
    def derivative(outputs):

        """
        Calculate derivative of Tanh activation.

        Args:
            outputs: Output data of shape (batch_size, input_size)
        Returns:
            Derivative of shape (batch_size, input_size)
        """

        return 1 - cp.square(outputs)

    def forward(self, inputs):

        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        self.outputs = self.calculate(inputs)
        return self.outputs

    def backward(self, output_gradient):

        """
        Perform a backward pass through the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer, of shape (batch_size, input_size)
        Returns:
            Gradient of the loss with respect to the input of the layer, of shape (batch_size, input_size)
        """

        return output_gradient * self.derivative(self.outputs)


class Softmax(Activation):
    def __init__(self):

        """
        Initialize a Softmax activation layer.
        """

        super().__init__()

    @staticmethod
    def calculate(inputs):

        """
        Perform Softmax activation on inputs.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        exps = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
        return exps / cp.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def derivative(outputs):

        """
        Calculate derivative of Softmax activation.

        Args:
            outputs: Output data of shape (batch_size, input_size)
        Returns:
            Derivative of shape (batch_size, input_size, input_size)
        """

        identity = cp.eye(outputs.shape[1])
        return outputs * (identity - cp.swapaxes(outputs, 1, 2))

    def forward(self, inputs):

        """
        Perform a forward pass through the layer.

        Args:
            inputs: Input data of shape (batch_size, input_size)
        Returns:
            Output of shape (batch_size, input_size)
        """

        self.outputs = self.calculate(inputs)
        return self.outputs

    def backward(self, output_gradient):

        """
        Perform a backward pass through the layer.

        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer, of shape (batch_size, input_size)
        Returns:
            Gradient of the loss with respect to the input of the layer, of shape (batch_size, input_size)
        """

        return self.derivative(self.outputs) @ output_gradient
