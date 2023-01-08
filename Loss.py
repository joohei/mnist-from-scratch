import cupy as cp


class Loss:
    def __init__(self):
        pass

    def calculate(self, outputs, expected_outputs):
        pass

    def derivative(self, outputs, expected_outputs):
        pass


class MeanSquaredError(Loss):
    def __init__(self):

        """
        Initialize the mean squared error loss function.
        """

        super().__init__()

    def calculate(self, outputs, expected_outputs):

        """
        Calculate the mean squared error loss.

        Args:
            outputs: Outputs of the model, of shape (batch_size, output_size)
            expected_outputs: Expected outputs, of shape (batch_size, output_size)
        Returns:
            Scalar mean squared error loss
        """

        return cp.mean(cp.square(expected_outputs - outputs))

    def derivative(self, outputs, expected_outputs):

        """
        Calculate the derivative of the mean squared error loss.

        Args:
            outputs: Outputs of the model, of shape (batch_size, output_size)
            expected_outputs: Expected outputs, of shape (batch_size, output_size)
        Returns:
            Derivative of the loss with respect to the outputs, of shape (batch_size, output_size)
        """

        return -2 * (expected_outputs - outputs)


class CrossEntropyLoss(Loss):
    def __init__(self, epsilon=1e-9):

        """
        Initialize the cross-entropy loss function.

        Args:
            epsilon: Small value to add to the output of the model to avoid taking the log of 0
        """

        super().__init__()
        self.epsilon = epsilon

    def calculate(self, outputs, expected_outputs):

        """
        Calculate the cross-entropy loss.

        Args:
            outputs: Outputs of the model, of shape (batch_size, output_size)
            expected_outputs: Expected outputs, of shape (batch_size, output_size)
        Returns:
            Scalar cross-entropy loss
        """

        return -cp.mean(cp.log(cp.clip(outputs[expected_outputs == 1], self.epsilon, 1 - self.epsilon)))

    def derivative(self, outputs, expected_outputs):

        """
        Calculate the derivative of the cross-entropy loss.

        Args:
            outputs: Outputs of the model, of shape (batch_size, output_size)
            expected_outputs: Expected outputs, of shape (batch_size, output_size)
        Returns:
            Derivative of the loss with respect to the outputs, of shape (batch_size, output_size)
        """

        return outputs - expected_outputs

