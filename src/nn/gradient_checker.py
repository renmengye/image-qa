import numpy as np

class GradientChecker():
    """
    Utility for checking gradient computation in a layer using finite
    difference.
    """
    def __init__(self, layer, tolerance=1e-4, epsilon=1e-5):
        """

        :param layer: a subclass instance of Layer
        :param tolerance: double, relative difference allowed between
        expected and
        actual gradient
        :param epsilon: double, finite difference step
        :return:
        """
        self._tolerance = tolerance
        self._epsilon = epsilon
        self._layer = layer

    def computeGradientToInput(self, inputValue):
        """
        Compute the gradient and numerical gradient w.r.t. the input
        vector evaluated at certain input value. Internally it uses a sum of
        squares cost function with target to be the zero vector.
        :param inputValue: numpy.ndarray or gnumpy.garray object, input value
        to the layer.
        :return: 2-tuple, gradient computed by the layer, and numerical
        gradient computed by finite difference, both in numpy.ndarray format.
        """
        outputValue = self._layer.forward(inputValue)
        gradientToOutput = -outputValue
        gradient = self._layer.backward(gradientToOutput)
        gradientNumerical = np.zeros(gradient.size)
        inputValueReshape = inputValue.reshape(inputValue.size)
        for i in range(inputValueReshape.size):
            inputValueReshape[i] += self._epsilon
            inputValueTmp = inputValueReshape.reshape(inputValue.shape)
            outputValueTmpPlus = self._layer.forward(inputValueTmp)
            lossPlus = .5 * np.sum(outputValueTmpPlus ** 2)
            inputValueReshape[i] -= 2 * self._epsilon
            inputValueTmp = inputValueReshape.reshape(inputValue.shape)
            outputValueTmpMinus = self._layer.forward(inputValueTmp)
            lossMinus = .5 * np.sum(outputValueTmpMinus ** 2)
            gradientNumerical[i] = (lossPlus - lossMinus)\
                                   / 2 / self._epsilon
        return gradient, gradientNumerical.reshape(gradient.shape)

    def computeGradientToWeight(self, inputValue):
        """
        Compute the gradient and numerical gradient w.r.t. the weight matrix
        evaluated at certain input value. Internally it uses a sum of squares
        cost function with target to be the zero vector.
        :param inputValue: numpy.ndarray or gnumpy.garray object, input value
        to the layer.
        :return: 2-tuple, gradient compute by the layer, and numerical
        gradient computed by finite difference, both in numpy.ndarray format.
        """
        outputValue = self._layer.forward(inputValue)
        gradientToOutput = -outputValue
        self._layer.backward(gradientToOutput)
        self._layer.weight.update()
        gradient = self._layer.weight.getGradient()
        print gradient
        gradientNumerical = np.zeros(gradient.size)
        weight = self._layer.weight.get()
        weightReshape = weight.reshape(weight.size)
        for i in range(weightReshape.size):
            weightReshape[i] += self._epsilon
            weightTmp = weightReshape.reshape(weight.shape)
            self._layer.weight.set(weightTmp)
            outputValueTmpPlus = self._layer.forward(inputValue)
            lossPlus = .5 * np.sum(outputValueTmpPlus ** 2)
            weightReshape[i] -= 2 * self._epsilon
            weightTmp = weightReshape.reshape(weight.shape)
            self._layer.weight.set(weightTmp)
            outputValueTmpMinus = self._layer.forward(inputValue)
            lossMinus = .5 * np.sum(outputValueTmpMinus ** 2)
            gradientNumerical[i] = (lossPlus - lossMinus)\
                                   / 2 / self._epsilon
        return gradient, gradientNumerical.reshape(gradient.shape)

    def checkGradient(self, testClass, gradient, gradientNumerical):
        """
        Check if the gradient and numerical gradient are similar enough.
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param gradient: numpy.ndarray, gradient computed by the layer
        :param gradientNumerical: numpy.ndarray, gradient compute by finite
        difference.
        :return:
        """
        print gradient / gradientNumerical
        gradientNumerical = gradientNumerical.reshape(gradient.size)
        gradient = gradient.reshape(gradient.size)
        for i in range(gradient.size):
            testClass.assertTrue(
                (gradient[i] == 0 and gradientNumerical[i] == 0) or
                (np.abs(gradient[i] / gradientNumerical[i] - 1) <
                 self._tolerance))

    def runInput(self, testClass, inputValue):
        """
        Compute the gradient w.r.t. the input value evaluated at certain input
        value and check the correctness
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param inputValue: numpy.ndarray or gnumpy.garray, input value to the
        layer.
        :return:
        """
        grd, grdNum = self.computeGradientToInput(inputValue)
        self.checkGradient(testClass, grd, grdNum)

    def runWeight(self, testClass, inputValue):
        """
        Compute the gradient w.r.t. the weight matrix evaluated at certain
        input value and check the correctness.
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param inputValue: numpy.ndarray or gnumpy.garray, input value to the
        layer.
        :return:
        """
        grd, grdNum = self.computeGradientToWeight(inputValue)
        self.checkGradient(testClass, grd, grdNum)

    def runAll(self, testClass, inputValue):
        """
        Compute the gradient w.r.t. the weight matrix and the input value
        evaluated at certain input value and check the correctness.
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param inputValue: numpy.ndarray or gnumpy.garray, input value to the layer
        :return:
        """
        self.runInput(testClass, inputValue)
        self.runWeight(testClass, inputValue)